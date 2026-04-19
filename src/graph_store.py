"""
Knowledge Graph Storage Module
-------------------------------
Manages the local NetworkX knowledge graph that sits alongside the
Parent-Child vector store. Each triplet extracted by graph_extractor.py
becomes a directed edge between two entity nodes, labeled with the
relationship verb.

The graph is persisted as GraphML (XML-based, widely readable) so it can
be inspected outside Python (e.g. with Gephi) and doesn't require a
database service.

Typical lifecycle:
    store = GraphStore()                   # loads ./db/graph.graphml if present
    store.add_triplets(llm_triplets)       # accumulate facts
    store.save()                           # persist to disk
    facts = store.get_neighborhood("Atharv", depth=2)  # retrieval at query time
"""

from pathlib import Path
from typing import Iterable

import networkx as nx

# Default on-disk location for the persisted graph. Lives under db/ next to
# the ChromaDB vector store so both retrieval indexes share one ignored dir.
DEFAULT_GRAPH_PATH = Path("./db/graph.graphml")


class GraphStore:
    """A thin wrapper around networkx.DiGraph with GraphML persistence.

    The store auto-loads the graph from disk on construction if the file
    exists; otherwise it starts empty. Call save() explicitly after mutation
    so ingestion pipelines control when I/O happens.
    """

    def __init__(self, graph_path: Path | str = DEFAULT_GRAPH_PATH):
        # Accept either Path or str for convenience; normalize internally.
        self.graph_path = Path(graph_path)

        # Ensure the parent directory exists so save() won't need a mkdir.
        # Safe to call even when the dir already exists.
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)

        # DiGraph preserves the direction of each relationship. Using an
        # undirected Graph would collapse ("Maryland located_in USA") and
        # ("USA located_in Maryland") into the same edge.
        self.graph: nx.DiGraph = nx.DiGraph()

        # Rehydrate from disk if a prior run persisted something.
        if self.graph_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_triplets(self, triplets: Iterable) -> None:
        """Insert triplets as directed, labeled edges.

        Each triplet should be a dict with "source", "target", and
        "relationship" keys — the format produced by extract_graph_triples.
        Malformed entries are silently skipped so a single noisy LLM output
        doesn't derail an ingestion batch.
        """
        for triplet in triplets:
            if not isinstance(triplet, dict):
                continue

            source = triplet.get("source")
            target = triplet.get("target")
            relationship = triplet.get("relationship")

            # All three fields must be non-empty strings for the edge to be
            # meaningful. Skip anything missing a piece.
            if not (source and target and relationship):
                continue
            if not all(isinstance(v, str) for v in (source, target, relationship)):
                continue

            # add_edge is idempotent — calling it twice with the same pair
            # just overwrites the edge attributes, it doesn't duplicate.
            self.graph.add_edge(source, target, relationship=relationship)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Write the graph to disk as GraphML."""
        nx.write_graphml(self.graph, self.graph_path)

    def _load(self) -> None:
        """Read the graph from disk. Called automatically in __init__."""
        loaded = nx.read_graphml(self.graph_path)
        # read_graphml can return a Graph or DiGraph depending on the file
        # header. Coerce to DiGraph so downstream code can rely on direction.
        if not isinstance(loaded, nx.DiGraph):
            loaded = nx.DiGraph(loaded)
        self.graph = loaded

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_neighborhood(self, entity: str, depth: int = 2) -> list[dict]:
        """Return every triplet reachable within `depth` hops of `entity`.

        Direction is ignored for traversal (we want to find related facts
        regardless of which side of the edge the entity appears on), but
        the returned triplets preserve the original edge direction.

        Args:
            entity: The node to start the walk from.
            depth: Maximum number of hops away to include. Defaults to 2.

        Returns:
            A list of {"source", "target", "relationship"} dicts. Empty list
            if the entity is not in the graph.
        """
        if entity not in self.graph:
            return []

        # ego_graph walks outward from the node along both in- and out-edges
        # (undirected=True) up to the given radius, returning the induced
        # subgraph. This pulls in second-hop nodes like USA (reached via
        # Maryland) and DataScience (reached via Python).
        subgraph = nx.ego_graph(
            self.graph,
            entity,
            radius=depth,
            undirected=True,
        )

        # Re-emit every directed edge in the subgraph as a triplet dict so
        # callers get back the same shape they put in via add_triplets.
        return [
            {
                "source": u,
                "target": v,
                "relationship": data.get("relationship", ""),
            }
            for u, v, data in subgraph.edges(data=True)
        ]
