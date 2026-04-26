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

import difflib
import re
from pathlib import Path
from typing import Iterable

import networkx as nx

# Default on-disk location for the persisted graph. Lives under db/ next to
# the ChromaDB vector store so both retrieval indexes share one ignored dir.
DEFAULT_GRAPH_PATH = Path("./db/graph.graphml")

# Cutoff for difflib.get_close_matches when fuzzy-resolving an entity against
# existing graph nodes. SequenceMatcher ratio: 1.0 = identical, 0.0 = nothing
# in common. 0.85 catches one-character typos in long names ("Maryland" vs
# "Mariland" — ratio ~0.875) without merging unrelated short tokens (e.g.
# "python" vs "panda" — ratio ~0.55). Tune up if false merges appear, down
# if real duplicates are slipping through.
_ENTITY_FUZZY_CUTOFF = 0.85

# Punctuation/separators that should be treated as word boundaries during
# normalization. Underscores, hyphens, and dots are the common ones LLMs
# mix in ("machine_learning", "machine-learning", "Machine.Learning").
_SEPARATOR_RE = re.compile(r"[_\-\.]+")

# Anything that's not a word char or whitespace gets stripped after the
# separators are converted to spaces. This handles trailing punctuation
# like "Atharv!" or "I-20."
_PUNCTUATION_RE = re.compile(r"[^\w\s]")

# Collapses runs of whitespace (including newlines and tabs) to a single
# space — applied last so the output is canonical.
_WHITESPACE_RE = re.compile(r"\s+")


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
    # Entity resolution (Ticket 17)
    # ------------------------------------------------------------------
    # The LLM doesn't emit consistent entity names. The same concept comes
    # back as "Machine Learning", "machine_learning", "machine-learning",
    # and occasionally "Mariland" (an OCR slip on "Maryland"). Without
    # resolution, each variant becomes its own node and the graph fragments
    # — graph walks miss connected facts and the Brain Map is unreadable.
    #
    # Strategy: aggressively normalize each entity (lowercase, separators
    # to spaces, strip punctuation, collapse whitespace) to a lookup key,
    # then check existing nodes for an exact match on that key. If none,
    # use difflib's fuzzy match to catch single-character typos. The first
    # display form wins and stays as the visible node id.

    def _normalize_entity(self, entity_str: str) -> str:
        """Aggressively normalize an entity string into a lookup key.

        Lowercases, converts underscores/hyphens/dots to spaces, strips
        remaining punctuation, and collapses runs of whitespace. Two strings
        that share the same normalized form are treated as the same entity.

        The output is NEVER stored as a node id — it's only used as the key
        for lookup against existing nodes' normalized forms. Display labels
        keep their original casing and punctuation.
        """
        s = entity_str.lower().strip()
        s = _SEPARATOR_RE.sub(" ", s)
        s = _PUNCTUATION_RE.sub("", s)
        s = _WHITESPACE_RE.sub(" ", s).strip()
        return s

    def _canonicalize(self, entity_str: str) -> str:
        """Resolve an entity to the canonical node id it should be added under.

        Two-step lookup:
          1. Exact normalized match — handles case and separator variants.
          2. difflib fuzzy match — handles one-character typos in longer
             names without merging unrelated short tokens.

        If no existing node is similar enough, returns the input unchanged
        so it becomes a new canonical for future variants.
        """
        normalized = self._normalize_entity(entity_str)
        # Empty/whitespace-only entities have no resolution target. Return
        # the original; downstream validation in add_triplets will reject it.
        if not normalized:
            return entity_str

        # normalized_form -> existing display id. Built fresh each call so
        # nodes added earlier in the same add_triplets pass are visible.
        existing = {self._normalize_entity(n): n for n in self.graph.nodes}

        # Step 1: exact post-normalization hit ("machine_learning" finds an
        # existing "Machine Learning" because both normalize to the same key).
        if normalized in existing:
            return existing[normalized]

        # Step 2: fuzzy fallback. n=1 -> only the closest match. cutoff
        # tuned to admit one-char diffs in 8+ char strings while rejecting
        # short-token coincidences. Pure exact-match would miss OCR typos.
        matches = difflib.get_close_matches(
            normalized,
            list(existing.keys()),
            n=1,
            cutoff=_ENTITY_FUZZY_CUTOFF,
        )
        if matches:
            return existing[matches[0]]

        # Nothing similar enough — this becomes a new canonical node.
        return entity_str

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_triplets(self, triplets: Iterable) -> None:
        """Insert triplets as directed, labeled edges.

        Each triplet should be a dict with "source", "target", and
        "relationship" keys — the format produced by extract_graph_triples.
        Malformed entries are silently skipped so a single noisy LLM output
        doesn't derail an ingestion batch.

        Both endpoints are run through entity resolution before the edge is
        added: visually-similar variants ("Machine Learning" and
        "machine_learning") collapse to a single node, the first display
        form encountered wins, and one-character typos are merged via
        difflib fuzzy matching.
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

            # Resolve both endpoints to canonical node ids. Done before the
            # add_edge call so the second variant of an entity reuses the
            # first variant's node instead of creating a duplicate.
            source = self._canonicalize(source)
            target = self._canonicalize(target)

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
