"""
Knowledge Graph Visualization Module
------------------------------------
Renders the persisted graph.graphml as an interactive HTML page using
PyVis. Used by the Streamlit "Brain Map" tab so the user can pan, zoom,
and drag entity nodes to explore what the corpus knows.

PyVis wraps vis-network and produces a self-contained HTML document we
can hand to st.components.v1.html. Physics is enabled so the layout
auto-arranges on first paint and nodes stay draggable afterward.
"""

from pathlib import Path

import networkx as nx
from pyvis.network import Network


# Canvas size for the embedded graph. 750px is tall enough to show a
# multi-cluster graph without forcing the user to scroll inside the iframe.
DEFAULT_HEIGHT = "750px"
DEFAULT_WIDTH = "100%"

# Shown when the graph file is missing or the graph is empty. Keeps the
# Brain Map tab from looking broken before the user has run ingestion.
_PLACEHOLDER_HTML = (
    "<div style='padding:2em;font-family:sans-serif;color:#bbb;'>"
    "No knowledge graph to display yet. Ingest documents from the sidebar "
    "to populate the Brain Map."
    "</div>"
)


def build_network(graph: nx.DiGraph) -> Network:
    """Build a configured PyVis Network from a NetworkX DiGraph.

    Each NX node becomes a PyVis node; each edge carries its `relationship`
    attribute as both the visible label and the hover title so the user
    can read what connects two entities at a glance. Physics is explicitly
    toggled on so the layout settles on its own and nodes are draggable.
    """
    net = Network(
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        directed=True,
        bgcolor="#1e1e1e",
        font_color="white",
        # in_line embeds vis-network's JS/CSS directly in the HTML so the
        # output is self-contained — important for our offline-first stack
        # and for st.components.v1.html, which can't resolve relative URLs.
        cdn_resources="in_line",
    )
    # Force physics on. PyVis defaults to enabled, but we set it explicitly
    # so this contract is loud and survives any future default change.
    net.toggle_physics(True)

    for node in graph.nodes:
        # Cast to str because GraphML occasionally rehydrates nodes as ints
        # or other types depending on attribute hints in the XML.
        name = str(node)
        net.add_node(name, label=name)

    for source, target, data in graph.edges(data=True):
        # Empty/missing relationship is unusual but possible if a triplet
        # was added without the attribute; render with no label rather than
        # crashing the whole graph.
        rel = data.get("relationship", "") or ""
        net.add_edge(str(source), str(target), label=rel, title=rel)

    return net


def render_graph_html(graph_path: Path | str) -> str:
    """Load a persisted graph.graphml and return interactive HTML.

    Returns placeholder HTML when the file is missing or the graph has no
    nodes. The Brain Map tab is rendered on every app load — including
    before ingestion — so we must never raise on absent state.
    """
    path = Path(graph_path)
    if not path.exists():
        return _PLACEHOLDER_HTML

    graph = nx.read_graphml(path)
    # read_graphml may return a Graph or DiGraph based on the file header.
    # Coerce so direction is preserved through the conversion.
    if not isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)

    if graph.number_of_nodes() == 0:
        return _PLACEHOLDER_HTML

    net = build_network(graph)
    # notebook=False -> standalone HTML suitable for st.components.v1.html
    return net.generate_html(notebook=False)
