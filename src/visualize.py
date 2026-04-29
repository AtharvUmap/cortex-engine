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


# --- Degree-gradient styling ---------------------------------------------
# Color and size both interpolate by degree/max_degree. The graph schema
# carries no entity-type attribute on nodes, so coloring by type isn't
# implementable without a separate extraction pass; coloring by degree
# delivers visual hierarchy (hubs vs leaves) using only what we have.

# Endpoints of the cyan-ish gradient. Tuned against the dark canvas
# background (#1e1e1e) — leaf is muted blue-gray, hub is saturated teal.
_LEAF_RGB = (94, 124, 146)    # #5e7c92
_HUB_RGB = (78, 201, 176)     # #4ec9b0

# Node size endpoints in PyVis pixels. Leaf size is kept high enough that
# single-edge nodes don't disappear; hub size is roughly 2.5x leaf so the
# hierarchy is obvious at a glance.
_LEAF_SIZE = 14
_HUB_SIZE = 38


def _ratio(degree: int, max_degree: int) -> float:
    """Clamp degree/max_degree into [0, 1]. Zero max → zero ratio (leaf).

    Defensive against both edge cases the gradient cares about: an empty
    graph (max=0, divide-by-zero) and a caller passing a degree that
    exceeds max (rare but possible during partial graph construction).
    """
    if max_degree <= 0:
        return 0.0
    return max(0.0, min(1.0, degree / max_degree))


def color_for_degree(degree: int, max_degree: int) -> str:
    """Hex color for a node, interpolated along the leaf→hub gradient.

    Degree 0 returns the leaf endpoint; max_degree returns the hub endpoint.
    Returns lowercase #rrggbb so PyVis accepts it directly.
    """
    t = _ratio(degree, max_degree)
    r = round(_LEAF_RGB[0] + (_HUB_RGB[0] - _LEAF_RGB[0]) * t)
    g = round(_LEAF_RGB[1] + (_HUB_RGB[1] - _LEAF_RGB[1]) * t)
    b = round(_LEAF_RGB[2] + (_HUB_RGB[2] - _LEAF_RGB[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def size_for_degree(degree: int, max_degree: int) -> int:
    """Node size in PyVis pixels, interpolated along the leaf→hub range."""
    t = _ratio(degree, max_degree)
    return round(_LEAF_SIZE + (_HUB_SIZE - _LEAF_SIZE) * t)


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
        # Match the page's secondaryBackgroundColor so the graph blends into
        # the surrounding UI instead of looking like an embedded panel.
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

    # Tune barnesHut so dense sub-clusters spread out instead of bunching
    # on top of each other. Defaults are gravity=-2000, central_gravity=0.3,
    # spring_length=95 — that bunches up tightly when a single hub has
    # 6+ leaves. Stronger repulsion + weaker central pull + longer spring
    # gives every cluster room to breathe.
    net.barnes_hut(
        gravity=-3800,
        central_gravity=0.26,
        spring_length=120,
        spring_strength=0.04,
        damping=0.09,
    )

    # Compute the most-connected node's degree once so every per-node
    # interpolation shares the same scale. default=0 keeps the call safe
    # on graphs with no edges.
    max_degree = max((deg for _, deg in graph.degree()), default=0)

    # Soft teal halos on every node + edge so the graph reads as the
    # hero element of the Brain Map tab. Applied per-element via add_node
    # / add_edge kwargs because PyVis's set_options() corrupts the
    # Options object (turns it into a plain dict, breaking attribute
    # access used by toggle_physics and the existing tests).
    node_shadow = {
        "enabled": True,
        "color": "rgba(78,201,176,0.55)",
        "size": 14,
        "x": 0,
        "y": 0,
    }
    edge_shadow = {
        "enabled": True,
        "color": "rgba(78,201,176,0.18)",
        "size": 4,
        "x": 0,
        "y": 0,
    }

    for node in graph.nodes:
        # Cast to str because GraphML occasionally rehydrates nodes as ints
        # or other types depending on attribute hints in the XML.
        name = str(node)
        degree = graph.degree(node)
        net.add_node(
            name,
            label=name,
            color=color_for_degree(degree, max_degree),
            size=size_for_degree(degree, max_degree),
            shadow=node_shadow,
        )

    for source, target, data in graph.edges(data=True):
        # Empty/missing relationship is unusual but possible if a triplet
        # was added without the attribute; render with no label rather than
        # crashing the whole graph.
        rel = data.get("relationship", "") or ""
        net.add_edge(
            str(source),
            str(target),
            label=rel,
            title=rel,
            shadow=edge_shadow,
        )

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
