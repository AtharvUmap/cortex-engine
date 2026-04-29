import re
from pathlib import Path

import networkx as nx

from src.visualize import (
    build_network,
    color_for_degree,
    render_graph_html,
    size_for_degree,
)


_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


# --- Test helpers ---

def _write_graph(path: Path, edges: list[tuple[str, str, str]]) -> None:
    """Build a tiny DiGraph with the given (source, target, relationship)
    edges and persist it as GraphML at `path`. Mirrors the shape produced
    by graph_extractor + GraphStore so render_graph_html sees realistic input.
    """
    g = nx.DiGraph()
    for source, target, relationship in edges:
        g.add_edge(source, target, relationship=relationship)
    nx.write_graphml(g, path)


# --- render_graph_html ---

def test_render_graph_html_returns_html_string(tmp_path):
    """A graphml file with edges should render to a non-empty HTML document."""
    graph_path = tmp_path / "graph.graphml"
    _write_graph(graph_path, [("Atharv", "Maryland", "attends")])

    html = render_graph_html(graph_path)

    assert isinstance(html, str)
    assert len(html) > 0
    # PyVis emits a full HTML document — at minimum it should look like one.
    lower = html.lower()
    assert "<html" in lower or "<!doctype" in lower


def test_render_graph_html_contains_node_names(tmp_path):
    """Node labels should be embedded in the rendered HTML so the user can read them."""
    graph_path = tmp_path / "graph.graphml"
    _write_graph(graph_path, [("Atharv", "Maryland", "attends")])

    html = render_graph_html(graph_path)

    assert "Atharv" in html
    assert "Maryland" in html


def test_render_graph_html_contains_relationship_label(tmp_path):
    """Edge `relationship` attribute should appear in the rendered HTML so the
    user can see how two entities are connected without hovering."""
    graph_path = tmp_path / "graph.graphml"
    _write_graph(graph_path, [("Atharv", "Maryland", "attends")])

    html = render_graph_html(graph_path)

    assert "attends" in html


def test_render_graph_html_handles_missing_file(tmp_path):
    """Missing graph file should return placeholder HTML, not raise.
    The Brain Map tab is rendered before any ingestion has happened, so this
    is the cold-start case."""
    graph_path = tmp_path / "does_not_exist.graphml"

    html = render_graph_html(graph_path)

    assert isinstance(html, str)
    assert len(html) > 0


def test_render_graph_html_handles_empty_graph(tmp_path):
    """A persisted-but-empty graph should yield placeholder HTML, not raise.
    PyVis can choke on a zero-node network, so we short-circuit instead."""
    graph_path = tmp_path / "empty.graphml"
    nx.write_graphml(nx.DiGraph(), graph_path)

    html = render_graph_html(graph_path)

    assert isinstance(html, str)
    assert len(html) > 0


def test_render_graph_html_accepts_string_path(tmp_path):
    """The function should accept a str path as well as a Path — app.py passes
    a hardcoded string."""
    graph_path = tmp_path / "graph.graphml"
    _write_graph(graph_path, [("A", "B", "rel")])

    html = render_graph_html(str(graph_path))

    assert isinstance(html, str)
    assert "A" in html and "B" in html


# --- build_network ---

def test_build_network_enables_physics():
    """Physics must be on so the user can drag nodes around the canvas — that's
    the whole point of using PyVis over a static image."""
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship="related_to")

    net = build_network(g)

    assert net.options.physics.enabled is True


def test_build_network_includes_all_nodes():
    """Every NX node must be carried into the PyVis network."""
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship="r1")
    g.add_edge("B", "C", relationship="r2")

    net = build_network(g)

    node_ids = {node["id"] for node in net.nodes}
    assert node_ids == {"A", "B", "C"}


def test_build_network_includes_all_edges():
    """Every NX edge must be carried into the PyVis network."""
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship="r1")
    g.add_edge("B", "C", relationship="r2")

    net = build_network(g)

    assert len(net.edges) == 2


def test_build_network_edges_carry_relationship_label():
    """The `relationship` attribute on each NX edge should become the visible
    PyVis edge label."""
    g = nx.DiGraph()
    g.add_edge("Atharv", "Maryland", relationship="attends")

    net = build_network(g)

    edge = net.edges[0]
    # PyVis stores edges as dicts with 'from'/'to' and label/title fields.
    assert edge.get("label") == "attends" or edge.get("title") == "attends"


# --- color_for_degree (Ticket 27 — degree-gradient styling) ---------------
# The Brain Map colors and sizes nodes by degree so hubs stand out from
# leaves. Both helpers are pure functions of (degree, max_degree) so they
# can be unit-tested deterministically; the resulting visual is verified
# by eyeballing the Streamlit app.


def test_color_for_degree_returns_hex_string():
    for degree in (0, 1, 5, 10):
        result = color_for_degree(degree, max_degree=10)
        assert _HEX.match(result), f"expected hex, got {result!r}"


def test_color_for_degree_endpoints_differ():
    """The whole point of a gradient — leaf and hub must not be the same color."""
    leaf = color_for_degree(0, max_degree=10)
    hub = color_for_degree(10, max_degree=10)
    assert leaf != hub


def test_color_for_degree_midpoint_distinct_from_endpoints():
    leaf = color_for_degree(0, max_degree=10)
    mid = color_for_degree(5, max_degree=10)
    hub = color_for_degree(10, max_degree=10)
    assert mid != leaf
    assert mid != hub


def test_color_for_degree_max_zero_does_not_crash():
    """Edge case: a graph with no edges — every node has degree 0 and
    max_degree=0. Must not divide by zero."""
    assert _HEX.match(color_for_degree(0, max_degree=0))


def test_color_for_degree_clamps_when_degree_exceeds_max():
    """Defensive: if degree > max_degree somehow, return a valid hex string
    (the hub endpoint is the natural clamp)."""
    hub = color_for_degree(10, max_degree=10)
    over = color_for_degree(20, max_degree=10)
    assert _HEX.match(over)
    assert over == hub


# --- size_for_degree -----------------------------------------------------


def test_size_for_degree_returns_numeric():
    """PyVis expects a number for the size attribute."""
    result = size_for_degree(5, max_degree=10)
    assert isinstance(result, (int, float))


def test_size_for_degree_hub_larger_than_leaf():
    leaf = size_for_degree(0, max_degree=10)
    hub = size_for_degree(10, max_degree=10)
    assert hub > leaf


def test_size_for_degree_monotone_increasing():
    sizes = [size_for_degree(d, max_degree=10) for d in range(11)]
    assert sizes == sorted(sizes)


def test_size_for_degree_max_zero_does_not_crash():
    """Graph with no edges — falls back to base size, doesn't divide by zero."""
    result = size_for_degree(0, max_degree=0)
    assert isinstance(result, (int, float))
    assert result > 0


def test_size_for_degree_leaf_size_visible():
    """Even single-edge nodes need to render at a readable size — the gradient
    shouldn't squash leaves down to invisible."""
    leaf = size_for_degree(0, max_degree=20)
    assert leaf >= 10


def test_size_for_degree_clamps_when_degree_exceeds_max():
    """Defensive: out-of-range input shouldn't produce gigantic nodes."""
    hub = size_for_degree(10, max_degree=10)
    over = size_for_degree(20, max_degree=10)
    assert over <= hub
