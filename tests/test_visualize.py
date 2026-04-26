from pathlib import Path

import networkx as nx

from src.visualize import build_network, render_graph_html


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
