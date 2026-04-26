from pathlib import Path

import networkx as nx

from src.graph_store import GraphStore


# --- Test helpers ---

def _sample_triplets():
    """A small connected set of facts we can use across tests.

    Graph shape:
        Atharv --attends--> Maryland --located_in--> USA
        Atharv --knows--> Python --used_for--> DataScience
        Maryland --has--> ComputerScience
    """
    return [
        {"source": "Atharv", "target": "Maryland", "relationship": "attends"},
        {"source": "Maryland", "target": "USA", "relationship": "located_in"},
        {"source": "Atharv", "target": "Python", "relationship": "knows"},
        {"source": "Python", "target": "DataScience", "relationship": "used_for"},
        {"source": "Maryland", "target": "ComputerScience", "relationship": "has"},
    ]


# --- Tests ---

def test_add_triplets_creates_nodes_and_edges(tmp_path):
    """add_triplets should insert both endpoints as nodes and a labeled edge."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets(_sample_triplets())

    # All five unique entities should be nodes in the graph
    expected_nodes = {"Atharv", "Maryland", "USA", "Python", "DataScience", "ComputerScience"}
    assert expected_nodes.issubset(set(store.graph.nodes()))

    # Directed edges should exist and carry the relationship label
    assert store.graph.has_edge("Atharv", "Maryland")
    assert store.graph["Atharv"]["Maryland"]["relationship"] == "attends"
    assert store.graph.has_edge("Maryland", "USA")
    assert store.graph["Maryland"]["USA"]["relationship"] == "located_in"


def test_add_triplets_is_idempotent(tmp_path):
    """Adding the same triplet twice should not duplicate edges."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    triplet = [{"source": "A", "target": "B", "relationship": "knows"}]

    store.add_triplets(triplet)
    store.add_triplets(triplet)

    assert store.graph.number_of_edges() == 1
    assert store.graph.number_of_nodes() == 2


def test_add_triplets_ignores_malformed_entries(tmp_path):
    """Triplets missing keys should be skipped, not crash the method."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets([
        {"source": "A", "target": "B", "relationship": "knows"},
        {"source": "C", "target": "D"},  # missing relationship
        {"relationship": "likes"},  # missing both endpoints
        "not a dict at all",
    ])

    assert store.graph.number_of_edges() == 1


def test_save_writes_graphml_file(tmp_path):
    """save() should persist the graph to the configured path as GraphML."""
    graph_path = tmp_path / "graph.graphml"
    store = GraphStore(graph_path=graph_path)
    store.add_triplets(_sample_triplets())

    store.save()

    assert graph_path.exists()
    # GraphML is XML-based — a quick sanity check that the file isn't empty
    assert graph_path.stat().st_size > 0


def test_load_restores_graph_from_disk(tmp_path):
    """A new GraphStore pointed at a saved file should see the same nodes/edges."""
    graph_path = tmp_path / "graph.graphml"

    # Build and persist
    writer = GraphStore(graph_path=graph_path)
    writer.add_triplets(_sample_triplets())
    writer.save()

    # Fresh instance should auto-load what the writer persisted
    reader = GraphStore(graph_path=graph_path)

    assert reader.graph.has_edge("Atharv", "Maryland")
    assert reader.graph["Atharv"]["Maryland"]["relationship"] == "attends"
    assert reader.graph.number_of_nodes() == writer.graph.number_of_nodes()
    assert reader.graph.number_of_edges() == writer.graph.number_of_edges()


def test_load_on_missing_file_starts_empty(tmp_path):
    """Pointing at a path with no existing file should give an empty graph, not error."""
    store = GraphStore(graph_path=tmp_path / "does_not_exist.graphml")
    assert store.graph.number_of_nodes() == 0
    assert store.graph.number_of_edges() == 0


def test_get_neighborhood_returns_one_hop_facts(tmp_path):
    """depth=1 should return only directly connected triplets."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets(_sample_triplets())

    facts = store.get_neighborhood("Atharv", depth=1)

    # Atharv has two direct outgoing edges: attends->Maryland, knows->Python
    assert any(
        f["source"] == "Atharv" and f["target"] == "Maryland" and f["relationship"] == "attends"
        for f in facts
    )
    assert any(
        f["source"] == "Atharv" and f["target"] == "Python" and f["relationship"] == "knows"
        for f in facts
    )
    # At depth 1, the second-hop facts like Maryland->USA should NOT be included
    assert not any(f["source"] == "Maryland" and f["target"] == "USA" for f in facts)


def test_get_neighborhood_returns_two_hop_facts(tmp_path):
    """depth=2 should include facts reachable within two hops."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets(_sample_triplets())

    facts = store.get_neighborhood("Atharv", depth=2)

    # Two-hop facts via Maryland and via Python should now appear
    assert any(
        f["source"] == "Maryland" and f["target"] == "USA" and f["relationship"] == "located_in"
        for f in facts
    )
    assert any(
        f["source"] == "Python" and f["target"] == "DataScience" and f["relationship"] == "used_for"
        for f in facts
    )


def test_get_neighborhood_for_missing_entity_returns_empty(tmp_path):
    """Querying an unknown entity should return [] rather than raising."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets(_sample_triplets())

    facts = store.get_neighborhood("NonexistentPerson", depth=2)

    assert facts == []


def test_graph_is_directed(tmp_path):
    """GraphStore must use a DiGraph so relationship direction is preserved."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    assert isinstance(store.graph, nx.DiGraph)


# --- Entity Resolution (Ticket 17) -----------------------------------------
# add_triplets must consolidate visually-similar entities into a single node
# so the graph doesn't fragment when the LLM emits "Machine Learning" in one
# triple and "machine_learning" in another.

def test_normalize_entity_lowercases(tmp_path):
    """Basic case-folding: capitalization should not split a node."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    assert store._normalize_entity("Atharv Umap") == "atharv umap"


def test_normalize_entity_treats_underscore_as_space(tmp_path):
    """Underscores and spaces must produce the same normalized form so
    'machine_learning' and 'Machine Learning' resolve to the same node."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    assert store._normalize_entity("machine_learning") == store._normalize_entity("Machine Learning")


def test_normalize_entity_collapses_separators_and_whitespace(tmp_path):
    """Hyphens, dots, and runs of whitespace all reduce to single spaces.
    Different transcriptions of the same multi-word phrase must agree."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    norm = store._normalize_entity
    assert norm("machine-learning") == norm("machine_learning") == norm("Machine   Learning")


def test_normalize_entity_strips_punctuation(tmp_path):
    """Trailing punctuation shouldn't fragment an otherwise identical entity."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    assert store._normalize_entity("Atharv!") == store._normalize_entity("atharv")


def test_add_triplets_collapses_case_variants(tmp_path):
    """'Machine Learning' and 'machine learning' must collapse to one node."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets([
        {"source": "Machine Learning", "target": "AI", "relationship": "is_a"},
        {"source": "machine learning", "target": "AI", "relationship": "is_a"},
    ])

    # 2 unique entities total (ML + AI), not 3
    assert store.graph.number_of_nodes() == 2


def test_add_triplets_collapses_separator_variants(tmp_path):
    """'Machine Learning' and 'machine_learning' must collapse to one node."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets([
        {"source": "Machine Learning", "target": "AI", "relationship": "is_a"},
        {"source": "machine_learning", "target": "AI", "relationship": "is_a"},
    ])

    assert store.graph.number_of_nodes() == 2


def test_add_triplets_collapses_typos_via_fuzzy_match(tmp_path):
    """A one-character typo in a long entity name should fuzzy-match the
    original. Catches real-world OCR slips like 'Mariland' vs 'Maryland'."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets([
        {"source": "Atharv", "target": "University of Maryland", "relationship": "attends"},
        {"source": "Atharv", "target": "University of Mariland", "relationship": "attends"},
    ])

    # Atharv + 1 university (not 2)
    assert store.graph.number_of_nodes() == 2


def test_add_triplets_keeps_distinct_entities_separate(tmp_path):
    """Entities that aren't visually similar must NOT be merged. Otherwise
    the fuzzy threshold is too loose and unrelated facts cluster together."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets([
        {"source": "Atharv Umap", "target": "Maryland", "relationship": "attends"},
        {"source": "Bijoya Chakraborty", "target": "Maryland", "relationship": "lives_in"},
    ])

    # Two distinct people + one university
    assert store.graph.number_of_nodes() == 3
    assert "Atharv Umap" in store.graph
    assert "Bijoya Chakraborty" in store.graph


def test_add_triplets_preserves_first_seen_display_form(tmp_path):
    """The first variant encountered remains as the visible node id, so the
    Brain Map shows readable labels ('Machine Learning') instead of the
    aggressively-normalized form ('machine learning')."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets([
        {"source": "Machine Learning", "target": "AI", "relationship": "is_a"},
        {"source": "machine_learning", "target": "Data Science", "relationship": "underlies"},
    ])

    assert "Machine Learning" in store.graph.nodes
    assert "machine_learning" not in store.graph.nodes


def test_add_triplets_normalizes_target_entities_too(tmp_path):
    """Target side must go through the same normalizer as source side —
    otherwise an entity used only as a target would still fragment."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets([
        {"source": "Atharv", "target": "Machine Learning", "relationship": "studies"},
        {"source": "Atharv", "target": "machine_learning", "relationship": "uses"},
    ])

    # Atharv + 1 ML node (not 2)
    assert store.graph.number_of_nodes() == 2


def test_add_triplets_resolves_within_a_single_call(tmp_path):
    """Variants in the SAME add_triplets call must also collapse — not just
    variants seen across separate calls."""
    store = GraphStore(graph_path=tmp_path / "graph.graphml")
    store.add_triplets([
        {"source": "Machine Learning", "target": "AI", "relationship": "is_a"},
        {"source": "MACHINE_LEARNING", "target": "Data Science", "relationship": "underlies"},
        {"source": "machine learning", "target": "Statistics", "relationship": "uses"},
    ])

    # 4 nodes: Machine Learning, AI, Data Science, Statistics — NOT 6
    assert store.graph.number_of_nodes() == 4
