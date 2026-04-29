"""Unit tests for src/index_status — sidebar status helper.

The sidebar displays four numbers (last-ingested timestamp, document count,
entity count, fact count) read from existing on-disk persistence. The
helper is pure I/O — no Ollama, no embedding — so it's worth testing
deterministically rather than eyeballing through the Streamlit app.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import pytest

from src.index_status import format_status_line, read_index_status


def _write_facts(path: Path, by_field: dict) -> None:
    """Helper: persist a {field: [(entity, value, source), ...]} dict in the
    same shape src/factual_index.py::save_facts uses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {f: [list(e) for e in entries] for f, entries in by_field.items()}
    path.write_text(json.dumps(serializable))


def _write_owners(path: Path, sources: list) -> None:
    """Helper: persist owners.json with one entry per source path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {src: {"owner": "x", "confidence": "high"} for src in sources}
    path.write_text(json.dumps(payload))


def _write_graph(path: Path, nodes: list) -> None:
    """Helper: persist a graphml file with the given node names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    nx.write_graphml(g, path)


# --- read_index_status ---------------------------------------------------

def test_read_index_status_missing_directory_is_safe(tmp_path):
    """First-run case: db/ doesn't exist yet. The helper must not raise —
    the sidebar would crash on app load otherwise."""
    status = read_index_status(tmp_path / "does_not_exist")
    assert status["doc_count"] == 0
    assert status["entity_count"] == 0
    assert status["fact_count"] == 0
    assert status["last_ingested"] is None


def test_read_index_status_empty_directory_returns_zeros(tmp_path):
    status = read_index_status(tmp_path)
    assert status == {
        "doc_count": 0,
        "entity_count": 0,
        "fact_count": 0,
        "last_ingested": None,
    }


def test_read_index_status_counts_documents_from_owners_json(tmp_path):
    _write_owners(tmp_path / "owners.json", ["a.pdf", "b.pdf", "c.pdf"])
    status = read_index_status(tmp_path)
    assert status["doc_count"] == 3


def test_read_index_status_counts_entities_from_graphml(tmp_path):
    _write_graph(tmp_path / "graph.graphml", ["Alex", "Bilal", "Maryland"])
    status = read_index_status(tmp_path)
    assert status["entity_count"] == 3


def test_read_index_status_counts_facts_summed_across_fields(tmp_path):
    """fact_count is total across all field buckets — emails + phones +
    SEVIS IDs + ... — not the number of fields."""
    _write_facts(tmp_path / "facts.json", {
        "email": [("Alex", "a@x.com", "a.pdf"), ("Bilal", "b@x.com", "b.pdf")],
        "phone": [("Alex", "240-555-1234", "a.pdf")],
    })
    status = read_index_status(tmp_path)
    assert status["fact_count"] == 3


def test_read_index_status_last_ingested_is_most_recent_mtime(tmp_path):
    """When several index files exist, the displayed timestamp is the most
    recent mtime — not pinned to one canonical file. A partial re-ingest
    that touches only facts.json should still bump the displayed time."""
    _write_owners(tmp_path / "owners.json", ["a.pdf"])
    _write_graph(tmp_path / "graph.graphml", ["Alex"])
    time.sleep(0.05)  # ensure facts.json mtime is strictly later
    _write_facts(tmp_path / "facts.json", {"email": [("Alex", "a@x.com", "a.pdf")]})

    status = read_index_status(tmp_path)
    facts_mtime = (tmp_path / "facts.json").stat().st_mtime
    assert status["last_ingested"] == datetime.fromtimestamp(facts_mtime)


def test_read_index_status_last_ingested_none_when_no_index_files(tmp_path):
    """Random unrelated files in the directory don't count — only the known
    index files contribute to the timestamp."""
    (tmp_path / "unrelated.txt").write_text("noise")
    status = read_index_status(tmp_path)
    assert status["last_ingested"] is None


# --- format_status_line --------------------------------------------------

def test_format_status_line_compact():
    """Format helper produces a one-line compact stats string. Tight copy
    is the goal — 'docs · entities · facts' beats verbose sentences."""
    line = format_status_line({
        "doc_count": 12,
        "entity_count": 47,
        "fact_count": 89,
        "last_ingested": None,
    })
    assert "12" in line and "47" in line and "89" in line
    # Compact separator — not a full sentence.
    assert "documents containing" not in line
    assert "·" in line or "•" in line or "|" in line


def test_format_status_line_singular_counts_still_render():
    line = format_status_line({
        "doc_count": 1,
        "entity_count": 1,
        "fact_count": 1,
        "last_ingested": None,
    })
    assert "1" in line


def test_format_status_line_handles_zero_counts():
    """Pre-ingest state — the line should still render something coherent,
    not blow up or produce weird empty strings."""
    line = format_status_line({
        "doc_count": 0,
        "entity_count": 0,
        "fact_count": 0,
        "last_ingested": None,
    })
    assert isinstance(line, str)
    assert line  # non-empty
