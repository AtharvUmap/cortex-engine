"""Sidebar status helper.

Reads counts and the last-ingested timestamp from existing on-disk
persistence so the Streamlit sidebar can show a one-line health summary
without anyone re-running ingestion. Pure I/O — no Ollama, no embedding —
which is why this is a separate module: trivially testable, callable from
app.py, won't grow the surface area of `src/embedding.py`.
"""

import json
from datetime import datetime
from pathlib import Path

import networkx as nx


_OWNERS_FILE = "owners.json"
_FACTS_FILE = "facts.json"
_GRAPH_FILE = "graph.graphml"

# The set of files that contribute to the "last-ingested" timestamp. The
# displayed timestamp is the most recent mtime across these — pinning to a
# single file would lie if a partial re-ingest touched only one of them.
_INDEX_FILES = (_OWNERS_FILE, _FACTS_FILE, _GRAPH_FILE)


def read_index_status(persist_directory) -> dict:
    """Return {doc_count, entity_count, fact_count, last_ingested}.

    All four fields are present in every return value — missing files
    contribute zero (or None for the timestamp) so the sidebar is safe to
    render before anything has been ingested.
    """
    persist = Path(persist_directory)

    return {
        "doc_count": _count_owners(persist / _OWNERS_FILE),
        "entity_count": _count_graph_nodes(persist / _GRAPH_FILE),
        "fact_count": _count_facts(persist / _FACTS_FILE),
        "last_ingested": _most_recent_mtime(persist),
    }


def format_status_line(status: dict) -> str:
    """Compact one-line summary for the sidebar.

    Tight copy — "12 docs · 47 entities · 89 facts" rather than a sentence.
    The middle-dot separator reads cleaner than commas at small widths.
    """
    return (
        f"{status['doc_count']} docs · "
        f"{status['entity_count']} entities · "
        f"{status['fact_count']} facts"
    )


# --- internals ------------------------------------------------------------

def _count_owners(path: Path) -> int:
    if not path.exists():
        return 0
    return len(json.loads(path.read_text()))


def _count_graph_nodes(path: Path) -> int:
    if not path.exists():
        return 0
    return nx.read_graphml(path).number_of_nodes()


def _count_facts(path: Path) -> int:
    if not path.exists():
        return 0
    by_field = json.loads(path.read_text())
    return sum(len(entries) for entries in by_field.values())


def _most_recent_mtime(persist: Path):
    if not persist.exists():
        return None
    mtimes = [
        (persist / name).stat().st_mtime
        for name in _INDEX_FILES
        if (persist / name).exists()
    ]
    if not mtimes:
        return None
    return datetime.fromtimestamp(max(mtimes))
