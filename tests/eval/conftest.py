"""Shared fixtures for the golden-query harness.

The corpus ingestion is session-scoped because it's the slow step — full
embedding + owner inference + fact extraction takes a minute or two against
local Ollama. Paying it once per `pytest tests/eval/` run is the only way
the harness is tolerable to run.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.embedding import embed_documents


@pytest.fixture(scope="session")
def ingested_db(tmp_path_factory):
    """Ingest the synthetic corpus once and yield the persist-directory path.

    build_graph=False because the goldens target the factual index and
    vector retrieval — the graph walk adds expensive LLM extraction calls
    that don't change pass/fail for any of the planned cases. Owners and
    facts indices are both required (cross-attribution checks depend on the
    owner index; refusal cases depend on the fact index returning empty).
    """
    corpus_dir = Path(__file__).parent / "corpus"
    if not corpus_dir.exists():
        pytest.skip(
            f"Corpus directory missing: {corpus_dir}. "
            "Author tests/eval/corpus/*.txt before running the harness."
        )

    txt_files = sorted(corpus_dir.glob("*.txt"))
    if not txt_files:
        pytest.skip(f"No corpus files in {corpus_dir}.")

    docs = [
        Document(
            page_content=path.read_text(),
            metadata={"source": str(path)},
        )
        for path in txt_files
    ]

    db_dir = tmp_path_factory.mktemp("eval_db")
    embed_documents(
        docs,
        persist_directory=str(db_dir),
        build_graph=False,
        build_facts=True,
        build_owners=True,
    )
    return str(db_dir)
