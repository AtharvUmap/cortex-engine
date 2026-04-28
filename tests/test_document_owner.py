"""Tests for src/document_owner.py — per-document owner inference.

Owner inference combines two signals:
  1. Filename hint (heuristic extraction from filename + known-name map)
  2. First-chunk LLM ask (single proper-noun answer or 'unknown')

The two are reconciled into one (owner, confidence) tuple per document.
Both helpers are isolated so tests can patch them without standing up
real LLM or known-name infrastructure.
"""

from unittest.mock import patch

from langchain_core.documents import Document

from src.document_owner import (
    build_owner_index,
    infer_owner,
    load_owners,
    save_owners,
)


def _doc(source: str, content: str = "Atharv Umap is a software engineer.") -> Document:
    return Document(page_content=content, metadata={"source": source})


# --- infer_owner ----------------------------------------------------------

@patch("src.document_owner._filename_owner_guess")
@patch("src.document_owner._llm_owner_guess")
def test_infer_owner_high_confidence_when_filename_and_llm_agree(
    mock_llm, mock_filename
):
    """Filename and LLM agreeing -> high confidence."""
    mock_filename.return_value = "Atharv Umap"
    mock_llm.return_value = "Atharv Umap"
    owner, confidence = infer_owner(_doc("aumap_resume_2026.pdf"))
    assert owner == "Atharv Umap"
    assert confidence == "high"


@patch("src.document_owner._filename_owner_guess")
@patch("src.document_owner._llm_owner_guess")
def test_infer_owner_medium_confidence_when_only_llm_returns(
    mock_llm, mock_filename
):
    """Filename has nothing -> trust the LLM with medium confidence."""
    mock_filename.return_value = ""
    mock_llm.return_value = "Atharv Umap"
    owner, confidence = infer_owner(_doc("scan001.pdf"))
    assert owner == "Atharv Umap"
    assert confidence == "medium"


@patch("src.document_owner._filename_owner_guess")
@patch("src.document_owner._llm_owner_guess")
def test_infer_owner_low_confidence_when_only_filename_matches(
    mock_llm, mock_filename
):
    """LLM returns 'unknown' -> fall back to filename hint with low confidence."""
    mock_filename.return_value = "Atharv Umap"
    mock_llm.return_value = "unknown"
    owner, confidence = infer_owner(_doc("aumap_resume_2026.pdf"))
    assert owner == "Atharv Umap"
    assert confidence == "low"


@patch("src.document_owner._filename_owner_guess")
@patch("src.document_owner._llm_owner_guess")
def test_infer_owner_none_when_both_signals_fail(mock_llm, mock_filename):
    """Both signals empty -> (None, 'none'). Document still indexes; just no owner."""
    mock_filename.return_value = ""
    mock_llm.return_value = "unknown"
    owner, confidence = infer_owner(_doc("scan001.pdf"))
    assert owner is None
    assert confidence == "none"


@patch("src.document_owner._filename_owner_guess")
@patch("src.document_owner._llm_owner_guess")
def test_infer_owner_disagreement_keeps_llm_result(mock_llm, mock_filename):
    """Filename suggests one name, LLM another -> trust the LLM. Confidence drops to medium."""
    mock_filename.return_value = "John Doe"
    mock_llm.return_value = "Atharv Umap"
    owner, confidence = infer_owner(_doc("jdoe_letter.pdf"))
    assert owner == "Atharv Umap"
    assert confidence == "medium"


@patch("src.document_owner._filename_owner_guess")
@patch("src.document_owner._llm_owner_guess")
def test_infer_owner_fuzzy_agreement_counts_as_agreement(mock_llm, mock_filename):
    """Filename hint and LLM result that resolve to the same canonical via the
    fuzzy matcher should still register as high confidence — not penalized for
    different casings or spacings."""
    mock_filename.return_value = "atharv umap"
    mock_llm.return_value = "Atharv Umap"
    owner, confidence = infer_owner(_doc("atharv_umap_passport.pdf"))
    # LLM-form casing wins as the canonical
    assert owner == "Atharv Umap"
    assert confidence == "high"


@patch("src.document_owner._filename_owner_guess")
@patch("src.document_owner._llm_owner_guess")
def test_infer_owner_handles_missing_source_metadata(mock_llm, mock_filename):
    """Doc without a 'source' key -> filename helper sees empty string.
    LLM result stands at medium confidence."""
    mock_filename.return_value = ""
    mock_llm.return_value = "Atharv Umap"
    doc = Document(page_content="Atharv's resume.", metadata={})
    owner, confidence = infer_owner(doc)
    assert owner == "Atharv Umap"
    assert confidence == "medium"


@patch("src.document_owner._filename_owner_guess")
@patch("src.document_owner._llm_owner_guess")
def test_infer_owner_skips_llm_on_empty_document(mock_llm, mock_filename):
    """Empty content -> no LLM call needed; falls through to filename hint or none."""
    mock_filename.return_value = ""
    doc = Document(page_content="", metadata={"source": "scan001.pdf"})
    owner, confidence = infer_owner(doc)
    assert owner is None
    assert confidence == "none"
    mock_llm.assert_not_called()


# --- build_owner_index ----------------------------------------------------

@patch("src.document_owner._filename_owner_guess", return_value="")
@patch("src.document_owner._llm_owner_guess")
def test_build_owner_index_one_entry_per_document(mock_llm, mock_filename):
    """Every input doc gets one entry in the output dict, keyed by source."""
    mock_llm.return_value = "Atharv Umap"
    docs = [_doc("a.pdf"), _doc("b.pdf"), _doc("c.pdf")]
    owners = build_owner_index(docs)
    assert set(owners.keys()) == {"a.pdf", "b.pdf", "c.pdf"}
    for entry in owners.values():
        assert "owner" in entry
        assert "confidence" in entry


@patch("src.document_owner._filename_owner_guess", return_value="")
@patch("src.document_owner._llm_owner_guess")
def test_build_owner_index_isolates_per_doc_failure(mock_llm, mock_filename):
    """If one doc's LLM call raises, the other docs still get processed."""
    mock_llm.side_effect = [
        "Atharv Umap",
        RuntimeError("simulated LLM failure"),
        "Sneha Umap",
    ]
    docs = [_doc("a.pdf"), _doc("b.pdf"), _doc("c.pdf")]
    owners = build_owner_index(docs)

    assert owners["a.pdf"]["owner"] == "Atharv Umap"
    # Failed doc should still appear with a None owner so callers can iterate
    # consistently — not omitted.
    assert owners["b.pdf"]["owner"] is None
    assert owners["b.pdf"]["confidence"] == "none"
    assert owners["c.pdf"]["owner"] == "Sneha Umap"


# --- save_owners / load_owners --------------------------------------------

def test_save_load_owners_roundtrip(tmp_path):
    """Saved and reloaded owners map is identical."""
    owners = {
        "a.pdf": {"owner": "Atharv Umap", "confidence": "high"},
        "b.pdf": {"owner": None, "confidence": "none"},
    }
    path = tmp_path / "owners.json"
    save_owners(owners, path)
    assert load_owners(path) == owners


def test_load_owners_returns_empty_dict_on_missing_file(tmp_path):
    """First-run tolerance — no owners.json yet must not raise."""
    assert load_owners(tmp_path / "nonexistent.json") == {}


def test_save_owners_creates_parent_directory(tmp_path):
    """save_owners should mkdir -p its parent so callers don't have to."""
    nested = tmp_path / "a" / "b" / "owners.json"
    save_owners({"x.pdf": {"owner": "X", "confidence": "high"}}, nested)
    assert nested.exists()
