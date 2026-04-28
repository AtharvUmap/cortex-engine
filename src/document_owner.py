"""
Per-document owner inference (Ticket 23).

A "document owner" is the single primary entity a document is *about*: the
person whose resume it is, the passport's holder, the I-20's recipient.
Owner inference combines two signals at ingest time:

  1. Filename hint — heuristic name extraction from the filename. Looks for
     name-like tokens (length >= 2, alphabetic) after stripping the extension
     and skipping common doc words ("resume", "letter", "passport"). Returns
     the cleaned name or "" when nothing useful is in the filename.

  2. First-chunk LLM ask — single prompt to the local LLM asking *whose*
     document this is. Strict single-line output: a proper name or the
     literal word "unknown".

The two signals are reconciled by infer_owner into one (owner, confidence)
tuple, where confidence is one of "high" / "medium" / "low" / "none":

  - high: both signals agree (via fuzzy matching on normalized form)
  - medium: only the LLM produced a name (filename empty or disagrees;
            the LLM result wins because it saw the document content)
  - low: LLM returned 'unknown' but filename has a name-like token
  - none: both signals failed

`build_owner_index` runs inference over a list of Documents, persisting any
per-doc failures as (None, "none") rather than letting them break the run.
The resulting dict is saved as ./db/owners.json next to the other indexes.

Both helper functions (_filename_owner_guess, _llm_owner_guess) are exposed
at module level so tests can patch them without standing up real LLM or
filename heuristic infrastructure.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from src.entity_resolution import normalize_entity, resolve_against

logger = logging.getLogger(__name__)


# How many characters of a document to feed the LLM when asking for the owner.
# 1500 covers a typical resume header / passport biographical page / I-20 first
# section, which is where the owner's name typically appears. More than this is
# a waste of inference time per doc.
_FIRST_CHUNK_CHARS = 1500

# Filename tokens that are obviously NOT names — common document words and
# numeric stems. Trimmed during filename-hint extraction so tokens like
# "resume" or "2026" don't get treated as candidate owner names.
_DOC_WORDS = frozenset({
    "resume", "cv", "letter", "passport", "scan", "scanned", "doc", "document",
    "i20", "i94", "i797", "ds160", "visa", "id", "license", "ticket", "form",
    "copy", "final", "draft", "v1", "v2", "page", "pages",
})

# Reuse the same env var as the rest of the app so one setting controls the LLM
# model used everywhere (synthesis entity extraction, graph extraction, owner
# inference).
DEFAULT_LLM_MODEL = "llama3.2"
LLM_MODEL = os.environ.get("CORTEX_LLM_MODEL", DEFAULT_LLM_MODEL)

# Strict single-line prompt — same shape as ENTITY_EXTRACTION_PROMPT in
# synthesis.py. Anything beyond a name or 'unknown' is treated as a refusal
# and downstream reconciliation falls back to the filename hint.
_OWNER_EXTRACTION_PROMPT = """You are reading the first page of a personal document. Identify the single person whose document this primarily is — the resume's subject, the passport's holder, the letter's recipient.

Rules:
- Return ONLY the person's full name, or the literal word "unknown".
- No quotes, no punctuation, no explanation, no prefix like "Owner:".
- Keep the answer to a few words at most.
- If the document is about an organization or has no clear single owner, return "unknown".

Document text:
{text}
"""


def _llm_owner_guess(text: str) -> str:
    """Ask the local LLM for the document owner.

    Returns the LLM's stripped answer, or "unknown" on any error / refusal.
    Importing OllamaLLM lazily keeps this module import-time cheap and lets
    tests patch the function without standing up Ollama.
    """
    if not text.strip():
        return "unknown"
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model=LLM_MODEL)
        raw = llm.invoke(_OWNER_EXTRACTION_PROMPT.format(text=text[:_FIRST_CHUNK_CHARS]))
    except Exception as exc:
        logger.warning("Owner LLM call failed: %s — treating as 'unknown'", exc)
        return "unknown"
    cleaned = (raw or "").strip().strip('"\'`').strip()
    return cleaned or "unknown"


def _filename_owner_guess(filename: str) -> str:
    """Heuristic owner-from-filename hint.

    Strips the extension and path, splits on common separators, drops doc
    words and pure-digit tokens, returns the remaining tokens space-joined
    with title casing. Empty string when nothing name-like remains.

    Examples:
        atharv_umap_resume_2026.pdf -> "Atharv Umap"
        scan001.pdf                 -> ""
        i20_2024.pdf                -> ""
    """
    if not filename:
        return ""
    stem = Path(filename).stem
    # Split on common filename separators
    tokens = re.split(r"[_\-\.\s]+", stem)
    name_like = []
    for t in tokens:
        if not t:
            continue
        lo = t.lower()
        if lo in _DOC_WORDS:
            continue
        if t.isdigit():
            continue
        # Single chars are noise (initials lose meaning without their pair)
        if len(t) < 2:
            continue
        # Reject tokens that are mostly digits (e.g. "scan001" → digits dominate)
        alpha_count = sum(c.isalpha() for c in t)
        if alpha_count < len(t) * 0.6:
            continue
        name_like.append(t)
    if not name_like:
        return ""
    return " ".join(name_like).title()


def infer_owner(doc: Document) -> tuple[str | None, str]:
    """Reconcile filename hint and LLM guess into (owner, confidence)."""
    source = (doc.metadata or {}).get("source", "")
    filename_guess = _filename_owner_guess(source)

    # Skip the LLM call entirely on empty documents — wasted inference.
    content = doc.page_content or ""
    if not content.strip():
        if filename_guess:
            return filename_guess, "low"
        return None, "none"

    llm_guess = _llm_owner_guess(content)
    has_llm = bool(llm_guess) and llm_guess.lower() != "unknown"
    has_filename = bool(filename_guess)

    if has_llm and has_filename:
        # Agreement check via fuzzy resolver — same one factual lookup uses.
        if resolve_against(filename_guess, [llm_guess]) is not None or \
           resolve_against(llm_guess, [filename_guess]) is not None or \
           normalize_entity(filename_guess) == normalize_entity(llm_guess):
            return llm_guess, "high"
        # Disagreement: trust the LLM (it saw the actual content) and log the
        # competing filename guess so a human can spot misnamed files.
        logger.info(
            "Owner inference disagreement: filename=%r llm=%r — keeping LLM",
            filename_guess, llm_guess,
        )
        return llm_guess, "medium"

    if has_llm:
        return llm_guess, "medium"

    if has_filename:
        return filename_guess, "low"

    return None, "none"


def build_owner_index(documents: Iterable[Document]) -> dict:
    """Run infer_owner over every document; return the owners-map shape.

    Per-doc failures are caught and recorded as (None, "none") so callers can
    iterate the dict consistently. One noisy doc cannot block the rest of the
    ingestion run.
    """
    owners: dict = {}
    for doc in documents:
        source = (doc.metadata or {}).get("source", "")
        try:
            owner, confidence = infer_owner(doc)
        except Exception as exc:
            logger.warning(
                "Owner inference failed for %s: %s — recording as 'none'",
                source, exc,
            )
            owner, confidence = None, "none"
        owners[source] = {"owner": owner, "confidence": confidence}
    return owners


def save_owners(owners: dict, path) -> None:
    """Persist as JSON. Parent directory is created if missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(owners, indent=2))


def load_owners(path) -> dict:
    """Return {} if missing — keeps the fallback tolerant on first run."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())
