"""End-to-end golden-query harness.

Loads tests/eval/golden_queries.yaml, ingests tests/eval/corpus/ once via
the session-scoped `ingested_db` fixture, runs each golden through
generate_answer, and scores against the spec.

This is the regression gate for the synthesis pipeline. Every future
ticket that touches retrieval, factual lookup, or owner filtering should
run this and confirm no goldens regress before shipping. The case for the
harness is exactly Ticket 23 -> Ticket 23.5: the cross-attribution
regression survived to manual Streamlit verification because nothing
automated would catch it. The relevant cases below (`amar_email_*`) make
that specific failure class fail loudly the next time it sneaks in.
"""

from pathlib import Path

import pytest

from src.eval import load_goldens, run_query, score, ScoreResult


_GOLDENS_PATH = Path(__file__).parent / "golden_queries.yaml"


def _load_specs():
    """Collect-time loader. Returns [] when the YAML hasn't been authored
    yet so pytest collection still succeeds — the harness reports "no
    tests collected" instead of an import-time crash."""
    if not _GOLDENS_PATH.exists():
        return []
    return load_goldens(_GOLDENS_PATH)


_SPECS = _load_specs()
_IDS = [s.name for s in _SPECS]


def _format_failure(result: ScoreResult) -> str:
    """Compose a multi-line failure message — pytest prints this when the
    assertion below fails, and it's the only diagnostic surface the user
    sees on a red harness run."""
    lines = [
        f"Golden '{result.spec.name}' failed.",
        f"  Query:  {result.spec.query}",
        f"  Answer: {result.answer[:300]!r}",
    ]
    if result.missing:
        lines.append(f"  Missing required substrings: {result.missing}")
    if result.forbidden:
        lines.append(f"  Forbidden substrings present: {result.forbidden}")
    if result.suppression_failed:
        lines.append("  Expected a refusal, but the answer asserted a fact.")
    return "\n".join(lines)


@pytest.mark.skipif(not _SPECS, reason="No golden specs authored yet.")
@pytest.mark.parametrize("spec", _SPECS, ids=_IDS)
def test_golden(spec, ingested_db):
    answer = run_query(spec.query, ingested_db)
    result = score(answer, spec)
    assert result.passed, _format_failure(result)
