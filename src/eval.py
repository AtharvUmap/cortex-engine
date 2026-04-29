"""
Eval utilities — golden-spec parsing + answer scoring.

The end-to-end harness in tests/eval/test_golden_queries.py loads a list of
GoldenSpecs from YAML, runs each query through generate_answer, and scores
the answer against the spec. This module is everything that runs *outside*
the actual LLM call: spec loading, scoring, diagnostics.

Why these three checks (must_contain / must_not_contain / should_suppress):

  - must_contain catches the obvious bug — the answer doesn't include the
    required fact. Loose substrings ("240" not "240-555-1234") so different
    formattings still pass.

  - must_not_contain catches cross-attribution — the case where the answer
    LOOKS right (contains a phone number) but is actually quoting the wrong
    person's data. This is the failure class Ticket 23 was added to fix and
    that the harness is meant to gate against in the future.

  - should_suppress catches the case where the system should refuse rather
    than answer (no relevant data, queried entity isn't in the corpus). The
    synthesis prompt enforces "I don't have enough context to answer that."
    but small models paraphrase, so refusal detection looks for any of a
    handful of fragments.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# Fragments that indicate the LLM refused to answer. Lowercase substring
# match; any one fragment in the answer counts as a refusal. The synthesis
# prompt asks for the first variant verbatim, but llama3.2 paraphrases
# routinely so the others are guard rails for "the model said no" in any of
# its usual phrasings.
_REFUSAL_FRAGMENTS = (
    "don't have",
    "do not have",
    "no information",
    "no relevant",
    "no record",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not enough context",
)


# --- Data classes ---------------------------------------------------------

@dataclass
class GoldenSpec:
    """One golden-query spec, loaded from YAML.

    must_contain / must_not_contain are lowercase-compared substrings.
    should_suppress=True means the answer should refuse rather than assert
    a fact (used for queries about entities not in the corpus).
    """
    name: str
    query: str
    must_contain: list = field(default_factory=list)
    must_not_contain: list = field(default_factory=list)
    should_suppress: bool = False


@dataclass
class ScoreResult:
    """Outcome of scoring one answer against one spec.

    The `missing`, `forbidden`, and `suppression_failed` fields exist so the
    harness can print a useful failure message — the assertion message is
    the only thing pytest shows when an end-to-end test fails, so the
    diagnostic detail has to be on the result object itself.
    """
    spec: GoldenSpec
    answer: str
    passed: bool
    missing: list = field(default_factory=list)
    forbidden: list = field(default_factory=list)
    suppression_failed: bool = False


# --- Loaders --------------------------------------------------------------

def load_goldens(path) -> list:
    """Parse a YAML file of golden specs into a list of GoldenSpec.

    Raises FileNotFoundError if the file is missing (rather than silently
    returning []) so a typo in the harness path can't masquerade as
    "everything passed because nothing ran." Raises ValueError on a spec
    missing the required `name` or `query` fields — same reason: better to
    fail loud than silently skip.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Goldens file not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return []

    specs = []
    for entry in raw:
        if "name" not in entry or "query" not in entry:
            raise ValueError(
                f"Golden spec missing required field (name or query): {entry!r}"
            )
        specs.append(GoldenSpec(
            name=entry["name"],
            query=entry["query"],
            must_contain=list(entry.get("must_contain") or []),
            must_not_contain=list(entry.get("must_not_contain") or []),
            should_suppress=bool(entry.get("should_suppress", False)),
        ))
    return specs


# --- Runner ---------------------------------------------------------------

def run_query(query: str, db_path: str) -> str:
    """Thin wrapper around generate_answer so the harness has one call site
    to mock or instrument (e.g. add timing later) without touching every
    test."""
    # Imported lazily so importing this module for unit-testing score() does
    # not pull in the synthesis pipeline (Ollama, ChromaDB, etc.).
    from src.synthesis import generate_answer
    return generate_answer(query, persist_directory=db_path)


# --- Scoring --------------------------------------------------------------

def _is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(frag in lower for frag in _REFUSAL_FRAGMENTS)


def score(answer: str, spec: GoldenSpec) -> ScoreResult:
    """Apply all three checks (must_contain / must_not_contain / suppression)
    and return a ScoreResult.

    All substring comparisons are case-insensitive. `passed` is True iff
    every check passes; the per-check fields (missing, forbidden,
    suppression_failed) are always populated so failure diagnostics are
    available regardless of which check actually tripped.
    """
    lower_answer = answer.lower()

    missing = [
        s for s in spec.must_contain
        if s.lower() not in lower_answer
    ]
    forbidden = [
        s for s in spec.must_not_contain
        if s.lower() in lower_answer
    ]
    suppression_failed = (
        spec.should_suppress and not _is_refusal(answer)
    )

    passed = (
        not missing
        and not forbidden
        and not suppression_failed
    )

    return ScoreResult(
        spec=spec,
        answer=answer,
        passed=passed,
        missing=missing,
        forbidden=forbidden,
        suppression_failed=suppression_failed,
    )
