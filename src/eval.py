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

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml


_UNCATEGORIZED = "uncategorized"


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

    category is an optional free-form string used by the report runner to
    group results into buckets (e.g. "per_entity_hit", "cross_attribution",
    "refusal"). Specs without a category are aggregated under a fallback
    bucket so adding a new spec without categorising it doesn't disappear
    it from the report.
    """
    name: str
    query: str
    must_contain: list = field(default_factory=list)
    must_not_contain: list = field(default_factory=list)
    should_suppress: bool = False
    category: str | None = None


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
            category=entry.get("category"),
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


# --- Suite report ---------------------------------------------------------
# These types support the `python eval/report.py` runner. The pytest gate
# (tests/eval/test_golden_queries.py) cares only about pass/fail per spec;
# the report runner adds aggregate structure on top: pass rate, latency
# percentiles, per-category breakdown. The split is deliberate — pytest is
# the merge gate, the report runner is the measurement tool.

@dataclass
class SpecRun:
    """One spec's run, including how long the LLM call took.

    `latency_seconds` is wall-clock time for the run_query call only; corpus
    ingestion happens once outside the loop and isn't billed against any
    individual spec.
    """
    result: "ScoreResult"
    latency_seconds: float


@dataclass
class CategoryStats:
    """Aggregate stats for one category bucket within a SuiteReport."""
    name: str
    total: int
    passed: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def _percentile(values, p):
    """Linear-interpolation percentile (matches numpy.percentile default).

    Stdlib has `statistics.quantiles`, but it requires n >= 2 and we want
    well-defined behavior at n=0 (return 0) and n=1 (return the single
    value) so the report runner is safe to invoke on a tiny suite.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


@dataclass
class SuiteReport:
    """Aggregate of all SpecRuns from one suite execution.

    All metrics are computed lazily so the report can be constructed cheaply
    from a list of runs and inspected by either the markdown formatter or
    direct attribute access (for the JSON dump or for ad-hoc analysis).
    """
    runs: list

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.runs if r.result.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def failures(self) -> list:
        return [r for r in self.runs if not r.result.passed]

    def _latencies(self):
        return [r.latency_seconds for r in self.runs]

    @property
    def latency_mean(self) -> float:
        latencies = self._latencies()
        return sum(latencies) / len(latencies) if latencies else 0.0

    @property
    def latency_p50(self) -> float:
        return _percentile(self._latencies(), 50)

    @property
    def latency_p95(self) -> float:
        return _percentile(self._latencies(), 95)

    @property
    def by_category(self) -> list:
        """Group runs by spec.category, returning sorted CategoryStats.

        None / missing categories collapse into a single "uncategorized"
        bucket. Sorting alphabetically by name keeps the report stable
        across runs (easier to diff before/after a change)."""
        buckets: dict = {}
        for run in self.runs:
            name = run.result.spec.category or _UNCATEGORIZED
            entry = buckets.setdefault(name, [0, 0])  # [total, passed]
            entry[0] += 1
            if run.result.passed:
                entry[1] += 1
        return [
            CategoryStats(name=name, total=t, passed=p)
            for name, (t, p) in sorted(buckets.items())
        ]


# --- Formatters -----------------------------------------------------------

def format_markdown(report: SuiteReport) -> str:
    """Human-readable summary of one suite execution.

    Three sections in fixed order: overall numbers, per-category breakdown,
    and (only when non-empty) a failures list with each failing spec's
    query and diagnostic. The per-category section is the part that turns
    "the suite is X% green" into "the suite is green except in bucket Y" —
    that's the actionable signal a refactor needs.
    """
    lines = ["# Eval suite report", ""]

    if report.total == 0:
        lines.append("No specs ran.")
        return "\n".join(lines)

    lines.extend([
        "## Overall",
        "",
        f"- Pass rate: **{report.passed}/{report.total} "
        f"({report.pass_rate * 100:.1f}%)**",
        f"- Latency mean: {report.latency_mean:.2f}s",
        f"- Latency p50: {report.latency_p50:.2f}s",
        f"- Latency p95: {report.latency_p95:.2f}s",
        "",
        "## By category",
        "",
        "| Category | Pass rate |",
        "| --- | --- |",
    ])
    for cat in report.by_category:
        lines.append(
            f"| {cat.name} | {cat.passed}/{cat.total} "
            f"({cat.pass_rate * 100:.1f}%) |"
        )

    if report.failures:
        lines.extend(["", "## Failures", ""])
        for run in report.failures:
            spec = run.result.spec
            lines.append(f"### `{spec.name}`")
            lines.append(f"- Query: {spec.query}")
            lines.append(f"- Answer: {run.result.answer[:300]!r}")
            if run.result.missing:
                lines.append(f"- Missing: {run.result.missing}")
            if run.result.forbidden:
                lines.append(f"- Forbidden present: {run.result.forbidden}")
            if run.result.suppression_failed:
                lines.append("- Expected refusal but answer asserted a fact.")
            lines.append("")

    return "\n".join(lines)


def to_dict(report: SuiteReport) -> dict:
    """JSON-safe representation of the report.

    Failures carry their full diagnostic (query, answer, missing, forbidden,
    suppression_failed) so the artifact is self-contained — a future reader
    shouldn't need the original YAML to interpret a red run.
    """
    return {
        "total": report.total,
        "passed": report.passed,
        "pass_rate": report.pass_rate,
        "latency_mean": report.latency_mean,
        "latency_p50": report.latency_p50,
        "latency_p95": report.latency_p95,
        "by_category": [
            {
                "name": c.name,
                "total": c.total,
                "passed": c.passed,
                "pass_rate": c.pass_rate,
            }
            for c in report.by_category
        ],
        "failures": [
            {
                "name": run.result.spec.name,
                "query": run.result.spec.query,
                "category": run.result.spec.category,
                "answer": run.result.answer,
                "missing": run.result.missing,
                "forbidden": run.result.forbidden,
                "suppression_failed": run.result.suppression_failed,
            }
            for run in report.failures
        ],
    }


# --- Driver ---------------------------------------------------------------

def run_suite(specs, db_path: str, runner=None) -> SuiteReport:
    """Run every spec through `runner` and return a SuiteReport.

    `runner` defaults to `run_query` (the real generate_answer wrapper).
    Tests inject a fake runner so they don't pull in Ollama; the end-to-end
    harness uses the default. Latency is wall-clock around the runner call
    only — corpus ingestion is amortised by the session-scoped fixture and
    isn't billed against any individual spec.
    """
    runner = runner or run_query
    runs = []
    for spec in specs:
        start = time.perf_counter()
        answer = runner(spec.query, db_path)
        elapsed = time.perf_counter() - start
        runs.append(SpecRun(
            result=score(answer, spec),
            latency_seconds=elapsed,
        ))
    return SuiteReport(runs=runs)
