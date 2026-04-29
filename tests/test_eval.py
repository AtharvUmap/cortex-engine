"""Unit tests for src/eval — golden-spec parsing + answer scoring.

These are pure-Python unit tests with no Ollama or ChromaDB dependency. The
end-to-end harness lives in tests/eval/test_golden_queries.py and is run
separately via `pytest tests/eval/`.
"""

import json

import pytest

from src.eval import (
    GoldenSpec,
    ScoreResult,
    SpecRun,
    SuiteReport,
    format_markdown,
    load_goldens,
    run_suite,
    score,
    to_dict,
)


def _make_run(name="t", passed=True, latency=0.1, category=None,
              answer="", missing=None, forbidden=None,
              suppression_failed=False):
    """Build a SpecRun without going through the full pipeline. Test-only."""
    spec = GoldenSpec(name=name, query=f"query for {name}", category=category)
    result = ScoreResult(
        spec=spec,
        answer=answer,
        passed=passed,
        missing=missing or [],
        forbidden=forbidden or [],
        suppression_failed=suppression_failed,
    )
    return SpecRun(result=result, latency_seconds=latency)


# --- load_goldens ---------------------------------------------------------

def test_load_goldens_parses_minimal_spec(tmp_path):
    """Minimal spec — only name + query — fills in all defaults."""
    path = tmp_path / "g.yaml"
    path.write_text(
        "- name: q1\n"
        "  query: What is Atharv's email?\n"
    )
    specs = load_goldens(path)
    assert len(specs) == 1
    s = specs[0]
    assert s.name == "q1"
    assert s.query == "What is Atharv's email?"
    assert s.must_contain == []
    assert s.must_not_contain == []
    assert s.should_suppress is False


def test_load_goldens_parses_full_spec(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(
        "- name: cross_attribution\n"
        "  query: \"What is Amar's email?\"\n"
        "  must_contain: [\"don't have\"]\n"
        "  must_not_contain: [\"atharvumap@gmail.com\"]\n"
        "  should_suppress: true\n"
    )
    specs = load_goldens(path)
    assert len(specs) == 1
    s = specs[0]
    assert s.name == "cross_attribution"
    assert s.must_contain == ["don't have"]
    assert s.must_not_contain == ["atharvumap@gmail.com"]
    assert s.should_suppress is True


def test_load_goldens_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_goldens(tmp_path / "missing.yaml")


def test_load_goldens_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("")
    assert load_goldens(path) == []


def test_load_goldens_loads_multiple_specs(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(
        "- name: a\n"
        "  query: q1\n"
        "- name: b\n"
        "  query: q2\n"
    )
    specs = load_goldens(path)
    assert [s.name for s in specs] == ["a", "b"]


def test_load_goldens_rejects_spec_missing_required_field(tmp_path):
    """A spec without `name` or `query` is malformed — reject loudly."""
    path = tmp_path / "g.yaml"
    path.write_text(
        "- query: orphan with no name\n"
    )
    with pytest.raises(ValueError):
        load_goldens(path)


def test_load_goldens_parses_optional_category(tmp_path):
    """`category` is optional and used by the report runner to group results.
    A missing field stays None; an explicit string is preserved verbatim."""
    path = tmp_path / "g.yaml"
    path.write_text(
        "- name: a\n"
        "  query: q1\n"
        "  category: per_entity_hit\n"
        "- name: b\n"
        "  query: q2\n"
    )
    specs = load_goldens(path)
    assert specs[0].category == "per_entity_hit"
    assert specs[1].category is None


# --- score: must_contain --------------------------------------------------

def test_score_passes_when_all_required_present():
    spec = GoldenSpec(name="t", query="q", must_contain=["maryland", "umap"])
    result = score("Atharv Umap goes to University of Maryland.", spec)
    assert result.passed is True
    assert result.missing == []
    assert result.forbidden == []


def test_score_fails_when_required_substring_missing():
    spec = GoldenSpec(name="t", query="q", must_contain=["maryland", "phd"])
    result = score("Atharv goes to Maryland.", spec)
    assert result.passed is False
    assert result.missing == ["phd"]


def test_score_required_match_is_case_insensitive():
    spec = GoldenSpec(name="t", query="q", must_contain=["MARYLAND"])
    result = score("Atharv goes to maryland.", spec)
    assert result.passed is True


# --- score: must_not_contain ---------------------------------------------

def test_score_fails_when_forbidden_substring_present():
    spec = GoldenSpec(
        name="t", query="q",
        must_not_contain=["atharvumap@gmail.com"],
    )
    result = score("The email is atharvumap@gmail.com.", spec)
    assert result.passed is False
    assert result.forbidden == ["atharvumap@gmail.com"]


def test_score_forbidden_match_is_case_insensitive():
    spec = GoldenSpec(name="t", query="q", must_not_contain=["AtHaRv"])
    result = score("atharv@gmail.com", spec)
    assert result.passed is False
    assert result.forbidden == ["AtHaRv"]


def test_score_forbidden_overrides_required_pass():
    """Even when must_contain is satisfied, a forbidden hit fails the case."""
    spec = GoldenSpec(
        name="t", query="q",
        must_contain=["umap"],
        must_not_contain=["atharvumap@gmail.com"],
    )
    result = score("Atharv Umap's email is atharvumap@gmail.com.", spec)
    assert result.passed is False
    assert result.missing == []
    assert result.forbidden == ["atharvumap@gmail.com"]


# --- score: should_suppress ----------------------------------------------

def test_score_should_suppress_passes_with_refusal_phrase():
    spec = GoldenSpec(name="t", query="q", should_suppress=True)
    result = score("I don't have enough context to answer that.", spec)
    assert result.passed is True
    assert result.suppression_failed is False


def test_score_should_suppress_fails_without_refusal_phrase():
    spec = GoldenSpec(name="t", query="q", should_suppress=True)
    result = score("The email is somebody@example.com.", spec)
    assert result.passed is False
    assert result.suppression_failed is True


def test_score_should_suppress_recognises_alternative_refusal_wording():
    """Refusal detection isn't tied to one exact phrase — paraphrases count."""
    spec = GoldenSpec(name="t", query="q", should_suppress=True)
    for refusal in [
        "I don't have enough context.",
        "I do not have that information.",
        "There is no information about that.",
        "I cannot answer that based on the context.",
    ]:
        assert score(refusal, spec).passed is True, refusal


def test_score_should_suppress_combines_with_must_not_contain():
    """The two checks are independent — a forbidden leak still fails the case."""
    spec = GoldenSpec(
        name="t", query="q",
        must_not_contain=["atharvumap@gmail.com"],
        should_suppress=True,
    )
    result = score(
        "I don't have enough context, but it's atharvumap@gmail.com.",
        spec,
    )
    assert result.passed is False
    # The refusal phrase is present, so suppression itself didn't fail —
    # the forbidden substring is what tripped the case.
    assert result.suppression_failed is False
    assert result.forbidden == ["atharvumap@gmail.com"]


def test_score_should_suppress_false_does_not_require_refusal():
    """When suppression isn't requested, a normal answer with the required
    keyword passes — refusal detection only runs when should_suppress=True."""
    spec = GoldenSpec(
        name="t", query="q",
        must_contain=["umap"],
        should_suppress=False,
    )
    result = score("Atharv Umap.", spec)
    assert result.passed is True


def test_score_returns_spec_and_answer_for_diagnostic_output():
    """The harness prints these on failure, so they have to be on the result."""
    spec = GoldenSpec(name="t", query="q", must_contain=["x"])
    result = score("y", spec)
    assert result.spec is spec
    assert result.answer == "y"
    assert isinstance(result, ScoreResult)


# --- SuiteReport: aggregate stats -----------------------------------------
# These tests exercise the report layer used by `python eval/report.py`. The
# pytest gate gives binary pass/fail; the report runner needs structured
# numbers (pass rate, percentiles, per-category breakdown) for the resume
# claim "X% pass rate at p95 latency Y s on N goldens" to be defensible.


def test_suite_report_empty_runs_is_safe():
    """An empty suite must not divide-by-zero — the runner could legitimately
    be invoked with zero specs (e.g. all filtered out by a future flag)."""
    report = SuiteReport(runs=[])
    assert report.total == 0
    assert report.passed == 0
    assert report.pass_rate == 0.0
    assert report.latency_mean == 0.0
    assert report.latency_p50 == 0.0
    assert report.latency_p95 == 0.0
    assert report.by_category == []
    assert report.failures == []


def test_suite_report_counts_pass_and_fail():
    runs = [
        _make_run(name="a", passed=True),
        _make_run(name="b", passed=False),
        _make_run(name="c", passed=True),
    ]
    report = SuiteReport(runs=runs)
    assert report.total == 3
    assert report.passed == 2
    assert report.pass_rate == pytest.approx(2 / 3)


def test_suite_report_failures_returns_only_failing_runs():
    runs = [
        _make_run(name="a", passed=True),
        _make_run(name="b", passed=False),
        _make_run(name="c", passed=False),
    ]
    failures = SuiteReport(runs=runs).failures
    assert [r.result.spec.name for r in failures] == ["b", "c"]


def test_suite_report_latency_single_run():
    """With one observation, p50 and p95 are both the single value — no
    percentile interpolation drama."""
    report = SuiteReport(runs=[_make_run(latency=2.5)])
    assert report.latency_mean == 2.5
    assert report.latency_p50 == 2.5
    assert report.latency_p95 == 2.5


def test_suite_report_latency_percentiles_match_linear_interpolation():
    """Latencies 0..9 → p50 = 4.5 (mean of 4 and 5), p95 = 8.55 under the
    linear-interpolation method (numpy.percentile default)."""
    runs = [_make_run(name=f"r{i}", latency=float(i)) for i in range(10)]
    report = SuiteReport(runs=runs)
    assert report.latency_mean == pytest.approx(4.5)
    assert report.latency_p50 == pytest.approx(4.5)
    assert report.latency_p95 == pytest.approx(8.55)


def test_suite_report_by_category_groups_and_counts():
    runs = [
        _make_run(name="a1", passed=True, category="alpha"),
        _make_run(name="a2", passed=False, category="alpha"),
        _make_run(name="b1", passed=True, category="beta"),
    ]
    by_cat = {c.name: c for c in SuiteReport(runs=runs).by_category}
    assert by_cat["alpha"].total == 2
    assert by_cat["alpha"].passed == 1
    assert by_cat["alpha"].pass_rate == pytest.approx(0.5)
    assert by_cat["beta"].total == 1
    assert by_cat["beta"].passed == 1
    assert by_cat["beta"].pass_rate == 1.0


def test_suite_report_by_category_buckets_uncategorized_specs():
    """Specs without a category mustn't be silently dropped from the report —
    they show up under an explicit 'uncategorized' bucket so the next author
    notices and adds the field."""
    runs = [
        _make_run(name="a", passed=True, category="alpha"),
        _make_run(name="orphan", passed=False, category=None),
    ]
    by_cat = {c.name: c for c in SuiteReport(runs=runs).by_category}
    assert "uncategorized" in by_cat
    assert by_cat["uncategorized"].total == 1
    assert by_cat["uncategorized"].passed == 0


def test_suite_report_by_category_sorted_by_name():
    """Stable order makes diffing two reports easier (e.g. "did refactor X
    drop pass rate in any bucket?")."""
    runs = [
        _make_run(name="b", passed=True, category="beta"),
        _make_run(name="a", passed=True, category="alpha"),
        _make_run(name="g", passed=True, category="gamma"),
    ]
    names = [c.name for c in SuiteReport(runs=runs).by_category]
    assert names == ["alpha", "beta", "gamma"]


# --- format_markdown ------------------------------------------------------
# The markdown output is what the user reads after `python eval/report.py`.
# Tests here pin the high-signal surface area (pass rate, percentiles, per-
# category breakdown, failure listing) so a refactor doesn't accidentally
# drop one of those. Formatting trivia (whitespace, exact heading levels)
# isn't asserted on — that's churn-prone and not load-bearing.


def test_format_markdown_includes_overall_pass_rate():
    runs = [
        _make_run(name="a", passed=True, category="x"),
        _make_run(name="b", passed=False, category="x"),
    ]
    md = format_markdown(SuiteReport(runs=runs))
    assert "1/2" in md
    assert "50.0%" in md


def test_format_markdown_includes_latency_percentiles():
    runs = [_make_run(name=f"r{i}", latency=float(i)) for i in range(10)]
    md = format_markdown(SuiteReport(runs=runs))
    assert "p50" in md.lower()
    assert "p95" in md.lower()


def test_format_markdown_includes_per_category_breakdown():
    runs = [
        _make_run(name="a", passed=True, category="alpha"),
        _make_run(name="b", passed=False, category="beta"),
    ]
    md = format_markdown(SuiteReport(runs=runs))
    assert "alpha" in md
    assert "beta" in md


def test_format_markdown_lists_failures_with_query_and_diagnostic():
    """A red run must show *why* each spec failed — query + missing/forbidden
    so the reader doesn't need to crack open the YAML to interpret it."""
    runs = [
        _make_run(
            name="failing_spec",
            passed=False,
            answer="some answer",
            forbidden=["leaked@example.com"],
        ),
    ]
    md = format_markdown(SuiteReport(runs=runs))
    assert "failing_spec" in md
    assert "leaked@example.com" in md


def test_format_markdown_empty_suite_does_not_crash():
    """Defensive — a report with no runs should still produce *some* output
    (e.g. "no specs ran"), not raise."""
    md = format_markdown(SuiteReport(runs=[]))
    assert isinstance(md, str)
    assert md  # non-empty


def test_format_markdown_all_passing_omits_failures_section():
    """Clean run shouldn't print an empty 'Failures' header — the absence
    is itself the signal."""
    runs = [
        _make_run(name="a", passed=True, category="x"),
        _make_run(name="b", passed=True, category="x"),
    ]
    md = format_markdown(SuiteReport(runs=runs))
    assert "Failures" not in md and "failures" not in md


# --- to_dict --------------------------------------------------------------


def test_to_dict_round_trips_through_json():
    """to_dict must emit a JSON-safe structure — the runner dumps it with
    json.dumps for the --json output mode and for archived eval artifacts."""
    runs = [
        _make_run(name="a", passed=True, category="x", latency=0.42),
        _make_run(name="b", passed=False, category="y", forbidden=["bad"]),
    ]
    payload = to_dict(SuiteReport(runs=runs))
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["total"] == 2
    assert decoded["passed"] == 1
    assert decoded["pass_rate"] == pytest.approx(0.5)


def test_to_dict_exposes_per_category_stats():
    runs = [
        _make_run(name="a", passed=True, category="alpha"),
        _make_run(name="b", passed=False, category="alpha"),
    ]
    payload = to_dict(SuiteReport(runs=runs))
    by_cat = {c["name"]: c for c in payload["by_category"]}
    assert by_cat["alpha"]["total"] == 2
    assert by_cat["alpha"]["passed"] == 1
    assert by_cat["alpha"]["pass_rate"] == pytest.approx(0.5)


# --- run_suite ------------------------------------------------------------
# The runner takes specs + a db path and produces a SuiteReport. We inject
# a fake runner so these tests don't pull in Ollama / ChromaDB — the real
# wiring to generate_answer is exercised by the end-to-end harness.


def test_run_suite_invokes_runner_per_spec_with_db_path():
    """Each spec's query goes through the runner once, with the provided
    db_path threaded through verbatim."""
    specs = [
        GoldenSpec(name="a", query="q1"),
        GoldenSpec(name="b", query="q2"),
    ]
    calls = []
    def fake(query, db_path):
        calls.append((query, db_path))
        return "ok"
    run_suite(specs, "/fake/db", runner=fake)
    assert calls == [("q1", "/fake/db"), ("q2", "/fake/db")]


def test_run_suite_scores_each_answer_against_its_spec():
    specs = [
        GoldenSpec(name="hit", query="q1", must_contain=["yes"]),
        GoldenSpec(name="miss", query="q2", must_contain=["expected"]),
    ]
    answers = {"q1": "yes here it is", "q2": "nope"}
    report = run_suite(specs, "/fake", runner=lambda q, _: answers[q])
    assert report.runs[0].result.passed is True
    assert report.runs[1].result.passed is False
    assert report.runs[1].result.missing == ["expected"]


def test_run_suite_records_non_negative_latency():
    """Latency is wall-clock — we don't pin a value, only that it's recorded
    and non-negative. Real LLM calls take seconds; a fake runner returns
    near-zero, both are valid."""
    specs = [GoldenSpec(name="a", query="q")]
    report = run_suite(specs, "/fake", runner=lambda q, _: "ok")
    assert report.runs[0].latency_seconds >= 0


def test_run_suite_empty_specs_returns_empty_report():
    report = run_suite([], "/fake", runner=lambda q, _: "ok")
    assert report.total == 0


def test_to_dict_failures_carry_diagnostic_fields():
    """Each failure entry needs spec name + query + answer + missing/forbidden
    so the JSON artifact is self-contained — a future reader shouldn't need
    the original YAML alongside it."""
    runs = [
        _make_run(
            name="bad_spec",
            passed=False,
            answer="leaked answer",
            missing=["expected"],
            forbidden=["leaked"],
        ),
    ]
    payload = to_dict(SuiteReport(runs=runs))
    assert len(payload["failures"]) == 1
    f = payload["failures"][0]
    assert f["name"] == "bad_spec"
    assert f["answer"] == "leaked answer"
    assert f["missing"] == ["expected"]
    assert f["forbidden"] == ["leaked"]
