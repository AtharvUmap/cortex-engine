"""Unit tests for src/eval — golden-spec parsing + answer scoring.

These are pure-Python unit tests with no Ollama or ChromaDB dependency. The
end-to-end harness lives in tests/eval/test_golden_queries.py and is run
separately via `pytest tests/eval/`.
"""

import pytest

from src.eval import (
    GoldenSpec,
    ScoreResult,
    load_goldens,
    score,
)


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
