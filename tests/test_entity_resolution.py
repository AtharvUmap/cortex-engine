"""Tests for src/entity_resolution.py — the shared normalize/fuzzy-match helper
extracted out of graph_store.py so factual_index.lookup and graph_store both
use the same resolver. Behavior parity with the original is the primary goal.
"""

from src.entity_resolution import (
    ENTITY_FUZZY_CUTOFF,
    normalize_entity,
    resolve_against,
)


# --- normalize_entity ------------------------------------------------------

def test_normalize_entity_lowercases():
    assert normalize_entity("Atharv Umap") == "atharv umap"


def test_normalize_entity_strips_edge_whitespace():
    assert normalize_entity("  Atharv Umap  ") == "atharv umap"


def test_normalize_entity_treats_underscore_as_space():
    """Same key for underscore- and space-separated variants."""
    assert normalize_entity("machine_learning") == normalize_entity("Machine Learning")


def test_normalize_entity_treats_hyphen_as_space():
    assert normalize_entity("machine-learning") == normalize_entity("Machine Learning")


def test_normalize_entity_treats_dot_as_space():
    assert normalize_entity("Machine.Learning") == normalize_entity("Machine Learning")


def test_normalize_entity_collapses_whitespace_runs():
    assert normalize_entity("Machine   Learning") == "machine learning"


def test_normalize_entity_strips_punctuation():
    """Trailing punctuation must not fragment an otherwise identical entity."""
    assert normalize_entity("Atharv!") == normalize_entity("atharv")
    assert normalize_entity("Atharv's") == "atharvs"


def test_normalize_entity_handles_empty_string():
    assert normalize_entity("") == ""
    assert normalize_entity("   ") == ""


# --- resolve_against -------------------------------------------------------

def test_resolve_against_exact_match_returns_canonical():
    """Exact normalized match returns the candidate as-given (not the normalized form)."""
    assert resolve_against("machine_learning", ["Machine Learning", "AI"]) == "Machine Learning"


def test_resolve_against_returns_none_when_no_match():
    """No similar candidate -> None. Caller decides whether to insert as new canonical."""
    assert resolve_against("Atharv Umap", ["AI", "Maryland"]) is None


def test_resolve_against_fuzzy_matches_one_char_typo_in_long_name():
    """OCR slips like 'Mariland' must resolve to existing 'Maryland'."""
    assert resolve_against("Mariland", ["Maryland", "Atharv Umap"]) == "Maryland"


def test_resolve_against_rejects_short_token_coincidences():
    """Short unrelated tokens (python vs panda) should not falsely merge."""
    assert resolve_against("python", ["panda"]) is None


def test_resolve_against_returns_none_for_empty_candidates():
    """Empty candidate list -> None, no error."""
    assert resolve_against("Atharv Umap", []) is None


def test_resolve_against_returns_none_for_empty_target():
    """Empty / whitespace-only target has no resolution target."""
    assert resolve_against("", ["Atharv Umap"]) is None
    assert resolve_against("   ", ["Atharv Umap"]) is None


def test_resolve_against_respects_custom_cutoff():
    """A loose cutoff should let weaker matches through; the default rejects them."""
    # 'cat' vs 'cot' is one-char-different in a 3-char string — far below 0.85
    assert resolve_against("cat", ["cot"]) is None
    # With a loose cutoff it merges
    assert resolve_against("cat", ["cot"], cutoff=0.5) == "cot"


def test_resolve_against_picks_closest_when_multiple_candidates():
    """Among multiple candidates the best fuzzy match wins."""
    assert (
        resolve_against("University of Mariland", ["University of Maryland", "Maryland"])
        == "University of Maryland"
    )


def test_entity_fuzzy_cutoff_constant_matches_spec():
    """The default cutoff is part of the contract; locking it down so any tuning
    is intentional and visible in diffs, not silent."""
    assert ENTITY_FUZZY_CUTOFF == 0.85


# --- Token containment fallback ------------------------------------------
# difflib's ratio for "atharv" vs "atharv umap" is only ~0.71 because the
# canonical's extra characters count as differences. But a first-name reference
# resolving to a full name is exactly what we want — that's why we layer a
# token-containment fallback on top of fuzzy matching.

def test_resolve_against_token_containment_matches_first_name():
    """First-name reference should resolve to full canonical when fuzzy alone
    is too strict (ratio below the cutoff)."""
    assert resolve_against("atharv", ["Atharv Umap", "Sneha Umap"]) == "Atharv Umap"


def test_resolve_against_token_containment_matches_last_name_alone():
    """Same idea on the last name when only one canonical contains it."""
    assert resolve_against("Chakraborty", ["Atharv Umap", "Bijoya Chakraborty"]) == "Bijoya Chakraborty"


def test_resolve_against_token_containment_returns_none_when_ambiguous():
    """If the target's tokens are contained by multiple candidates, return None.
    'Umap' alone could be Atharv or Sneha — better to fall through than guess."""
    assert resolve_against("Umap", ["Atharv Umap", "Sneha Umap"]) is None


def test_resolve_against_token_containment_requires_all_tokens():
    """All target tokens must appear in the candidate. 'Atharv Singh' must NOT
    resolve to 'Atharv Umap' just because they share 'Atharv'."""
    assert resolve_against("Atharv Singh", ["Atharv Umap"]) is None
