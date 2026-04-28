"""
Shared entity-resolution helpers.

Originally lived inside graph_store.py as private methods. With Ticket 23
(per-entity binding for facts and retrieval) the same logic is needed in
factual_index.lookup, so it's been promoted to a stand-alone module.

Two pieces:
  - normalize_entity(s): aggressive lowercase + separator-to-space + strip
    punctuation + collapse whitespace. Output is a lookup key, never a
    display label.
  - resolve_against(target, candidates, cutoff=0.85): exact-match-on-normalized
    first, then difflib fuzzy fallback against the same normalized forms.
    Returns the matched candidate (in its original casing) or None when no
    candidate is similar enough.

The display-form-wins behavior of GraphStore stays in graph_store.py — that
module knows about "first variant encountered" and the visible node id.
This module is intentionally smaller in scope: it answers "does X resolve
to anything in this list?" without taking ownership of the canonical set.
"""

import difflib
import re

# Cutoff for difflib.get_close_matches when fuzzy-resolving an entity against
# existing names. SequenceMatcher ratio: 1.0 = identical, 0.0 = nothing in
# common. 0.85 catches one-character typos in long names ("Maryland" vs
# "Mariland" — ratio ~0.875) without merging unrelated short tokens (e.g.
# "python" vs "panda" — ratio ~0.55). Tune up if false merges appear, down
# if real duplicates are slipping through.
ENTITY_FUZZY_CUTOFF = 0.85

# Punctuation/separators that should be treated as word boundaries during
# normalization. Underscores, hyphens, and dots are the common ones LLMs
# mix in ("machine_learning", "machine-learning", "Machine.Learning").
_SEPARATOR_RE = re.compile(r"[_\-\.]+")

# Anything that's not a word char or whitespace gets stripped after the
# separators are converted to spaces. Handles trailing punctuation
# like "Atharv!" or "I-20."
_PUNCTUATION_RE = re.compile(r"[^\w\s]")

# Collapses runs of whitespace (including newlines and tabs) to a single
# space — applied last so the output is canonical.
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_entity(entity_str: str) -> str:
    """Aggressively normalize an entity string into a lookup key.

    Lowercases, converts underscores/hyphens/dots to spaces, strips remaining
    punctuation, and collapses runs of whitespace. Two strings that share the
    same normalized form should be treated as the same entity.

    The output is NEVER a display label — only used as the key for lookup
    against existing canonicals' normalized forms.
    """
    s = entity_str.lower().strip()
    s = _SEPARATOR_RE.sub(" ", s)
    s = _PUNCTUATION_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def resolve_against(
    target: str,
    candidates: list[str],
    cutoff: float = ENTITY_FUZZY_CUTOFF,
) -> str | None:
    """Resolve `target` to its best matching candidate, or None.

    Two-step lookup, matching what graph_store.py used internally before this
    module existed:
      1. Exact match on normalized form — handles case and separator variants.
      2. difflib fuzzy match against normalized forms — handles one-character
         typos in long names without merging unrelated short tokens.

    Args:
        target: The entity name to resolve.
        candidates: Pool of canonical names to match against. Returned values
            are taken from this list as-given (preserving original casing).
        cutoff: difflib SequenceMatcher ratio threshold. Defaults to
            ENTITY_FUZZY_CUTOFF; lower values admit weaker matches.

    Returns:
        The matching candidate (in its original casing), or None when no
        candidate is similar enough.
    """
    normalized_target = normalize_entity(target)
    if not normalized_target:
        return None
    if not candidates:
        return None

    # Build a {normalized_form: original_casing} map for the candidates.
    # Multiple candidates can normalize to the same form (e.g. "Atharv Umap"
    # and "atharv umap"); the first one wins.
    by_norm: dict[str, str] = {}
    for c in candidates:
        n = normalize_entity(c)
        if n and n not in by_norm:
            by_norm[n] = c

    # Step 1: exact normalized hit.
    if normalized_target in by_norm:
        return by_norm[normalized_target]

    # Step 2: fuzzy fallback. n=1 — only the closest match counts.
    matches = difflib.get_close_matches(
        normalized_target,
        list(by_norm.keys()),
        n=1,
        cutoff=cutoff,
    )
    if matches:
        return by_norm[matches[0]]

    # Step 3: token-containment fallback. A short reference like "atharv"
    # won't fuzzy-match the full canonical "atharv umap" — the ratio is
    # only ~0.7 because the canonical has extra characters difflib counts
    # as differences. But "atharv" being a token of the canonical is the
    # textbook case of a person being referred to by first name; we want
    # that to resolve. So: if every token of `target` appears as a token
    # of exactly one candidate, return that candidate. Multiple candidates
    # match → ambiguous → None (rather than picking arbitrarily).
    target_tokens = set(normalized_target.split())
    if not target_tokens:
        return None
    token_matches = [
        original
        for norm, original in by_norm.items()
        if target_tokens.issubset(set(norm.split()))
    ]
    if len(token_matches) == 1:
        return token_matches[0]

    return None
