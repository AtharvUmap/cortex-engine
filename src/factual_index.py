"""
Factual index — regex-first lookup for structured-fact queries.

Vector retrieval is brittle on questions with a single deterministic answer
("what's X's email?"). Same question, different phrasing, different answer.
This module short-circuits that path: at ingestion we scan every chunk with
regex and persist the matches; at query time we detect whether the user is
asking about a known field and inject the answer directly into the LLM's
context as a "Direct facts:" header.

Ticket 23 added per-entity binding. Each persisted fact now carries the
document's owner alongside its value and source, and lookup filters by the
entity named in the user's query. Without this, "what is amar umap's email"
returns Atharv's email (the only one in the corpus) — see the spec for the
demonstrated bug.

The patterns mirror ATTRIBUTE_SHAPE_PATTERNS in src/graph_extractor.py but
are unanchored — they need to find values inside multi-sentence chunk text
instead of validating a fully-extracted single value.
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document

from src.entity_resolution import resolve_against


# Unanchored regexes that scan within arbitrary chunk text. \b boundaries
# prevent partial matches against longer alphanumeric strings.
FACT_PATTERNS = {
    "email":           re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone_number":    re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "passport_number": re.compile(r"\b[A-Z]\d{7,8}\b"),
    "sevis_id":        re.compile(r"\bN\d{8,10}\b"),
    "visa_class":      re.compile(r"\b[A-Z]-?\d[A-Z]?\b"),
}


# Lowercase substring matching against the user's query. First field whose
# keyword list has any match wins. Order = most specific phrases first so
# "visa class" beats "visa" if both appeared.
_INTENT_KEYWORDS = {
    "visa_class":      ["visa class", "visa type", "what visa"],
    "sevis_id":        ["sevis"],
    "passport_number": ["passport"],
    "email":           ["email", "e-mail"],
    "phone_number":    ["phone", "cell", "mobile", "call", "reach"],
}


# --- Query → entity signal extraction (Ticket 23 + 23.5) -----------------
# Question stopwords that must be excluded when reading a possessive phrase
# from the query. "what's" is what — not a person — so we don't want it
# treated as the queried entity.
_QUERY_STOPWORDS = frozenset({
    "what", "whats", "where", "when", "who", "whose", "why", "how",
    "i", "my", "his", "her", "their", "your", "our",
    "the", "a", "an", "this", "that",
    "is", "are", "was", "were", "do", "does", "did",
})

# Generic nouns that look like a possessive subject syntactically but aren't
# specific people. Ticket 23.5: "the company's email" should fall through to
# generic retrieval, not get treated as a strong-signal person query and
# suppressed when the company isn't a known owner.
_GENERIC_NOUNS = frozenset({
    "company", "school", "university", "registrar", "department",
    "office", "agency", "embassy", "consulate", "government",
    "team", "group", "organization", "organisation",
    "college", "institute", "association", "society",
})

# Possessive phrase: capture everything before "'s" or "s'". Used to find
# the named subject of the question ("atharv's email" -> "atharv").
_POSSESSIVE_RE = re.compile(r"([a-z][\w\s]*?)(?:'s|s')\b", re.IGNORECASE)

# Mid-sentence capitalized phrase. Catches "Atharv" in "What does Atharv do?"
# where there's no possessive marker. Skips sentence-start capitals (those are
# always capitalized regardless of being a name) by requiring a space before.
_CAPITALIZED_NAME_RE = re.compile(r"\s([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")


def _entity_signal_from_query(query: str) -> tuple[str | None, str | None]:
    """Extract a queried entity AND classify how confident the reference is.

    Returns (entity, signal) where signal is one of:

      "strong" — possessive form whose subject is a specific (non-generic)
          phrase. The user is naming a particular person/thing. Used by
          callers to suppress retrieval on no-match (the screenshot fix).
      "weak"   — mid-sentence capitalized phrase. Could be a person, place,
          company, or concept. Callers do NOT suppress on no-match — they
          fall through to normal retrieval.
      None     — no entity reference (generic question, or a generic
          possessive like "the company's email" where the subject is a
          common noun rather than a specific name).

    Caller side: strong+no_match suppresses; weak+no_match passes through;
    None never filters.
    """
    if not query:
        return None, None

    # Step 1: possessive subject. Strips question stopwords from the captured
    # phrase. If what remains is empty OR contains only generic nouns, this
    # isn't a specific-person query — fall through.
    matches = _POSSESSIVE_RE.findall(query)
    for raw in matches:
        tokens = [
            t for t in raw.strip().lower().split()
            if t and t not in _QUERY_STOPWORDS
        ]
        if not tokens:
            continue
        # All tokens generic ("the company") → not a specific reference.
        if all(t in _GENERIC_NOUNS for t in tokens):
            continue
        return " ".join(tokens), "strong"

    # Step 2: mid-sentence capitalized phrase. Sentence-start words are always
    # capitalized regardless of being a name, so the regex requires a space
    # before the phrase — naturally skipping the first word.
    cap_matches = _CAPITALIZED_NAME_RE.findall(query)
    for raw in cap_matches:
        tokens = [
            t for t in raw.strip().lower().split()
            if t and t not in _QUERY_STOPWORDS
        ]
        if tokens:
            return " ".join(tokens), "weak"

    return None, None


# --- Public API ------------------------------------------------------------

def extract_facts(documents: list[Document], owners: dict | None = None) -> dict:
    """Scan every document with FACT_PATTERNS, attaching the document's owner.

    Each match is recorded as (entity, value, source) where entity comes from
    owners[source]['owner']. Docs not present in the owners map (or with
    owner=None) get entity=None — the fact is still indexed but lookup will
    only surface it for generic queries.

    No dedup at this layer so callers can see how many documents back each
    fact while iterating.
    """
    owners = owners or {}
    facts: dict[str, list] = {field: [] for field in FACT_PATTERNS}
    for doc in documents:
        text = doc.page_content or ""
        source = (doc.metadata or {}).get("source", "")
        owner_entry = owners.get(source) or {}
        entity = owner_entry.get("owner")
        for field, pattern in FACT_PATTERNS.items():
            for match in pattern.findall(text):
                facts[field].append((entity, match, source))
    return facts


def save_facts(facts: dict, path) -> None:
    """Persist as JSON. Tuples are stored as 3-element lists (entity, value, source)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        field: [list(entry) for entry in entries]
        for field, entries in facts.items()
    }
    path.write_text(json.dumps(serializable, indent=2))


def load_facts(path) -> dict:
    """Return {} if missing — keeps the fallback tolerant on first run.

    Pre-Ticket-23 facts.json files used 2-element [value, source] entries.
    Loading one of those is rejected with a clear error rather than silently
    miscoercing the value into the entity slot.
    """
    path = Path(path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    for field, entries in raw.items():
        for entry in entries:
            if len(entry) != 3:
                raise ValueError(
                    f"facts.json at {path} has {len(entry)}-element entries; "
                    f"Ticket 23 requires 3-element (entity, value, source). "
                    f"Clear db/ and re-ingest to upgrade."
                )
    return {
        field: [tuple(entry) for entry in entries]
        for field, entries in raw.items()
    }


def detect_intent(query: str) -> str | None:
    """Return the field key the query is asking about, or None.

    Lowercase substring match against _INTENT_KEYWORDS. Dumb-on-purpose —
    no LLM call, no NER — keeps this layer fully deterministic.
    """
    q = query.lower()
    for field, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return field
    return None


def lookup(query: str, facts: dict, owners: dict | None = None) -> list:
    """Return [(field, entity, value, source), ...] for the matched intent.

    Filtering matrix (Ticket 23 + 23.5):

      | Signal | Owner match | Behavior                                   |
      |--------|-------------|--------------------------------------------|
      | None   | n/a         | All facts (generic query)                  |
      | weak   | match       | Matched-owner facts only                   |
      | weak   | no match    | All facts (don't suppress on weak signal)  |
      | strong | match       | Matched-owner facts only                   |
      | strong | no match    | [] (the screenshot fix)                    |

    Without owners — or with an owners map containing only `owner=null`
    entries (every inference failed) — fall through to legacy non-filtered
    behavior.

    Empty list when intent is None or the matched field has no recorded facts.
    """
    field = detect_intent(query)
    if field is None:
        return []
    matches = facts.get(field, [])
    if not matches:
        return []

    all_matches = [
        (field, entity, value, source) for entity, value, source in matches
    ]

    # Owner index missing or all-null → legacy non-filtered behavior.
    owner_names = sorted({
        e["owner"] for e in (owners or {}).values()
        if isinstance(e, dict) and e.get("owner")
    })
    if not owner_names:
        return all_matches

    queried_entity, signal = _entity_signal_from_query(query)
    if queried_entity is None or signal is None:
        # Generic question — surface every owner's matches.
        return all_matches

    matched_owner = resolve_against(queried_entity, owner_names)
    if matched_owner is None:
        # Strong+no_match: the screenshot fix — suppress entirely.
        # Weak+no_match: don't filter; the capitalized phrase could be a
        # place/company/concept, not a person. Let vector retrieval answer.
        return [] if signal == "strong" else all_matches

    return [
        match for match in all_matches
        if match[1] == matched_owner
    ]
