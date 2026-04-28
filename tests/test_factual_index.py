import pytest
from langchain_core.documents import Document

from src.factual_index import (
    _entity_signal_from_query,
    detect_intent,
    extract_facts,
    load_facts,
    lookup,
    save_facts,
)


# --- extract_facts ---------------------------------------------------------

def test_extract_facts_finds_all_field_types():
    """A single chunk that mentions every supported field type should produce
    matches for every field — proves the regex table covers what it claims to."""
    text = (
        "Contact Atharv at atharvumap@gmail.com or call (240) 782-5464. "
        "His passport number is T3859852 and SEVIS ID N0034363393. "
        "He is on an F-1 visa."
    )
    doc = Document(page_content=text, metadata={"source": "test.pdf"})

    facts = extract_facts([doc])

    email_values = [v for _, v, _ in facts["email"]]
    passport_values = [v for _, v, _ in facts["passport_number"]]
    sevis_values = [v for _, v, _ in facts["sevis_id"]]
    visa_values = [v for _, v, _ in facts["visa_class"]]

    assert "atharvumap@gmail.com" in email_values
    assert "T3859852" in passport_values
    assert "N0034363393" in sevis_values
    assert "F-1" in visa_values
    # Phone-number regex variations make exact-match brittle; assert only that
    # something plausible was extracted from the (240) 782-5464 substring.
    assert any("5464" in v for _, v, _ in facts["phone_number"])


def test_extract_facts_records_source_metadata():
    """Each match must carry the source filename so downstream code can cite it."""
    doc = Document(
        page_content="Email: a@b.com",
        metadata={"source": "/path/resume.pdf"},
    )
    facts = extract_facts([doc])
    assert (None, "a@b.com", "/path/resume.pdf") in facts["email"]


def test_extract_facts_preserves_duplicates_across_documents():
    """Same value in two docs should appear twice (no dedup in this ticket)."""
    docs = [
        Document(page_content="email: x@y.com", metadata={"source": "doc1"}),
        Document(page_content="email: x@y.com", metadata={"source": "doc2"}),
    ]
    facts = extract_facts(docs)
    assert (None, "x@y.com", "doc1") in facts["email"]
    assert (None, "x@y.com", "doc2") in facts["email"]
    assert len(facts["email"]) == 2


def test_extract_facts_handles_missing_metadata():
    """A doc without a 'source' key must not crash — fall back to empty string."""
    doc = Document(page_content="email: no-meta@example.com", metadata={})
    facts = extract_facts([doc])
    assert (None, "no-meta@example.com", "") in facts["email"]


# --- extract_facts: per-entity binding (Ticket 23) -------------------------

def test_extract_facts_attaches_owner_from_owners_map():
    """When owners[source] = 'Atharv Umap', every fact extracted from that doc
    must carry 'Atharv Umap' as its entity."""
    doc = Document(
        page_content="Email: a@b.com",
        metadata={"source": "resume.pdf"},
    )
    owners = {"resume.pdf": {"owner": "Atharv Umap", "confidence": "high"}}

    facts = extract_facts([doc], owners=owners)

    assert ("Atharv Umap", "a@b.com", "resume.pdf") in facts["email"]


def test_extract_facts_stores_null_owner_when_doc_has_no_mapping():
    """Doc not present in owners map -> entity stored as None. The fact is
    still indexed (vector retrieval still wants it) but lookup will only
    surface it for generic queries."""
    doc = Document(page_content="Email: a@b.com", metadata={"source": "stray.pdf"})
    owners = {"resume.pdf": {"owner": "Atharv Umap", "confidence": "high"}}

    facts = extract_facts([doc], owners=owners)

    assert (None, "a@b.com", "stray.pdf") in facts["email"]


def test_extract_facts_stores_null_owner_when_owner_value_is_none():
    """Doc IS in owners map but owner=None (inference returned 'none' confidence).
    Same outcome as not being in the map at all — entity slot stays None."""
    doc = Document(page_content="Email: a@b.com", metadata={"source": "scan.pdf"})
    owners = {"scan.pdf": {"owner": None, "confidence": "none"}}

    facts = extract_facts([doc], owners=owners)

    assert (None, "a@b.com", "scan.pdf") in facts["email"]


def test_extract_facts_partitions_facts_by_owner_across_documents():
    """Two docs with different owners → emails partitioned by owner. This is the
    payload that fixes the demonstrated cross-attribution bug."""
    docs = [
        Document(page_content="email: a@b.com", metadata={"source": "atharv.pdf"}),
        Document(page_content="email: c@d.com", metadata={"source": "sneha.pdf"}),
    ]
    owners = {
        "atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"},
        "sneha.pdf":  {"owner": "Sneha Umap",  "confidence": "high"},
    }

    facts = extract_facts(docs, owners=owners)

    assert ("Atharv Umap", "a@b.com", "atharv.pdf") in facts["email"]
    assert ("Sneha Umap",  "c@d.com", "sneha.pdf")  in facts["email"]


# --- save_facts / load_facts ----------------------------------------------

def test_save_then_load_facts_roundtrip(tmp_path):
    """3-tuples must survive the JSON round-trip as 3-tuples (not lists)."""
    facts = {
        "email": [
            ("Atharv Umap", "a@b.com", "doc1.pdf"),
            (None,          "c@d.com", "doc2.pdf"),
        ],
        "phone_number": [],
        "passport_number": [("Atharv Umap", "T1234567", "passport.pdf")],
        "sevis_id": [],
        "visa_class": [],
    }
    path = tmp_path / "facts.json"

    save_facts(facts, path)
    loaded = load_facts(path)

    assert loaded == facts


def test_load_facts_missing_path_returns_empty_dict(tmp_path):
    """Tolerant on first run — no facts.json yet must not be an error."""
    assert load_facts(tmp_path / "nonexistent.json") == {}


def test_save_facts_creates_parent_directory(tmp_path):
    """save_facts should mkdir -p its parent so callers don't have to."""
    nested = tmp_path / "a" / "b" / "facts.json"
    save_facts({"email": []}, nested)
    assert nested.exists()


def test_load_facts_rejects_old_2tuple_schema(tmp_path):
    """A facts.json from before Ticket 23 has 2-element inner lists.
    Loading it must raise a clear error directing the user to re-ingest,
    NOT silently treat the value as the owner."""
    import json
    path = tmp_path / "facts.json"
    # Old shape — pre-Ticket-23 — must be rejected.
    path.write_text(json.dumps({
        "email": [["a@b.com", "resume.pdf"]],
    }))

    with pytest.raises(ValueError, match="re-ingest"):
        load_facts(path)


# --- detect_intent --------------------------------------------------------

def test_detect_intent_email():
    assert detect_intent("what is atharv's email?") == "email"
    assert detect_intent("What is Atharv's e-mail address") == "email"


def test_detect_intent_phone():
    assert detect_intent("how can I reach atharv?") == "phone_number"
    assert detect_intent("What is his phone number") == "phone_number"


def test_detect_intent_passport():
    assert detect_intent("what is the passport number?") == "passport_number"


def test_detect_intent_sevis():
    assert detect_intent("what is the SEVIS ID?") == "sevis_id"


def test_detect_intent_visa_class():
    assert detect_intent("what visa is he on") == "visa_class"


def test_detect_intent_unknown_returns_none():
    """Unknown intent must fall through cleanly — caller does normal retrieval."""
    assert detect_intent("summarize my I-20 finances") is None
    assert detect_intent("where did Atharv go to school?") is None


# --- lookup (legacy: no owners argument) ----------------------------------

def test_lookup_returns_all_matches_for_known_intent():
    """Without owners, every fact under the matched field is returned, in stored order."""
    facts = {
        "email": [
            ("Atharv Umap", "a@b.com", "resume.pdf"),
            ("Atharv Umap", "c@d.com", "i20.pdf"),
        ],
        "phone_number": [("Atharv Umap", "240-555-1234", "resume.pdf")],
    }
    hits = lookup("what is atharv's email?", facts)

    assert hits == [
        ("email", "Atharv Umap", "a@b.com", "resume.pdf"),
        ("email", "Atharv Umap", "c@d.com", "i20.pdf"),
    ]


def test_lookup_returns_empty_for_unknown_intent():
    facts = {"email": [("Atharv Umap", "a@b.com", "resume.pdf")]}
    assert lookup("summarize my finances", facts) == []


def test_lookup_returns_empty_when_field_has_no_facts():
    """Known intent + empty field → empty list, not KeyError."""
    facts = {"email": []}
    assert lookup("what is atharv's email?", facts) == []


def test_lookup_returns_empty_when_field_missing_from_facts():
    """Robust against an older facts.json missing some fields."""
    facts = {"phone_number": [("Atharv Umap", "240-555", "x")]}
    assert lookup("what is atharv's email?", facts) == []


# --- lookup: owner-aware (Ticket 23) --------------------------------------

def test_lookup_filters_by_queried_entity_when_owners_provided():
    """Two owners' facts in the index, query names one — only that owner's
    facts come back."""
    facts = {
        "email": [
            ("Atharv Umap", "a@b.com", "atharv.pdf"),
            ("Sneha Umap",  "c@d.com", "sneha.pdf"),
        ],
    }
    owners = {
        "atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"},
        "sneha.pdf":  {"owner": "Sneha Umap",  "confidence": "high"},
    }
    hits = lookup("what is atharv's email?", facts, owners=owners)

    assert hits == [("email", "Atharv Umap", "a@b.com", "atharv.pdf")]


def test_lookup_returns_empty_when_queried_entity_unknown():
    """Query names someone who isn't a known owner — suppress the block.
    This is the fix for the 'amar umap' wrong-attribution bug."""
    facts = {
        "email": [("Atharv Umap", "a@b.com", "atharv.pdf")],
    }
    owners = {"atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"}}

    hits = lookup("what is amar umap's email?", facts, owners=owners)

    assert hits == []


def test_lookup_fuzzy_matches_owner_typo():
    """Typo in the query should still resolve to a known owner via the same
    fuzzy resolver graph_store uses."""
    facts = {
        "email": [("Atharv Umap", "a@b.com", "atharv.pdf")],
    }
    owners = {"atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"}}

    # 'Atharv Uma' — last char dropped — should still resolve
    hits = lookup("what is atharv uma's email?", facts, owners=owners)

    assert hits == [("email", "Atharv Umap", "a@b.com", "atharv.pdf")]


def test_lookup_returns_all_facts_when_query_has_no_name():
    """Generic factual query (no proper-noun candidate in the question) →
    return facts for every owner. The prompt will list each with its owner."""
    facts = {
        "email": [
            ("Atharv Umap", "a@b.com", "atharv.pdf"),
            ("Sneha Umap",  "c@d.com", "sneha.pdf"),
        ],
    }
    owners = {
        "atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"},
        "sneha.pdf":  {"owner": "Sneha Umap",  "confidence": "high"},
    }

    # No proper noun in the question — generic
    hits = lookup("what's the email on file?", facts, owners=owners)

    # Both owners' facts come back
    assert ("email", "Atharv Umap", "a@b.com", "atharv.pdf") in hits
    assert ("email", "Sneha Umap",  "c@d.com", "sneha.pdf")  in hits


def test_lookup_first_name_only_resolves_to_full_owner():
    """User types 'atharv' alone → resolves to full owner name 'Atharv Umap'."""
    facts = {
        "email": [
            ("Atharv Umap", "a@b.com", "atharv.pdf"),
            ("Sneha Umap",  "c@d.com", "sneha.pdf"),
        ],
    }
    owners = {
        "atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"},
        "sneha.pdf":  {"owner": "Sneha Umap",  "confidence": "high"},
    }

    hits = lookup("what is atharv's email?", facts, owners=owners)

    # Only Atharv Umap's email — Sneha's is filtered out
    assert hits == [("email", "Atharv Umap", "a@b.com", "atharv.pdf")]


# --- Signal classification (Ticket 23.5) ---------------------------------
# _entity_signal_from_query returns (entity, signal) where signal classifies
# how confident we are that the user named a specific person:
#
#   strong  – possessive form whose subject isn't a generic noun
#   weak    – mid-sentence capitalized phrase (could be a person, place, concept)
#   None    – generic question or generic possessive ("the company's")
#
# Strong+no_match suppresses (the screenshot fix). Weak+no_match falls through.
# Generic queries don't filter at all.

def test_entity_signal_strong_for_personal_possessive():
    assert _entity_signal_from_query("what is atharv's email") == ("atharv", "strong")


def test_entity_signal_strong_for_multi_token_personal_possessive():
    assert _entity_signal_from_query("what is amar umap's email") == ("amar umap", "strong")


def test_entity_signal_none_for_generic_possessive():
    """A possessive whose subject is a generic noun ('the company') is NOT a
    person reference. Treated as no-entity so retrieval doesn't suppress."""
    assert _entity_signal_from_query("what is the company's email") == (None, None)


def test_entity_signal_weak_for_capitalized_only():
    """No possessive but a mid-sentence capital — could be person, place,
    company. Classified weak so no-match falls through instead of suppressing."""
    assert _entity_signal_from_query("Tell me about Maryland") == ("maryland", "weak")


def test_entity_signal_weak_for_capitalized_person_name():
    """Capitalized name with no possessive form. Even though we suspect it's
    a person, classification stays weak — confidence comes from owner-match,
    not from the syntactic pattern alone."""
    assert _entity_signal_from_query("What does Atharv do?") == ("atharv", "weak")


def test_entity_signal_none_for_truly_generic_question():
    """No possessive subject and no capitalized phrase mid-sentence."""
    assert _entity_signal_from_query("what's the email on file") == (None, None)


def test_entity_signal_none_for_empty_query():
    assert _entity_signal_from_query("") == (None, None)


# --- Signal-aware lookup behavior (Ticket 23.5) --------------------------

def test_lookup_does_not_suppress_on_weak_signal_no_match():
    """Query mentions a capitalized non-owner ('Google') without any possessive
    form. Ticket 23 wrongly suppressed retrieval here; 23.5 falls through to
    legacy behavior because the signal is weak (we don't know if 'Google' is
    a person, place, or concept).

    Note: 'what is Google's email' DOES suppress — that's a strong-signal
    possessive about an unknown entity, which is exactly the screenshot bug
    pattern (just with Google instead of Amar Umap). The test below covers
    that case as a regression guard."""
    facts = {
        "email": [("Atharv Umap", "atharvumap@gmail.com", "atharv.pdf")],
    }
    owners = {"atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"}}

    hits = lookup("Find an email mentioning Google", facts, owners=owners)

    # Capitalized 'Google' but no possessive — weak signal, no-match falls
    # through to legacy non-filtered behavior. All facts surface.
    assert hits == [("email", "Atharv Umap", "atharvumap@gmail.com", "atharv.pdf")]


def test_lookup_treats_generic_possessive_as_no_entity():
    """'the company's email' is a generic question, not a person query.
    Should return all facts, not suppress."""
    facts = {
        "email": [("Atharv Umap", "atharvumap@gmail.com", "atharv.pdf")],
    }
    owners = {"atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"}}

    hits = lookup("what is the company's email?", facts, owners=owners)

    assert hits == [("email", "Atharv Umap", "atharvumap@gmail.com", "atharv.pdf")]


def test_lookup_falls_through_when_all_owners_are_null():
    """If owners.json exists but every entry has owner=null (every inference
    failed), behave like no owners.json — don't suppress on every named query."""
    facts = {
        "email": [(None, "a@b.com", "scan1.pdf"), (None, "c@d.com", "scan2.pdf")],
    }
    owners = {
        "scan1.pdf": {"owner": None, "confidence": "none"},
        "scan2.pdf": {"owner": None, "confidence": "none"},
    }

    hits = lookup("what is atharv's email?", facts, owners=owners)

    # Both unfiltered facts come through — legacy behavior
    assert ("email", None, "a@b.com", "scan1.pdf") in hits
    assert ("email", None, "c@d.com", "scan2.pdf") in hits


def test_lookup_still_suppresses_on_strong_signal_no_match():
    """Regression guard: the screenshot fix from Ticket 23 must stay intact.
    Strong signal + no owner match = []."""
    facts = {
        "email": [("Atharv Umap", "atharvumap@gmail.com", "atharv.pdf")],
    }
    owners = {"atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"}}

    hits = lookup("what is amar umap's email?", facts, owners=owners)

    assert hits == []
