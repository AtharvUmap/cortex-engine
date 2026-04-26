from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.graph_extractor import extract_graph_triples, extract_attributes


# --- Test helpers ---

def _make_mock_llm(mock_llm_cls, response: str):
    """Wire up the OllamaLLM mock so .invoke() returns the given string."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = response
    mock_llm_cls.return_value = mock_llm
    return mock_llm


# --- Tests ---

@patch("src.graph_extractor.OllamaLLM")
def test_extract_returns_list_of_dicts(mock_llm_cls):
    """A clean JSON array from the LLM should parse into a list of dicts."""
    _make_mock_llm(
        mock_llm_cls,
        '[{"source": "Atharv", "target": "Maryland", "relationship": "attends"}]',
    )

    doc = Document(page_content="Atharv attends the University of Maryland.")
    triples = extract_graph_triples(doc)

    assert isinstance(triples, list)
    assert len(triples) == 1
    assert triples[0] == {
        "source": "Atharv",
        "target": "Maryland",
        "relationship": "attends",
    }


@patch("src.graph_extractor.OllamaLLM")
def test_extract_parses_multiple_triples(mock_llm_cls):
    """Multiple triples in the LLM output should all be parsed."""
    _make_mock_llm(
        mock_llm_cls,
        '['
        '{"source": "Atharv", "target": "Maryland", "relationship": "attends"},'
        '{"source": "Maryland", "target": "USA", "relationship": "located_in"},'
        '{"source": "Atharv", "target": "Python", "relationship": "knows"}'
        ']',
    )

    doc = Document(page_content="Atharv attends Maryland in the USA and knows Python.")
    triples = extract_graph_triples(doc)

    assert len(triples) == 3
    assert all("source" in t and "target" in t and "relationship" in t for t in triples)


@patch("src.graph_extractor.OllamaLLM")
def test_extract_strips_markdown_code_fences(mock_llm_cls):
    """LLMs often wrap JSON in ```json ... ``` fences; those should be stripped."""
    _make_mock_llm(
        mock_llm_cls,
        '```json\n'
        '[{"source": "Earth", "target": "Sun", "relationship": "orbits"}]\n'
        '```',
    )

    doc = Document(page_content="The Earth orbits the Sun.")
    triples = extract_graph_triples(doc)

    assert len(triples) == 1
    assert triples[0]["relationship"] == "orbits"


@patch("src.graph_extractor.OllamaLLM")
def test_extract_returns_empty_list_on_malformed_json(mock_llm_cls):
    """If the LLM returns garbage, the function should not crash — return []."""
    _make_mock_llm(mock_llm_cls, "this is definitely not json")

    doc = Document(page_content="Some text.")
    triples = extract_graph_triples(doc)

    assert triples == []


@patch("src.graph_extractor.OllamaLLM")
def test_extract_returns_empty_list_for_empty_document(mock_llm_cls):
    """An empty document should short-circuit without calling the LLM."""
    _make_mock_llm(mock_llm_cls, "[]")

    doc = Document(page_content="")
    triples = extract_graph_triples(doc)

    assert triples == []
    mock_llm_cls.return_value.invoke.assert_not_called()


@patch("src.graph_extractor.OllamaLLM")
def test_extract_passes_document_content_to_llm(mock_llm_cls):
    """The document's page_content must make it into the prompt sent to the LLM."""
    mock_llm = _make_mock_llm(mock_llm_cls, "[]")

    doc = Document(page_content="Marie Curie discovered radium.")
    extract_graph_triples(doc)

    mock_llm.invoke.assert_called_once()
    prompt_arg = mock_llm.invoke.call_args[0][0]
    # The prompt text (whether a string or message list) must contain our content.
    prompt_str = str(prompt_arg)
    assert "Marie Curie discovered radium." in prompt_str


@patch("src.graph_extractor.OllamaLLM")
def test_extract_filters_out_malformed_triples(mock_llm_cls):
    """Triples missing required keys should be dropped, not crash the function."""
    _make_mock_llm(
        mock_llm_cls,
        '['
        '{"source": "A", "target": "B", "relationship": "knows"},'
        '{"source": "C", "target": "D"},'
        '{"relationship": "likes"},'
        '"not even a dict"'
        ']',
    )

    doc = Document(page_content="Some relational text.")
    triples = extract_graph_triples(doc)

    # Only the first entry is a complete triple
    assert len(triples) == 1
    assert triples[0]["source"] == "A"


# --- extract_attributes (Ticket 16.5: two-pass extraction) -----------------
# Pass 2 extracts (entity, field, value) attribute facts — phone numbers,
# expiry dates, identification numbers — that the relationship-extraction
# prompt was steering the model away from.

@patch("src.graph_extractor.OllamaLLM")
def test_extract_attributes_returns_list_of_dicts(mock_llm_cls):
    """A clean JSON array from the LLM should parse into a list of attribute dicts."""
    _make_mock_llm(
        mock_llm_cls,
        '[{"entity": "Atharv Umap", "field": "phone_number", "value": "240-555-1234"}]',
    )

    doc = Document(page_content="Atharv Umap's phone number is 240-555-1234.")
    attrs = extract_attributes(doc)

    assert isinstance(attrs, list)
    assert len(attrs) == 1
    assert attrs[0] == {
        "entity": "Atharv Umap",
        "field": "phone_number",
        "value": "240-555-1234",
    }


@patch("src.graph_extractor.OllamaLLM")
def test_extract_attributes_parses_multiple_attributes(mock_llm_cls):
    """Multiple attributes in one document should all be parsed."""
    _make_mock_llm(
        mock_llm_cls,
        '['
        '{"entity": "Atharv Umap", "field": "phone_number", "value": "240-555-1234"},'
        '{"entity": "Atharv Umap", "field": "email", "value": "atharv@umd.edu"},'
        '{"entity": "I-20", "field": "expiry_date", "value": "2027-05-15"}'
        ']',
    )

    doc = Document(page_content="Resume + I-20 contents...")
    attrs = extract_attributes(doc)

    assert len(attrs) == 3
    assert all("entity" in a and "field" in a and "value" in a for a in attrs)


@patch("src.graph_extractor.OllamaLLM")
def test_extract_attributes_strips_markdown_code_fences(mock_llm_cls):
    """LLMs sometimes wrap JSON in ```json fences; those should be stripped."""
    _make_mock_llm(
        mock_llm_cls,
        '```json\n'
        '[{"entity": "Passport", "field": "expiry_date", "value": "2030-01-01"}]\n'
        '```',
    )

    doc = Document(page_content="Passport expires 2030-01-01.")
    attrs = extract_attributes(doc)

    assert len(attrs) == 1
    assert attrs[0]["value"] == "2030-01-01"


@patch("src.graph_extractor.OllamaLLM")
def test_extract_attributes_returns_empty_list_on_malformed_json(mock_llm_cls):
    """Garbage LLM output must not crash — return []."""
    _make_mock_llm(mock_llm_cls, "this is not json at all")

    doc = Document(page_content="Some text.")
    attrs = extract_attributes(doc)

    assert attrs == []


@patch("src.graph_extractor.OllamaLLM")
def test_extract_attributes_returns_empty_list_for_empty_document(mock_llm_cls):
    """An empty document should short-circuit without calling the LLM."""
    _make_mock_llm(mock_llm_cls, "[]")

    doc = Document(page_content="")
    attrs = extract_attributes(doc)

    assert attrs == []
    mock_llm_cls.return_value.invoke.assert_not_called()


@patch("src.graph_extractor.OllamaLLM")
def test_extract_attributes_passes_document_content_to_llm(mock_llm_cls):
    """The document's page_content must reach the prompt sent to the LLM."""
    mock_llm = _make_mock_llm(mock_llm_cls, "[]")

    doc = Document(page_content="Phone: 240-555-1234. Email: a@b.com.")
    extract_attributes(doc)

    mock_llm.invoke.assert_called_once()
    prompt_str = str(mock_llm.invoke.call_args[0][0])
    assert "Phone: 240-555-1234. Email: a@b.com." in prompt_str


@patch("src.graph_extractor.OllamaLLM")
def test_extract_attributes_filters_out_malformed_entries(mock_llm_cls):
    """Entries missing entity/field/value should be dropped, not crash the function."""
    _make_mock_llm(
        mock_llm_cls,
        '['
        '{"entity": "Atharv Umap", "field": "phone_number", "value": "240-555-1234"},'
        '{"entity": "Atharv Umap", "field": "email"},'
        '{"value": "no entity here"},'
        '"not even a dict"'
        ']',
    )

    doc = Document(page_content="Mixed-quality output.")
    attrs = extract_attributes(doc)

    assert len(attrs) == 1
    assert attrs[0]["entity"] == "Atharv Umap"


@patch("src.graph_extractor.OllamaLLM")
def test_extract_attributes_uses_separate_prompt_from_relationships(mock_llm_cls):
    """Attribute prompt must instruct the LLM differently from the relationship
    prompt. Otherwise the two passes would just produce duplicate output and
    the whole point of separating them is lost.
    """
    mock_llm = _make_mock_llm(mock_llm_cls, "[]")

    doc = Document(page_content="Atharv attends Maryland.")
    extract_attributes(doc)

    prompt_str = str(mock_llm.invoke.call_args[0][0])
    # The attribute prompt must reference attribute-style keys (entity/field/value)
    # rather than the relationship triple keys (source/target/relationship).
    assert "entity" in prompt_str.lower()
    assert "field" in prompt_str.lower()
    assert "value" in prompt_str.lower()
