from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.synthesis import generate_answer


# --- Test helpers ---

def _mock_vector_chunks():
    """Fake parent chunks the vector retriever would produce."""
    return [
        Document(page_content="Atharv attended the University of Maryland."),
        Document(page_content="He studied Computer Science."),
    ]


def _mock_graph_facts():
    """Fake neighborhood triplets the graph store would produce."""
    return [
        {"source": "Atharv", "target": "Maryland", "relationship": "attends"},
        {"source": "Maryland", "target": "USA", "relationship": "located_in"},
    ]


def _setup_llm_mock(mock_llm_cls, entity: str, final_answer: str):
    """Two LLM calls happen in generate_answer: one for entity extraction,
    one for the final answer. Queue them in order.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [entity, final_answer]
    mock_llm_cls.return_value = mock_llm
    return mock_llm


# --- Tests ---

@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_returns_string(mock_llm_cls, mock_search, mock_graph_cls):
    """generate_answer should return a string final answer from the LLM."""
    _setup_llm_mock(mock_llm_cls, "Atharv", "Atharv attended Maryland.")
    mock_search.return_value = _mock_vector_chunks()

    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = _mock_graph_facts()
    mock_graph_cls.return_value = mock_graph

    response = generate_answer("Where did Atharv go to school?")

    assert isinstance(response, str)
    assert response == "Atharv attended Maryland."


@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_extracts_entity_first(mock_llm_cls, mock_search, mock_graph_cls):
    """The first LLM call should be for entity extraction from the user's query."""
    mock_llm = _setup_llm_mock(mock_llm_cls, "Atharv", "Some answer.")
    mock_search.return_value = _mock_vector_chunks()
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("What does Atharv do?")

    # First invoke call is the entity extraction prompt
    first_call_prompt = str(mock_llm.invoke.call_args_list[0][0][0])
    assert "What does Atharv do?" in first_call_prompt


@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_queries_graph_with_extracted_entity(
    mock_llm_cls, mock_search, mock_graph_cls
):
    """The extracted entity should be passed to graph_store.get_neighborhood."""
    _setup_llm_mock(mock_llm_cls, "Atharv", "Some answer.")
    mock_search.return_value = []
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("Who is Atharv?")

    mock_graph.get_neighborhood.assert_called_once()
    entity_arg = mock_graph.get_neighborhood.call_args[0][0]
    assert entity_arg == "Atharv"


@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_calls_vector_search_with_query(
    mock_llm_cls, mock_search, mock_graph_cls
):
    """The user's full query (not the extracted entity) should go to vector search."""
    _setup_llm_mock(mock_llm_cls, "passport", "Some answer.")
    mock_search.return_value = _mock_vector_chunks()
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("When does my passport expire?")

    mock_search.assert_called_once()
    query_arg = mock_search.call_args[0][0]
    assert query_arg == "When does my passport expire?"


@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_final_prompt_includes_both_vector_and_graph_context(
    mock_llm_cls, mock_search, mock_graph_cls
):
    """The final LLM call must receive context from BOTH retrievers when
    both have results. They are folded into one flat context block."""
    mock_llm = _setup_llm_mock(mock_llm_cls, "Atharv", "Final answer.")
    mock_search.return_value = _mock_vector_chunks()
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = _mock_graph_facts()
    mock_graph_cls.return_value = mock_graph

    generate_answer("Tell me about Atharv.")

    # Second invoke call is the final answer generation
    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])

    # Vector context chunks should be present
    assert "Atharv attended the University of Maryland." in final_prompt
    assert "He studied Computer Science." in final_prompt

    # Graph triplets should be present in arrow-syntax form
    assert "attends" in final_prompt
    assert "located_in" in final_prompt


@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_final_prompt_includes_original_question(mock_llm_cls, mock_search, mock_graph_cls):
    """The user's original question must appear in the final prompt."""
    mock_llm = _setup_llm_mock(mock_llm_cls, "Atharv", "Final answer.")
    mock_search.return_value = _mock_vector_chunks()
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = _mock_graph_facts()
    mock_graph_cls.return_value = mock_graph

    generate_answer("What does Atharv study?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    assert "What does Atharv study?" in final_prompt


@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_handles_empty_graph_neighborhood(
    mock_llm_cls, mock_search, mock_graph_cls
):
    """When the graph has no facts for the entity, synthesis should still work."""
    _setup_llm_mock(mock_llm_cls, "UnknownPerson", "I don't have enough context to answer that.")
    mock_search.return_value = []
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    response = generate_answer("Who is UnknownPerson?")

    assert isinstance(response, str)
    assert len(response) > 0


@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_strips_entity_whitespace_and_quotes(
    mock_llm_cls, mock_search, mock_graph_cls
):
    """LLMs often wrap entity names in quotes or add whitespace — both should be stripped
    before being passed to the graph.
    """
    _setup_llm_mock(mock_llm_cls, '  "Atharv"  \n', "Final answer.")
    mock_search.return_value = []
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("Tell me about Atharv.")

    entity_arg = mock_graph.get_neighborhood.call_args[0][0]
    assert entity_arg == "Atharv"
