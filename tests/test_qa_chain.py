from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.qa_chain import ask_question


# --- Test helpers ---

def _make_mock_context():
    """Create fake retrieved documents to simulate retriever output."""
    return [
        Document(page_content="Photosynthesis converts sunlight into chemical energy."),
        Document(page_content="Plants use chlorophyll to absorb light during photosynthesis."),
    ]


# --- Tests ---

@patch("src.qa_chain.OllamaLLM")
@patch("src.qa_chain.search")
def test_ask_question_returns_string(mock_retriever, mock_llm_cls):
    """The QA chain should return a string response given a query."""
    # Mock the retriever to return fake context chunks
    mock_retriever.return_value = _make_mock_context()

    # Mock the LLM to return a canned answer instead of calling Ollama
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Photosynthesis converts sunlight into energy."
    mock_llm_cls.return_value = mock_llm

    response = ask_question("What is photosynthesis?")

    assert isinstance(response, str)
    assert len(response) > 0


@patch("src.qa_chain.OllamaLLM")
@patch("src.qa_chain.search")
def test_ask_question_calls_retriever_with_query(mock_retriever, mock_llm_cls):
    """The QA chain should pass the user's query to the retriever."""
    mock_retriever.return_value = _make_mock_context()

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Some answer."
    mock_llm_cls.return_value = mock_llm

    ask_question("What is photosynthesis?")

    # Verify the retriever was called with the user's query
    mock_retriever.assert_called_once()
    args = mock_retriever.call_args
    assert args[0][0] == "What is photosynthesis?"


@patch("src.qa_chain.OllamaLLM")
@patch("src.qa_chain.search")
def test_ask_question_with_no_context_returns_response(mock_retriever, mock_llm_cls):
    """When the retriever finds no relevant chunks, the chain should still return a string."""
    # Simulate an empty retrieval result
    mock_retriever.return_value = []

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "I don't have enough context to answer that."
    mock_llm_cls.return_value = mock_llm

    response = ask_question("What is dark matter?")

    assert isinstance(response, str)
    assert len(response) > 0
