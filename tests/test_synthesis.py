from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.synthesis import generate_answer, _rerank_chunks, CROSS_ENCODER_MODEL


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


# Default _rerank_chunks stub used by all existing tests so they don't load
# the real CrossEncoder model. Returns chunks unchanged (capped at top_k)
# so existing assertions about chunk presence in the final prompt still hold.
def _passthrough_rerank(query, chunks, top_k=3):
    return chunks[:top_k]


# --- Tests ---

@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_returns_string(
    mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """generate_answer should return a string final answer from the LLM."""
    _setup_llm_mock(mock_llm_cls, "Atharv", "Atharv attended Maryland.")
    mock_search.return_value = _mock_vector_chunks()

    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = _mock_graph_facts()
    mock_graph_cls.return_value = mock_graph

    response = generate_answer("Where did Atharv go to school?")

    assert isinstance(response, str)
    assert response == "Atharv attended Maryland."


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_extracts_entity_first(
    mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
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


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_queries_graph_with_extracted_entity(
    mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
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


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_calls_vector_search_with_query(
    mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
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


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_final_prompt_includes_both_vector_and_graph_context(
    mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
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


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_final_prompt_includes_original_question(
    mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """The user's original question must appear in the final prompt."""
    mock_llm = _setup_llm_mock(mock_llm_cls, "Atharv", "Final answer.")
    mock_search.return_value = _mock_vector_chunks()
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = _mock_graph_facts()
    mock_graph_cls.return_value = mock_graph

    generate_answer("What does Atharv study?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    assert "What does Atharv study?" in final_prompt


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_handles_empty_graph_neighborhood(
    mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
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


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_strips_entity_whitespace_and_quotes(
    mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
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


# --- Re-ranking integration (Ticket 18) ------------------------------------
# The vector chunks coming out of search() must be passed through the
# CrossEncoder reranker before being merged with graph context. Verifies the
# wiring at the generate_answer level.

@patch("src.synthesis._rerank_chunks")
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_invokes_reranker_with_query_and_chunks(
    mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """generate_answer must call the reranker with the user's query and the
    raw vector chunks before merging into the final context."""
    chunks = _mock_vector_chunks()
    _setup_llm_mock(mock_llm_cls, "Atharv", "Final answer.")
    mock_search.return_value = chunks
    mock_rerank.return_value = chunks  # passthrough for this test
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("Tell me about Atharv.")

    mock_rerank.assert_called_once()
    args = mock_rerank.call_args[0]
    assert args[0] == "Tell me about Atharv."
    assert args[1] == chunks


@patch("src.synthesis._rerank_chunks")
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
def test_generate_answer_only_uses_top_k_after_rerank(
    mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """When the reranker keeps only the top-k chunks, the chunks it dropped
    must NOT appear in the final prompt — that's the whole point of reranking."""
    six_chunks = [Document(page_content=f"Chunk {i}") for i in range(6)]
    mock_llm = _setup_llm_mock(mock_llm_cls, "Atharv", "Final answer.")
    mock_search.return_value = six_chunks

    # Reranker picks chunks 3, 4, 5 as the top-3 (chunks 0/1/2 are dropped)
    mock_rerank.return_value = six_chunks[3:]

    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("any query")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    # Dropped chunks must NOT make it into context
    assert "Chunk 0" not in final_prompt
    assert "Chunk 1" not in final_prompt
    assert "Chunk 2" not in final_prompt
    # Kept chunks must be present
    assert "Chunk 3" in final_prompt
    assert "Chunk 4" in final_prompt
    assert "Chunk 5" in final_prompt


# --- _rerank_chunks unit tests (Ticket 18) ---------------------------------
# Direct tests of the reranker function. Patches _get_cross_encoder so no
# real model is loaded, then verifies score-based ordering and edge cases.

@patch("src.synthesis._get_cross_encoder")
def test_rerank_chunks_orders_by_cross_encoder_score_descending(mock_loader):
    """Chunks with higher cross-encoder scores should appear earlier in the
    output. That's what makes the highest-relevance passage land first in
    the prompt."""
    chunks = [
        Document(page_content="A"),
        Document(page_content="B"),
        Document(page_content="C"),
    ]
    mock_encoder = MagicMock()
    # Score B highest (0.9), then A (0.5), then C (0.1)
    mock_encoder.predict.return_value = [0.5, 0.9, 0.1]
    mock_loader.return_value = mock_encoder

    result = _rerank_chunks("query", chunks, top_k=3)

    assert [doc.page_content for doc in result] == ["B", "A", "C"]


@patch("src.synthesis._get_cross_encoder")
def test_rerank_chunks_returns_only_top_k(mock_loader):
    """When more than top_k chunks are passed in, only the top_k highest-scoring
    chunks should be returned. That's the whole point of the function."""
    chunks = [Document(page_content=f"Chunk {i}") for i in range(5)]
    mock_encoder = MagicMock()
    # Ascending scores so top-3 are chunks 4, 3, 2 (in that order)
    mock_encoder.predict.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_loader.return_value = mock_encoder

    result = _rerank_chunks("query", chunks, top_k=3)

    assert len(result) == 3
    assert [doc.page_content for doc in result] == ["Chunk 4", "Chunk 3", "Chunk 2"]


def test_rerank_chunks_handles_empty_input():
    """Empty input must short-circuit to [] without trying to load the
    CrossEncoder model — saves a multi-second load on every empty-vector
    query (e.g. before the user has ingested anything)."""
    # No patch on _get_cross_encoder — if the function tries to load the
    # real model, the test would be slow/flaky.
    result = _rerank_chunks("any query", [])
    assert result == []


@patch("src.synthesis._get_cross_encoder")
def test_rerank_chunks_returns_all_when_fewer_than_top_k(mock_loader):
    """When the input has fewer chunks than top_k, return them all (sorted) —
    don't pad, don't error."""
    chunks = [Document(page_content="A"), Document(page_content="B")]
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.5, 0.9]
    mock_loader.return_value = mock_encoder

    result = _rerank_chunks("query", chunks, top_k=3)

    assert len(result) == 2
    # Higher-scoring chunk first
    assert [doc.page_content for doc in result] == ["B", "A"]


@patch("src.synthesis._get_cross_encoder")
def test_rerank_chunks_passes_query_chunk_pairs_to_encoder(mock_loader):
    """CrossEncoder.predict expects (query, chunk_text) pairs — one per chunk."""
    chunks = [Document(page_content="alpha"), Document(page_content="beta")]
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.1, 0.9]
    mock_loader.return_value = mock_encoder

    _rerank_chunks("test query", chunks)

    mock_encoder.predict.assert_called_once()
    pairs = mock_encoder.predict.call_args[0][0]
    assert len(pairs) == 2
    # Each pair: query first, chunk text second
    assert pairs[0][0] == "test query" and pairs[0][1] == "alpha"
    assert pairs[1][0] == "test query" and pairs[1][1] == "beta"


def test_cross_encoder_model_constant_matches_spec():
    """The reranker must use the exact model the ticket specifies. Hard-coded
    so the constant is part of the contract, not an implementation detail."""
    assert CROSS_ENCODER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
