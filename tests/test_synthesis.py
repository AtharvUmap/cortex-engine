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


# --- Direct-facts injection (Ticket 20) -----------------------------------

@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_injects_direct_facts_when_intent_matches(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """When the fact index has the answer for a known-intent query, the LLM
    prompt must contain a 'Direct facts' block with the actual value."""
    mock_load_facts.return_value = {
        "email": [("Atharv Umap", "atharvumap@gmail.com", "resume.pdf")],
    }
    mock_load_owners.return_value = {
        "resume.pdf": {"owner": "Atharv Umap", "confidence": "high"},
    }
    mock_llm = _setup_llm_mock(mock_llm_cls, "atharv", "atharvumap@gmail.com")
    mock_search.return_value = []
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("what is atharv's email address?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    assert "atharvumap@gmail.com" in final_prompt
    assert "Direct facts (from indexed documents):" in final_prompt


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_omits_direct_facts_when_index_empty(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """An empty fact index must not emit a stray 'Direct facts' header —
    the section is gated on having actual hits to display."""
    mock_load_facts.return_value = {}
    mock_load_owners.return_value = {}
    mock_llm = _setup_llm_mock(
        mock_llm_cls, "atharv", "I don't have enough context to answer that."
    )
    mock_search.return_value = _mock_vector_chunks()
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("what is atharv's email address?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    assert "Direct facts (from indexed documents):" not in final_prompt


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_omits_direct_facts_for_unknown_intent(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """A query whose intent doesn't match any field must skip the direct-facts
    block even if the index has data — prevents irrelevant fact dumps."""
    mock_load_facts.return_value = {
        "email": [("Atharv Umap", "atharvumap@gmail.com", "resume.pdf")],
    }
    mock_load_owners.return_value = {
        "resume.pdf": {"owner": "Atharv Umap", "confidence": "high"},
    }
    mock_llm = _setup_llm_mock(
        mock_llm_cls, "Atharv", "Atharv attended Maryland."
    )
    mock_search.return_value = _mock_vector_chunks()
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("Where did Atharv go to school?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    assert "Direct facts (from indexed documents):" not in final_prompt


# --- Owner-aware filtering (Ticket 23) ------------------------------------

@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_omits_direct_facts_when_query_owner_unknown(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """The screenshot bug: query asks about a person who isn't an owner of any
    indexed document. Direct-facts block must be suppressed entirely so the
    LLM cannot misattribute the only email in the corpus."""
    mock_load_facts.return_value = {
        "email": [("Atharv Umap", "atharvumap@gmail.com", "resume.pdf")],
    }
    mock_load_owners.return_value = {
        "resume.pdf": {"owner": "Atharv Umap", "confidence": "high"},
    }
    mock_llm = _setup_llm_mock(
        mock_llm_cls, "Amar Umap", "I don't have enough context to answer that."
    )
    mock_search.return_value = []
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("what is amar umap's email?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    assert "Direct facts (from indexed documents):" not in final_prompt
    # And critically, the email must not appear anywhere in the prompt
    assert "atharvumap@gmail.com" not in final_prompt


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_filters_vector_chunks_by_owner_when_query_owner_unknown(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """When the query names someone unknown to the corpus, vector chunks (which
    inherit owner metadata at ingest) must also be filtered out — otherwise the
    LLM still sees Atharv's resume content and may misattribute it."""
    mock_load_facts.return_value = {}
    mock_load_owners.return_value = {
        "resume.pdf": {"owner": "Atharv Umap", "confidence": "high"},
    }
    chunks = [
        Document(
            page_content="Atharv attended Maryland.",
            metadata={"source": "resume.pdf", "owner": "Atharv Umap"},
        ),
    ]
    mock_search.return_value = chunks
    mock_llm = _setup_llm_mock(
        mock_llm_cls, "Amar Umap", "I don't have enough context to answer that."
    )
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("what is amar umap's email?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    # Atharv's content must NOT appear when the user asked about a different,
    # unknown person.
    assert "Atharv attended Maryland" not in final_prompt


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_keeps_vector_chunks_when_owner_matches(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """Sanity check the inverse: when the queried entity DOES match an owner,
    that owner's chunks must reach the prompt unchanged."""
    mock_load_facts.return_value = {}
    mock_load_owners.return_value = {
        "resume.pdf": {"owner": "Atharv Umap", "confidence": "high"},
    }
    chunks = [
        Document(
            page_content="Atharv attended Maryland.",
            metadata={"source": "resume.pdf", "owner": "Atharv Umap"},
        ),
    ]
    mock_search.return_value = chunks
    mock_llm = _setup_llm_mock(mock_llm_cls, "Atharv", "Atharv attended Maryland.")
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("Where did Atharv go to school?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    assert "Atharv attended Maryland" in final_prompt


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_filters_chunks_by_owner_in_multi_owner_corpus(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """Multi-owner corpus, query about one owner — only that owner's chunks
    survive into the prompt."""
    mock_load_facts.return_value = {}
    mock_load_owners.return_value = {
        "atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"},
        "sneha.pdf":  {"owner": "Sneha Umap",  "confidence": "high"},
    }
    chunks = [
        Document(
            page_content="Atharv works in software.",
            metadata={"source": "atharv.pdf", "owner": "Atharv Umap"},
        ),
        Document(
            page_content="Sneha works in finance.",
            metadata={"source": "sneha.pdf", "owner": "Sneha Umap"},
        ),
    ]
    mock_search.return_value = chunks
    mock_llm = _setup_llm_mock(mock_llm_cls, "Atharv", "Atharv works in software.")
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("What does Atharv do?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    assert "Atharv works in software" in final_prompt
    assert "Sneha works in finance" not in final_prompt


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_direct_facts_block_includes_owner_in_rendered_line(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """The rendered direct-facts line must include the owner so the LLM
    knows whose fact this is — and whose it isn't."""
    mock_load_facts.return_value = {
        "email": [("Atharv Umap", "atharvumap@gmail.com", "resume.pdf")],
    }
    mock_load_owners.return_value = {
        "resume.pdf": {"owner": "Atharv Umap", "confidence": "high"},
    }
    mock_llm = _setup_llm_mock(mock_llm_cls, "Atharv", "atharvumap@gmail.com")
    mock_search.return_value = []
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("what is atharv's email?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    # Owner must be visible inline with the fact, not just in the source tag
    assert "Atharv Umap" in final_prompt
    assert "atharvumap@gmail.com" in final_prompt


# --- Signal-aware filtering (Ticket 23.5) --------------------------------

@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_keeps_chunks_for_weak_signal_no_match(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """Query about a non-person capitalized noun ('Maryland') against a corpus
    owned by Atharv. Ticket 23 wrongly suppressed all chunks here; 23.5 keeps
    them because the signal is weak (no possessive 'Maryland's')."""
    mock_load_facts.return_value = {}
    mock_load_owners.return_value = {
        "atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"},
    }
    chunks = [
        Document(
            page_content="The University of Maryland is in College Park.",
            metadata={"source": "atharv.pdf", "owner": "Atharv Umap"},
        ),
    ]
    mock_search.return_value = chunks
    mock_llm = _setup_llm_mock(mock_llm_cls, "Maryland", "Maryland is in College Park.")
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("Tell me about Maryland")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    # Weak-signal query about a non-owner must NOT suppress vector context.
    assert "University of Maryland" in final_prompt


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_keeps_unattributed_chunks_alongside_matched_owner(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """Owner-tagged chunk + unattributed chunk both reach the prompt when the
    query matches the tagged owner. Ticket 23 dropped the unattributed one,
    making failed-inference docs invisible. 23.5 includes them."""
    mock_load_facts.return_value = {}
    mock_load_owners.return_value = {
        "atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"},
        "scan.pdf":   {"owner": None,         "confidence": "none"},
    }
    chunks = [
        Document(
            page_content="Atharv works in software.",
            metadata={"source": "atharv.pdf", "owner": "Atharv Umap"},
        ),
        Document(
            page_content="Note found in unsorted scan.",
            metadata={"source": "scan.pdf"},  # no 'owner' key — inference failed
        ),
    ]
    mock_search.return_value = chunks
    mock_llm = _setup_llm_mock(mock_llm_cls, "Atharv", "Atharv works in software.")
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("What does Atharv do?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    assert "Atharv works in software" in final_prompt
    # The unattributed chunk must also reach the prompt.
    assert "Note found in unsorted scan" in final_prompt


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_falls_through_when_all_owners_null(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """owners.json with every entry null (all inferences failed) must behave
    like no owners.json at all — no filtering, every named query gets all
    chunks. Otherwise every named query suppresses against an empty owner set."""
    mock_load_facts.return_value = {}
    mock_load_owners.return_value = {
        "scan1.pdf": {"owner": None, "confidence": "none"},
        "scan2.pdf": {"owner": None, "confidence": "none"},
    }
    chunks = [
        Document(page_content="First scan content.", metadata={"source": "scan1.pdf"}),
        Document(page_content="Second scan content.", metadata={"source": "scan2.pdf"}),
    ]
    mock_search.return_value = chunks
    mock_llm = _setup_llm_mock(mock_llm_cls, "Atharv", "Some answer.")
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("What does Atharv do?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    assert "First scan content" in final_prompt
    assert "Second scan content" in final_prompt


@patch("src.synthesis._rerank_chunks", side_effect=_passthrough_rerank)
@patch("src.synthesis.GraphStore")
@patch("src.synthesis.search")
@patch("src.synthesis.OllamaLLM")
@patch("src.synthesis.load_owners")
@patch("src.synthesis.load_facts")
def test_generate_answer_still_suppresses_on_strong_signal_no_match(
    mock_load_facts, mock_load_owners, mock_llm_cls, mock_search, mock_graph_cls, mock_rerank
):
    """Regression guard: the Ticket 23 cross-attribution fix must survive.
    Strong-signal possessive about an unknown person → no chunks, no facts."""
    mock_load_facts.return_value = {
        "email": [("Atharv Umap", "atharvumap@gmail.com", "atharv.pdf")],
    }
    mock_load_owners.return_value = {
        "atharv.pdf": {"owner": "Atharv Umap", "confidence": "high"},
    }
    chunks = [
        Document(
            page_content="Atharv attended Maryland.",
            metadata={"source": "atharv.pdf", "owner": "Atharv Umap"},
        ),
    ]
    mock_search.return_value = chunks
    mock_llm = _setup_llm_mock(
        mock_llm_cls, "Amar Umap", "I don't have enough context to answer that."
    )
    mock_graph = MagicMock()
    mock_graph.get_neighborhood.return_value = []
    mock_graph_cls.return_value = mock_graph

    generate_answer("what is amar umap's email?")

    final_prompt = str(mock_llm.invoke.call_args_list[1][0][0])
    # Both direct facts and vector chunks must be suppressed for strong+no_match.
    assert "atharvumap@gmail.com" not in final_prompt
    assert "Atharv attended Maryland" not in final_prompt
