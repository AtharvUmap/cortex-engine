from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.embedding import embed_documents, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE
from src.retriever import search


# --- Test helpers ---

def _fake_embedding_function(texts):
    """Return fake vectors so tests run without Ollama.
    Each text gets a deterministic vector based on its hash for differentiation.
    """
    return [[float(hash(t) % 1000) / 1000, 0.5, 0.5] for t in texts]


def _make_long_document_with_keyword(keyword: str):
    """Build a long Document containing the keyword, large enough to create
    multiple parent chunks of PARENT_CHUNK_SIZE.
    """
    # Filler text that mentions the keyword in exactly one spot
    filler = "Generic filler text with unrelated content. " * 200  # ~9000 chars
    # Inject the keyword roughly halfway through the text
    midpoint = len(filler) // 2
    injected = filler[:midpoint] + f" The secret keyword here is {keyword}. " + filler[midpoint:]
    return [Document(page_content=injected, metadata={"source": "test_doc.pdf"})]


def _setup_mock_embeddings(mock_ollama_cls):
    """Configure the mocked OllamaEmbeddings class for both ingestion and search."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings
    return mock_embeddings


def _setup_mock_llm(mock_llm_cls, query: str):
    """Configure OllamaLLM mock to return fake query variations.
    MultiQueryRetriever expects the LLM to produce newline-separated rephrasings.
    """
    mock_llm = MagicMock()
    # Return a few rephrasings of the original query so the multi-query path
    # exercises its full pipeline (rephrase -> search each -> dedupe).
    variations = f"{query}\nWhat about {query}?\nTell me about {query}\n"
    mock_llm.invoke.return_value = variations
    # MultiQueryRetriever uses `|` to pipe through an LLMChain, so we also
    # make sure the mock behaves as a Runnable.
    mock_llm.__or__ = lambda self, other: MagicMock(invoke=lambda x: variations)
    mock_llm_cls.return_value = mock_llm
    return mock_llm


# --- Tests ---

@patch("src.retriever.OllamaLLM")
@patch("src.retriever.OllamaEmbeddings")
@patch("src.embedding.OllamaEmbeddings")
def test_search_returns_list_of_documents(
    mock_embedding_ollama, mock_retriever_ollama, mock_llm_cls, tmp_path
):
    """search() should return a list of LangChain Document objects."""
    _setup_mock_embeddings(mock_embedding_ollama)
    _setup_mock_embeddings(mock_retriever_ollama)
    _setup_mock_llm(mock_llm_cls, "photosynthesis")

    db_path = str(tmp_path / "test_db")
    docs = _make_long_document_with_keyword("photosynthesis")
    embed_documents(docs, persist_directory=db_path)

    results = search("photosynthesis", persist_directory=db_path)

    assert isinstance(results, list)
    assert all(isinstance(doc, Document) for doc in results)


@patch("src.retriever.OllamaLLM")
@patch("src.retriever.OllamaEmbeddings")
@patch("src.embedding.OllamaEmbeddings")
def test_search_returns_parent_sized_chunks(
    mock_embedding_ollama, mock_retriever_ollama, mock_llm_cls, tmp_path
):
    """Returned documents should be parent-sized, not tiny child chunks.

    This is the core assertion of Parent-Child retrieval: even though we search
    over 400-char children, we return the full 2000-char parents for LLM context.
    """
    _setup_mock_embeddings(mock_embedding_ollama)
    _setup_mock_embeddings(mock_retriever_ollama)
    _setup_mock_llm(mock_llm_cls, "quantum")

    db_path = str(tmp_path / "test_db")
    docs = _make_long_document_with_keyword("quantum")
    embed_documents(docs, persist_directory=db_path)

    results = search("quantum", persist_directory=db_path)

    # Every returned chunk should be clearly longer than a child chunk.
    assert len(results) > 0
    for doc in results:
        assert len(doc.page_content) > CHILD_CHUNK_SIZE, (
            f"Expected parent-sized chunk (>{CHILD_CHUNK_SIZE} chars), "
            f"got {len(doc.page_content)} chars — looks like a child chunk was returned."
        )


@patch("src.retriever.OllamaLLM")
@patch("src.retriever.OllamaEmbeddings")
@patch("src.embedding.OllamaEmbeddings")
def test_search_results_respect_parent_chunk_size(
    mock_embedding_ollama, mock_retriever_ollama, mock_llm_cls, tmp_path
):
    """Returned chunks should not exceed the parent chunk size."""
    _setup_mock_embeddings(mock_embedding_ollama)
    _setup_mock_embeddings(mock_retriever_ollama)
    _setup_mock_llm(mock_llm_cls, "mitochondria")

    db_path = str(tmp_path / "test_db")
    docs = _make_long_document_with_keyword("mitochondria")
    embed_documents(docs, persist_directory=db_path)

    results = search("mitochondria", persist_directory=db_path)

    assert len(results) > 0
    # Allow a small overflow margin — splitters may include separators/overlap
    for doc in results:
        assert len(doc.page_content) <= PARENT_CHUNK_SIZE + 200


@patch("src.retriever.OllamaLLM")
@patch("src.retriever.OllamaEmbeddings")
@patch("src.embedding.OllamaEmbeddings")
def test_search_from_empty_database_returns_empty(
    mock_embedding_ollama, mock_retriever_ollama, mock_llm_cls, tmp_path
):
    """Searching before any documents are ingested should return an empty list."""
    _setup_mock_embeddings(mock_embedding_ollama)
    _setup_mock_embeddings(mock_retriever_ollama)
    _setup_mock_llm(mock_llm_cls, "anything")

    # Pointed at a fresh directory with no ingested documents
    db_path = str(tmp_path / "empty_db")

    results = search("anything", persist_directory=db_path)

    assert results == []


@patch("src.retriever.USE_MULTI_QUERY", True)
@patch("src.retriever.MultiQueryRetriever")
@patch("src.retriever.OllamaLLM")
@patch("src.retriever.OllamaEmbeddings")
@patch("src.embedding.OllamaEmbeddings")
def test_search_invokes_llm_for_query_variations(
    mock_embedding_ollama, mock_retriever_ollama, mock_llm_cls, mock_mqr_cls, tmp_path
):
    """When multi-query is enabled, search() should construct a MultiQueryRetriever
    and instantiate an OllamaLLM for it. Multi-query is OFF by default (see
    retriever.py for the cross-document drift reason), so we force it ON here.

    We stub out MultiQueryRetriever entirely — the real class builds an internal
    Runnable chain (prompt | llm | parser) that requires a concrete string from
    the LLM step, which our MagicMock can't satisfy. The assertion is about the
    wrapping path being taken, not the chain executing end-to-end.
    """
    _setup_mock_embeddings(mock_embedding_ollama)
    _setup_mock_embeddings(mock_retriever_ollama)
    _setup_mock_llm(mock_llm_cls, "roles")

    # Stub the MultiQueryRetriever so its .invoke() returns an empty list
    # instead of running the real Runnable pipeline.
    mock_mqr = MagicMock()
    mock_mqr.invoke.return_value = []
    mock_mqr_cls.from_llm.return_value = mock_mqr

    db_path = str(tmp_path / "test_db")
    docs = _make_long_document_with_keyword("roles")
    embed_documents(docs, persist_directory=db_path)

    search("What roles does X have", persist_directory=db_path)

    # The wrapper was constructed — proof we took the multi-query path.
    assert mock_mqr_cls.from_llm.called
    # And an LLM was instantiated to pass into the wrapper.
    assert mock_llm_cls.called
