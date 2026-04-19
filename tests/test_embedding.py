from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.embedding import embed_documents, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE


# --- Test helpers ---

def _make_long_document():
    """Create a single long Document that will produce multiple parent and child chunks.
    The text must exceed PARENT_CHUNK_SIZE to trigger parent splitting.
    """
    # Repeat a varied sentence to produce enough characters for both splits
    text = (
        "Artificial intelligence is transforming the modern world. "
        "Machine learning models can identify patterns in data. "
        "Deep learning networks learn hierarchical representations. "
        "Natural language processing enables computers to understand text. "
    ) * 40  # ~8000 chars - produces multiple parent chunks of 2000 chars
    return [Document(page_content=text, metadata={"source": "test_doc.pdf"})]


def _fake_embedding_function(texts):
    """Return fake vectors so tests run without Ollama.
    Each text gets a unique vector based on its hash for differentiation.
    """
    return [[float(hash(t) % 100) / 100, 0.5, 0.5] for t in texts]


# --- Tests ---

@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_returns_parent_document_retriever(mock_ollama_cls, tmp_path):
    """embed_documents should return a ParentDocumentRetriever instance."""
    # Mock Ollama so we don't need a running server
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    docs = _make_long_document()
    db_path = str(tmp_path / "test_db")

    retriever = embed_documents(docs, persist_directory=db_path)

    # The retriever should expose the ParentDocumentRetriever interface
    assert hasattr(retriever, "vectorstore")
    assert hasattr(retriever, "docstore")
    assert hasattr(retriever, "invoke")


@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_stores_children_in_chroma(mock_ollama_cls, tmp_path):
    """Child chunks should be stored in the ChromaDB vectorstore."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    docs = _make_long_document()
    db_path = str(tmp_path / "test_db")

    retriever = embed_documents(docs, persist_directory=db_path)

    # ChromaDB should contain the smaller child chunks
    child_count = retriever.vectorstore._collection.count()
    assert child_count > 0


@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_stores_parents_in_filestore(mock_ollama_cls, tmp_path):
    """Parent chunks should be stored in the LocalFileStore under ./db/docstore."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    docs = _make_long_document()
    db_path = str(tmp_path / "test_db")

    retriever = embed_documents(docs, persist_directory=db_path)

    # Collect keys from the docstore — at least one parent should exist
    parent_keys = list(retriever.docstore.yield_keys())
    assert len(parent_keys) > 0


@patch("src.embedding.OllamaEmbeddings")
def test_children_outnumber_parents(mock_ollama_cls, tmp_path):
    """There should be more child chunks than parent chunks.
    (Children are smaller, so each parent produces multiple children.)
    """
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    docs = _make_long_document()
    db_path = str(tmp_path / "test_db")

    retriever = embed_documents(docs, persist_directory=db_path)

    child_count = retriever.vectorstore._collection.count()
    parent_count = len(list(retriever.docstore.yield_keys()))

    # Each parent (2000 chars) should produce multiple children (400 chars each)
    assert child_count > parent_count


def test_chunk_size_constants():
    """Confirm the chunk size constants match the Parent-Child retrieval spec."""
    assert PARENT_CHUNK_SIZE == 2000
    assert CHILD_CHUNK_SIZE == 400
