from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.embedding import (
    embed_documents,
    is_meaningful_chunk,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
)
from src.graph_store import GraphStore


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

@patch("src.embedding.extract_attributes", return_value=[])
@patch("src.embedding.extract_graph_triples", return_value=[])
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_returns_parent_document_retriever(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """embed_documents should return a ParentDocumentRetriever instance."""
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


@patch("src.embedding.extract_attributes", return_value=[])
@patch("src.embedding.extract_graph_triples", return_value=[])
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_stores_children_in_chroma(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """Child chunks should be stored in the ChromaDB vectorstore."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    docs = _make_long_document()
    db_path = str(tmp_path / "test_db")

    retriever = embed_documents(docs, persist_directory=db_path)

    child_count = retriever.vectorstore._collection.count()
    assert child_count > 0


@patch("src.embedding.extract_attributes", return_value=[])
@patch("src.embedding.extract_graph_triples", return_value=[])
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_stores_parents_in_filestore(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """Parent chunks should be stored in the LocalFileStore under ./db/docstore."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    docs = _make_long_document()
    db_path = str(tmp_path / "test_db")

    retriever = embed_documents(docs, persist_directory=db_path)

    parent_keys = list(retriever.docstore.yield_keys())
    assert len(parent_keys) > 0


@patch("src.embedding.extract_attributes", return_value=[])
@patch("src.embedding.extract_graph_triples", return_value=[])
@patch("src.embedding.OllamaEmbeddings")
def test_children_outnumber_parents(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
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

    assert child_count > parent_count


def test_chunk_size_constants():
    """Confirm the chunk size constants match the Parent-Child retrieval spec."""
    assert PARENT_CHUNK_SIZE == 2000
    assert CHILD_CHUNK_SIZE == 400


# --- is_meaningful_chunk filter tests ---

def test_is_meaningful_chunk_accepts_normal_prose():
    """Real document text should pass the filter."""
    text = (
        "Atharv Umap is a senior at the University of Maryland studying "
        "Computer Science. He has experience building web applications "
        "and working with machine learning frameworks."
    )
    assert is_meaningful_chunk(text) is True


def test_is_meaningful_chunk_rejects_whitespace_only():
    """Pure whitespace / near-empty chunks should be filtered out."""
    assert is_meaningful_chunk("   \n\n\n   \t   ") is False
    assert is_meaningful_chunk("") is False


def test_is_meaningful_chunk_rejects_short_chunks():
    """Chunks shorter than the minimum length (like lone headers) should be filtered."""
    # "Nanopore Raw Signal" style header — under the 100-char floor
    assert is_meaningful_chunk("Nanopore Raw Signal") is False


def test_is_meaningful_chunk_rejects_base64_blobs():
    """Font/image streams extracted from PDFs should be filtered out.

    Real-world example from the RawHash PDF: long unbroken strings of mixed-
    case letters with no whitespace. Alpha ratio is high (~0.9), so the
    filter must reject them on word-count grounds instead.
    """
    blob = "LKP2Hggv37h0Vc6zkvfCRHZqJv1ryPlamOoot1TUjSECLwtVDYNGwnYUsKSKYMPmHiCsqNcK8QQphI" * 3
    assert is_meaningful_chunk(blob) is False


def test_is_meaningful_chunk_accepts_prose_with_some_numbers():
    """Normal prose that contains dates or numbers should still pass."""
    text = (
        "The I-20 document was issued on 2026-01-15 with program end date "
        "2026-12-20. Degree level is Bachelor of Science in Computer Science. "
        "The student has maintained full-time enrollment for the entire period."
    )
    assert is_meaningful_chunk(text) is True


# --- Graph ingestion wiring (Ticket 15) ---

def _multi_docs():
    """Two documents, each long enough to survive the is_meaningful_chunk
    filter (>= 100 chars + >= 10 words). Needed because short docs get
    dropped by the child splitter and Chroma errors on empty upserts."""
    doc1 = (
        "Atharv Umap attends the University of Maryland in College Park. "
        "He is currently studying Computer Science and has been working on "
        "several interesting software engineering and research projects."
    )
    doc2 = (
        "The University of Maryland is located in the United States of America, "
        "specifically in the state of Maryland. It is a large public research "
        "university with a strong computer science and engineering program."
    )
    return [
        Document(page_content=doc1, metadata={"source": "resume.pdf"}),
        Document(page_content=doc2, metadata={"source": "about_umd.pdf"}),
    ]


@patch("src.embedding.extract_attributes", return_value=[])
@patch("src.embedding.extract_graph_triples", return_value=[])
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_extracts_triples_per_document(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """Graph extraction should run exactly once per input document by default."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    docs = _multi_docs()
    db_path = str(tmp_path / "test_db")

    embed_documents(docs, persist_directory=db_path)

    assert mock_extract.call_count == len(docs)


@patch("src.embedding.extract_attributes", return_value=[])
@patch("src.embedding.extract_graph_triples")
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_persists_graph_to_disk(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """The extracted triples should be saved to graph.graphml under persist_directory."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    # Return a distinct triple for each doc so we can verify persistence
    mock_extract.side_effect = [
        [{"source": "Atharv", "target": "Maryland", "relationship": "attends"}],
        [{"source": "Maryland", "target": "USA", "relationship": "located_in"}],
    ]

    docs = _multi_docs()
    db_path = str(tmp_path / "test_db")

    embed_documents(docs, persist_directory=db_path)

    # File must exist on disk
    graph_file = Path(db_path) / "graph.graphml"
    assert graph_file.exists()

    # Reload from disk and confirm both triples landed
    reloaded = GraphStore(graph_path=graph_file)
    assert reloaded.graph.has_edge("Atharv", "Maryland")
    assert reloaded.graph["Atharv"]["Maryland"]["relationship"] == "attends"
    assert reloaded.graph.has_edge("Maryland", "USA")
    assert reloaded.graph["Maryland"]["USA"]["relationship"] == "located_in"


@patch("src.embedding.extract_attributes")
@patch("src.embedding.extract_graph_triples")
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_skips_graph_when_build_graph_false(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """build_graph=False should skip BOTH extraction passes entirely — useful for
    fast re-indexing when the graph is already up to date or explicitly not wanted."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    docs = _multi_docs()
    db_path = str(tmp_path / "test_db")

    embed_documents(docs, persist_directory=db_path, build_graph=False)

    mock_extract.assert_not_called()
    mock_extract_attrs.assert_not_called()
    assert not (Path(db_path) / "graph.graphml").exists()


@patch("src.embedding.extract_attributes", return_value=[])
@patch("src.embedding.extract_graph_triples")
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_continues_on_extraction_failure(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """If relationship extraction raises for one doc, ingestion should continue
    with the rest. Graph extraction is best-effort — a single noisy doc or LLM
    hiccup should not fail the whole vector ingestion."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    # First doc raises, second returns a valid triple
    mock_extract.side_effect = [
        RuntimeError("simulated ollama failure"),
        [{"source": "Maryland", "target": "USA", "relationship": "located_in"}],
    ]

    docs = _multi_docs()
    db_path = str(tmp_path / "test_db")

    # Should NOT raise
    retriever = embed_documents(docs, persist_directory=db_path)

    # Vector ingestion completed
    assert retriever.vectorstore._collection.count() > 0

    # The second doc's triple still made it into the graph
    reloaded = GraphStore(graph_path=Path(db_path) / "graph.graphml")
    assert reloaded.graph.has_edge("Maryland", "USA")


# --- Two-pass extraction wiring (Ticket 16.5) ------------------------------
# Pass 1 (relationships) and Pass 2 (attributes) must both run per document
# and both feed into the same persisted graph.

@patch("src.embedding.extract_attributes", return_value=[])
@patch("src.embedding.extract_graph_triples", return_value=[])
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_runs_attribute_extraction_per_document(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """Attribute extraction must run exactly once per input document — same
    cadence as relationship extraction so neither pass starves the other."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    docs = _multi_docs()
    db_path = str(tmp_path / "test_db")

    embed_documents(docs, persist_directory=db_path)

    assert mock_extract_attrs.call_count == len(docs)


@patch("src.embedding.extract_attributes")
@patch("src.embedding.extract_graph_triples", return_value=[])
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_persists_attributes_to_graph(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """Attributes extracted by Pass 2 should land in the same graph as
    relationship triples, encoded as (entity --[field]--> value) edges so the
    graph walker and visualizer can treat them uniformly."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    # Distinct attributes per doc so we can verify both reached the graph.
    mock_extract_attrs.side_effect = [
        [{"entity": "Atharv Umap", "field": "phone_number", "value": "240-555-1234"}],
        [{"entity": "I-20", "field": "expiry_date", "value": "2027-05-15"}],
    ]

    docs = _multi_docs()
    db_path = str(tmp_path / "test_db")

    embed_documents(docs, persist_directory=db_path)

    reloaded = GraphStore(graph_path=Path(db_path) / "graph.graphml")
    # Each attribute becomes an edge: source=entity, target=value, relationship=field.
    assert reloaded.graph.has_edge("Atharv Umap", "240-555-1234")
    assert reloaded.graph["Atharv Umap"]["240-555-1234"]["relationship"] == "phone_number"
    assert reloaded.graph.has_edge("I-20", "2027-05-15")
    assert reloaded.graph["I-20"]["2027-05-15"]["relationship"] == "expiry_date"


@patch("src.embedding.extract_attributes")
@patch("src.embedding.extract_graph_triples", return_value=[])
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_continues_when_attribute_extraction_fails(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """If attribute extraction raises for one doc, the other doc's attributes
    should still land. The two passes are independent — neither failure mode
    should poison the rest of the run."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    mock_extract_attrs.side_effect = [
        RuntimeError("simulated attribute extraction failure"),
        [{"entity": "I-20", "field": "expiry_date", "value": "2027-05-15"}],
    ]

    docs = _multi_docs()
    db_path = str(tmp_path / "test_db")

    # Should NOT raise
    retriever = embed_documents(docs, persist_directory=db_path)
    assert retriever.vectorstore._collection.count() > 0

    reloaded = GraphStore(graph_path=Path(db_path) / "graph.graphml")
    assert reloaded.graph.has_edge("I-20", "2027-05-15")


@patch("src.embedding.extract_attributes")
@patch("src.embedding.extract_graph_triples")
@patch("src.embedding.OllamaEmbeddings")
def test_embed_documents_attribute_failure_does_not_block_relationships(
    mock_ollama_cls, mock_extract, mock_extract_attrs, tmp_path
):
    """A failure in Pass 2 (attributes) for a given doc must not prevent
    Pass 1 (relationships) results from that same doc reaching the graph."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = _fake_embedding_function
    mock_embeddings.embed_query.side_effect = lambda t: _fake_embedding_function([t])[0]
    mock_ollama_cls.return_value = mock_embeddings

    # Doc 1: relationship works, attribute pass blows up.
    mock_extract.side_effect = [
        [{"source": "Atharv", "target": "Maryland", "relationship": "attends"}],
        [],
    ]
    mock_extract_attrs.side_effect = [
        RuntimeError("simulated failure on attribute pass"),
        [],
    ]

    docs = _multi_docs()
    db_path = str(tmp_path / "test_db")

    embed_documents(docs, persist_directory=db_path)

    reloaded = GraphStore(graph_path=Path(db_path) / "graph.graphml")
    assert reloaded.graph.has_edge("Atharv", "Maryland")
