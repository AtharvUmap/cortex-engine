from langchain_core.documents import Document

from src.splitter import split_documents


def _make_long_document(length=3000):
    """Create a single Document with a long repeating text."""
    text = "The quick brown fox jumps over the lazy dog. " * (length // 45 + 1)
    return [Document(page_content=text[:length])]


def test_split_returns_multiple_chunks():
    """A long document should be split into more than one chunk."""
    docs = _make_long_document(3000)

    chunks = split_documents(docs)

    assert isinstance(chunks, list)
    assert len(chunks) > 1


def test_no_chunk_exceeds_max_size():
    """No chunk should exceed the 1000 character limit."""
    docs = _make_long_document(3000)

    chunks = split_documents(docs)

    for chunk in chunks:
        assert len(chunk.page_content) <= 1000


def test_chunks_are_document_objects():
    """Each chunk should be a LangChain Document object."""
    docs = _make_long_document(3000)

    chunks = split_documents(docs)

    assert all(isinstance(chunk, Document) for chunk in chunks)


def test_short_document_returns_single_chunk():
    """A document shorter than chunk_size should not be split."""
    docs = [Document(page_content="Short text.")]

    chunks = split_documents(docs)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."
