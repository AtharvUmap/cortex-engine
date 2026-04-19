"""
Parent-Child + Multi-Query Retrieval Search Module
---------------------------------------------------
Reconstructs the ParentDocumentRetriever from disk (ChromaDB + LocalFileStore)
and wraps it with MultiQueryRetriever for robustness against synonym mismatches.

The retriever:
  1. An LLM rephrases the user's query into several semantically equivalent
     variations (e.g. "roles" -> "positions", "responsibilities", "job titles").
  2. For each variation, embed it and run similarity search over the small
     CHILD vectors in ChromaDB.
  3. For each matching child, fetch its PARENT document from the LocalFileStore.
  4. Deduplicate parents across all query variations and return the union.

This dramatically reduces failures caused by the user phrasing a question
differently from the language in the source document.
"""

import os
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embedding import (
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    DEFAULT_PERSIST_DIR,
)

# Number of child chunks to retrieve from ChromaDB per query variation.
# With multi-query, the final parent count is the dedup'd union across queries.
#
# Raised from 6 to 15 after diagnosing a case where the resume's top-scoring
# child ranked 12th for a query like "What is Atharv Umap's experience."
# Noise chunks (near-empty headers, font-stream blobs from PDFs) were pushing
# legitimate content below a tight k=6 cutoff. Parent dedup collapses the
# redundant children so the final context stays focused.
RETRIEVAL_K = 15

# LLM used to generate query variations for multi-query retrieval.
# Pulled from the same env var as the main QA LLM so users can change both
# at once with `CORTEX_LLM_MODEL=llama3.1:8b streamlit run app.py`.
DEFAULT_LLM_MODEL = "llama3.2"
MULTI_QUERY_LLM_MODEL = os.environ.get("CORTEX_LLM_MODEL", DEFAULT_LLM_MODEL)

# Toggle for multi-query retrieval. Default is OFF because for mixed-document
# use cases (resume + visa + I-20 + receipts in the same data/ folder), the
# LLM's rephrasings tend to drift across documents and pull in noise.
# Set CORTEX_MULTI_QUERY=true to re-enable for single-document workloads
# where synonym mismatches within one doc are the main failure mode.
USE_MULTI_QUERY = os.environ.get("CORTEX_MULTI_QUERY", "false").lower() == "true"


def _build_retriever(persist_directory: str) -> ParentDocumentRetriever:
    """Rebuild the ParentDocumentRetriever from persisted storage.

    This connects to the existing ChromaDB collection and LocalFileStore that
    were populated during ingestion, so we can run queries without re-ingesting.
    """
    # Use the same embedding model that was used during ingestion, otherwise
    # the query vector would live in a different vector space than the stored
    # child vectors and similarity search would produce garbage.
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")

    # Reconnect to the child vectorstore (ChromaDB on disk).
    # The collection_name must match what was used in src/embedding.py.
    vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    # Reconnect to the parent docstore (LocalFileStore at ./db/docstore).
    docstore_path = Path(persist_directory) / "docstore"
    docstore_path.mkdir(parents=True, exist_ok=True)
    file_store = LocalFileStore(str(docstore_path))
    docstore = create_kv_docstore(file_store)

    # The splitters must match ingestion so the retriever knows the chunk
    # hierarchy. We don't re-split here, but the retriever keeps these as
    # configuration for any future ingestion calls.
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    # search_kwargs tunes how many child chunks are pulled from ChromaDB
    # per query. Each unique parent is returned once regardless of how many
    # of its children matched.
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": RETRIEVAL_K},
    )


def _wrap_with_multi_query(base_retriever: ParentDocumentRetriever) -> MultiQueryRetriever:
    """Wrap the Parent-Child retriever so the LLM generates query variations.

    MultiQueryRetriever internally:
      - Uses the LLM to produce N alternative phrasings of the user's query
      - Runs each phrasing through the wrapped retriever in parallel
      - Deduplicates results by document content
    """
    # A small local LLM is enough here — generating 3-5 rephrasings is a
    # light task compared to answering the final question.
    query_gen_llm = OllamaLLM(model=MULTI_QUERY_LLM_MODEL)
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=query_gen_llm,
    )


def search(query: str, persist_directory: str = DEFAULT_PERSIST_DIR) -> list:
    """Search for documents relevant to the query.

    By default, uses Multi-Query + Parent-Child retrieval for robustness.
    Set CORTEX_MULTI_QUERY=false to disable multi-query for speed.

    Args:
        query: The user's search query string.
        persist_directory: Path to the persisted ChromaDB + docstore directory.

    Returns:
        A list of parent-sized Document objects. Empty list if no documents
        have been ingested yet.
    """
    base_retriever = _build_retriever(persist_directory)

    # If no children have been indexed yet, there's nothing to search.
    # Returning early avoids confusing errors from an empty collection and
    # also skips the expensive LLM rephrasing step.
    if base_retriever.vectorstore._collection.count() == 0:
        return []

    # Pick the retriever based on the multi-query toggle:
    #   - Multi-query: LLM rephrases -> multiple searches -> union of parents.
    #   - Single-query: one direct search through Parent-Child retrieval.
    if USE_MULTI_QUERY:
        retriever = _wrap_with_multi_query(base_retriever)
    else:
        retriever = base_retriever

    # invoke() runs the full pipeline and returns parent-sized Documents.
    return retriever.invoke(query)
