"""
Parent-Child Document Embedding Module
---------------------------------------
Implements the "small-to-big" retrieval pattern:
  - Documents are split into LARGE parent chunks (2000 chars) for rich context
  - Each parent is further split into SMALL child chunks (400 chars) for precise search
  - Children are embedded into ChromaDB; their vectors are what we actually search against
  - Parents live in a LocalFileStore on disk — when a child matches a query,
    we return its parent so the LLM gets more surrounding context

This gives us the precision of small-chunk similarity search AND the context
richness of large-chunk retrieval, all in one retriever.
"""

from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default directory where ChromaDB stores its vector data on disk
DEFAULT_PERSIST_DIR = "./db"

# Parent chunks: larger pieces of text returned as context for the LLM.
# Bigger = more context, but each parent still has to fit in the model's window.
PARENT_CHUNK_SIZE = 2000

# Overlap between adjacent parent chunks. Ensures ideas that straddle a chunk
# boundary aren't lost — both chunks carry a bit of the shared context.
PARENT_CHUNK_OVERLAP = 400

# Child chunks: small, focused pieces we actually embed and search against.
# Smaller chunks produce more precise semantic matches.
CHILD_CHUNK_SIZE = 400

# Overlap between adjacent child chunks. Small overlap prevents a sentence
# from being sliced in half at a chunk boundary, which would hurt retrieval.
CHILD_CHUNK_OVERLAP = 50


def embed_documents(documents: list, persist_directory: str = DEFAULT_PERSIST_DIR):
    """Embed documents using Parent-Child retrieval and store them persistently.

    Args:
        documents: A list of LangChain Document objects (one per source file).
        persist_directory: Root directory for the vector DB and parent docstore.

    Returns:
        A ParentDocumentRetriever that searches children and returns parents.
    """
    # Step 1: Define the two splitters.
    # Overlap is important — without it, a sentence split across two chunks
    # loses its meaning in both halves, hurting retrieval accuracy.
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    # Step 2: Set up the child vectorstore (ChromaDB).
    # This is where the *child* chunk embeddings live — the retriever searches here.
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    # Step 3: Set up the parent docstore (LocalFileStore on disk).
    # Parents are stored as key-value pairs under ./db/docstore so they persist
    # across restarts. `create_kv_docstore` wraps the raw byte store so it can
    # save/retrieve LangChain Document objects.
    docstore_path = Path(persist_directory) / "docstore"
    docstore_path.mkdir(parents=True, exist_ok=True)
    file_store = LocalFileStore(str(docstore_path))
    docstore = create_kv_docstore(file_store)

    # Step 4: Build the ParentDocumentRetriever.
    # It internally: splits docs -> parents -> children, embeds children into Chroma,
    # and keeps a parent_id -> parent mapping in the docstore.
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Step 5: Ingest the documents. This populates both ChromaDB (children) and
    # the LocalFileStore (parents) in a single call.
    retriever.add_documents(documents)

    return retriever
