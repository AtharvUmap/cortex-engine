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

import logging
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.graph_extractor import extract_graph_triples, extract_attributes
from src.graph_store import GraphStore
from src.factual_index import extract_facts, save_facts
from src.document_owner import build_owner_index, save_owners

logger = logging.getLogger(__name__)


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

# --- Chunk-quality filter -------------------------------------------------
# PDFs produce a lot of garbage chunks: standalone headers like "Nanopore
# Raw Signal", near-empty whitespace blocks from page breaks, and base64
# blobs from embedded font streams. These chunks have embeddings close to
# many queries (they contain little distinguishing content), so they pollute
# top-K retrieval. We drop them at ingestion time so Chroma never sees them.

# Minimum useful chunk length after stripping whitespace. Short chunks are
# usually headers or page-break artifacts, not searchable content.
MIN_CHUNK_CHARS = 100

# Minimum ratio of letters to total characters. Real prose runs 70-90%
# alphabetic; base64 and binary streams can dip below 10% when digit- or
# symbol-heavy. 0.3 is a comfortable floor that keeps legitimate numeric-heavy
# text (dates, IDs, account numbers).
MIN_ALPHA_RATIO = 0.3

# Minimum count of whitespace-separated tokens. A 400-char prose chunk has
# 60-80 tokens; a pure-letter base64 blob like "LKP2Hggv37h0Vc6zkvfCRHZqJv..."
# has exactly 1 (no spaces anywhere). This is what distinguishes prose from
# embedded PDF font streams when alpha ratio alone can't tell them apart.
MIN_WORD_COUNT = 10


def is_meaningful_chunk(text: str) -> bool:
    """True if a text chunk is worth embedding and storing.

    Returns False for chunks that are too short, too non-alphabetic, or
    too-few-tokens to contribute useful search signal (blank pages, headers,
    base64 font blobs extracted from embedded PDF streams).
    """
    stripped = text.strip()
    if len(stripped) < MIN_CHUNK_CHARS:
        return False
    alpha_count = sum(c.isalpha() for c in stripped)
    if (alpha_count / len(stripped)) < MIN_ALPHA_RATIO:
        return False
    # A chunk with only one or two whitespace-separated tokens is almost
    # certainly not prose, even if its alpha ratio is high.
    return len(stripped.split()) >= MIN_WORD_COUNT


class _FilteringTextSplitter(RecursiveCharacterTextSplitter):
    """A RecursiveCharacterTextSplitter that drops low-signal chunks.

    Subclass of the standard splitter; after splitting, it removes any
    chunks that fail is_meaningful_chunk. Used for the child splitter so
    junk never makes it into the vector index.
    """

    def split_documents(self, documents):
        chunks = super().split_documents(documents)
        return [c for c in chunks if is_meaningful_chunk(c.page_content)]


def _annotate_owners(documents: list, owners: dict) -> None:
    """Set metadata['owner'] on every input doc from the owners map.

    Done in-place before the splitter runs so every parent and child chunk
    inherits the owner tag — the langchain text splitters propagate metadata
    automatically. Docs whose source isn't in the owners map (or whose owner
    is None) get no 'owner' key, so vector retrieval can still surface them
    on generic queries that don't filter by owner.
    """
    for doc in documents:
        source = (doc.metadata or {}).get("source", "")
        entry = owners.get(source) or {}
        owner = entry.get("owner")
        if owner:
            # Mutate in place — splitter sees the updated metadata.
            doc.metadata = {**(doc.metadata or {}), "owner": owner}


def embed_documents(
    documents: list,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    build_graph: bool = True,
    build_facts: bool = True,
    build_owners: bool = True,
):
    """Embed documents using Parent-Child retrieval and store them persistently.

    Also (when build_graph=True) runs each input document through the LLM-backed
    triple extractor and persists the resulting knowledge graph to
    {persist_directory}/graph.graphml. And (when build_facts=True) scans every
    document with FACT_PATTERNS and persists a regex-built factual index to
    {persist_directory}/facts.json. Ticket 23 added (when build_owners=True)
    a per-document owner inference pass that writes
    {persist_directory}/owners.json and binds each fact + chunk + graph entity
    to the document's primary owner so synthesis-time filtering can reject
    cross-owner contamination.

    Args:
        documents: A list of LangChain Document objects (one per source file).
        persist_directory: Root directory for the vector DB and parent docstore.
        build_graph: If True (default), extract triples and save the knowledge
            graph. Set False for fast re-indexing when the graph is already
            current or intentionally not wanted.
        build_facts: If True (default), scan documents for structured facts
            (emails, phones, IDs) and write facts.json. Set False for fast
            re-indexing when the fact index is already current.
        build_owners: If True (default), infer per-document primary entity and
            write owners.json. Without this, every fact is recorded with
            entity=None and lookup falls back to legacy non-filtered behavior.

    Returns:
        A ParentDocumentRetriever that searches children and returns parents.
    """
    # Step 0 (optional): Per-document owner inference (Ticket 23). Run first
    # so the inferred owners can flow into vector chunk metadata, graph
    # canonicals, and the factual index in subsequent steps.
    owners: dict = {}
    if build_owners:
        owners = build_owner_index(documents)
        save_owners(owners, Path(persist_directory) / "owners.json")
        # Annotate docs in place so chunks inherit owner metadata at split time.
        _annotate_owners(documents, owners)

    # Step 1: Define the two splitters.
    # Overlap is important — without it, a sentence split across two chunks
    # loses its meaning in both halves, hurting retrieval accuracy.
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    # Child splitter uses the filtering variant: after chunking, it drops
    # low-signal chunks (whitespace, short headers, base64 blobs) so they
    # never land in Chroma and never compete for a top-K slot at query time.
    child_splitter = _FilteringTextSplitter(
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

    # Step 6 (optional): Build the knowledge graph alongside the vector store.
    # Two extraction passes per document, both feeding into one NetworkX DiGraph:
    #   Pass 1 (extract_graph_triples) — entity-to-entity relationships.
    #   Pass 2 (extract_attributes) — entity-field-value attribute facts
    #       (phone numbers, expiry dates, identification numbers) that the
    #       relationship prompt would otherwise crowd out.
    # Each pass is best-effort and wrapped in its own try/except so a single
    # noisy doc or LLM hiccup cannot wreck the whole ingestion run, and so
    # a failure on one pass does not block the other pass for the same doc.
    if build_graph:
        graph_path = Path(persist_directory) / "graph.graphml"
        graph_store = GraphStore(graph_path=graph_path)

        # Seed canonical names from the inferred owners BEFORE any triplets
        # land. An OCR variant of a known owner ("ATHARV AMAR" for "Atharv
        # Umap") will fuzzy-resolve into the seeded canonical instead of
        # spawning a duplicate node.
        for entry in owners.values():
            if isinstance(entry, dict) and entry.get("owner"):
                graph_store.register_canonical(entry["owner"])

        for doc in documents:
            source = doc.metadata.get("source", "<unknown>")

            # Pass 1: relationships
            try:
                triples = extract_graph_triples(doc)
            except Exception as exc:
                logger.warning(
                    "Relationship extraction failed for %s: %s — continuing.",
                    source, exc,
                )
                triples = []
            graph_store.add_triplets(triples)

            # Pass 2: attributes. Convert to triplet shape
            # (entity, value, field) so the existing graph store can ingest
            # them without a schema change. The visualizer and graph walker
            # then treat attribute edges identically to relationship edges.
            try:
                attributes = extract_attributes(doc)
            except Exception as exc:
                logger.warning(
                    "Attribute extraction failed for %s: %s — continuing.",
                    source, exc,
                )
                attributes = []
            attribute_triplets = [
                {
                    "source": a["entity"],
                    "target": a["value"],
                    "relationship": a["field"],
                }
                for a in attributes
            ]
            graph_store.add_triplets(attribute_triplets)

        graph_store.save()

    # Step 7 (optional): Build the regex-first factual index (Ticket 20).
    # Scans the raw input documents for emails, phones, and structured IDs;
    # persists the matches to facts.json. Ticket 23 attaches each match to
    # the document's owner (from Step 0) so generate_answer can filter by
    # the entity named in the user's query.
    if build_facts:
        facts = extract_facts(documents, owners=owners)
        save_facts(facts, Path(persist_directory) / "facts.json")

    return retriever
