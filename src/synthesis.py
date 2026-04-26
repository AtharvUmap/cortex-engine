"""
Hybrid Synthesis Engine
-----------------------
The final stage of Cortex Engine's retrieval pipeline. Combines the two
retrieval sources — Parent-Child vector search and the NetworkX knowledge
graph — into a single answer.

Pipeline:
  1. Extract the core entity from the user's query (small LLM call).
  2. Vector search: fetch parent chunks semantically similar to the query.
  3. Graph walk: fetch every triplet within 2 hops of the extracted entity.
  4. Format both into one context block and prompt the LLM for a final answer,
     still constrained to "context-only, no outside knowledge."

The split matters: vector retrieval is great at "find passages that sound
like the question"; graph retrieval is great at "what else is connected
to this entity?" Neither alone is as strong as both together.
"""

import os
import re
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from src.retriever import search
from src.graph_store import GraphStore

# Reuse the shared LLM env var so one setting controls the whole app.
DEFAULT_LLM_MODEL = "llama3.2"
LLM_MODEL = os.environ.get("CORTEX_LLM_MODEL", DEFAULT_LLM_MODEL)

# How many hops out from the entity we walk in the knowledge graph. 2 is
# the same default used in GraphStore and has been reliable for personal
# document vaults where entity clusters stay small and tight.
GRAPH_DEPTH = 2

# --- Cross-encoder re-ranking (Ticket 18) ---------------------------------
# Bi-encoder vector retrieval (the Parent-Child store) is fast but coarse:
# it ranks chunks by approximate semantic similarity in a shared embedding
# space. A cross-encoder runs the query and each candidate chunk through a
# transformer together, scoring them on actual relevance — much more
# accurate but too slow to run over the entire corpus. The trick is to use
# the bi-encoder to fetch a short candidate list (e.g. RETRIEVAL_K=15 in
# retriever.py) and then have the cross-encoder rerank just those, keeping
# only the top few. That's what _rerank_chunks does below.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Number of chunks to keep after re-ranking. 3 is small enough to leave
# room for graph context in the final prompt, large enough to capture
# multiple supporting passages when the answer spans documents.
RERANK_TOP_K = 3

# Lazy-loaded module-level cache. Loading the cross-encoder takes several
# seconds (model weights + tokenizer) and we don't want to pay that cost on
# import — only the first time a query actually needs reranking.
_cross_encoder_instance = None

# --- Entity extraction prompt ---------------------------------------------
# Ask the LLM for the main noun phrase so we can query the graph with it.
# Strict single-line output keeps post-processing simple. If the LLM adds
# quotes or whitespace, _clean_entity handles it.
ENTITY_EXTRACTION_PROMPT = """You are an entity extraction engine. Read the user's question and return the single most important named entity it asks about.

Rules:
- Return ONLY the entity name, nothing else.
- No quotes, no punctuation, no explanation, no prefix like "Entity:".
- If the question has no clear named entity, return the most important noun phrase.
- Keep the answer to a few words at most.

Question: {query}
"""

# --- Final answer prompt --------------------------------------------------
# Flat string prompt (OllamaLLM is a completion model, not a chat model).
# Kept intentionally close to the old qa_chain prompt that was working —
# small models like llama3.2 (3B) get refusal-happy when given multiple
# labeled context sections and redundant instructions.
#
# When the graph is empty, _build_context() omits the graph block entirely
# so the model sees a single flat "Context: ..." — the shape it handled best.
FINAL_ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant that answers questions strictly based on the provided context.
Do NOT use any outside knowledge. Only use the information given in the context below.
If the context does not contain enough information to answer the question, say:
"I don't have enough context to answer that."

Context:
{context}

Question: {question}
Answer:"""
)

# Strip leading/trailing quotes and whitespace from an LLM-returned entity.
# Matches both straight and smart quotes.
_ENTITY_STRIP_RE = re.compile(r'^[\s"\'`“”‘’]+|[\s"\'`“”‘’]+$')


def _clean_entity(raw: str) -> str:
    """Normalize the LLM's entity output: strip quotes, trim whitespace."""
    if not raw:
        return ""
    return _ENTITY_STRIP_RE.sub("", raw).strip()


def _get_cross_encoder():
    """Lazy-load and cache the CrossEncoder model.

    The import happens inside this function so sentence-transformers doesn't
    have to be installed at module-import time (useful for tests that mock
    the reranker entirely). The instance is cached on the module so
    subsequent queries reuse the same loaded model.
    """
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        # Imported here, not at top-of-file, so test environments and
        # tooling that doesn't need the reranker can skip the heavy import.
        from sentence_transformers import CrossEncoder
        _cross_encoder_instance = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder_instance


def _rerank_chunks(query: str, chunks: list, top_k: int = RERANK_TOP_K) -> list:
    """Re-rank parent chunks against the query and return the top_k.

    The bi-encoder vector store has already returned a short candidate list
    (typically RETRIEVAL_K=15 chunks). This function runs each chunk through
    the cross-encoder paired with the query, sorts by score descending, and
    returns the top_k highest-scoring chunks. Order matters: the LLM gets
    the most relevant passage first in the context block.

    Empty input short-circuits before loading the model — important because
    a fresh app instance can be queried before any documents are ingested,
    and we'd rather not pay the multi-second model-load cost just to return
    an empty list.
    """
    if not chunks:
        return []

    encoder = _get_cross_encoder()
    # CrossEncoder.predict expects (query, candidate) pairs — one per chunk.
    # The model returns one relevance score per pair.
    pairs = [(query, doc.page_content) for doc in chunks]
    scores = encoder.predict(pairs)

    # Pair each score with its chunk and sort desc. Ties keep the order the
    # bi-encoder produced (Python sort is stable), so when two passages are
    # equally relevant the one the vector store ranked higher wins.
    scored = sorted(zip(scores, chunks), key=lambda sc: sc[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def _format_graph_triplets(triplets: list[dict]) -> str:
    """Render graph triplets as one human-readable line per fact."""
    return "\n".join(
        f"({t['source']}) --[{t['relationship']}]--> ({t['target']})"
        for t in triplets
    )


def _format_vector_chunks(chunks: list) -> str:
    """Render parent chunks with their source filename prepended.

    Small models often refuse relevant passages because the passage itself
    does not repeat the entity the user asked about (e.g. a resume body
    rarely repeats the owner's name). Tagging each chunk with its source
    file gives the LLM the missing bridge: "from aumap_resume_2026.pdf" is
    enough for it to connect a Unity project bullet to "Atharv's experience."
    """
    blocks = []
    for doc in chunks:
        source = doc.metadata.get("source", "")
        source_name = Path(source).name if source else "unknown source"
        blocks.append(f"[from {source_name}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _build_context(vector_chunks: list, graph_triplets: list[dict]) -> str:
    """Combine retrieval sources into a single context block.

    When the graph is empty (typical before graph ingestion is wired up),
    the output is just the vector chunks — identical to the shape the old
    QA chain used, which the 3B model handled well. When graph facts ARE
    present, they are appended under a short labeled sub-header.
    """
    parts = []
    if vector_chunks:
        parts.append(_format_vector_chunks(vector_chunks))
    if graph_triplets:
        parts.append("Related facts:\n" + _format_graph_triplets(graph_triplets))
    if not parts:
        return "(no relevant context found)"
    return "\n\n".join(parts)


def generate_answer(query: str, persist_directory: str = "./db") -> str:
    """Run the full hybrid (vector + graph) RAG pipeline.

    Args:
        query: The user's natural-language question.
        persist_directory: Root directory holding both the vector store
            (./db) and graph.graphml.

    Returns:
        The final LLM answer as a string.
    """
    # One LLM client instance is reused for both the entity-extraction call
    # and the final-answer call; constructing it once avoids a second cold
    # start on the same model.
    llm = OllamaLLM(model=LLM_MODEL)

    # Step 1: Extract the core entity so we know where to root the graph walk.
    entity_raw = llm.invoke(ENTITY_EXTRACTION_PROMPT.format(query=query))
    entity = _clean_entity(entity_raw)

    # Step 2: Vector retrieval — use the user's full query, not the entity.
    # Parent-Child retrieval pulls rich surrounding context.
    vector_chunks = search(query, persist_directory=persist_directory)

    # Step 2b: Cross-encoder re-ranking. The vector store returns a wide
    # candidate set (RETRIEVAL_K=15) optimized for recall; the cross-encoder
    # rescores each chunk against the query for true relevance and we keep
    # only the top-K. Final context stays focused on the most relevant
    # passages instead of getting diluted by near-misses from the bi-encoder.
    vector_chunks = _rerank_chunks(query, vector_chunks)

    # Step 3: Graph retrieval — walk outward from the entity up to GRAPH_DEPTH
    # hops. If the entity isn't in the graph, this returns [] and the prompt
    # degrades gracefully to vector-only.
    graph = GraphStore(graph_path=f"{persist_directory}/graph.graphml")
    graph_facts = graph.get_neighborhood(entity, depth=GRAPH_DEPTH) if entity else []

    # Step 4: Build one flat context block and ask for the final answer.
    # Keeping graph and vector in a single block (rather than two labeled
    # sections) keeps the prompt shape that small models handle reliably.
    context = _build_context(vector_chunks, graph_facts)

    prompt = FINAL_ANSWER_PROMPT.format(
        context=context,
        question=query,
    )

    return llm.invoke(prompt)
