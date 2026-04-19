import os

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from src.retriever import search

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# Switch between speed and reasoning power by changing LLM_MODEL below, or
# by setting the CORTEX_LLM_MODEL environment variable before running the app.
#
# Recommended options (all local via Ollama):
#   "llama3.2"       — 3B params, fast (~2-5s), best for factual extraction
#                      (passport dates, receipts, appointment times, etc.)
#   "llama3.1:8b"    — 8B params, medium speed, strong at logical reasoning
#                      and multi-document comparison
#   "qwen2.5:7b"     — 7B params, medium speed, strong at reasoning and math
#
# To use a non-default model: `CORTEX_LLM_MODEL=llama3.1:8b streamlit run app.py`
DEFAULT_LLM_MODEL = "llama3.2"
LLM_MODEL = os.environ.get("CORTEX_LLM_MODEL", DEFAULT_LLM_MODEL)

# System prompt that strictly constrains the LLM to only use provided context.
# This prevents hallucination by telling the model to refuse when context is insufficient.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.
Do NOT use any outside knowledge. Only use the information given in the context below.
If the context does not contain enough information to answer the question, say:
"I don't have enough context to answer that."

Context:
{context}
"""

# Prompt template combining the system instructions with the user's question
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def ask_question(query: str, persist_directory: str = "./db") -> str:
    """Run the full RAG pipeline: retrieve relevant chunks and generate an answer.

    Args:
        query: The user's question.
        persist_directory: Path to the ChromaDB directory for retrieval.

    Returns:
        The LLM's response as a string.
    """
    # Step 1: Retrieve the most relevant parent chunks using Parent-Child retrieval.
    # Children (400 chars) are searched for precision; their parents (2000 chars)
    # are returned so the LLM gets rich surrounding context.
    relevant_chunks = search(query, persist_directory=persist_directory)

    # Step 2: Combine the retrieved chunks into a single context string
    # Each chunk's text is joined with newlines so the LLM sees all relevant info
    context_text = "\n\n".join(doc.page_content for doc in relevant_chunks)

    # Step 3: Build the final prompt by injecting context and the user's question
    prompt = PROMPT_TEMPLATE.format_messages(
        context=context_text,
        question=query,
    )

    # Step 4: Send the prompt to the local Ollama LLM and get a response.
    # The model is configurable via the LLM_MODEL constant / env var above.
    llm = OllamaLLM(model=LLM_MODEL)
    response = llm.invoke(prompt)

    return response
