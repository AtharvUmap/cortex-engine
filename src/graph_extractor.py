"""
Knowledge Graph Extraction Module
----------------------------------
Takes a LangChain Document and asks the local LLM to extract a list of
(source, relationship, target) triples that describe the entities and
relationships mentioned in the text.

The output feeds into a NetworkX knowledge graph (GraphRAG), which sits
alongside Parent-Child vector retrieval to give the system relational
context in addition to semantic context.

Design notes:
  - Uses a strict system prompt demanding pure JSON output. Small local
    models often add prose or markdown fences; we defensively strip those
    before parsing.
  - Returns [] on any parse failure rather than raising — one noisy chunk
    should not kill a full ingestion run.
  - Filters out malformed triples so downstream graph code can assume every
    dict has source/target/relationship keys.
"""

import json
import os
import re

from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

# Reuse the same LLM env var as the QA chain so one setting controls the
# whole app. llama3.2 (3B) is usually sufficient for triple extraction;
# upgrade to llama3.1:8b if you see many malformed outputs.
DEFAULT_LLM_MODEL = "llama3.2"
GRAPH_LLM_MODEL = os.environ.get("CORTEX_LLM_MODEL", DEFAULT_LLM_MODEL)

# Strict instructions: the LLM must return ONLY a JSON array. Any prose,
# explanation, or markdown fences will break json.loads — the prompt
# explicitly forbids them, and the post-processor strips them if the model
# disobeys anyway.
SYSTEM_PROMPT = """You are an information extraction engine. Extract entities and the relationships between them from the given text.

Return ONLY a valid JSON array. Do NOT include any explanation, prose, or markdown code fences. Do NOT wrap the output in ```json ... ```.

Each element of the array must be an object with exactly these three keys:
  - "source": the subject entity (a short noun phrase)
  - "target": the object entity (a short noun phrase)
  - "relationship": a concise verb or verb phrase describing how source relates to target

Example output format:
[{{"source": "Marie Curie", "target": "radium", "relationship": "discovered"}}]

If the text contains no clear relationships, return an empty array: []

Text to analyze:
{content}
"""

# Matches ```json ... ``` or ``` ... ``` fences that some models add despite
# being told not to. Captures the inner body so we can feed it to json.loads.
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _clean_llm_output(raw: str) -> str:
    """Strip markdown fences and surrounding whitespace so json.loads can parse."""
    fence_match = _CODE_FENCE_RE.search(raw)
    if fence_match:
        return fence_match.group(1).strip()
    return raw.strip()


def _is_valid_triple(item) -> bool:
    """A triple must be a dict with all three required string keys populated."""
    if not isinstance(item, dict):
        return False
    required = ("source", "target", "relationship")
    return all(k in item and isinstance(item[k], str) and item[k] for k in required)


def extract_graph_triples(document: Document) -> list[dict]:
    """Extract (source, relationship, target) triples from a document.

    Args:
        document: A LangChain Document whose page_content will be analyzed.

    Returns:
        A list of dicts with keys "source", "target", "relationship".
        Returns [] if the document is empty or the LLM output can't be parsed.
    """
    # Skip empty documents — no point spending an LLM call on no text, and
    # the model might hallucinate triples from the system prompt alone.
    content = (document.page_content or "").strip()
    if not content:
        return []

    # Build the extraction prompt with the document's text injected.
    prompt = SYSTEM_PROMPT.format(content=content)

    # Call the local LLM. Any Ollama failure will propagate — callers can
    # decide whether to wrap ingestion in a try/except for resilience.
    llm = OllamaLLM(model=GRAPH_LLM_MODEL)
    raw_output = llm.invoke(prompt)

    # Models sometimes ignore the "no fences" instruction; strip them.
    cleaned = _clean_llm_output(raw_output)

    # Parse the JSON. On failure, return [] rather than raising so one
    # bad chunk doesn't sink a whole ingestion batch.
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return []

    # The LLM should return a list; anything else is malformed.
    if not isinstance(parsed, list):
        return []

    # Keep only well-formed triples so downstream graph construction can
    # assume every dict has the three required keys.
    return [item for item in parsed if _is_valid_triple(item)]
