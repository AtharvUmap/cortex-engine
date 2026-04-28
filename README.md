# Cortex Engine

A 100% local, privacy-first RAG (Retrieval-Augmented Generation) pipeline that turns your documents into a searchable, conversational knowledge base. Nothing leaves your machine.

Built with **Ollama**, **ChromaDB**, **NetworkX**, **LangChain**, and **Streamlit**.

## What It Does

Drop PDFs into a folder, click a button, and start asking questions. Cortex Engine reads your documents through a hybrid text-and-OCR loader, indexes them three ways (vector embeddings, a knowledge graph, and a regex-built factual index), and uses a local LLM to answer your questions using only the content from your files.

- **Fully offline** — no API keys, no cloud, no data leaving your machine
- **Hybrid retrieval** — Parent-Child vector search + GraphRAG + deterministic fact lookup, combined into one prompt
- **OCR fallback** — Apple Vision Framework on macOS, Tesseract on other platforms, so scanned PDFs index alongside text-native ones
- **Conversational UI** — Streamlit chat interface with session history and an interactive Brain Map view of the extracted graph
- **Strict grounding** — the LLM is prompted to only answer from your documents, reducing hallucination

## Current Architecture

```
PDF / image files (data/)
    │
    ▼
Hybrid Document Loader              ← PyPDF first; falls back to Apple Vision (macOS)
    │                                 or Tesseract (other platforms) for scanned pages.
    │                                 See docs/DOCUMENT_LOADING.md
    ▼
Parent-Child Splitter               ← 2000-char parents for context,
    │                                 400-char children for precise vector matching.
    │                                 See docs/PARENT_CHILD_RETRIEVAL.md
    ▼
┌───────────────────┬──────────────────────┬──────────────────────┐
▼                   ▼                      ▼                      │
Vector Store        Knowledge Graph        Factual Index          │
(ChromaDB,          (NetworkX DiGraph,     (regex on raw chunks,  │
 nomic-embed-text,  graph.graphml,         facts.json,            │
 child chunks)      two-pass LLM extract)  email/phone/IDs)       │
│                   │                      │                      │
│ docs/PARENT_      │ docs/KNOWLEDGE_      │ docs/FACTUAL_         │
│  CHILD_RETRIEVAL  │  GRAPH               │  INDEX                │
└───────────────────┴──────────────────────┴──────────────────────┘
                              │
                              ▼  query time
                    ┌─────────────────────┐
                    │ Hybrid Synthesis    │
                    │ (src/synthesis.py)  │
                    │                     │
                    │ 1. detect intent →  │
                    │    direct facts     │
                    │ 2. extract entity → │
                    │    graph walk (k=2) │
                    │ 3. vector search    │
                    │    (k=15)           │
                    │ 4. cross-encoder    │
                    │    rerank → top 3   │
                    │ 5. compose prompt   │
                    │ 6. Ollama llama3.2  │
                    └─────────────────────┘
                              │
                              ▼
                    Streamlit Chat + Brain Map UI
```

Each retrieval index is independent — if one fails or is empty, the others still answer. See `docs/` for the deep dive on each component.

## Project Structure

```
cortex-engine/
├── app.py                  # Streamlit UI — chat tab + Brain Map tab + ingestion sidebar
├── data/                   # Drop your PDF / image files here
├── db/                     # Auto-generated persistence (gitignored)
│   ├── chroma.sqlite3      #   Vector store (child chunk embeddings)
│   ├── docstore/           #   LocalFileStore for parent chunks
│   ├── graph.graphml       #   NetworkX knowledge graph
│   ├── facts.json          #   Regex-built factual index
│   └── owners.json         #   Per-document owner inference (Ticket 23)
├── src/
│   ├── document_loader.py  # Hybrid PDF loader: PyPDF + OCR fallback dispatcher
│   ├── embedding.py        # Parent-Child ingestion, runs all three index passes
│   ├── retriever.py        # Vector retrieval (Parent-Child + optional Multi-Query)
│   ├── graph_extractor.py  # Two-pass LLM extraction: relationships + attributes
│   ├── graph_store.py      # NetworkX DiGraph wrapper (uses entity_resolution)
│   ├── entity_resolution.py# Shared normalize + fuzzy-match helpers
│   ├── document_owner.py   # Per-document owner inference (Ticket 23)
│   ├── factual_index.py    # Regex shapes + intent keywords + owner-aware lookup
│   ├── synthesis.py        # Hybrid query-time pipeline (vector + graph + facts)
│   ├── visualize.py        # PyVis Brain Map renderer
│   ├── splitter.py         # (legacy — used only by eval harness)
│   └── qa_chain.py         # (legacy — used only by eval harness)
├── docs/                   # Architecture deep dives (see below)
├── eval/                   # Evaluation harness
├── tests/                  # Pytest test suite (TDD)
├── requirements.txt
├── SETUP.md                # Detailed setup and usage guide
└── README.md
```

## Documentation

| Doc | What it covers |
|-----|----------------|
| `docs/DOCUMENT_LOADING.md` | Hybrid loader, Apple Vision OCR (macOS), Tesseract fallback, image ingestion |
| `docs/PARENT_CHILD_RETRIEVAL.md` | Small-to-big chunking, parent docstore, child vector index |
| `docs/MULTI_QUERY_RETRIEVAL.md` | LLM-rephrased query union, when to enable it, why it's off by default |
| `docs/KNOWLEDGE_GRAPH.md` | Two-pass extraction, entity resolution, graph walk synthesis, Brain Map |
| `docs/FACTUAL_INDEX.md` | Regex-first short-circuit for structured-fact queries (emails, phones, IDs) |
| `docs/TOOLS_AND_DEPENDENCIES.md` | Every external tool and Python package, what it's for, why it was chosen |

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- **macOS users:** OCR ships out of the box via Apple Vision Framework (no extra install)
- **Other platforms:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for scanned PDF fallback and image ingestion

### Install Dependencies

```bash
pip install -r requirements.txt
```

The `ocrmac` package is gated on `sys_platform == 'darwin'`, so non-Mac installs skip it automatically and the loader falls back to Tesseract.

### Pull Ollama Models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

| Model | Purpose |
|-------|---------|
| `nomic-embed-text` | Converts text chunks into vector embeddings |
| `llama3.2` | Local LLM for entity extraction, two-pass graph extraction, and final answer synthesis |

The first query also lazy-loads `cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face for retrieval reranking — one-time download, cached locally.

### Run the App

```bash
streamlit run app.py
```

1. Place your PDFs / images in the `data/` folder.
2. Click **"Ingest Documents"** in the sidebar. This builds all three indexes (vector + graph + facts).
3. Ask questions in the chat input.
4. Switch to the **Brain Map** tab to inspect the extracted knowledge graph.

### Run Tests

```bash
pytest tests/ -v
```

## Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `CORTEX_LLM_MODEL` | `llama3.2` | Ollama model for entity extraction, graph extraction, and final answer synthesis |
| `CORTEX_MULTI_QUERY` | `false` | Set to `true` to wrap the retriever with LLM-rephrased query expansion. See `docs/MULTI_QUERY_RETRIEVAL.md` for when this helps and when it hurts. |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (llama3.2) |
| Embeddings | Ollama (nomic-embed-text) |
| Vector Store | ChromaDB |
| Knowledge Graph | NetworkX (GraphML on disk) |
| Reranker | sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2) |
| OCR (macOS) | Apple Vision Framework via `ocrmac` + `pypdfium2` |
| OCR (other) | Tesseract via `unstructured` |
| Framework | LangChain |
| Frontend | Streamlit + PyVis (Brain Map) |
| Testing | pytest (TDD) |

## License

This project is for personal and educational use.
