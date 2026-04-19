# Cortex Engine

A 100% local, privacy-first RAG (Retrieval-Augmented Generation) pipeline that turns your documents into a searchable, conversational knowledge base. Nothing leaves your machine.

Built with **Ollama**, **ChromaDB**, **LangChain**, and **Streamlit**.

## What It Does

Drop PDFs into a folder, click a button, and start asking questions. Cortex Engine reads your documents, breaks them into chunks, embeds them into a local vector database, and uses a local LLM to answer your questions using only the content from your files.

- **Fully offline** — no API keys, no cloud, no data leaving your machine
- **Conversational UI** — Streamlit chat interface with session history
- **Strict grounding** — the LLM is prompted to only answer from your documents, reducing hallucination

## Current Architecture

```
PDF files (data/)
    │
    ▼
Document Loader (PyPDFDirectoryLoader)
    │
    ▼
Text Splitter (RecursiveCharacterTextSplitter, 1000 chars, 200 overlap)
    │
    ▼
Embedding (OllamaEmbeddings + nomic-embed-text)
    │
    ▼
Vector Store (ChromaDB, persisted to db/)
    │
    ▼
Retriever (top-3 similarity search)
    │
    ▼
QA Chain (Ollama llama3.2 + strict context-only prompt)
    │
    ▼
Streamlit Chat UI
```

## Project Structure

```
cortex-engine/
├── app.py                 # Streamlit UI with chat and ingestion sidebar
├── data/                  # Drop your PDF files here
├── db/                    # ChromaDB vector storage (auto-generated)
├── src/
│   ├── document_loader.py # Loads PDFs from a directory
│   ├── splitter.py        # Splits documents into chunks
│   ├── embedding.py       # Embeds chunks into ChromaDB
│   ├── retriever.py       # Similarity search over stored vectors
│   └── qa_chain.py        # RAG chain connecting retriever to LLM
├── tests/                 # Pytest test suite (TDD)
├── requirements.txt       # Python dependencies
├── SETUP.md               # Detailed setup and usage guide
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (required for upcoming OCR features)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pull Ollama Models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

| Model | Purpose |
|-------|---------|
| `nomic-embed-text` | Converts text chunks into vector embeddings |
| `llama3.2` | Local LLM for answering questions from context |

### Run the App

```bash
streamlit run app.py
```

1. Place your PDFs in the `data/` folder.
2. Click **"Ingest Documents"** in the sidebar.
3. Ask questions in the chat input.

### Run Tests

```bash
pytest tests/ -v
```

## Upcoming Features

### OCR for Scanned Documents
Extract text from scanned PDFs, receipts, passports, and handwritten notes using Tesseract OCR. This will allow Cortex Engine to ingest documents that don't have selectable text.

### Diagram & Image Extraction via Vision Models
Use vision-capable models to interpret charts, diagrams, and figures embedded in documents. Extract descriptions and data from visual content so it becomes searchable and queryable alongside text.

### Agentic Routing
Intelligent query routing that classifies incoming questions and directs them to specialized pipelines. For example, factual lookups go straight to retrieval, summarization requests trigger a different chain, and ambiguous queries ask for clarification. This enables multi-step reasoning and tool use beyond simple Q&A.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (llama3.2) |
| Embeddings | Ollama (nomic-embed-text) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Frontend | Streamlit |
| Testing | pytest (TDD) |

## License

This project is for personal and educational use.
