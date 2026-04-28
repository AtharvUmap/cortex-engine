# Second Brain - Setup & Running Guide

A 100% local RAG (Retrieval-Augmented Generation) app that lets you chat with your PDF documents using Ollama, ChromaDB, and Streamlit.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running on your machine
- **macOS:** OCR works out of the box via Apple Vision Framework — no extra install
- **Linux / Windows:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for scanned PDF fallback and image ingestion

## 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The `ocrmac` package is gated on `sys_platform == 'darwin'`, so non-Mac installs skip it and the loader falls back to Tesseract automatically.

## 2. Pull the Required Ollama Models

These models run entirely on your machine — no data leaves your computer.

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

- **nomic-embed-text**: Converts your PDF text into vectors for search.
- **llama3.2**: The LLM that reads your documents and answers your questions.

## 3. Add Your PDF Files

Place any PDF files you want to chat with into the `data/` folder:

```
second-brain/
└── data/
    ├── your-notes.pdf
    ├── research-paper.pdf
    └── ...
```

## 4. Run the App

```bash
streamlit run app.py
```

This will open the app in your browser (usually at http://localhost:8501).

## 5. Using the App

1. **Ingest your documents**: Click the **"Ingest Documents"** button in the sidebar. This reads your PDFs (with OCR fallback for scanned pages), builds three retrieval indexes — Parent-Child vector store, knowledge graph, and regex-built factual index — and persists them to `db/`. You only need to do this once per set of documents.

2. **Ask questions**: Type your question in the chat input at the bottom of the page. The app combines all three indexes into the LLM's context: direct facts (when the query has a structured intent like "what's my email"), graph-walk results around the queried entity, and reranked vector chunks.

3. **Inspect the graph**: Switch to the **Brain Map** tab to see the extracted entities and relationships rendered as an interactive network. Useful for spotting bad extractions and understanding what the system "knows."

4. **Add more documents**: Drop new PDFs into `data/`, click "Ingest Documents" again, and they will be added to your knowledge base.

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
second-brain/
├── app.py                  # Streamlit UI — chat tab + Brain Map tab + ingestion sidebar
├── data/                   # Place your PDF / image files here
├── db/                     # Auto-generated persistence (gitignored)
│   ├── chroma.sqlite3      #   Vector store (child chunk embeddings)
│   ├── docstore/           #   Parent chunks (LocalFileStore)
│   ├── graph.graphml       #   NetworkX knowledge graph
│   └── facts.json          #   Regex-built factual index
├── src/
│   ├── document_loader.py  # Hybrid PDF loader: PyPDF + OCR fallback
│   ├── embedding.py        # Parent-Child ingestion, builds all three indexes
│   ├── retriever.py        # Vector retrieval (Parent-Child + optional Multi-Query)
│   ├── graph_extractor.py  # Two-pass LLM extraction: relationships + attributes
│   ├── graph_store.py      # NetworkX DiGraph wrapper + entity resolution
│   ├── factual_index.py    # Regex shapes + intent keywords + lookup
│   ├── synthesis.py        # Hybrid query-time pipeline (vector + graph + facts)
│   └── visualize.py        # PyVis Brain Map renderer
├── docs/                   # Architecture deep dives
├── eval/                   # Evaluation harness
├── tests/                  # Pytest test suite
└── requirements.txt
```

For deep dives on each component, see the docs:

- `docs/DOCUMENT_LOADING.md` — hybrid loader and OCR strategy
- `docs/PARENT_CHILD_RETRIEVAL.md` — small-to-big chunking
- `docs/MULTI_QUERY_RETRIEVAL.md` — LLM-rephrased query expansion
- `docs/KNOWLEDGE_GRAPH.md` — graph extraction and synthesis
- `docs/FACTUAL_INDEX.md` — regex-first short-circuit for structured-fact queries
- `docs/TOOLS_AND_DEPENDENCIES.md` — every external tool and library

## Troubleshooting

- **"Connection refused" errors**: Make sure Ollama is running (`ollama serve`).
- **No documents found**: Make sure your PDFs are in the `data/` folder and you clicked "Ingest Documents".
- **Slow first response**: The first query may take longer as Ollama loads the model into memory.
