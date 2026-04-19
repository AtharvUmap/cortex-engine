# Second Brain - Setup & Running Guide

A 100% local RAG (Retrieval-Augmented Generation) app that lets you chat with your PDF documents using Ollama, ChromaDB, and Streamlit.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running on your machine

## 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

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

1. **Ingest your documents**: Click the **"Ingest Documents"** button in the sidebar. This reads your PDFs, splits them into chunks, and stores them in the local vector database. You only need to do this once per set of documents.

2. **Ask questions**: Type your question in the chat input at the bottom of the page. The app will find the most relevant sections from your documents and generate an answer.

3. **Add more documents**: Drop new PDFs into `data/`, click "Ingest Documents" again, and they will be added to your knowledge base.

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
second-brain/
├── app.py                 # Streamlit UI
├── data/                  # Place your PDF files here
├── db/                    # ChromaDB stores vector data here (auto-generated)
├── src/
│   ├── document_loader.py # Loads PDFs from the data/ folder
│   ├── splitter.py        # Splits documents into smaller chunks
│   ├── embedding.py       # Embeds chunks and stores them in ChromaDB
│   ├── retriever.py       # Searches ChromaDB for relevant chunks
│   └── qa_chain.py        # Connects the retriever to the LLM
├── tests/                 # Pytest test suite
└── requirements.txt       # Python dependencies
```

## Troubleshooting

- **"Connection refused" errors**: Make sure Ollama is running (`ollama serve`).
- **No documents found**: Make sure your PDFs are in the `data/` folder and you clicked "Ingest Documents".
- **Slow first response**: The first query may take longer as Ollama loads the model into memory.
