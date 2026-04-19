import streamlit as st

from src.document_loader import load_documents
from src.embedding import embed_documents
from src.qa_chain import ask_question

# --- Page configuration ---
st.set_page_config(page_title="Second Brain", layout="wide")
st.title("Second Brain")

# --- Session state initialization ---
# Keep track of chat history across reruns so the conversation persists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Ingestion pipeline ---
with st.sidebar:
    st.header("Document Ingestion")
    st.write("Place your PDF files in the `data/` folder, then click below to process them.")

    # Button triggers the full ingestion pipeline: load -> embed.
    # Splitting happens inside embed_documents via ParentDocumentRetriever:
    # it creates 2000-char parents for context and 400-char children for search.
    if st.button("Ingest Documents"):
        with st.spinner("Loading PDFs and images..."):
            # Step 1: Load all supported files from the data directory
            docs = load_documents("./data")

        if not docs:
            st.warning("No supported files found in the data/ folder.")
        else:
            with st.spinner("Splitting into parent/child chunks and embedding..."):
                # Step 2: Hand raw documents to the ParentDocumentRetriever.
                # It handles both parent (2000) and child (400) splitting internally,
                # embeds children into ChromaDB, and persists parents to disk.
                embed_documents(docs)

            st.success(f"Ingested {len(docs)} pages into the knowledge base.")

# --- Main area: Chat interface ---

# Display all previous messages from the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input field for the user to type their question
user_input = st.chat_input("Ask your Second Brain a question...")

if user_input:
    # Show the user's message immediately
    with st.chat_message("user"):
        st.write(user_input)

    # Save the user's message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Run the RAG pipeline to generate an answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_question(user_input)
        st.write(response)

    # Save the assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
