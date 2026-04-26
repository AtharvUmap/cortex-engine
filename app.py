import streamlit as st
import streamlit.components.v1 as components

from src.document_loader import load_documents
from src.embedding import embed_documents
from src.synthesis import generate_answer
from src.visualize import render_graph_html

# On-disk location of the persisted knowledge graph. Must stay in sync with
# the path embed_documents() writes to during ingestion.
GRAPH_PATH = "./db/graph.graphml"

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

# --- Main area: tabs ---
# Chat is the primary view; Brain Map is a secondary explorer for the
# knowledge graph that sits alongside the vector store.
chat_tab, brain_map_tab = st.tabs(["Chat", "Brain Map"])

with chat_tab:
    # Layout note: st.chat_input inside a tab renders INLINE — it does not
    # dock to the viewport bottom the way a top-level chat_input does. To
    # keep the input visually pinned below the history (and to keep new
    # messages from rendering *under* the input during the rerun where they
    # were submitted), we render history into a fixed-height scrollable
    # container declared before the input. Content is added to the container
    # after we've handled the new user message, so the latest round always
    # lands inside the scroll area.
    history_container = st.container(height=600)

    # Chat input field for the user to type their question
    user_input = st.chat_input("Ask your Second Brain a question...")

    if user_input:
        # Save the user's message to chat history first so the loop below
        # renders it inside the container (not below the input).
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Run the hybrid RAG pipeline (vector + knowledge graph) to generate an answer
        with st.spinner("Thinking..."):
            response = generate_answer(user_input)

        # Save the assistant's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Render the full chat history inside the scrollable container above
    # the input. Placing this after the input-handling block ensures the
    # newly appended round is included in the same rerun.
    with history_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

with brain_map_tab:
    st.subheader("Knowledge Graph")
    st.caption(
        "Drag nodes to explore. Each edge shows how two entities are related."
    )
    # render_graph_html handles missing/empty graph gracefully, so this is
    # safe to call even before the user has ingested any documents.
    graph_html = render_graph_html(GRAPH_PATH)
    components.html(graph_html, height=800, scrolling=True)
