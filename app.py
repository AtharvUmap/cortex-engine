import os

import streamlit as st
import streamlit.components.v1 as components

from src.document_loader import load_documents
from src.embedding import embed_documents
from src.index_status import format_status_line, read_index_status
from src.synthesis import generate_answer
from src.visualize import render_graph_html

# On-disk location of the persisted knowledge graph. Must stay in sync with
# the path embed_documents() writes to during ingestion.
DB_DIR = "./db"
GRAPH_PATH = f"{DB_DIR}/graph.graphml"

# --- Page configuration ---
st.set_page_config(
    page_title="Cortex Engine",
    layout="wide",
    # Force the sidebar open on every page load so it can't get stuck
    # collapsed if a chrome-hiding CSS rule trapped the reopen arrow.
    initial_sidebar_state="expanded",
)

# Strip Streamlit's developer chrome (deploy button, hamburger, "Made with
# Streamlit" footer) so the app reads as a standalone product. Selectors
# cover the various names Streamlit has used across versions; missing ones
# are harmless. Pairs with toolbarMode = "minimal" in .streamlit/config.toml.
st.markdown(
    """
    <style>
      /* Hide individual chrome elements rather than the whole toolbar
         wrapper — the toolbar contains the sidebar reopen arrow in some
         Streamlit versions, so blanket-hiding it traps the user when the
         sidebar is collapsed. */
      [data-testid="stDecoration"] { display: none; }
      [data-testid="stHeaderActionElements"] { display: none; }
      #MainMenu { visibility: hidden; }
      .stDeployButton { display: none; }
      footer { visibility: hidden; }
      /* Tighten the page wrapper's default ~6rem top padding so the
         header sits closer to the top and the chat input is visible
         on first load without scrolling. */
      .block-container { padding-top: 1.5rem; }
      /* Subtle breathing room between sidebar sections — keeps the
         pane structured without adding visible dividers. */
      [data-testid="stSidebar"] h3 { margin-top: 1.25rem; }
      [data-testid="stSidebar"] [data-testid="stExpander"] { margin-top: 0.75rem; }
      /* Route every accent through the theme's primaryColor so links
         and any future highlights match the active-tab teal. */
      a, a:visited { color: #4ec9b0; }
      a:hover { color: #6fdcc1; }
      /* Card-style chat container — slight inner tint, soft border,
         and rounded corners give the history area depth without
         shouting at the user. Targets the bordered-container wrapper
         that Streamlit emits when st.container has a fixed height. */
      [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.75rem;
        background: rgba(255, 255, 255, 0.02);
      }
      /* Make the Brain Map iframe feel like the hero element — soft
         teal halo around the container that brightens on hover. The
         iframe is the only one in the app, so a generic selector is
         safe and avoids the gymnastics of digging into Streamlit's
         component ID. */
      .stApp iframe {
        border-radius: 0.75rem;
        box-shadow:
          0 0 30px rgba(78, 201, 176, 0.12),
          0 0 4px rgba(78, 201, 176, 0.25);
        transition: box-shadow 0.4s ease;
        /* Fade up on first load so the graph arrives gracefully rather
           than popping in. */
        animation: cortex-fade-in 0.6s ease;
      }
      .stApp iframe:hover {
        box-shadow:
          0 0 60px rgba(78, 201, 176, 0.22),
          0 0 6px rgba(78, 201, 176, 0.45);
      }
      @keyframes cortex-fade-in {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      /* Micro-interactions — buttons, tabs, sidebar. Goal is "feels
         alive when you brush past it" without becoming a distraction.
         Single-property transitions where possible to avoid the
         performance cost of `all`. */
      .stButton button {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }
      .stButton button:hover {
        transform: scale(1.02);
      }
      [data-baseweb="tab"] {
        transition: color 0.2s ease;
      }
      [data-baseweb="tab-highlight"] {
        transition: left 0.25s ease, width 0.25s ease !important;
      }
      [data-testid="stSidebar"] {
        transition: transform 0.3s ease, margin-left 0.3s ease;
      }

      /* === Depth layering =====================================
         Four elevation tiers. Each step up is signalled by a
         slightly brighter surface tint plus a softer shadow,
         matching real-world depth cues so the eye knows what's
         "in front." Tints are kept under 5% alpha so the dark
         identity stays intact.
            L0 — page bg (#0e1117, no decoration)
            L1 — panels (sidebar): drop shadow + top highlight
            L2 — cards (chat container, expander): tint + shadow
            L3 — interactive (buttons): tint + shadow + lift
      */
      [data-testid="stSidebar"] {
        box-shadow: 2px 0 18px rgba(0, 0, 0, 0.4);
      }
      [data-testid="stSidebar"]::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: rgba(255, 255, 255, 0.06);
        pointer-events: none;
        z-index: 1;
      }
      /* Card layer — strengthen what we had so it reads as a
         distinct surface, not just a border. */
      [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.035);
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
      }
      [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.025);
        border-radius: 0.5rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
      }
      /* Interactive layer — buttons sit one step above their card. */
      .stButton button {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
      }
      .stButton button:hover {
        background: rgba(255, 255, 255, 0.06);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.4);
      }
      /* Living background — two faint teal glows that slowly drift over
         ~24s, giving the page a sense of motion without distracting from
         content. Pseudo-element sits behind everything, no pointer events,
         transparent gradient endpoints so it composes with the theme bg. */
      @keyframes cortex-ambient {
        from { transform: translate3d(0, 0, 0); }
        to   { transform: translate3d(2%, -2%, 0); }
      }
      [data-testid="stApp"]::before {
        content: "";
        position: fixed;
        inset: -10%;
        pointer-events: none;
        z-index: 0;
        background:
          radial-gradient(circle at 20% 30%, rgba(78, 201, 176, 0.08), transparent 45%),
          radial-gradient(circle at 75% 70%, rgba(78, 201, 176, 0.05), transparent 45%);
        animation: cortex-ambient 24s ease-in-out infinite alternate;
      }
      /* Sidebar ambient glow — same idea but in muted white tones to
         stay within the sidebar's grey/black palette. Opacity pulse
         instead of translate so the gradient stays bounded inside the
         sidebar without needing overflow tricks (which would break
         scroll on long sidebar content). Slower cadence (30s) so the
         two effects feel independent, not synchronized. */
      @keyframes cortex-sidebar-ambient {
        from { opacity: 1; }
        to   { opacity: 0.4; }
      }
      [data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background:
          radial-gradient(circle at 30% 25%, rgba(255, 255, 255, 0.08), transparent 50%),
          radial-gradient(circle at 70% 75%, rgba(255, 255, 255, 0.05), transparent 50%);
        animation: cortex-sidebar-ambient 30s ease-in-out infinite alternate;
      }
      /* Mini neural-network ornament anchored to the sidebar's bottom-
         left corner. Tiny, very faint (composite ~10–18%), staggered
         pulses give each node its own beat — reads as "the brain is
         alive" without competing with sidebar content. SVG strokes/fills
         pick up the teal accent so it's part of the same visual system. */
      .cortex-neural-mini {
        /* Pushed near the bottom of the sidebar via viewport-relative
           top margin. `calc(100vh - 27rem)` accounts for the sidebar
           content above (~19rem) + the SVG itself (~7rem) + a small
           bottom gap, so it adapts to taller/shorter viewports rather
           than pinning to a fixed pixel offset. */
        margin: calc(100vh - 34rem) 0 1rem 0.25rem;
        width: 170px;
        pointer-events: none;
        opacity: 0.25;
      }
      .cortex-neural-mini svg { width: 100%; height: auto; display: block; }
      .cortex-neural-mini line {
        stroke: #4ec9b0;
        stroke-width: 0.6;
        opacity: 0.55;
      }
      .cortex-neural-mini .n {
        fill: #4ec9b0;
        animation: cortex-mini-pulse 4s ease-in-out infinite;
      }
      .cortex-neural-mini .n1 { animation-delay: 0s; }
      .cortex-neural-mini .n2 { animation-delay: 0.6s; }
      .cortex-neural-mini .n3 { animation-delay: 1.2s; }
      .cortex-neural-mini .n4 { animation-delay: 1.8s; }
      .cortex-neural-mini .n5 { animation-delay: 2.4s; }
      .cortex-neural-mini .n6 { animation-delay: 3.0s; }
      @keyframes cortex-mini-pulse {
        0%, 100% { opacity: 0.4; }
        50%      { opacity: 1; }
      }

      /* Respect users who've opted out of motion. */
      @media (prefers-reduced-motion: reduce) {
        [data-testid="stApp"]::before,
        [data-testid="stSidebar"]::before { animation: none; }
        .stApp iframe { animation: none; }
        .stButton button:hover { transform: none; }
        .cortex-neural-mini .n { animation: none; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 🧠 Cortex Engine")

# --- Session state initialization ---
# Keep track of chat history across reruns so the conversation persists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar ---
# Three sections, kept terse so the sidebar reads at a glance:
#   Ingestion — primary action, top of pane.
#   Indices   — one-line health summary read from on-disk persistence.
#   Advanced  — collapsed expander with read-only config (env-driven).
with st.sidebar:
    st.subheader("Ingestion")
    if st.button("Ingest Documents", use_container_width=True):
        with st.spinner("Loading..."):
            docs = load_documents("./data")
        if not docs:
            st.warning("No files in `data/`.")
        else:
            with st.spinner("Embedding..."):
                embed_documents(docs)
            st.success(f"{len(docs)} pages ingested.")

    st.subheader("Indices")
    _status = read_index_status(DB_DIR)
    st.markdown(format_status_line(_status))
    if _status["last_ingested"] is not None:
        st.caption(_status["last_ingested"].strftime("Updated %d %b · %H:%M"))

    with st.expander("Advanced"):
        st.caption(f"LLM: `{os.getenv('CORTEX_LLM_MODEL', 'llama3.2')}`")
        st.caption(
            f"Multi-Query: `{os.getenv('CORTEX_MULTI_QUERY', 'false').lower()}`"
        )

    # Mini neural-network ornament — six nodes + connecting lines that
    # pulse in staggered phase. Anchored at the sidebar's bottom-left
    # corner via position: absolute (see CSS), so it lives in the empty
    # space below the sidebar's main controls without competing with them.
    st.markdown(
        """
        <div class="cortex-neural-mini">
          <svg viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg">
            <line x1="20" y1="22" x2="55" y2="38" />
            <line x1="55" y1="38" x2="92" y2="20" />
            <line x1="55" y1="38" x2="35" y2="62" />
            <line x1="55" y1="38" x2="80" y2="60" />
            <line x1="35" y1="62" x2="80" y2="60" />
            <line x1="20" y1="22" x2="35" y2="62" />
            <line x1="92" y1="20" x2="80" y2="60" />
            <line x1="92" y1="20" x2="105" y2="55" />
            <line x1="80" y1="60" x2="105" y2="55" />
            <circle cx="20" cy="22" r="3" class="n n1"/>
            <circle cx="55" cy="38" r="4" class="n n2"/>
            <circle cx="92" cy="20" r="3" class="n n3"/>
            <circle cx="35" cy="62" r="3" class="n n4"/>
            <circle cx="80" cy="60" r="3" class="n n5"/>
            <circle cx="105" cy="55" r="2.5" class="n n6"/>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Main area: tabs ---
# Chat is the primary view; Brain Map is a secondary explorer for the
# knowledge graph that sits alongside the vector store.
chat_tab, brain_map_tab = st.tabs(["💬 Chat", "🕸 Brain Map"])

with chat_tab:
    # Layout note: st.chat_input inside a tab renders INLINE — it does not
    # dock to the viewport bottom the way a top-level chat_input does. To
    # keep the input visually pinned below the history (and to keep new
    # messages from rendering *under* the input during the rerun where they
    # were submitted), we render history into a fixed-height scrollable
    # container declared before the input. Content is added to the container
    # after we've handled the new user message, so the latest round always
    # lands inside the scroll area.
    history_container = st.container(height=560)

    # Chat input field for the user to type their question
    user_input = st.chat_input("Ask a question...")

    if user_input:
        # Save the user's message to chat history first so the loop below
        # renders it inside the container (not below the input).
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Run the hybrid RAG pipeline (vector + knowledge graph) to generate an answer
        with st.spinner("Searching graph..."):
            response = generate_answer(user_input)

        # Save the assistant's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Render the full chat history inside the scrollable container above
    # the input. Placing this after the input-handling block ensures the
    # newly appended round is included in the same rerun.
    with history_container:
        if not st.session_state.chat_history:
            # Empty state — keeps the chat tab from looking like a broken
            # blank panel before the user has asked anything.
            st.markdown(
                "<p style='color: #888;'>Ask anything in your knowledge base.</p>",
                unsafe_allow_html=True,
            )
        else:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

with brain_map_tab:
    st.subheader("Knowledge Graph")
    st.caption("Larger, brighter nodes are more connected.")
    # render_graph_html handles missing/empty graph gracefully, so this is
    # safe to call even before the user has ingested any documents.
    graph_html = render_graph_html(GRAPH_PATH)
    components.html(graph_html, height=800, scrolling=True)
