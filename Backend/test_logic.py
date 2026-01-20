import streamlit as st
import os
import sys
import tempfile
import uuid
from datetime import datetime

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.crag_service import CRAGService
from app.services.vector_store import VectorService
from app.core.security import RBAC, UserRole

st.set_page_config(layout="wide", page_title="CRAG Logic Tester")
st.title("🏗️ CRAG Backend Logic Tester")


# --- CACHED BRAIN ---
@st.cache_resource(show_spinner="Loading AI Models...")
def load_crag_brain():
    return CRAGService()


try:
    crag_brain = load_crag_brain()
    # st.success("System Ready") # Removed to reduce clutter
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- 1. SESSION MANAGEMENT LOGIC (NEW) ---

if "sessions" not in st.session_state:
    # Initialize with one empty session
    default_id = str(uuid.uuid4())[:8]
    st.session_state.sessions = {
        default_id: {
            "title": "New Chat",
            "history": [],
            "context": {"user_name": None}  # Per-session memory
        }
    }
    st.session_state.current_session_id = default_id


def create_new_session():
    new_id = str(uuid.uuid4())[:8]
    st.session_state.sessions[new_id] = {
        "title": f"New Chat {len(st.session_state.sessions) + 1}",
        "history": [],
        "context": {"user_name": None}
    }
    st.session_state.current_session_id = new_id


def delete_session(session_id):
    if len(st.session_state.sessions) > 1:
        del st.session_state.sessions[session_id]
        # Switch to another available session
        st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]


# Get Active Session Data
active_id = st.session_state.current_session_id
active_session = st.session_state.sessions[active_id]
active_history = active_session["history"]
active_context = active_session["context"]

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Chat Sessions")

    # NEW CHAT BUTTON
    if st.button("➕ Start New Chat", use_container_width=True):
        create_new_session()
        st.rerun()

    st.markdown("---")
    st.caption("Active Conversations:")

    # SESSION LIST SELECTOR
    # We use a radio button to simulate switching chats
    session_options = {sid: s["title"] for sid, s in st.session_state.sessions.items()}

    selected_session = st.radio(
        "Select Chat:",
        options=list(session_options.keys()),
        format_func=lambda x: session_options[x],
        index=list(session_options.keys()).index(active_id),
        key="session_selector"
    )

    # Handle Switching
    if selected_session != active_id:
        st.session_state.current_session_id = selected_session
        st.rerun()

    # RENAME / DELETE CURRENT CHAT
    with st.expander("⚙️ Session Options"):
        new_title = st.text_input("Rename Chat:", value=active_session["title"])
        if new_title != active_session["title"]:
            active_session["title"] = new_title
            st.rerun()

        if st.button("🗑️ Delete This Chat", type="primary"):
            delete_session(active_id)
            st.rerun()

    st.markdown("---")
    st.header("2. Security & Data")
    current_role = st.selectbox("Simulate Role:", [UserRole.ADMIN, UserRole.AGENT, UserRole.GUEST])

    # Debug Info
    if active_context.get("user_name"):
        st.info(f"🧠 Memory: Name = {active_context['user_name']}")

    if current_role == UserRole.ADMIN:
        if st.button("🔄 Check Knowledge Base"):
            vs = VectorService()
            files = vs.list_ingested_files()
            st.session_state.doc_list = files

        if "doc_list" in st.session_state:
            st.caption("Files in Qdrant:")
            for f in st.session_state.doc_list:
                st.code(f, language="text")

    if st.button("⚠️ Reset Database (Admin)"):
        if RBAC.check_access(current_role, "delete_documents"):
            vs = VectorService()
            st.warning(vs.clear_database())
            st.cache_resource.clear()
        else:
            st.error("Access Denied")

    uploaded_file = st.file_uploader("Upload PDF")
    if uploaded_file and st.button("Ingest"):
        if RBAC.check_access(current_role, "ingest_documents"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            vs = VectorService()
            st.info(vs.ingest_document(tmp_path))
            os.remove(tmp_path)
        else:
            st.error("Access Denied")

# --- MAIN CHAT INTERFACE ---
st.subheader(f"💬 {active_session['title']}")

# Display History for CURRENT Session
for msg in active_history:
    role = "👤 You" if msg["role"] == "user" else "🤖 Bot"
    # Helper to render clean markdown
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there are sources, show them in a collapsed view
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.caption(src)

# Input
query = st.chat_input("Ask a question...")

if query:
    if not RBAC.check_access(current_role, "chat_rag"):
        st.error("Access Denied")
    else:
        # 1. Display User Message Immediately
        with st.chat_message("user"):
            st.markdown(query)
        active_history.append({"role": "user", "content": query})

        # 2. Prepare Context (Last 4 turns of THIS session)
        history_text_list = [f"{m['role']}: {m['content']}" for m in active_history[-5:]]

        with st.spinner("Thinking..."):
            # 3. Call Backend
            response_data = crag_brain.generate_response(query, history_text_list, active_context)

            answer = response_data["answer"]
            sources = response_data.get("sources", [])
            updates = response_data.get("session_updates", {})

            # 4. Handle Memory Updates (Per Session)
            if "user_name" in updates:
                active_context["user_name"] = updates["user_name"]
                st.toast(f"I'll remember to call you {updates['user_name']} in this chat.")

            # 5. Display Bot Response
            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources"):
                        for src in sources:
                            st.caption(src)

            # 6. Save to History
            active_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            # Auto-Rename Chat if it's the first message
            if len(active_history) <= 2 and active_session["title"].startswith("New Chat"):
                # Use the user's first query as the title (truncated)
                new_title = query[:25] + "..." if len(query) > 25 else query
                active_session["title"] = new_title
                st.rerun()