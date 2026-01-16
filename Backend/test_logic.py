import streamlit as st
import os
import sys
import tempfile

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
    st.success("System Ready")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Security Simulation")
    current_role = st.selectbox("Role:", [UserRole.ADMIN, UserRole.AGENT, UserRole.GUEST])

    st.divider()

    # --- NEW: VIEW STORED DOCS (ADMIN ONLY) ---
    st.header("2. Knowledge Base")
    if current_role == UserRole.ADMIN:
        if st.button("🔄 Refresh Document List"):
            # We instantiate VectorService purely for this check
            vs = VectorService()
            files = vs.list_ingested_files()
            st.session_state.doc_list = files

        if "doc_list" in st.session_state:
            st.caption("Files currently in Qdrant:")
            for f in st.session_state.doc_list:
                st.code(f, language="text")
    else:
        st.caption("🔒 Document list hidden (Admin only)")

    st.divider()

    st.header("3. Ingestion")
    if st.button("⚠️ Reset Database"):
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

# --- MAIN CHAT ---
st.header("4. Chat Interface")

# Display History
for msg in st.session_state.chat_history:
    role = "👤 You" if msg["role"] == "user" else "🤖 Bot"
    st.text(f"{role}: {msg['content']}")

query = st.text_input("Ask a question:", key="q_input")

if st.button("Send") and query:
    if not RBAC.check_access(current_role, "chat_rag"):
        st.error("Access Denied")
    else:
        # FILTER CONTEXT: We only want to pass 'informational' turns to the AI,
        # but for simplicity in this test tool, we pass the last 4 messages.
        # The backend 'Gatekeeper' handles the actual retrieval logic.
        history_text = [f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-4:]]

        with st.spinner("Processing..."):
            # The response is now a DICTIONARY
            response_data = crag_brain.generate_response(query, history_text)

            answer = response_data["answer"]
            sources = response_data["sources"]

            st.markdown("### 🤖 Answer:")
            st.write(answer)

            # DISPLAY SOURCES (NEW)
            if sources:
                with st.expander("📚 Reference Documents"):
                    for src in sources:
                        st.caption(f"📄 {src}")

            # Update History (Store just the answer string)
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})