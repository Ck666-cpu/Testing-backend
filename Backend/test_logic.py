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

# --- SESSION STATE SETUP ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ 1. Session Memory: User Name
if "user_name" not in st.session_state:
    st.session_state.user_name = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Security Simulation")
    current_role = st.selectbox("Role:", [UserRole.ADMIN, UserRole.AGENT, UserRole.GUEST])

    # ✅ Debug Info: Show current session memory
    if st.session_state.user_name:
        st.info(f"🧠 Memory: User Name = {st.session_state.user_name}")

    st.divider()

    st.header("2. Knowledge Base")
    if current_role == UserRole.ADMIN:
        if st.button("🔄 Refresh Document List"):
            vs = VectorService()
            files = vs.list_ingested_files()
            st.session_state.doc_list = files

        if "doc_list" in st.session_state:
            st.caption("Files in Qdrant:")
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
        history_text = [f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-4:]]
        user_context = {"user_name": st.session_state.user_name}

        with st.spinner("Processing..."):
            response_data = crag_brain.generate_response(query, history_text, user_context)

            answer = response_data["answer"]
            sources = response_data.get("sources", [])
            intent = response_data.get("intent", "UNKNOWN")
            updates = response_data.get("session_updates", {})
            debug_nodes = response_data.get("debug_nodes", [])  # NEW

            if "user_name" in updates:
                st.session_state.user_name = updates["user_name"]
                st.toast(f"Memory Updated: Call me {updates['user_name']}!")

            st.markdown("### 🤖 Answer:")
            st.write(answer)

            if sources:
                with st.expander("📚 Reference Documents (Sources)"):
                    for src in sources:
                        st.caption(f"📄 {src}")

            # --- TUNING 3.5: Admin Debug Mode ---
            # Shows why retrieval happened (or failed)
            if current_role == UserRole.ADMIN:
                with st.expander("🛠️ ADMIN DEBUG: Retrieval Details"):
                    st.write(f"**Intent:** {intent}")
                    st.write("**Retrieved Chunks (Top 5 Reranked):**")
                    if debug_nodes:
                        for d_node in debug_nodes:
                            st.code(d_node, language="text")
                    else:
                        st.warning("No nodes passed the confidence threshold (0.35).")

            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})