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

# --- 1. SESSION STATE SETUP ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- 2. CACHED RESOURCE LOADING (THE FIX) ---
# This prevents the app from reloading the AI models on every interaction
@st.cache_resource(show_spinner="Loading AI Models (Phi-3 & Embeddings)...")
def load_crag_brain():
    print(" [Streamlit] Loading CRAG Service for the first time...")
    return CRAGService()


try:
    # Load the brain
    crag_brain = load_crag_brain()
    st.success("System Ready & Loaded!")
except Exception as e:
    st.error(f"Failed to load backend: {e}")
    st.stop()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("1. Security Simulation")
    current_role = st.selectbox("Simulate Role:", [UserRole.ADMIN, UserRole.AGENT, UserRole.GUEST])

    st.header("2. Data Ingestion")
    uploaded_file = st.file_uploader("Upload Knowledge (PDF/TXT)")

    if uploaded_file and st.button("Ingest"):
        if RBAC.check_access(current_role, "ingest_documents"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Note: VectorService is light enough to instantiate here, 
            # or you could add a method to crag_brain to handle ingestion
            service = VectorService()
            result = service.ingest_document(tmp_path)
            st.success(result)
            os.remove(tmp_path)
        else:
            st.error("⛔ ACCESS DENIED")

# --- 4. MAIN CHAT INTERFACE ---
st.header("3. Chat Test")

# Display History
for msg in st.session_state.chat_history:
    role = "👤 You" if msg["role"] == "user" else "🤖 Bot"
    st.text(f"{role}: {msg['content']}")

query = st.text_input("Ask a question:", key="query_input")

if st.button("Generate Answer") and query:
    if not RBAC.check_access(current_role, "chat_rag"):
        st.error("⛔ ACCESS DENIED")
    else:
        # Prepare history
        history_text_list = [f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-4:]]

        with st.spinner("Thinking..."):
            # Use the cached brain
            response = crag_brain.generate_response(query, history_text_list)

            st.markdown("### 🤖 Answer:")
            st.write(str(response))

            # Save to history
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": str(response)})

            with st.expander("Debug Details"):
                if hasattr(response, 'source_nodes'):
                    st.write(f"Sources: {len(response.source_nodes)}")