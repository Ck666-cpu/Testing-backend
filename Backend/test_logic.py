import streamlit as st
import os
import sys
import tempfile

# Add current folder to path so we can import 'app'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.crag_service import CRAGService
from app.services.vector_store import VectorService
from app.core.security import RBAC, UserRole

st.set_page_config(layout="wide", page_title="CRAG Logic Tester")

st.title("🏗️ CRAG Backend Logic Tester")
st.markdown("Use this to test the **Phi-3**, **Reranker**, and **RBAC** logic directly.")

# --- SIDEBAR: CONFIG & SECURITY ---
with st.sidebar:
    st.header("1. Security Simulation (RBAC)")
    # Select a Role to simulate
    current_role = st.selectbox(
        "Simulate User Role:",
        [UserRole.ADMIN, UserRole.AGENT, UserRole.GUEST]
    )
    st.info(f"Current Permissions: {RBAC.PERMISSIONS[current_role]}")

    st.header("2. Data Ingestion")
    uploaded_file = st.file_uploader("Upload Knowledge (PDF/TXT)")

    if uploaded_file:
        if st.button("Ingest to Qdrant"):
            # Check RBAC Permission First!
            if RBAC.check_access(current_role, "ingest_documents"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                service = VectorService()
                result = service.ingest_document(tmp_path)
                st.success(result)
                os.remove(tmp_path)
            else:
                st.error("⛔ ACCESS DENIED: You need 'ingest_documents' permission.")

# --- MAIN: CHAT ---
st.header("3. Chat Test (Phi-3 + CRAG)")

# Initialize Service Logic
if "crag_brain" not in st.session_state:
    with st.spinner("Loading Phi-3 & Rerankers..."):
        st.session_state.crag_brain = CRAGService()
    st.success("System Ready")

query = st.text_input("Ask a question:")

if st.button("Generate Answer"):
    # 1. Security Check
    if not RBAC.check_access(current_role, "chat_rag"):
        st.error("⛔ ACCESS DENIED: Guests cannot use the AI Chat.")
    else:
        # 2. Run Logic
        with st.spinner("Running CRAG Pipeline (Retrieve -> Rerank -> Correct -> Generate)..."):
            response = st.session_state.crag_brain.generate_response(query)

            # Display Result
            st.markdown("### 🤖 Answer:")
            st.write(str(response))

            # Debug Info (Show what logic happened)
            with st.expander("See Backend Details"):
                if hasattr(response, 'source_nodes'):
                    st.write(f"Sources Found: {len(response.source_nodes)}")
                    for node in response.source_nodes:
                        st.caption(f"Score: {node.score:.4f}")
                        st.text(node.get_content()[:200] + "...")
                else:
                    st.warning("No sources used (Fallback triggered).")