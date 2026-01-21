import streamlit as st
import os
import sys
import tempfile
import uuid

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.crag_service import CRAGService
from app.services.vector_store import VectorService
from app.core.security import RBAC, UserRole

st.set_page_config(layout="wide", page_title="RBAC Logic Tester")
st.title("🛡️ RBAC & CRAG Logic Tester")


# --- CACHED BRAIN ---
@st.cache_resource(show_spinner="Loading AI Models...")
def load_crag_brain():
    return CRAGService()


try:
    crag_brain = load_crag_brain()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- SESSION MANAGEMENT ---
if "sessions" not in st.session_state:
    default_id = str(uuid.uuid4())[:8]
    st.session_state.sessions = {
        default_id: {
            "title": "New Chat",
            "history": [],
            "context": {"user_name": None}
        }
    }
    st.session_state.current_session_id = default_id

# --- SIDEBAR: ROLE SIMULATION ---
with st.sidebar:
    st.header("🔐 Security Context")
    current_role = st.selectbox(
        "Simulate Logged-in Role:",
        [UserRole.STAFF, UserRole.ADMIN, UserRole.MASTER_ADMIN]
    )

    st.info(f"Active Role: **{current_role.value.upper()}**")

    # Show active permissions for debugging
    with st.expander("View Permissions"):
        st.write(RBAC.PERMISSIONS[current_role])

    st.divider()

    # --- DOCUMENT UPLOAD (Ingestion) ---
    st.header("📂 Document Upload")

    # Toggle: Private vs Global
    doc_type = st.radio("Document Type:", ["Private", "Global"])

    uploaded_file = st.file_uploader("Upload PDF")

    if uploaded_file and st.button("Ingest Document"):
        # PERMISSION CHECK: UPLOAD
        required_perm = "upload_private_document" if doc_type == "Private" else "upload_global_document"

        if RBAC.check_access(current_role, required_perm):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            vs = VectorService()
            # In a real app, we would tag metadata={'visibility': doc_type, 'owner_id': 'current_user'}
            st.info(vs.ingest_document(tmp_path))
            os.remove(tmp_path)
        else:
            st.error(f"⛔ ACCESS DENIED: {current_role.value} cannot upload {doc_type} documents.")

# --- TABS FOR DIFFERENT FUNCTIONAL AREAS ---
tab_chat, tab_docs, tab_admin = st.tabs(["💬 Chat Workspace", "📚 Knowledge Base", "👥 User Management"])

# ==========================================
# TAB 1: CHAT WORKSPACE (Staff & Admin Only)
# ==========================================
with tab_chat:
    if not RBAC.check_access(current_role, "start_chat_session"):
        st.error("⛔ ACCESS DENIED: Master Admin does not have access to Chat Workspaces.")
    else:
        # ... (Existing Chat Logic) ...
        active_id = st.session_state.current_session_id
        if active_id not in st.session_state.sessions:
            active_id = list(st.session_state.sessions.keys())[0]
            st.session_state.current_session_id = active_id

        active_session = st.session_state.sessions[active_id]

        # Chat UI
        for msg in active_session["history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("Sources"):
                        for src in msg["sources"]: st.caption(src)

        query = st.chat_input("Ask a question...")

        if query:
            if RBAC.check_access(current_role, "submit_chat_query"):
                # ... Processing logic ...
                with st.chat_message("user"):
                    st.markdown(query)
                active_session["history"].append({"role": "user", "content": query})

                # Mock context for response
                history_list = [f"{m['role']}: {m['content']}" for m in active_session["history"][-5:]]

                with st.spinner("Thinking..."):
                    response = crag_brain.generate_response(query, history_list, active_session["context"])

                    with st.chat_message("assistant"):
                        st.markdown(response["answer"])

                    active_session["history"].append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    })
            else:
                st.error("⛔ Authorization Failed.")

# ==========================================
# TAB 2: KNOWLEDGE BASE (View Docs)
# ==========================================
with tab_docs:
    st.subheader("Document Repository")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Global Documents**")
        if RBAC.check_access(current_role, "view_global_documents"):
            if st.button("List Global Docs"):
                vs = VectorService()
                st.code(vs.list_ingested_files())  # Mocking filter
        else:
            st.warning("⛔ No Access to Global Docs")

    with col2:
        st.markdown("**My Private Documents**")
        if RBAC.check_access(current_role, "view_own_private_documents"):
            if st.button("List My Private Docs"):
                st.info("Displaying private docs for current user...")
        else:
            st.warning("⛔ No Access to Private Docs (Master Admin Restriction)")

# ==========================================
# TAB 3: USER MANAGEMENT (Admin & Master)
# ==========================================
with tab_admin:
    st.subheader("User Administration Console")

    # 1. CREATE USER (Admin & Master)
    with st.container(border=True):
        st.write("###### Create New User")
        new_username = st.text_input("Username")
        new_role = st.selectbox("Assign Role", ["staff", "admin"])

        if st.button("Create User"):
            if RBAC.check_access(current_role, "create_user"):
                st.success(f"✅ User '{new_username}' created as {new_role}")
            else:
                st.error("⛔ Access Denied: Cannot create users.")

    # 2. UPDATE ROLE (Master Only)
    with st.container(border=True):
        st.write("###### Update User Role")
        target_user = st.text_input("Target Username for Role Change")
        updated_role = st.selectbox("New Role", ["staff", "admin", "master_admin"])

        if st.button("Update Role"):
            if RBAC.check_access(current_role, "update_user_role"):
                st.success(f"✅ User '{target_user}' promoted to {updated_role}")
            else:
                st.error(f"⛔ Access Denied: {current_role.value} cannot change roles.")

    # 3. DELETE USER (Master Only)
    with st.container(border=True):
        st.write("###### Delete User")
        del_user = st.text_input("Username to Delete")

        if st.button("Delete User", type="primary"):
            if RBAC.check_access(current_role, "delete_user"):
                st.warning(f"✅ User '{del_user}' has been deleted.")
            else:
                st.error(f"⛔ Access Denied: {current_role.value} cannot delete users.")