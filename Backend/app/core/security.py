from enum import Enum


class UserRole(str, Enum):
    STAFF = "staff"
    ADMIN = "admin"
    MASTER_ADMIN = "master_admin"


class RBAC:
    """
    Strict Role-Based Access Control based on Permission Tables.
    """
    PERMISSIONS = {
        # 1. STAFF PERMISSIONS
        UserRole.STAFF: [
            # Chat & Session
            "start_chat_session",
            "submit_chat_query",
            "view_own_chat_history",

            # Private Documents
            "upload_private_document",
            "view_own_private_documents",

            # Global Documents
            "view_global_documents"
        ],

        # 2. ADMIN PERMISSIONS
        UserRole.ADMIN: [
            # Chat & Session
            "start_chat_session",
            "submit_chat_query",
            "view_own_chat_history",

            # Private Documents
            "upload_private_document",
            "view_own_private_documents",

            # Global Documents
            "view_global_documents",
            "upload_global_document",  # Exclusive to Admin

            # User Management
            "create_user"
        ],

        # 3. MASTER ADMIN PERMISSIONS (Strict Management Only)
        UserRole.MASTER_ADMIN: [
            # User Management
            "create_user",
            "update_user_role",  # Exclusive
            "delete_user"  # Exclusive

            # NOTE: NO CHAT, NO DOCUMENT ACCESS allowed.
        ]
    }

    @staticmethod
    def check_access(user_role: UserRole, action: str) -> bool:
        """Returns True if the user_role is allowed to perform action."""
        allowed_actions = RBAC.PERMISSIONS.get(user_role, [])
        return action in allowed_actions