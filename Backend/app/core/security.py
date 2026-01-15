from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    AGENT = "agent"
    GUEST = "guest"

class RBAC:
    """
    Simple Role-Based Access Control Logic.
    Defines who can do what.
    """
    PERMISSIONS = {
        UserRole.GUEST: ["read_general_info"],
        UserRole.AGENT: ["read_general_info", "read_property_prices", "chat_rag"],
        UserRole.ADMIN: ["read_general_info", "read_property_prices", "chat_rag", "ingest_documents", "delete_documents"]
    }

    @staticmethod
    def check_access(user_role: UserRole, action: str) -> bool:
        """Returns True if the user_role is allowed to perform action."""
        allowed_actions = RBAC.PERMISSIONS.get(user_role, [])
        if action in allowed_actions:
            return True
        return False