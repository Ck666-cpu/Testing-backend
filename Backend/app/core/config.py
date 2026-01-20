import os


class Settings:
    # 1. AI Configuration
    LLM_MODEL = "phi3:mini"  # User requirement: SLM phi-3-mini (Consider 'qwen2:7b' for better Chinese/Malay)
    # UPGRADE: Switch to Multilingual Embedding (Supports EN, ZH, MY)
    EMBEDDING_MODEL = "BAAI/bge-m3"

    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

    # 2. Database
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME = "crag_collects"
    # collection name : "crag_collects"


settings = Settings()