import os


class Settings:
    # 1. AI Configuration
    LLM_MODEL = "phi3:mini"  # User requirement: SLM phi-3-mini
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" # Efficient local embeddings
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # User requirement: Cross-Encoder

    # 2. Database
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME = "crag_llamaindex"
    # collection name : crag_llamaindex, "real_estate_crag"


settings = Settings()