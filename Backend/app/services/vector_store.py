from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings as LlamaSettings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings


class VectorService:
    def __init__(self):
        print(" [VectorStore] Initializing Embedding Model & Settings...")

        # 1. Tuning for BGE-Small (Max 512 tokens)
        LlamaSettings.chunk_size = 512
        LlamaSettings.chunk_overlap = 50

        # Load Embedding Model locally
        LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)
        print(" [VectorStore] Embedding Model Loaded.")

        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.COLLECTION_NAME

        if not self.client.collection_exists(self.collection_name):
            print(f" [VectorStore] Collection '{self.collection_name}' not found. Creating it...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )

        self.vector_store = QdrantVectorStore(client=self.client, collection_name=self.collection_name)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def get_index(self):
        return VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context
        )

    def ingest_document(self, file_path: str):
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=True
        )
        return f"Successfully ingested {len(documents)} pages."

    def clear_database(self):
        self.client.delete_collection(self.collection_name)
        return "Database cleared! Please re-ingest your documents."

    def list_ingested_files(self) -> List[str]:
        """
        Scrolls through the Qdrant database to find unique file names in metadata.
        """
        try:
            # We scroll through points to get metadata
            # This is a basic implementation; for millions of points, you'd use a Payload index.
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,  # Check first 100 chunks (usually enough to see files)
                with_payload=True,
                with_vectors=False
            )

            seen_files = set()
            points, _ = response

            for point in points:
                payload = point.payload or {}
                # LlamaIndex usually stores file_name in metadata keys like 'file_name' or 'node_info'
                # Depending on version, it might be directly in payload
                f_name = payload.get("file_name") or payload.get("metadata", {}).get("file_name")
                if f_name:
                    # Clean path to just show filename
                    clean_name = f_name.split("/")[-1].split("\\")[-1]
                    seen_files.add(clean_name)

            return list(seen_files) if seen_files else ["No metadata found (Index might be empty)"]
        except Exception as e:
            return [f"Error fetching files: {str(e)}"]