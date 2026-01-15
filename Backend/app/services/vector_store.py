from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings as LlamaSettings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings


class VectorService:
    def __init__(self):
        print(" [VectorStore] Initializing Embedding Model (This may take time)...")
        # FIX: Move this INSIDE the init, so it doesn't block imports
        LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)
        print(" [VectorStore] Embedding Model Loaded.")

        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.COLLECTION_NAME

        # Auto-create collection if missing
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
        )
        return f"Successfully ingested {len(documents)} pages."