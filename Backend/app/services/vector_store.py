from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings as LlamaSettings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings


class VectorService:
    def __init__(self):
        print(" [VectorStore] Initializing Embedding Model & Settings...")

        # --- TUNING 4.3: Legal Chunking Strategy ---
        # Legal texts are dense. Smaller chunks (300-500) ensure the embedding
        # captures the specific clause/rule without noise.
        LlamaSettings.chunk_size = 512
        LlamaSettings.chunk_overlap = 100  # Good overlap ensures clauses aren't cut in half

        # --- TUNING 4.2: Query Instruction (BGE Specific) ---
        # BGE models perform better when told what the query represents.
        LlamaSettings.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            query_instruction="Represent this sentence for searching relevant legal and real estate documents: "
        )
        print(" [VectorStore] Embedding Model Loaded.")

        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.COLLECTION_NAME

        if not self.client.collection_exists(self.collection_name):
            print(f" [VectorStore] Collection '{self.collection_name}' not found. Creating it...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # BGE-BASE uses 768 dimensions (Small uses 384)
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
        # --- TUNING 4.4: Metadata ---
        # SimpleDirectoryReader automatically adds file_name.
        # We enforce the specific splitter here to ensure chunking is applied during ingestion.
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        # Apply parser
        nodes = parser.get_nodes_from_documents(documents)

        VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=True
        )
        return f"Successfully ingested {len(nodes)} chunks (Legal Optimized)."

    def clear_database(self):
        self.client.delete_collection(self.collection_name)
        return "Database cleared! Please re-ingest your documents."

    def list_ingested_files(self) -> List[str]:
        try:
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            seen_files = set()
            points, _ = response
            for point in points:
                payload = point.payload or {}
                # Handle different metadata structures
                f_name = payload.get("file_name") or payload.get("metadata", {}).get("file_name") or "Unknown"
                if f_name != "Unknown":
                    clean_name = f_name.split("/")[-1].split("\\")[-1]
                    seen_files.add(clean_name)
            return list(seen_files) if seen_files else ["No metadata found (Index might be empty)"]
        except Exception as e:
            return [f"Error fetching files: {str(e)}"]