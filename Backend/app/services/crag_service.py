from llama_index.llms.ollama import Ollama
from llama_index.core import Settings as LlamaSettings, get_response_synthesizer, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from app.core.config import settings
from app.services.vector_store import VectorService


class CRAGService:
    def __init__(self):
        print(f" [CRAG] Initializing with Model: {settings.LLM_MODEL}...")

        # 1. Setup Phi-3 (SLM)
        self.llm = Ollama(
            model=settings.LLM_MODEL,
            request_timeout=300.0,
            additional_kwargs={
                "num_ctx": 2048,  # Limits memory usage to ~2GB
                "num_predict": 512  # Limits the answer length
            }
        )
        LlamaSettings.llm = self.llm

        # 2. Setup Vector Store Connection
        self.vector_service = VectorService()
        self.index = self.vector_service.get_index()

        # 3. Setup Reranker (Cross-Encoder)
        self.reranker = SentenceTransformerRerank(
            model=settings.RERANKER_MODEL,
            top_n=3  # Only keep top 3 most relevant chunks
        )

        # 4. Corrective Evaluator Prompt (Self-Correction)
        self.eval_prompt = PromptTemplate(
            "Context: {context_str}\n"
            "Question: {query_str}\n"
            "Instruction: Rate if the Context contains the answer to the Question.\n"
            "Answer ONLY 'YES' or 'NO'."
        )

    def generate_response(self, query: str):
        """
        Full CRAG Pipeline: Retrieve -> Rerank -> Correct/Evaluate -> Generate
        """
        print(f" [CRAG] Processing: {query}")

        # Step A: Retrieve
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=10)
        nodes = retriever.retrieve(query)

        # Step B: Rerank (Cross-Encoder)
        if nodes:
            nodes = self.reranker.postprocess_nodes(nodes, query_str=query)

        # Step C: Corrective Evaluation (The "C" in CRAG)
        # We check if the top node is actually relevant before generating.
        is_relevant = self._evaluate_relevance(query, nodes)

        if not is_relevant:
            # FALLBACK MECHANISM (Since we are local/intranet, we can't Google Search)
            # We return a safe fallback message.
            return "I searched the internal database, but the retrieved documents do not seem relevant to your specific question. Please contact an admin to upload more data."

        # Step D: Generate Answer (Phi-3)
        # Using a synthesizer to combine context and question
        synthesizer = get_response_synthesizer(response_mode="compact")
        response = synthesizer.synthesize(query, nodes=nodes)

        return response

    def _evaluate_relevance(self, query, nodes):
        """Uses the LLM to double-check if the docs are actually useful."""
        if not nodes:
            return False

        # Check the top 1 result for speed
        context_text = nodes[0].get_content()
        prompt = self.eval_prompt.format(context_str=context_text, query_str=query)

        # Ask Phi-3 if it's relevant
        verdict = self.llm.complete(prompt).text.strip().upper()
        print(f" [CRAG] Relevance Check: {verdict}")

        return "YES" in verdict