from typing import List
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings as LlamaSettings, get_response_synthesizer, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from app.core.config import settings
from app.services.vector_store import VectorService


class CRAGService:
    def __init__(self):
        print(f" [CRAG] Initializing with Model: {settings.LLM_MODEL}...")

        # 1. Setup Phi-3 (SLM) with Memory Limits
        self.llm = Ollama(
            model=settings.LLM_MODEL,
            request_timeout=300.0,
            additional_kwargs={"num_ctx": 2048, "num_predict": 512}
        )
        LlamaSettings.llm = self.llm

        # 2. Setup Vector Store & Reranker
        self.vector_service = VectorService()
        self.index = self.vector_service.get_index()
        self.reranker = SentenceTransformerRerank(
            model=settings.RERANKER_MODEL, top_n=3
        )

        # 3. PROMPTS
        # A. Relevance Check (CRAG)
        self.eval_prompt = PromptTemplate(
            "Context: {context_str}\n"
            "Question: {query_str}\n"
            "Instruction: Does the Context answer the Question? Answer YES or NO."
        )

        # B. History Rewriter (NEW)
        self.rewrite_prompt = PromptTemplate(
            "History:\n{history_str}\n\n"
            "Current Question: {query_str}\n"
            "Task: Rewrite the Current Question to be standalone, including necessary details from History.\n"
            "Rewritten Question:"
        )

    def generate_response(self, query: str, history: List[str] = []):
        """
        Full Pipeline: Rewrite -> Retrieve -> Rerank -> Correct -> Generate
        """
        # Step 0: Contextualize (Rewrite Query)
        search_query = query
        if history:
            print(f" [CRAG] Rewriting query based on history...")
            search_query = self._rewrite_query(query, history)
            print(f" [CRAG] Rewritten: '{search_query}'")

        # Step A: Retrieve
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=10)
        nodes = retriever.retrieve(search_query)

        # Step B: Rerank
        if nodes:
            nodes = self.reranker.postprocess_nodes(nodes, query_str=search_query)

        # Step C: Corrective Evaluation
        is_relevant = self._evaluate_relevance(search_query, nodes)

        if not is_relevant:
            return "I couldn't find relevant information in the internal database for that specific question."

        # Step D: Generate Answer
        synthesizer = get_response_synthesizer(response_mode="compact")
        response = synthesizer.synthesize(search_query, nodes=nodes)

        return response

    def _rewrite_query(self, query: str, history: List[str]) -> str:
        """Uses Phi-3 to merge history into the new question"""
        try:
            # Only use last 2 turns to keep it fast
            history_str = "\n".join(history[-2:])
            prompt = self.rewrite_prompt.format(history_str=history_str, query_str=query)

            response = self.llm.complete(prompt).text.strip()

            # Sanity check: If rewrite is empty or fails, use original
            return response if len(response) > 3 else query
        except Exception as e:
            print(f"Warning: Rewrite failed ({e}). Using original query.")
            return query

    def _evaluate_relevance(self, query, nodes):
        """Double-check if documents match the query"""
        if not nodes: return False
        context_text = nodes[0].get_content()[:500]  # Check first 500 chars of top result
        prompt = self.eval_prompt.format(context_str=context_text, query_str=query)
        verdict = self.llm.complete(prompt).text.strip().upper()
        return "YES" in verdict