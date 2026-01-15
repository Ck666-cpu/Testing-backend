from typing import List, Dict
import re
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
            additional_kwargs={"num_ctx": 2048, "num_predict": 512}
        )
        LlamaSettings.llm = self.llm

        # 2. Setup Vector Store & Reranker
        self.vector_service = VectorService()
        self.index = self.vector_service.get_index()
        self.reranker = SentenceTransformerRerank(
            model=settings.RERANKER_MODEL, top_n=5
        )

        # --- PROMPTS ---

        # A. INTENT CLASSIFIER (The Gatekeeper)
        self.classify_prompt = PromptTemplate(
            "Analyze the User Query. Classify it into exactly one category:\n"
            "1. GREETING: (Hello, Hi, Thanks, Bye)\n"
            "2. GENERAL: (Weather, Date, Time, Jokes, General Knowledge not related to Real Estate)\n"
            "3. DOMAIN: (Real Estate, Tenancy, Contracts, Rent, Property, Fees)\n"
            "4. DEPENDENT: (Ambiguous questions referring to previous context, e.g., 'Who pays it?', 'How much?')\n\n"
            "Query: {query_str}\n"
            "Answer ONLY with the Category Name (GREETING, GENERAL, DOMAIN, DEPENDENT)."
        )

        # B. REWRITE PROMPT
        self.rewrite_prompt = PromptTemplate(
            "Task: Rewrite the Follow-up Question to be standalone based on the Context.\n"
            "Context: {history_str}\n"
            "Follow-up: {query_str}\n"
            "Rewritten Question:"
        )

    def generate_response(self, query: str, history: List[str] = []) -> str:
        """
        Pipeline: Classify -> Gatekeep -> Route -> Execute
        """
        # STEP 1: CLASSIFY INTENT
        category = self._classify_input(query)
        print(f" [CRAG] Intent: {category} | Query: '{query}'")

        # STEP 2: HANDLE NON-RETRIEVAL CATEGORIES

        if category == "GREETING":
            return "Hello! I am your Real Estate AI Assistant. I can help with tenancy agreements and property questions."

        if category == "GENERAL":
            return "I am designed specifically for Real Estate queries. I cannot answer general questions about the weather, date, or outside topics."

        # STEP 3: HANDLE RETRIEVAL CATEGORIES (DOMAIN & DEPENDENT)
        search_query = query

        # Only rewrite if it is DEPENDENT and we actually have history
        if category == "DEPENDENT":
            if history:
                print(" [CRAG] Context dependency detected. Rewriting...")
                raw_rewrite = self._rewrite_query(query, history)
                search_query = self._clean_rewrite(raw_rewrite, query)
                print(f" [CRAG] Rewritten Query: '{search_query}'")
            else:
                # If dependent but no history, ask for clarification
                return "Could you please clarify what specific property or agreement you are asking about?"

        # STEP 4: EXECUTE RAG
        return self._run_rag_pipeline(search_query)

    def _classify_input(self, query: str) -> str:
        """Determines Intent using Phi-3"""
        try:
            # Fast keyword check for greetings
            greetings = {'hello', 'hi', 'hey', 'good morning', 'thanks'}
            if query.lower().strip().strip('!.?') in greetings:
                return "GREETING"

            prompt = self.classify_prompt.format(query_str=query)
            response = self.llm.complete(prompt).text.strip().upper()

            if "GREETING" in response: return "GREETING"
            if "GENERAL" in response: return "GENERAL"
            if "DEPENDENT" in response: return "DEPENDENT"
            return "DOMAIN"  # Default to Domain if unsure
        except:
            return "DOMAIN"

    def _run_rag_pipeline(self, search_query: str):
        """Standard RAG: Retrieve -> Rerank -> Synthesize"""
        # 1. Retrieve
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=15)
        nodes = retriever.retrieve(search_query)

        # 2. Rerank
        if nodes:
            nodes = self.reranker.postprocess_nodes(nodes, query_str=search_query)

        # 3. Check if empty after rerank
        if not nodes:
            return "I searched the internal database, but I couldn't find specific information regarding that."

        # 4. Generate Answer
        synthesizer = get_response_synthesizer(response_mode="compact")
        response = synthesizer.synthesize(search_query, nodes=nodes)
        return str(response)

    def _rewrite_query(self, query: str, history: List[str]) -> str:
        try:
            # Use last 2 turns
            history_str = "\n".join(history[-2:])
            prompt = self.rewrite_prompt.format(history_str=history_str, query_str=query)
            return self.llm.complete(prompt).text.strip()
        except:
            return query

    def _clean_rewrite(self, rewrite: str, original: str) -> str:
        clean = re.sub(r'^(Rewritten Question:|Rewritten:|Question:)', '', rewrite, flags=re.IGNORECASE).strip()
        clean = clean.strip('"').strip("'")
        # Safety valve: if rewrite is huge or apologetic, revert
        if len(clean) > len(original) * 4 or "apologize" in clean.lower():
            return original
        return clean if clean else original