from typing import List, Dict, Any
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

        self.llm = Ollama(
            model=settings.LLM_MODEL,
            request_timeout=300.0,
            additional_kwargs={"num_ctx": 2048, "num_predict": 512}
        )
        LlamaSettings.llm = self.llm

        self.vector_service = VectorService()
        self.index = self.vector_service.get_index()
        self.reranker = SentenceTransformerRerank(
            model=settings.RERANKER_MODEL, top_n=5
        )

        # --- PROMPTS ---

        self.classify_prompt = PromptTemplate(
            "Classify the User Input into exactly one category:\n"
            "1. GREETING: (Hello, Hi, Thanks, Bye)\n"
            "2. SESSION: (User name, preferences)\n"
            "3. GENERAL: (Weather, Jokes, General Knowledge)\n"
            "4. DOMAIN: (Real Estate, Tenancy, Contracts, Rent, Property)\n"
            "5. DEPENDENT: (Ambiguous follow-ups)\n"
            "Query: {query_str}\n"
            "Answer ONLY with the Category Name."
        )

        self.rewrite_prompt = PromptTemplate(
            "Task: Rewrite the Follow-up Question to be standalone based on Context.\n"
            "Context: {history_str}\n"
            "Follow-up: {query_str}\n"
            "Rewritten Question:"
        )

        self.session_prompt = PromptTemplate(
            "User Input: {query_str}\n"
            "Extract the name they want to be called. If none, output NONE.\n"
            "Name:"
        )

        # D. MULTI-QUERY PROMPT
        self.multiquery_prompt = PromptTemplate(
            "You are an AI assistant. Your task is to generate 3 different search queries based on the user's follow-up question and conversation history.\n"
            "1. A direct rewrite of the question.\n"
            "2. A search for related keywords (e.g., 'fees', 'legal', 'clause').\n"
            "3. A hypothetical answer snippet (what the document might say).\n"
            "Context: {history_str}\n"
            "Follow-up: {query_str}\n"
            "Output ONLY the 3 queries, separated by a newline."
        )

        # --- TUNING 3.1: Strict Answer Generation Prompt ---
        # This prevents the LLM from using its own training data.
        self.qa_prompt_tmpl = (
            "You are a strict Real Estate Assistant. \n"
            "Answer the question using ONLY the context provided below. \n"
            "If the answer is not explicitly in the context, say: 'I could not find this information in the provided documents.'\n"
            "Do NOT hallucinate. Do NOT write essays. Do NOT use outside knowledge. \n"
            "IMPORTANT: Answer in the same language as the user's question (e.g. Malay -> Malay). \n"
            "---------------------\n"
            "Context:\n"
            "{context_str}\n"
            "---------------------\n"
            "Question: {query_str}\n"
            "Answer:"
        )
        self.qa_prompt = PromptTemplate(self.qa_prompt_tmpl)

    def generate_response(self, query: str, history: List[str] = [], user_context: Dict[str, Any] = {}) -> Dict[
        str, Any]:
        user_name = user_context.get("user_name", "")
        friendly_prefix = f"{user_name}, " if user_name else ""

        # STEP 1: CLASSIFY
        category = self._classify_input(query)
        print(f" [CRAG] Intent: {category} | Query: '{query}'")

        result_template = {
            "answer": "", "sources": [], "intent": category, "session_updates": {}
        }

        # STEP 2: HANDLE NON-RETRIEVAL
        if category == "GREETING":
            result_template[
                "answer"] = f"Hello {user_name}! I can help with tenancy agreements and property questions." if user_name else "Hello! I am your Real Estate Assistant."
            return result_template

        if category == "SESSION":
            extracted_name = self.llm.complete(self.session_prompt.format(query_str=query)).text.strip()
            extracted_name = re.sub(r"[^\w\s]", "", extracted_name)
            if extracted_name and "NONE" not in extracted_name and len(extracted_name) < 20:
                result_template["answer"] = f"Nice to meet you, {extracted_name}."
                result_template["session_updates"] = {"user_name": extracted_name}
            else:
                result_template["answer"] = "Understood."
            return result_template

        if category == "GENERAL":
            result_template[
                "answer"] = f"{friendly_prefix}I focus specifically on Tenancy and Real Estate. I cannot answer general questions."
            return result_template

        # STEP 3: PREPARE QUERY
        search_query = query

        # --- TUNING 3.4: Query Normalization ---
        # Fix common grammar mistakes to improve embedding recall
        search_query = self._normalize_query(search_query)

        if category == "DEPENDENT":
            if history:
                print(" [CRAG] Rewriting...")
                raw_rewrite = self._rewrite_query(query, history)
                search_query = self._clean_rewrite(raw_rewrite, query)
                print(f" [CRAG] Rewritten: '{search_query}'")
            else:
                result_template[
                    "answer"] = f"{friendly_prefix}could you please clarify which agreement you are referring to?"
                return result_template

        # STEP 4: EXECUTE RAG WITH SAFETY
        rag_result = self._run_rag_pipeline(search_query)

        # --- TUNING 3.2: Block Free-Form Hallucination ---
        # Final check: If the model generated an essay despite our prompts, kill it.
        final_answer = rag_result["answer"]
        forbidden_keywords = ["essay", "methodology", "urban planning", "renewable energy", "introduction",
                              "conclusion"]

        # If answer is long (>500 chars) AND contains forbidden words, it's likely a hallucination
        if len(final_answer) > 500 and any(word in final_answer.lower() for word in forbidden_keywords):
            print(" [CRAG] 🚨 Hallucination Detected and Blocked.")
            result_template[
                "answer"] = "I apologize, but I could not find a specific answer in the documents, and I am restricted from guessing."
            result_template["sources"] = []
        else:
            result_template["answer"] = final_answer
            result_template["sources"] = rag_result["sources"]
            # Pass debug info
            result_template["debug_nodes"] = rag_result.get("debug_nodes", [])

        return result_template

    def _normalize_query(self, query: str) -> str:
        """Fixes common grammar issues for better retrieval"""
        q = query.lower()
        # Fix "what should included" -> "what should include"
        q = q.replace("should included", "should be included")
        q = q.replace("what include", "what is included")
        return q

    def _classify_input(self, query: str) -> str:
        try:
            greetings = {'hello', 'hi', 'hey', 'thanks', 'good morning'}
            if query.lower().strip().strip('!.?') in greetings: return "GREETING"

            prompt = self.classify_prompt.format(query_str=query)
            response = self.llm.complete(prompt).text.strip().upper()

            if "GREETING" in response: return "GREETING"
            if "SESSION" in response: return "SESSION"
            if "GENERAL" in response: return "GENERAL"
            if "DEPENDENT" in response: return "DEPENDENT"
            return "DOMAIN"
        except:
            return "DOMAIN"

    def _run_rag_pipeline(self, search_query: str) -> dict:
        # 1. Retrieve (Top K = 15)
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=15)
        nodes = retriever.retrieve(search_query)

        # 2. Rerank (Top N = 5)
        if nodes:
            nodes = self.reranker.postprocess_nodes(nodes, query_str=search_query)

        # --- TUNING 3.3: Strengthen Confidence Threshold ---
        # If the best score is below 0.35, the retrieval failed.
        if not nodes or (nodes[0].score is not None and nodes[0].score < 0.35):
            print(f" [CRAG] Low Confidence: {nodes[0].score if nodes else 0}")
            return {
                "answer": "I searched the internal documents, but I couldn't find information closely related to that. Could you try rephrasing?",
                "sources": [],
                "debug_nodes": nodes  # Return even if rejected for Admin view
            }

        # 3. Generate Answer (Strict)
        synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt,  # Apply STRICT prompt
            response_mode="compact"
        )
        response_obj = synthesizer.synthesize(search_query, nodes=nodes)

        # 4. Extract Sources
        source_list = []
        debug_nodes_data = []
        for node in response_obj.source_nodes:
            file_name = node.metadata.get("file_name", "Unknown")
            page_label = node.metadata.get("page_label", "N/A")
            score = f"{node.score:.2f}" if node.score else "N/A"
            text_preview = node.get_content()[:100]

            source_list.append(f"{file_name} (Page {page_label}) - Score: {score}")
            debug_nodes_data.append(f"[{score}] {file_name}: {text_preview}...")

        return {
            "answer": str(response_obj),
            "sources": source_list[:3],
            "debug_nodes": debug_nodes_data
        }

    def _rewrite_query(self, query: str, history: List[str]) -> str:
        try:
            # Use last 3 turns for better context
            history_str = "\n".join(history[-3:])

            prompt = self.multiquery_prompt.format(history_str=history_str, query_str=query)
            response = self.llm.complete(prompt).text.strip()

            # Combine the user's original query with the AI's 3 variations
            # This creates a "Mega Query" that hits multiple keywords in the vector store
            combined_query = f"{query}\n{response}"
            return combined_query
        except:
            return query

    def _clean_rewrite(self, rewrite: str, original: str) -> str:
        clean = re.sub(r'^(Rewritten Question:|Rewritten:|Question:)', '', rewrite, flags=re.IGNORECASE).strip()
        clean = clean.strip('"').strip("'")
        if len(clean) > len(original) * 4 or "apologize" in clean.lower(): return original
        return clean if clean else original