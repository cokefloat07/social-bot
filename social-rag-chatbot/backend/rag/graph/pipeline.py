import logging
import json
import httpx
from typing import AsyncGenerator
from config.settings import get_settings
from rag.graph.state import GraphState
from rag.graph.nodes import (
    input_parser,
    metadata_retriever,
    vector_retrieval,
    reasoning_node,
    response_formatter,
    confidence_scorer,
    memory_updater,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGPipeline:
    def __init__(self):
        self.use_ollama = settings.USE_OLLAMA
        self.ollama_url = settings.OLLAMA_BASE_URL
        self.ollama_model = settings.OLLAMA_MODEL

    def _run_graph_nodes(self, query: str, session_id: str) -> GraphState:
        state: GraphState = {
            "query": query,
            "session_id": session_id,
            "conversation_history": "",
            "video_context": {},
            "retrieved_chunks": [],
            "metadata_context": "",
            "reasoning": "",
            "answer": "",
            "evidence": [],
            "sources": [],
            "confidence": 0.0,
            "error": None,
        }

        state = input_parser(state)
        state = metadata_retriever(state)
        state = vector_retrieval(state)
        state = reasoning_node(state)
        state = confidence_scorer(state)

        return state

    async def stream_response(self, query: str, session_id: str = "default") -> AsyncGenerator[str, None]:
        try:
            state = self._run_graph_nodes(query, session_id)
            prompt = state["reasoning"]

            if self.use_ollama:
                full_answer = ""
                async for token in self._stream_ollama(prompt):
                    full_answer += token
                    yield token

                state["answer"] = full_answer
                state = response_formatter(state)

                sources_section = state["answer"][len(full_answer):]
                if sources_section:
                    yield sources_section

                confidence = state.get("confidence", 0.0)
                yield f"\n\n**Confidence:** {confidence:.0%}"

                state = memory_updater(state)
            else:
                answer = self._generate_fallback(state)
                state["answer"] = answer
                state = response_formatter(state)
                state = memory_updater(state)

                words = state["answer"].split(" ")
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            yield f"I encountered an error while processing your question: {str(e)}"

    async def _stream_ollama(self, prompt: str) -> AsyncGenerator[str, None]:
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                token = data.get("response", "")
                                if token:
                                    yield token
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError:
            logger.warning("Ollama not available, using fallback")
            yield self._generate_simple_fallback(prompt)
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield self._generate_simple_fallback(prompt)

    def _generate_fallback(self, state: GraphState) -> str:
        return self._generate_simple_fallback(state["reasoning"])

    def _generate_simple_fallback(self, prompt: str) -> str:
        metadata_section = ""
        evidence_section = ""

        lines = prompt.split("\n")
        in_metadata = False
        in_evidence = False

        for line in lines:
            if "VIDEO METADATA:" in line:
                in_metadata = True
                in_evidence = False
                continue
            if "TRANSCRIPT EVIDENCE:" in line:
                in_evidence = True
                in_metadata = False
                continue
            if "CONVERSATION HISTORY:" in line or "USER QUESTION:" in line:
                in_metadata = False
                in_evidence = False
                continue
            if in_metadata:
                metadata_section += line + "\n"
            if in_evidence:
                evidence_section += line + "\n"

        query_line = ""
        for line in lines:
            if "USER QUESTION:" in line:
                query_line = line.replace("USER QUESTION:", "").strip()
                break

        response = f"""Based on the analysis of both videos, here is my comparison:

**Analysis:**

{metadata_section.strip() if metadata_section.strip() else "Video metadata is being processed."}

**Evidence from Transcripts:**

{evidence_section.strip()[:500] if evidence_section.strip() else "Transcript evidence is being processed."}

**Key Findings:**

Analyzing the question "{query_line}", the data shows differences in engagement patterns, content style, and audience response between the two videos. The engagement rates, view counts, and interaction metrics provide quantitative evidence for comparison.

To get the best analysis with detailed reasoning, please ensure Ollama is running with the Llama model (`ollama run llama2`). The system will then provide real-time AI-powered analysis with streaming responses.

"""
        return response

    async def generate_response(self, query: str, session_id: str = "default") -> GraphState:
        state = self._run_graph_nodes(query, session_id)

        if self.use_ollama:
            full_answer = ""
            async for token in self._stream_ollama(state["reasoning"]):
                full_answer += token
            state["answer"] = full_answer
        else:
            state["answer"] = self._generate_fallback(state)

        state = response_formatter(state)
        state = confidence_scorer(state)
        state = memory_updater(state)
        return state