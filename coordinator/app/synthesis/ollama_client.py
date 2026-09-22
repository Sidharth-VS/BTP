"""Ollama generation client — deferred to Phase 2."""
import logging
import os
from typing import List
import ollama
from shared.schemas.common import SourceResult

logger = logging.getLogger("coordinator.synthesis")


class OllamaSynthesizer:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
        )
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.client = ollama.Client(host=self.base_url)

    def synthesize(self, query: str, sources: List[SourceResult], max_chunks: int = 4) -> str:
        if not sources:
            return "No relevant context was retrieved from the federated nodes to answer the query."

        # Take the top ranked chunks
        selected_sources = sources[:max_chunks]
        context_blocks = []
        for idx, src in enumerate(selected_sources, start=1):
            context_blocks.append(
                f"[Source {idx} | Node: {src.node_id} | Domain: {src.metadata.get('domain', 'general')}]:\n{src.content}"
            )
        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are the central LLM for a Federated Retrieval-Augmented Generation (FedRAG) system. "
            "Synthesize an accurate, concise answer to the user query based ONLY on the provided context retrieved "
            "from federated nodes. If the context does not contain sufficient details, state that clearly."
        )

        user_prompt = f"Context:\n{context_str}\n\nUser Question:\n{query}\n\nAnswer:"

        try:
            response = self.client.generate(
                model=self.model,
                prompt=user_prompt,
                system=system_prompt,
                options={"temperature": 0.2, "top_p": 0.9},
            )
            return response.get("response", "").strip()
        except Exception as e:
            logger.error("Ollama generation error on model %s: %s", self.model, e, exc_info=True)
            return f"Error communicating with local LLM ({self.model}) at {self.base_url}: {e}"