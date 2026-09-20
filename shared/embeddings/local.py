"""
Local embedding provider using ChromaDB's built-in DefaultEmbeddingFunction.

Uses all-MiniLM-L6-v2 via onnxruntime — no PyTorch required.
"""
from typing import List

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from shared.embeddings.base import BaseEmbeddingProvider


class LocalEmbeddings(BaseEmbeddingProvider):
    """
    Wraps ChromaDB's DefaultEmbeddingFunction (all-MiniLM-L6-v2, ONNX-based).
    Output: 384-dimensional normalised float vectors.
    """

    def __init__(self) -> None:
        self._fn = DefaultEmbeddingFunction()
        self._dim = 384  # all-MiniLM-L6-v2 output dimension

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    @property
    def dimension(self) -> int:
        return self._dim


# Backwards-compatible alias used by existing imports
LocalSentenceTransformerEmbeddings = LocalEmbeddings