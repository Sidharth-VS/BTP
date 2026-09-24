import os
from typing import List

# Limit ONNX runtime / OpenMP CPU thread contention in multi-node environments
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from shared.embeddings.base import BaseEmbeddingProvider


class LocalEmbeddings(BaseEmbeddingProvider):
    """
    Wraps ChromaDB's DefaultEmbeddingFunction (all-MiniLM-L6-v2, ONNX-based).
    Output: 384-dimensional normalised float vectors.
    """

    def __init__(self, warmup: bool = True) -> None:
        self._fn = DefaultEmbeddingFunction()
        self._dim = 384  # all-MiniLM-L6-v2 output dimension
        if warmup:
            try:
                # Pre-warm ONNX session and tokenizer at startup to eliminate first-query cold start
                self._fn(["warmup"])
            except Exception:
                pass

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    @property
    def dimension(self) -> int:
        return self._dim


# Backwards-compatible alias used by existing imports
LocalSentenceTransformerEmbeddings = LocalEmbeddings