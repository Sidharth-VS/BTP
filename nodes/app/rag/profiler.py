from typing import List, Tuple
import numpy as np
from nodes.app.rag.chroma_db import ChromaStore


class NodeProfiler:
    def __init__(self, chroma_store: ChromaStore):
        self.chroma_store = chroma_store

    def compute_profile(self, n_clusters: int = 3) -> Tuple[List[float], List[List[float]]]:
        """Calculates global centroid C_i and multi-cluster centroids P_i from local documents."""
        embeddings = self.chroma_store.get_all_embeddings()
        if not embeddings:
            dim = self.chroma_store.embedder.dimension
            return [0.0] * dim, []

        matrix = np.array(embeddings, dtype=np.float32)

        # 1. Global centroid C_i (mean vector)
        global_centroid = np.mean(matrix, axis=0)
        global_centroid_norm = global_centroid / (np.linalg.norm(global_centroid) + 1e-12)
        c_i = global_centroid_norm.tolist()

        # 2. Multi-centroid profile P_i (simple uniform clustering or sample centroids)
        p_i: List[List[float]] = []
        if len(matrix) <= n_clusters:
            p_i = matrix.tolist()
        else:
            # Deterministic interval sampling to avoid external scikit-learn dependency
            indices = np.linspace(0, len(matrix) - 1, n_clusters, dtype=int)
            samples = matrix[indices]
            p_i = (samples / (np.linalg.norm(samples, axis=1, keepdims=True) + 1e-12)).tolist()

        return c_i, p_i