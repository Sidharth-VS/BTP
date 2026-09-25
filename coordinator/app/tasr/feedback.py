"""
TASR Feedback Engine matching base implementation (arXiv:2605.28112v2).
Calculates:
  1. Retrieval Relevance (f_rel)
  2. Profile Consistency (f_cons)
  3. Cross-Client Agreement (f_agr)
"""
from typing import Dict, List, Optional, Union
import numpy as np


class TASRFeedbackEngine:
    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            return x / (norm + 1e-8) if norm > 1e-12 else x
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return np.where(norms > 1e-12, x / (norms + 1e-8), x)

    @classmethod
    def calculate_relevance(
        cls,
        query_embedding: Union[List[float], np.ndarray],
        doc_embeddings: Union[List[List[float]], np.ndarray],
        chunk_scores: Optional[List[float]] = None,
    ) -> float:
        """
        f_rel: Mean cosine similarity between query and returned evidence.
        Falls back to chunk_scores if doc_embeddings is empty.
        """
        if doc_embeddings is not None and len(doc_embeddings) > 0:
            q = cls._normalize(np.asarray(query_embedding, dtype=np.float64))
            docs = cls._normalize(np.asarray(doc_embeddings, dtype=np.float64))
            return float(np.mean(docs @ q))
        elif chunk_scores:
            return float(np.mean(chunk_scores))
        return 0.0

    @classmethod
    def calculate_consistency(
        cls,
        query_embedding: Union[List[float], np.ndarray],
        profile_centroids: Union[List[List[float]], np.ndarray],
        doc_embeddings: Union[List[List[float]], np.ndarray],
        rho: float = 0.6,
    ) -> float:
        """
        f_cons: Match between returned evidence and registered profile centroids.
        Matches exact formula from base/fedrag/rag/trust_defense.py.
        """
        if doc_embeddings is None or len(doc_embeddings) == 0:
            return 1.0
        if profile_centroids is None or len(profile_centroids) == 0:
            return 1.0

        q = cls._normalize(np.asarray(query_embedding, dtype=np.float64))
        docs = cls._normalize(np.asarray(doc_embeddings, dtype=np.float64))
        centroids = cls._normalize(np.asarray(profile_centroids, dtype=np.float64))

        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)

        winner = int(np.argmax(centroids @ q))
        if centroids.shape[0] == 1:
            return float(np.mean(docs @ centroids[winner]))

        all_sims = centroids @ docs.T  # (K, M)
        weight = rho
        other_weight = (1.0 - weight) / (centroids.shape[0] - 1)
        weighted = weight * all_sims[winner]
        for idx in range(centroids.shape[0]):
            if idx != winner:
                weighted = weighted + other_weight * all_sims[idx]
        return float(np.mean(weighted))

    @classmethod
    def calculate_cross_client_agreement(
        cls,
        feedback_nodes: List[str],
        node_doc_embeddings: Dict[str, Union[List[List[float]], np.ndarray]],
        node_relevance_scores: Dict[str, float],
        node_trust_rel: Dict[str, float],
        rel_threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        f_agr: Relevance-trust-weighted peer agreement across qualified clients.
        Matches exact formula from base/fedrag/rag/trust_defense.py.
        """
        agreements = {nid: 1.0 for nid in feedback_nodes}
        if len(feedback_nodes) < 2:
            return agreements

        rel_scores = [node_relevance_scores.get(nid, 0.0) for nid in feedback_nodes]
        if rel_threshold is None:
            rel_threshold = float(np.median(rel_scores)) if rel_scores else 0.5

        relevant = [nid for nid in feedback_nodes if node_relevance_scores.get(nid, 0.0) >= rel_threshold]
        if len(relevant) < 2:
            return agreements

        doc_centroids: Dict[str, Optional[np.ndarray]] = {}
        for nid in relevant:
            docs = node_doc_embeddings.get(nid)
            if docs is not None and len(docs) > 0:
                docs_arr = cls._normalize(np.asarray(docs, dtype=np.float64))
                mean_vec = np.mean(docs_arr, axis=0)
                doc_centroids[nid] = cls._normalize(mean_vec)
            else:
                doc_centroids[nid] = None

        for nid in feedback_nodes:
            if nid not in relevant or doc_centroids.get(nid) is None:
                agreements[nid] = 1.0
                continue

            weighted_sum = 0.0
            weight_total = 0.0
            for other in relevant:
                if other == nid or doc_centroids.get(other) is None:
                    continue
                peer_weight = node_trust_rel.get(other, 1.0)
                sim = float(np.dot(doc_centroids[nid], doc_centroids[other]))
                weighted_sum += peer_weight * sim
                weight_total += peer_weight

            agreements[nid] = float(weighted_sum / weight_total) if weight_total > 1e-12 else 1.0

        return agreements
