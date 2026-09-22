"""
TASR Feedback Engine matching arXiv:2605.28112v2 (Appendix G).
Calculates:
  1. Retrieval Relevance (f_rel)
  2. Profile Consistency (f_cons)
  3. Cross-Client Agreement (f_agr)
"""
from typing import Dict, List, Optional
import numpy as np


class TASRFeedbackEngine:
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-12:
            return 0.0
        return float(np.dot(a, b) / norm)

    @classmethod
    def calculate_relevance(
        cls,
        query_embedding: List[float],
        doc_embeddings: List[List[float]],
        chunk_scores: Optional[List[float]] = None,
    ) -> float:
        """
        f_rel: Mean cosine similarity between query and returned evidence.
        Falls back to ChromaDB cosine similarity scores if doc_embeddings is empty.
        """
        if doc_embeddings:
            q = np.array(query_embedding, dtype=np.float32)
            sims = [cls._cosine_similarity(q, np.array(z, dtype=np.float32)) for z in doc_embeddings]
            return float(np.mean(sims))
        elif chunk_scores:
            return float(np.mean(chunk_scores))
        return 0.0

    @classmethod
    def calculate_consistency(
        cls,
        query_embedding: List[float],
        profile_centroids: List[List[float]],
        doc_embeddings: List[List[float]],
        rho: float = 0.6,
    ) -> float:
        """f_cons: Match between returned evidence and registered profile centroids."""
        if not doc_embeddings or not profile_centroids:
            return 1.0

        q = np.array(query_embedding, dtype=np.float32)
        k_prof = len(profile_centroids)

        if k_prof == 1:
            p = np.array(profile_centroids[0], dtype=np.float32)
            sims = [cls._cosine_similarity(p, np.array(z, dtype=np.float32)) for z in doc_embeddings]
            return float(np.mean(sims))

        p_sims = [cls._cosine_similarity(q, np.array(p, dtype=np.float32)) for p in profile_centroids]
        best_k = int(np.argmax(p_sims))

        weights = []
        for k in range(k_prof):
            if k == best_k:
                weights.append(rho)
            else:
                weights.append((1.0 - rho) / (k_prof - 1))

        doc_consistencies = []
        for z in doc_embeddings:
            z_arr = np.array(z, dtype=np.float32)
            weighted_sim = sum(
                weights[k] * cls._cosine_similarity(np.array(profile_centroids[k], dtype=np.float32), z_arr)
                for k in range(k_prof)
            )
            doc_consistencies.append(weighted_sim)

        return float(np.mean(doc_consistencies))

    @classmethod
    def calculate_cross_client_agreement(
        cls,
        feedback_nodes: List[str],
        node_doc_embeddings: Dict[str, List[List[float]]],
        node_relevance_scores: Dict[str, float],
        node_trust_rel: Dict[str, float],
    ) -> Dict[str, float]:
        """f_agr: Relevance-trust-weighted peer agreement across qualified clients."""
        agreements = {nid: 1.0 for nid in feedback_nodes}
        if len(feedback_nodes) < 2:
            return agreements

        rel_scores = [node_relevance_scores.get(nid, 0.0) for nid in feedback_nodes]
        theta_rel = float(np.median(rel_scores))

        b_q = [nid for nid in feedback_nodes if node_relevance_scores.get(nid, 0.0) >= theta_rel]
        if len(b_q) < 2:
            return agreements

        centroids: Dict[str, np.ndarray] = {}
        for nid in b_q:
            docs = node_doc_embeddings.get(nid, [])
            if docs:
                mean_vec = np.mean(docs, axis=0)
                norm = np.linalg.norm(mean_vec)
                centroids[nid] = (mean_vec / norm) if norm > 1e-12 else mean_vec
            else:
                centroids[nid] = np.zeros(384, dtype=np.float32)

        for nid in b_q:
            peers = [p for p in b_q if p != nid]
            denom = sum(node_trust_rel.get(p, 1.0) for p in peers)
            if denom < 1e-12:
                agreements[nid] = 1.0
                continue

            numer = sum(
                node_trust_rel.get(p, 1.0) * cls._cosine_similarity(centroids[nid], centroids[p])
                for p in peers
            )
            agreements[nid] = max(0.0, min(1.0, float(numer / denom)))

        return agreements
