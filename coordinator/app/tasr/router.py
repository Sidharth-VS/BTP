"""
Trust-Aware Secure Routing (TASR) Router matching arXiv:2605.28112v2 and base reference.
Implements Algorithm 2, Table 5, and Appendix H.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from coordinator.app.tasr.feedback import TASRFeedbackEngine

logger = logging.getLogger("coordinator.tasr.router")


class ClientTrustState:
    """Wrapper exposing individual client trust components for coordinator schemas/telemetry."""

    def __init__(self, node_id: str, router: "TrustAwareRouter") -> None:
        self.node_id = str(node_id)
        self._router = router

    @property
    def u_rel(self) -> float:
        return self._router.reputation.get(self.node_id, 1.0)

    @u_rel.setter
    def u_rel(self, val: float) -> None:
        self._router.reputation[self.node_id] = val

    @property
    def u_cons(self) -> float:
        return self._router.consistency_trust.get(self.node_id, 1.0)

    @u_cons.setter
    def u_cons(self, val: float) -> None:
        self._router.consistency_trust[self.node_id] = val

    @property
    def u_agr(self) -> float:
        return self._router.agreement_trust.get(self.node_id, 1.0)

    @u_agr.setter
    def u_agr(self, val: float) -> None:
        self._router.agreement_trust[self.node_id] = val

    @property
    def feedback_count(self) -> int:
        return self._router.feedback_count.get(self.node_id, 0)

    @feedback_count.setter
    def feedback_count(self, val: int) -> None:
        self._router.feedback_count[self.node_id] = val

    @property
    def s_i(self) -> float:
        return self._router._cold_start_factor(self.node_id)


class TrustAwareRouter:
    """Trust-aware post-routing reweighting for profile-based routers."""

    def __init__(
        self,
        decay_factor: float = 0.9,
        recovery_factor: float = 1.02,
        min_reputation: float = 0.01,
        warmup_queries: int = 50,
        cold_start_s0: float = 0.7,
        cold_start_tau: float = 30.0,
        docs_for_feedback: int = 5,
        threshold_mode: str = "dynamic",
        fixed_threshold: float = 0.5,
        alpha_r: float = 1.0,
        alpha_c: float = 1.0,
        alpha_a: float = 0.5,
        delta_c: float = 0.3,
        delta_a: float = 0.5,
        cons_winner_weight: float = 0.6,
        explore_interval: int = 20,
        explore_extra: int = 1,
        defense_mode: str = "rel_cons_agr",
        k_route: int = 3,
        # Alias parameters for backwards compatibility
        decay_gamma: Optional[float] = None,
        recovery_gamma: Optional[float] = None,
        u_min: Optional[float] = None,
        warmup_w: Optional[int] = None,
        cold_start_t: Optional[float] = None,
        exploration_interval: Optional[int] = None,
    ) -> None:
        self.decay_factor = decay_gamma if decay_gamma is not None else decay_factor
        self.recovery_factor = recovery_gamma if recovery_gamma is not None else recovery_factor
        self.min_reputation = u_min if u_min is not None else min_reputation
        self.warmup_queries = warmup_w if warmup_w is not None else warmup_queries
        self.cold_start_s0 = cold_start_s0
        self.cold_start_tau = cold_start_t if cold_start_t is not None else cold_start_tau
        self.docs_for_feedback = docs_for_feedback
        self.threshold_mode = threshold_mode
        self.fixed_threshold = fixed_threshold
        self.alpha_r = alpha_r
        self.alpha_c = alpha_c
        self.alpha_a = alpha_a
        self.delta_c = delta_c
        self.delta_a = delta_a
        self.cons_winner_weight = cons_winner_weight
        self.explore_interval = exploration_interval if exploration_interval is not None else explore_interval
        self.explore_extra = explore_extra
        self.defense_mode = defense_mode
        self.k_route = k_route

        self.centroids: Dict[Any, np.ndarray] = {}
        self.profile_centroids: Dict[Any, np.ndarray] = {}
        self.doc_embeddings: Dict[Any, np.ndarray] = {}

        self.reputation: Dict[Any, float] = {}
        self.consistency_trust: Dict[Any, float] = {}
        self.agreement_trust: Dict[Any, float] = {}
        self.feedback_count: Dict[Any, int] = {}
        self.query_count = 0

        self.reputation_history: Dict[Any, List[float]] = {}
        self.consistency_history: Dict[Any, List[float]] = {}
        self.agreement_history: Dict[Any, List[float]] = {}
        self.feedback_history: Dict[Any, List[float]] = {}
        self.rel_feedback_history: Dict[Any, List[float]] = {}
        self.cons_feedback_history: Dict[Any, List[float]] = {}
        self.agr_feedback_history: Dict[Any, List[float]] = {}

    @property
    def trust_states(self) -> Dict[str, ClientTrustState]:
        return {str(cid): ClientTrustState(str(cid), self) for cid in self.centroids}

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            return x / (norm + 1e-8) if norm > 1e-12 else x
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return np.where(norms > 1e-12, x / (norms + 1e-8), x)

    def _cold_start_factor(self, client_id: Any) -> float:
        count = self.feedback_count.get(client_id, 0)
        return self.cold_start_s0 + (1.0 - self.cold_start_s0) * (
            1.0 - math.exp(-count / self.cold_start_tau)
        )

    @staticmethod
    def _soft_gate(x: float, alpha: float, delta: float) -> float:
        return delta + (1.0 - delta) * (max(float(x), 0.0) ** alpha)

    def register_client(
        self,
        client_id: Any,
        centroid: Union[np.ndarray, List[float]],
        doc_embeddings: Optional[Union[np.ndarray, List[List[float]]]] = None,
        profile_centroids: Optional[Union[np.ndarray, List[List[float]]]] = None,
    ) -> None:
        """Register a client profile and feedback evidence pool."""
        c_arr = np.asarray(centroid, dtype=np.float64)
        self.centroids[client_id] = self._normalize(c_arr)

        if doc_embeddings is not None and len(doc_embeddings) > 0:
            docs = np.asarray(doc_embeddings, dtype=np.float64)
        else:
            docs = c_arr.reshape(1, -1) if c_arr.ndim == 1 else c_arr
        self.doc_embeddings[client_id] = self._normalize(docs)

        if profile_centroids is None:
            profile = c_arr.reshape(1, -1) if c_arr.ndim == 1 else c_arr
        elif np.asarray(profile_centroids).ndim == 1:
            profile = np.asarray(profile_centroids, dtype=np.float64).reshape(1, -1)
        else:
            profile = np.asarray(profile_centroids, dtype=np.float64)
        self.profile_centroids[client_id] = self._normalize(profile)

        self.reputation[client_id] = 1.0
        self.consistency_trust[client_id] = 1.0
        self.agreement_trust[client_id] = 1.0
        self.feedback_count[client_id] = 0

        self.reputation_history[client_id] = [1.0]
        self.consistency_history[client_id] = [1.0]
        self.agreement_history[client_id] = [1.0]
        self.feedback_history[client_id] = []
        self.rel_feedback_history[client_id] = []
        self.cons_feedback_history[client_id] = []
        self.agr_feedback_history[client_id] = []
        logger.info("TASR: Registered client '%s'", client_id)

    def register_node(
        self,
        node_id: str,
        centroids: Optional[Union[List[List[float]], np.ndarray]] = None,
        doc_embeddings: Optional[Union[List[List[float]], np.ndarray]] = None,
        centroid: Optional[Union[List[float], np.ndarray]] = None,
        profile_centroids: Optional[Union[List[List[float]], np.ndarray]] = None,
    ) -> None:
        """Alias for register_client providing multi-format parameter support."""
        if centroid is None and centroids is not None:
            c_arr = np.asarray(centroids, dtype=np.float64)
            if c_arr.ndim == 2 and c_arr.shape[0] > 0:
                c_val = np.mean(c_arr, axis=0)
                p_val = c_arr
            else:
                c_val = c_arr
                p_val = c_arr.reshape(1, -1) if c_arr.ndim == 1 else c_arr
        else:
            c_val = centroid
            p_val = profile_centroids

        if c_val is None:
            c_val = np.zeros(384, dtype=np.float64)

        self.register_client(
            client_id=node_id,
            centroid=c_val,
            doc_embeddings=doc_embeddings,
            profile_centroids=p_val,
        )

    def _effective_trust(self, client_id: Any, top_k: int = 3) -> float:
        score = self._cold_start_factor(client_id)
        if self.defense_mode != "none":
            score *= self.reputation.get(client_id, 1.0) ** self.alpha_r
        if self.defense_mode in ("rel_cons", "rel_cons_agr"):
            score *= self._soft_gate(self.consistency_trust.get(client_id, 1.0), self.alpha_c, self.delta_c)
        if self.defense_mode == "rel_cons_agr" and top_k >= 2:
            score *= self._soft_gate(self.agreement_trust.get(client_id, 1.0), self.alpha_a, self.delta_a)
        return float(score)

    def compute_trust_weight(self, node_id: Any, top_k: int = 3) -> float:
        return self._effective_trust(node_id, top_k=top_k)

    def route(self, query_emb: Union[np.ndarray, List[float]], top_k: int = 3) -> Tuple[List[Any], Dict[Any, float]]:
        """Route a query using base cosine scores reweighted by trust."""
        query = self._normalize(np.asarray(query_emb, dtype=np.float64))
        raw_scores: Dict[Any, float] = {}
        weighted_scores: Dict[Any, float] = {}
        for cid, centroid in self.centroids.items():
            prof = self.profile_centroids.get(cid)
            if prof is not None and prof.ndim == 2 and prof.shape[0] > 1:
                raw = float(np.max(prof @ query))
            else:
                raw = float(np.dot(query, centroid))
            raw_scores[cid] = raw
            weighted_scores[cid] = raw * self._effective_trust(cid, top_k=top_k)

        sorted_clients = sorted(weighted_scores, key=lambda x: weighted_scores[x], reverse=True)
        selected = sorted_clients[:top_k]
        if self.explore_interval > 0 and self.query_count > 0 and self.query_count % self.explore_interval == 0:
            remaining = [cid for cid in sorted_clients[top_k:] if cid not in selected]
            selected = selected + remaining[: self.explore_extra]
        return selected, raw_scores

    def route_query(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: Optional[int] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Coordinator interface returning (primary_routed, feedback_targets)."""
        k = top_k if top_k is not None else self.k_route
        query = self._normalize(np.asarray(query_embedding, dtype=np.float64))
        raw_scores: Dict[Any, float] = {}
        weighted_scores: Dict[Any, float] = {}

        for cid, centroid in self.centroids.items():
            prof = self.profile_centroids.get(cid)
            if prof is not None and prof.ndim == 2 and prof.shape[0] > 1:
                raw = float(np.max(prof @ query))
            else:
                raw = float(np.dot(query, centroid))
            raw_scores[cid] = raw
            weighted_scores[cid] = raw * self._effective_trust(cid, top_k=k)

        sorted_clients = sorted(weighted_scores, key=lambda x: weighted_scores[x], reverse=True)
        primary_routed = sorted_clients[:k]
        feedback_set = list(primary_routed)

        self.query_count += 1
        if self.explore_interval > 0 and self.query_count % self.explore_interval == 0:
            remaining = [cid for cid in sorted_clients[k:] if cid not in primary_routed]
            if remaining:
                feedback_set.extend(remaining[: self.explore_extra])

        return primary_routed, feedback_set

    def _simulate_retrieval(self, query_emb: np.ndarray, client_id: Any) -> np.ndarray:
        docs = self.doc_embeddings.get(client_id)
        if docs is None or len(docs) == 0:
            return np.zeros((0, query_emb.shape[0]), dtype=np.float64)
        scores = docs @ query_emb
        k = min(self.docs_for_feedback, len(scores))
        return docs[np.argsort(scores)[-k:]]

    def compute_relevance_feedback(self, query_emb: np.ndarray, returned_docs: np.ndarray) -> float:
        return TASRFeedbackEngine.calculate_relevance(query_emb, returned_docs)

    def compute_consistency_feedback(
        self,
        client_id: Any,
        query_emb: np.ndarray,
        returned_docs: np.ndarray,
    ) -> float:
        centroids = self.profile_centroids.get(client_id)
        return TASRFeedbackEngine.calculate_consistency(
            query_emb, centroids, returned_docs, rho=self.cons_winner_weight
        )

    def compute_agreement_feedback(
        self,
        query_emb: np.ndarray,
        selected_ids: List[Any],
        all_returned_docs: Dict[Any, np.ndarray],
        rel_feedbacks: Dict[Any, float],
        rel_threshold: float,
    ) -> Dict[Any, float]:
        return TASRFeedbackEngine.calculate_cross_client_agreement(
            feedback_nodes=selected_ids,
            node_doc_embeddings=all_returned_docs,
            node_relevance_scores=rel_feedbacks,
            node_trust_rel=self.reputation,
            rel_threshold=rel_threshold,
        )

    def _threshold(self, values: List[float]) -> float:
        if self.threshold_mode == "dynamic":
            return float(np.median(values)) if values else 0.5
        return self.fixed_threshold

    def _update_one_score(self, current: float, feedback: float, threshold: float) -> float:
        if feedback < threshold:
            return max(current * self.decay_factor, self.min_reputation)
        return min(current * self.recovery_factor, 1.0)

    def update_trust(
        self,
        arg1: Any,
        arg2: Any = None,
        arg3: Any = None,
        arg4: Any = None,
    ) -> None:
        """
        Flexible update_trust supporting both:
        - Base: update_trust(query_emb, selected_ids, returned_docs=None)
        - Coordinator: update_trust(feedback_nodes, f_rel, f_cons, f_agr)
        """
        if isinstance(arg1, (list, tuple)) and isinstance(arg2, dict) and isinstance(arg3, dict):
            feedback_nodes = list(arg1)
            f_rel = arg2
            f_cons = arg3
            f_agr = arg4 if isinstance(arg4, dict) else {}
            self._update_from_signals(feedback_nodes, f_rel, f_cons, f_agr)
        else:
            query_emb = arg1
            selected_ids = list(arg2) if arg2 is not None else []
            returned_docs = arg3 if isinstance(arg3, dict) else None
            self._update_from_docs(query_emb, selected_ids, returned_docs)

    def _update_from_docs(
        self,
        query_emb: Union[np.ndarray, List[float]],
        selected_ids: List[Any],
        returned_docs: Optional[Dict[Any, np.ndarray]] = None,
    ) -> None:
        query = self._normalize(np.asarray(query_emb, dtype=np.float64))
        overrides = returned_docs or {}
        returned = {
            cid: self._normalize(np.asarray(overrides[cid], dtype=np.float64))
            if cid in overrides
            else self._simulate_retrieval(query, cid)
            for cid in selected_ids
        }
        rel = {cid: self.compute_relevance_feedback(query, docs) for cid, docs in returned.items()}
        cons: Dict[Any, float] = {}
        if self.defense_mode in ("rel_cons", "rel_cons_agr"):
            cons = {cid: self.compute_consistency_feedback(cid, query, returned[cid]) for cid in selected_ids}
        agr: Dict[Any, float] = {}
        if self.defense_mode == "rel_cons_agr" and len(selected_ids) >= 2:
            rel_thresh = self._threshold(list(rel.values()))
            agr = self.compute_agreement_feedback(query, selected_ids, returned, rel, rel_thresh)

        self._update_from_signals(selected_ids, rel, cons, agr)

    def _update_from_signals(
        self,
        selected_ids: List[Any],
        rel: Dict[Any, float],
        cons: Dict[Any, float],
        agr: Dict[Any, float],
    ) -> None:
        for cid in selected_ids:
            rel_val = rel.get(cid, 0.0)
            cons_val = cons.get(cid, 1.0) if cons else 1.0
            agr_val = agr.get(cid, 1.0) if agr else 1.0

            if cid in self.rel_feedback_history:
                self.rel_feedback_history[cid].append(rel_val)
                self.cons_feedback_history[cid].append(cons_val)
                self.agr_feedback_history[cid].append(agr_val)
                combined = 0.5 * rel_val + 0.5 * cons_val if cons else rel_val
                self.feedback_history[cid].append(combined)

            self.feedback_count[cid] = self.feedback_count.get(cid, 0) + 1

        if self.query_count <= self.warmup_queries:
            self._record_histories()
            return

        if self.defense_mode != "none":
            threshold = self._threshold(list(rel.values()))
            for cid in selected_ids:
                if cid in self.reputation:
                    self.reputation[cid] = self._update_one_score(self.reputation[cid], rel.get(cid, 0.0), threshold)

        if self.defense_mode in ("rel_cons", "rel_cons_agr") and cons:
            threshold = self._threshold(list(cons.values()))
            for cid in selected_ids:
                if cid in self.consistency_trust:
                    self.consistency_trust[cid] = self._update_one_score(
                        self.consistency_trust[cid], cons.get(cid, 1.0), threshold
                    )

        if self.defense_mode == "rel_cons_agr" and agr:
            threshold = self._threshold([agr.get(cid, 1.0) for cid in selected_ids])
            for cid in selected_ids:
                if cid in self.agreement_trust:
                    self.agreement_trust[cid] = self._update_one_score(
                        self.agreement_trust[cid], agr.get(cid, 1.0), threshold
                    )

        self._record_histories()

    def _record_histories(self) -> None:
        for cid in self.centroids:
            if cid in self.reputation_history:
                self.reputation_history[cid].append(self.reputation[cid])
                self.consistency_history[cid].append(self.consistency_trust[cid])
                self.agreement_history[cid].append(self.agreement_trust[cid])

    def get_effective_score(self, client_id: Any, top_k: int = 3) -> float:
        return self._effective_trust(client_id, top_k=top_k)

    def get_trust_summary(self, malicious_ids: List[Any]) -> Dict[str, Any]:
        malicious = set(malicious_ids)
        honest = [cid for cid in self.reputation if cid not in malicious]

        def avg(values: Dict[Any, float], ids: List[Any]) -> float:
            present = [values[cid] for cid in ids if cid in values]
            return float(np.mean(present)) if present else 0.0

        return {
            "defense_mode": self.defense_mode,
            "total_queries": self.query_count,
            "malicious_reputation": {cid: self.reputation[cid] for cid in malicious if cid in self.reputation},
            "honest_avg_reputation": avg(self.reputation, honest),
            "malicious_consistency": {
                cid: self.consistency_trust[cid] for cid in malicious if cid in self.consistency_trust
            },
            "honest_avg_consistency": avg(self.consistency_trust, honest),
            "malicious_agreement": {
                cid: self.agreement_trust[cid] for cid in malicious if cid in self.agreement_trust
            },
            "honest_avg_agreement": avg(self.agreement_trust, honest),
        }
