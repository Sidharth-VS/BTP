"""
Trust-Aware Secure Routing (TASR) Router matching arXiv:2605.28112v2.
Implements Algorithm 2, Table 5, and Appendix H.
"""
import logging
import math
from typing import Dict, List, Tuple
import numpy as np

from coordinator.app.tasr.feedback import TASRFeedbackEngine

logger = logging.getLogger("coordinator.tasr.router")


class ClientTrustState:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.u_rel: float = 1.0
        self.u_cons: float = 1.0
        self.u_agr: float = 1.0
        self.feedback_count: int = 0


class TrustAwareRouter:
    def __init__(
        self,
        decay_gamma: float = 0.9,
        recovery_gamma: float = 1.02,
        u_min: float = 0.01,
        warmup_w: int = 50,
        cold_start_s0: float = 0.7,
        cold_start_t: float = 30.0,
        alpha_r: float = 1.0,
        alpha_c: float = 1.0,
        alpha_a: float = 0.5,
        delta_c: float = 0.3,
        delta_a: float = 0.5,
        k_route: int = 3,
        exploration_interval: int = 20,
    ) -> None:
        self.gamma = decay_gamma
        self.gamma_rec = recovery_gamma
        self.u_min = u_min
        self.warmup_w = warmup_w
        self.s0 = cold_start_s0
        self.t_horizon = cold_start_t
        self.alpha_r = alpha_r
        self.alpha_c = alpha_c
        self.alpha_a = alpha_a
        self.delta_c = delta_c
        self.delta_a = delta_a
        self.k_route = k_route
        self.exploration_interval = exploration_interval

        self.query_count: int = 0
        self.trust_states: Dict[str, ClientTrustState] = {}
        self.profile_centroids: Dict[str, List[List[float]]] = {}

    def register_node(self, node_id: str, centroids: List[List[float]]) -> None:
        self.profile_centroids[node_id] = centroids
        if node_id not in self.trust_states:
            self.trust_states[node_id] = ClientTrustState(node_id=node_id)
            logger.info("TASR: Registered node '%s'", node_id)

    def compute_trust_weight(self, node_id: str) -> float:
        state = self.trust_states.get(node_id)
        if not state:
            return 1.0

        # Cold start schedule
        s_i = self.s0 + (1.0 - self.s0) * (1.0 - math.exp(-state.feedback_count / self.t_horizon))

        # Soft gates
        g_c = self.delta_c + (1.0 - self.delta_c) * (state.u_cons ** self.alpha_c)
        g_a = self.delta_a + (1.0 - self.delta_a) * (state.u_agr ** self.alpha_a)

        # Multiplicative trust weight
        tau_i = s_i * (state.u_rel ** self.alpha_r) * g_c * g_a
        return float(tau_i)

    def route_query(
        self,
        query_embedding: List[float],
    ) -> Tuple[List[str], List[str]]:
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        scores: List[Tuple[float, str]] = []
        for node_id, centroids in self.profile_centroids.items():
            if not centroids:
                sim = 0.0
            else:
                sims = [TASRFeedbackEngine._cosine_similarity(q, np.array(c, dtype=np.float32)) for c in centroids]
                sim = max(sims)

            tau_i = self.compute_trust_weight(node_id)
            reweighted = sim * tau_i
            scores.append((reweighted, node_id))

        scores.sort(reverse=True)
        ranked_nodes = [nid for _, nid in scores]

        primary_routed = ranked_nodes[:self.k_route]
        feedback_set = list(primary_routed)

        # Scheduled Exploration
        self.query_count += 1
        if self.query_count % self.exploration_interval == 0 and len(ranked_nodes) > self.k_route:
            extra_node = ranked_nodes[self.k_route]
            feedback_set.append(extra_node)

        return primary_routed, feedback_set

    def update_trust(
        self,
        feedback_nodes: List[str],
        f_rel: Dict[str, float],
        f_cons: Dict[str, float],
        f_agr: Dict[str, float],
    ) -> None:
        for nid in feedback_nodes:
            if nid in self.trust_states:
                self.trust_states[nid].feedback_count += 1

        if self.query_count <= self.warmup_w:
            return

        signals = {"rel": f_rel, "cons": f_cons, "agr": f_agr}
        for sig_name, scores in signals.items():
            val_list = [scores.get(nid, 0.0) for nid in feedback_nodes]
            if not val_list:
                continue
            theta_h = float(np.median(val_list))

            for nid in feedback_nodes:
                state = self.trust_states.get(nid)
                if not state:
                    continue

                curr_val = getattr(state, f"u_{sig_name}")
                obs_score = scores.get(nid, 0.0)

                # Median-thresholded step update
                if obs_score < theta_h:
                    new_val = max(self.u_min, self.gamma * curr_val)
                else:
                    new_val = min(1.0, self.gamma_rec * curr_val)

                setattr(state, f"u_{sig_name}", new_val)
