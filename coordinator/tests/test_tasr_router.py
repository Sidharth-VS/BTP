#Pseudo-Testing code, 
#Currently not the part of main project

"""Unit tests for TASR registration, routing, feedback, and trust updates."""

import math

import numpy as np
import pytest

from coordinator.app.tasr.router import TrustAwareRouter


def make_router(**overrides) -> TrustAwareRouter:
    options = {
        "cold_start_s0": 1.0,
        "cold_start_tau": 2.0,
        "warmup_queries": 0,
        "explore_interval": 0,
        "defense_mode": "rel_cons_agr",
        "threshold_mode": "fixed",
        "fixed_threshold": 0.5,
        "k_route": 2,
    }
    options.update(overrides)
    return TrustAwareRouter(**options)


def register_two_directional_nodes(router: TrustAwareRouter) -> None:
    router.register_client(
        "x-node",
        centroid=[1.0, 0.0],
        profile_centroids=[[1.0, 0.0]],
        doc_embeddings=[[1.0, 0.0]],
    )
    router.register_client(
        "y-node",
        centroid=[0.0, 1.0],
        profile_centroids=[[0.0, 1.0]],
        doc_embeddings=[[0.0, 1.0]],
    )


def test_register_client_normalizes_centroid_profile_and_documents():
    router = make_router()

    router.register_client(
        "node-a",
        centroid=[3.0, 4.0],
        profile_centroids=[[0.0, 5.0], [2.0, 0.0]],
        doc_embeddings=[[0.0, 2.0]],
    )

    np.testing.assert_allclose(router.centroids["node-a"], [0.6, 0.8])
    np.testing.assert_allclose(
        router.profile_centroids["node-a"], [[0.0, 1.0], [1.0, 0.0]]
    )
    np.testing.assert_allclose(router.doc_embeddings["node-a"], [[0.0, 1.0]])
    assert router.reputation["node-a"] == 1.0
    assert router.consistency_trust["node-a"] == 1.0
    assert router.agreement_trust["node-a"] == 1.0
    assert router.feedback_count["node-a"] == 0


def test_register_client_defaults_documents_and_profiles_to_centroid():
    router = make_router()

    router.register_client("node-a", centroid=[0.0, 7.0])

    np.testing.assert_allclose(router.centroids["node-a"], [0.0, 1.0])
    np.testing.assert_allclose(router.profile_centroids["node-a"], [[0.0, 1.0]])
    np.testing.assert_allclose(router.doc_embeddings["node-a"], [[0.0, 1.0]])


def test_profile_routing_uses_best_matching_profile_centroid():
    router = make_router()
    router.register_client(
        "multi-profile",
        centroid=[0.0, 1.0],
        profile_centroids=[[0.0, 1.0], [1.0, 0.0]],
    )
    router.register_client("x-node", centroid=[0.8, 0.6])

    primary, feedback = router.route_query([1.0, 0.0], top_k=1)

    assert primary == ["multi-profile"]
    assert feedback == ["multi-profile"]
    assert router.query_affinity([1.0, 0.0])["multi-profile"] == pytest.approx(
        1.0
    )


def test_effective_trust_combines_defense_components_and_cold_start():
    router = make_router(
        cold_start_s0=0.5,
        cold_start_tau=2.0,
        delta_c=0.2,
        delta_a=0.4,
    )
    router.register_client(
        "node-a",
        centroid=[1.0, 0.0],
        initial_trust_state={
            "u_rel": 0.25,
            "u_cons": 0.5,
            "u_agr": 0.25,
            "feedback_count": 2,
        },
    )
    cold_start = 0.5 + 0.5 * (1.0 - math.exp(-1.0))
    expected = cold_start * 0.25 * (0.2 + 0.8 * 0.5) * (
        0.4 + 0.6 * math.sqrt(0.25)
    )

    assert router.compute_trust_weight("node-a", top_k=2) == pytest.approx(
        expected
    )
    assert router.compute_trust_weight("node-a", top_k=1) == pytest.approx(
        cold_start * 0.25 * (0.2 + 0.8 * 0.5)
    )


def test_route_orders_by_similarity_times_effective_trust():
    router = make_router()
    router.register_client("aligned", centroid=[1.0, 0.0])
    router.register_client("less-aligned", centroid=[0.9, 0.43589])
    router.reputation["aligned"] = 0.2

    selected, raw_scores = router.route([1.0, 0.0], top_k=2)

    assert raw_scores["aligned"] > raw_scores["less-aligned"]
    assert selected[0] == "less-aligned"
    assert selected[1] == "aligned"


def test_register_node_alias_averages_profiles_and_restores_trust():
    router = make_router()
    router.register_node(
        "node-a",
        centroids=[[1.0, 0.0], [0.0, 1.0]],
        doc_embeddings=[[1.0, 0.0]],
        initial_trust_state={
            "u_rel": 0.4,
            "u_cons": 0.6,
            "u_agr": 0.8,
            "feedback_count": 7,
            "reputation_history": [1.0, 0.4],
        },
    )

    assert router.centroids["node-a"] == pytest.approx(
        [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]
    )
    np.testing.assert_allclose(
        router.profile_centroids["node-a"], [[1.0, 0.0], [0.0, 1.0]]
    )
    assert router.reputation["node-a"] == pytest.approx(0.4)
    assert router.consistency_trust["node-a"] == pytest.approx(0.6)
    assert router.agreement_trust["node-a"] == pytest.approx(0.8)
    assert router.feedback_count["node-a"] == 7
    assert router.reputation_history["node-a"] == [1.0, 0.4]


def test_reregister_without_snapshot_preserves_in_memory_trust():
    router = make_router()
    router.register_client("node-a", centroid=[1.0, 0.0])
    router.reputation["node-a"] = 0.3
    router.feedback_count["node-a"] = 9

    router.register_client("node-a", centroid=[0.0, 1.0])

    assert router.reputation["node-a"] == pytest.approx(0.3)
    assert router.feedback_count["node-a"] == 9
    assert router.centroids["node-a"] == pytest.approx([0.0, 1.0])


def test_trust_state_properties_are_readable_and_writable():
    router = make_router()
    router.register_client("node-a", centroid=[1.0, 0.0])

    state = router.trust_states["node-a"]
    state.u_rel = 0.4
    state.u_cons = 0.5
    state.u_agr = 0.6
    state.feedback_count = 3

    assert state.node_id == "node-a"
    assert state.u_rel == pytest.approx(0.4)
    assert state.u_cons == pytest.approx(0.5)
    assert state.u_agr == pytest.approx(0.6)
    assert state.feedback_count == 3
    assert state.s_i == pytest.approx(router._cold_start_factor("node-a"))


def test_relevance_and_consistency_feedback_for_aligned_evidence():
    router = make_router()
    router.register_client(
        "node-a",
        centroid=[1.0, 0.0],
        profile_centroids=[[1.0, 0.0]],
    )
    query = np.asarray([1.0, 0.0])
    aligned = np.asarray([[1.0, 0.0], [0.8, 0.6]])
    unrelated = np.asarray([[0.0, 1.0]])

    assert router.compute_relevance_feedback(query, aligned) == pytest.approx(0.9)
    assert router.compute_relevance_feedback(query, unrelated) == pytest.approx(0.0)
    assert router.compute_consistency_feedback(
        "node-a", query, aligned
    ) == pytest.approx(0.9)
    assert router.compute_consistency_feedback(
        "node-a", query, unrelated
    ) == pytest.approx(0.0)


def test_consistency_and_agreement_feedback_default_to_neutral_without_evidence():
    router = make_router()
    router.register_client("node-a", centroid=[1.0, 0.0])

    assert router.compute_consistency_feedback(
        "node-a", np.asarray([1.0, 0.0]), np.empty((0, 2))
    ) == pytest.approx(1.0)
    agreement = router.compute_agreement_feedback(
        np.asarray([1.0, 0.0]),
        ["node-a"],
        {"node-a": np.empty((0, 2))},
        {"node-a": 0.8},
        0.5,
    )
    assert agreement == {"node-a": 1.0}


def test_update_trust_decays_bad_signals_and_recovers_good_signals():
    router = make_router(
        decay_factor=0.5,
        recovery_factor=1.1,
        fixed_threshold=0.5,
    )
    register_two_directional_nodes(router)
    router.reputation["x-node"] = 0.8
    router.reputation["y-node"] = 0.8
    router.consistency_trust["x-node"] = 0.8
    router.consistency_trust["y-node"] = 0.8

    router.update_trust(
        ["x-node", "y-node"],
        {"x-node": 0.9, "y-node": 0.1},
        {"x-node": 0.9, "y-node": 0.1},
        {"x-node": 0.9, "y-node": 0.1},
    )

    assert router.query_count == 1
    assert router.feedback_count["x-node"] == 1
    assert router.feedback_count["y-node"] == 1
    assert router.reputation["x-node"] == pytest.approx(0.88)
    assert router.reputation["y-node"] == pytest.approx(0.4)
    assert router.consistency_trust["x-node"] == pytest.approx(0.88)
    assert router.consistency_trust["y-node"] == pytest.approx(0.4)
    assert router.agreement_trust["x-node"] == pytest.approx(0.88)
    assert router.agreement_trust["y-node"] == pytest.approx(0.4)
    assert router.feedback_history["x-node"] == pytest.approx([0.9])
    assert len(router.reputation_history["x-node"]) == 2


def test_document_feedback_update_applies_similarity_and_profile_consistency():
    router = make_router(defense_mode="rel_cons", fixed_threshold=0.5)
    register_two_directional_nodes(router)
    router.update_trust(
        [1.0, 0.0],
        ["x-node", "y-node"],
        {
            "x-node": [[1.0, 0.0]],
            "y-node": [[0.0, 1.0]],
        },
    )

    assert router.query_count == 1
    assert router.feedback_count["x-node"] == 1
    assert router.feedback_count["y-node"] == 1
    assert router.rel_feedback_history["x-node"] == pytest.approx([1.0])
    assert router.rel_feedback_history["y-node"] == pytest.approx([0.0])
    assert router.cons_feedback_history["x-node"] == pytest.approx([1.0])
    assert router.cons_feedback_history["y-node"] == pytest.approx([0.0])
    assert router.reputation["x-node"] == pytest.approx(1.0)
    assert router.reputation["y-node"] < 1.0


def test_eligible_gate_records_feedback_but_does_not_penalize_excluded_node():
    router = make_router()
    register_two_directional_nodes(router)

    router.update_trust(
        ["x-node", "y-node"],
        {"x-node": 1.0, "y-node": 0.0},
        {"x-node": 1.0, "y-node": 0.0},
        {"x-node": 1.0, "y-node": 0.0},
        eligible={"x-node"},
    )

    assert router.feedback_count["x-node"] == 1
    assert router.feedback_count["y-node"] == 1
    assert router.reputation["x-node"] == pytest.approx(1.0)
    assert router.reputation["y-node"] == pytest.approx(1.0)
    assert router.rel_feedback_history["y-node"] == [0.0]


def test_warmup_suppresses_updates_but_records_feedback_histories():
    router = make_router(warmup_queries=1)
    register_two_directional_nodes(router)
    state_before = router.export_trust_state("y-node")

    router.update_trust(
        ["x-node", "y-node"],
        {"x-node": 1.0, "y-node": 0.0},
        {"x-node": 1.0, "y-node": 0.0},
        {"x-node": 1.0, "y-node": 0.0},
    )

    assert router.query_count == 1
    assert router.reputation["y-node"] == state_before["u_rel"]
    assert router.consistency_trust["y-node"] == state_before["u_cons"]
    assert router.agreement_trust["y-node"] == state_before["u_agr"]
    assert router.feedback_count["y-node"] == 1
    assert router.rel_feedback_history["y-node"] == [0.0]


def test_exploration_adds_extra_feedback_target_on_configured_update_interval():
    router = make_router(explore_interval=2, explore_extra=1)
    register_two_directional_nodes(router)
    router.register_client("z-node", centroid=[0.7, 0.7])

    primary_before, feedback_before = router.route_query([1.0, 0.0], top_k=1)
    assert len(primary_before) == len(feedback_before) == 1

    router.update_trust(
        ["x-node"],
        {"x-node": 1.0},
        {"x-node": 1.0},
        {"x-node": 1.0},
    )
    primary_after_one, feedback_after_one = router.route_query(
        [1.0, 0.0], top_k=1
    )
    assert len(primary_after_one) == len(feedback_after_one) == 1

    router.update_trust(
        ["x-node"],
        {"x-node": 1.0},
        {"x-node": 1.0},
        {"x-node": 1.0},
    )
    primary_after_two, feedback_after_two = router.route_query(
        [1.0, 0.0], top_k=1
    )

    assert len(primary_after_two) == 1
    assert len(feedback_after_two) == 2
    assert feedback_after_two[0] == primary_after_two[0]
    assert len(set(feedback_after_two)) == 2


def test_defense_mode_none_does_not_weight_routing_by_reputation():
    router = make_router(defense_mode="none")
    router.register_client("best-match", centroid=[1.0, 0.0])
    router.register_client("second-match", centroid=[0.8, 0.6])
    router.reputation["best-match"] = 0.01

    selected, _ = router.route([1.0, 0.0], top_k=1)

    assert selected == ["best-match"]
