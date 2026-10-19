#Pseudo-Testing code, 
#Currently not the part of main project



"""End-to-end tests for TASR feedback updates and trust-aware routing."""

import numpy as np
import pytest

from coordinator.app.tasr.db import TASRDatabase
from coordinator.app.tasr.router import TrustAwareRouter


@pytest.fixture
def router() -> TrustAwareRouter:
    return TrustAwareRouter(
        decay_factor=0.8,
        recovery_factor=1.02,
        min_reputation=0.01,
        warmup_queries=0,
        cold_start_s0=1.0,
        cold_start_tau=1.0,
        threshold_mode="fixed",
        fixed_threshold=0.5,
        delta_c=0.3,
        delta_a=0.5,
        explore_interval=0,
        defense_mode="rel_cons_agr",
        k_route=1,
    )


def register_similar_nodes(router: TrustAwareRouter) -> None:
    """Register similarly ranked nodes with distinct evidence quality."""
    claimed_profile = [[1.0, 0.0]]
    router.register_client(
        client_id="honest",
        centroid=[1.0, 0.0],
        profile_centroids=claimed_profile,
        doc_embeddings=[[1.0, 0.0]],
    )
    router.register_client(
        client_id="unreliable",
        centroid=[1.0, 0.0],
        profile_centroids=claimed_profile,
        doc_embeddings=[[0.0, 1.0]],
    )


def test_repeated_feedback_reduces_unreliable_node_trust_and_changes_route(router):
    register_similar_nodes(router)
    query = np.asarray([1.0, 0.0])
    selected_nodes = ["honest", "unreliable"]
    returned_evidence = {
        "honest": np.asarray([[1.0, 0.0]]),
        "unreliable": np.asarray([[0.0, 1.0]]),
    }

    initial_unreliable_reputation = router.reputation["unreliable"]
    initial_unreliable_consistency = router.consistency_trust["unreliable"]

    for _ in range(5):
        router.update_trust(query, selected_nodes, returned_evidence)

    assert router.query_count == 5
    assert router.feedback_count["honest"] == 5
    assert router.feedback_count["unreliable"] == 5
    assert router.reputation["honest"] == pytest.approx(1.0)
    assert router.consistency_trust["honest"] == pytest.approx(1.0)
    assert router.reputation["unreliable"] == pytest.approx(
        initial_unreliable_reputation * router.decay_factor**5
    )
    assert router.consistency_trust["unreliable"] == pytest.approx(
        initial_unreliable_consistency * router.decay_factor**5
    )
    assert router.rel_feedback_history["honest"] == pytest.approx([1.0] * 5)
    assert router.rel_feedback_history["unreliable"] == pytest.approx([0.0] * 5)
    assert router.cons_feedback_history["honest"] == pytest.approx([1.0] * 5)
    assert router.cons_feedback_history["unreliable"] == pytest.approx([0.0] * 5)

    primary, feedback_targets = router.route_query(query, top_k=1)
    assert primary == ["honest"]
    assert feedback_targets == ["honest"]
    assert router.compute_trust_weight("honest") > router.compute_trust_weight(
        "unreliable"
    )


def test_warmup_records_feedback_without_changing_trust():
    router = TrustAwareRouter(
        warmup_queries=2,
        cold_start_s0=1.0,
        explore_interval=0,
        defense_mode="rel_cons",
    )
    register_similar_nodes(router)
    query = np.asarray([1.0, 0.0])
    evidence = {
        "honest": np.asarray([[1.0, 0.0]]),
        "unreliable": np.asarray([[0.0, 1.0]]),
    }
    initial_state = {
        node_id: router.export_trust_state(node_id)
        for node_id in ("honest", "unreliable")
    }

    router.update_trust(query, ["honest", "unreliable"], evidence)
    router.update_trust(query, ["honest", "unreliable"], evidence)

    assert router.query_count == 2
    for node_id, state in initial_state.items():
        assert router.reputation[node_id] == state["u_rel"]
        assert router.consistency_trust[node_id] == state["u_cons"]
        assert router.feedback_count[node_id] == 2
        assert len(router.rel_feedback_history[node_id]) == 2


def test_trust_state_round_trip_preserves_converged_routing(tmp_path, router):
    register_similar_nodes(router)
    query = [1.0, 0.0]
    router.update_trust(
        query,
        ["honest", "unreliable"],
        {
            "honest": [[1.0, 0.0]],
            "unreliable": [[0.0, 1.0]],
        },
    )
    saved_state = {
        node_id: router.export_trust_state(node_id)
        for node_id in ("honest", "unreliable")
    }
    database = TASRDatabase(str(tmp_path / "tasr_state.json"))
    database.save_state(saved_state)

    restored_state = database.load_state()
    assert restored_state is not None
    restored_router = TrustAwareRouter(
        warmup_queries=0,
        cold_start_s0=1.0,
        explore_interval=0,
        defense_mode="rel_cons_agr",
        k_route=1,
    )
    for node_id in ("honest", "unreliable"):
        restored_router.register_client(
            client_id=node_id,
            centroid=[1.0, 0.0],
            profile_centroids=[[1.0, 0.0]],
            doc_embeddings=[[1.0, 0.0]],
            initial_trust_state=restored_state[node_id],
        )

    assert restored_router.export_trust_state("honest") == saved_state["honest"]
    assert restored_router.export_trust_state("unreliable") == saved_state[
        "unreliable"
    ]
    assert restored_router.route_query(query, top_k=1)[0] == ["honest"]


def test_unscored_routed_node_is_counted_but_not_penalized(router):
    register_similar_nodes(router)
    router.update_trust(
        ["honest", "unreliable"],
        {"honest": 1.0, "unreliable": 0.0},
        {"honest": 1.0, "unreliable": 0.0},
        {"honest": 1.0, "unreliable": 0.0},
        eligible={"honest"},
    )

    assert router.feedback_count["honest"] == 1
    assert router.feedback_count["unreliable"] == 1
    assert router.reputation["honest"] == pytest.approx(1.0)
    assert router.reputation["unreliable"] == pytest.approx(1.0)
    assert router.rel_feedback_history["unreliable"] == [0.0]
