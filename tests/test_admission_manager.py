import json
import logging
import os
import re
import shutil
import statistics
import tempfile
import time
import pytest
from coordinator.app.flwr_server.admission import _LatencyStats, NodeAdmissionManager
from shared.schemas.common import AttestationRequest
from coordinator.app.tasr.router import TrustAwareRouter


@pytest.fixture
def temp_storage():
    tmp_dir = tempfile.mkdtemp()
    storage_file = os.path.join(tmp_dir, "attestation_registry.json")
    yield storage_file
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_admission_and_host_quota_enforcement(temp_storage):
    manager = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=1,
        storage_path=temp_storage,
    )

    req1 = AttestationRequest(
        node_id="node-1",
        fingerprint="a" * 64,
        timestamp=time.time(),
        nonce="nonce-1",
        hardware_summary={"cpu": "core-i7"},
    )

    # 1. First node on host 'aaaa...' is admitted
    admitted1, msg1, lease1 = manager.attest_and_admit(req1)
    assert admitted1 is True
    assert lease1 != ""
    assert manager.is_authorized("node-1") is True

    # 2. Second node on the SAME host ('aaaa...') is REJECTED (Sybil block)
    req2 = AttestationRequest(
        node_id="node-2",
        fingerprint="a" * 64,
        timestamp=time.time(),
        nonce="nonce-2",
        hardware_summary={"cpu": "core-i7"},
    )
    admitted2, msg2, lease2 = manager.attest_and_admit(req2)
    assert admitted2 is False
    assert "Host quota exceeded" in msg2
    assert lease2 == ""
    assert manager.is_authorized("node-2") is False

    # 3. Third node on a DIFFERENT host ('bbbb...') is admitted
    req3 = AttestationRequest(
        node_id="node-3",
        fingerprint="b" * 64,
        timestamp=time.time(),
        nonce="nonce-3",
        hardware_summary={"cpu": "ryzen-9"},
    )
    admitted3, msg3, lease3 = manager.attest_and_admit(req3)
    assert admitted3 is True
    assert manager.is_authorized("node-3") is True


def test_attestation_latency_is_logged_per_node(temp_storage, caplog):
    """Every admission attempt must emit one parseable latency line, including
    rejections, so the reported distribution is not biased to successes."""
    manager = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=1,
        storage_path=temp_storage,
    )
    req = AttestationRequest(
        node_id="node-latency",
        fingerprint="e" * 64,
        timestamp=time.time(),
        nonce="nonce-latency",
        hardware_summary={"cpu": "core-i7"},
    )

    with caplog.at_level(logging.INFO, logger="coordinator.admission"):
        admitted, _, _ = manager.attest_and_admit(req, channel="rest-preflight")
        # Same host, second node -> quota rejection, which must be measured too.
        rejected, _, _ = manager.attest_and_admit(
            req.model_copy(update={"node_id": "node-latency-2", "nonce": "nonce-2"}),
            channel="rest-preflight",
        )

    assert admitted is True
    assert rejected is False

    lines = [
        record.getMessage()
        for record in caplog.records
        if "Attestation latency" in record.getMessage()
    ]
    assert len(lines) == 2

    assert "channel=rest-preflight" in lines[0]
    assert "node='node-latency'" in lines[0]
    assert "outcome=admitted:new" in lines[0]
    assert "verify=" in lines[0] and "total=" in lines[0]
    assert "outcome=rejected:host_quota" in lines[1]


def test_latency_stats_moments():
    """Welford accumulator must produce the sample mean and std (n-1)."""
    stats = _LatencyStats()
    for value in (10.0, 20.0, 30.0):
        stats.add(value)

    assert stats.count == 3
    assert stats.mean_ms == pytest.approx(20.0)
    assert stats.std_ms == pytest.approx(10.0)
    assert stats.min_ms == 10.0
    assert stats.max_ms == 30.0

    empty = _LatencyStats().as_dict()
    assert empty == {
        "count": 0,
        "mean_ms": 0.0,
        "std_ms": 0.0,
        "min_ms": None,
        "max_ms": None,
    }


def test_admission_latency_summary_matches_logged_samples(temp_storage, caplog):
    """Summary mean/std must agree with the per-attempt latency lines."""
    manager = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=1,
        storage_path=temp_storage,
    )
    req = AttestationRequest(
        node_id="node-1",
        fingerprint="a" * 64,
        timestamp=time.time(),
        nonce="nonce-1",
    )

    with caplog.at_level(logging.INFO, logger="coordinator.admission"):
        for _ in range(4):  # first admission, then refreshes
            admitted, _, _ = manager.attest_and_admit(req)
            assert admitted is True

    lines = [
        record.getMessage()
        for record in caplog.records
        if "Attestation latency" in record.getMessage()
    ]
    assert len(lines) == 4
    samples = [
        float(re.search(r"total=([\d.]+) ms", line).group(1)) for line in lines
    ]

    summary = manager.get_summary()["latency"]["admission_handshake"]
    assert summary["count"] == 4
    assert summary["mean_ms"] == pytest.approx(statistics.fmean(samples), abs=2e-3)
    assert summary["std_ms"] == pytest.approx(statistics.stdev(samples), abs=2e-3)
    assert summary["by_channel"]["internal"]["count"] == 4


def test_registration_latency_logged_on_new_node_boot(temp_storage, caplog):
    """Registration latency is sampled on first registry appearance and on a
    new node boot, but not on a refresh."""
    manager = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=1,
        storage_path=temp_storage,
    )
    started_at = time.time() - 3.0
    req = AttestationRequest(
        node_id="node-reg",
        fingerprint="b" * 64,
        timestamp=time.time(),
        nonce="nonce-reg",
        node_started_at=started_at,
    )

    with caplog.at_level(logging.INFO, logger="coordinator.admission"):
        admitted, _, _ = manager.attest_and_admit(req, channel="rest-preflight")
        # Refresh of the same boot: must not add a second sample.
        refreshed, _, _ = manager.attest_and_admit(req, channel="grpc-handshake")

    assert admitted is True
    assert refreshed is True

    lines = [
        record.getMessage()
        for record in caplog.records
        if "Registration latency" in record.getMessage()
    ]
    assert len(lines) == 1
    assert "node='node-reg'" in lines[0]
    assert "registry=first_admission" in lines[0]
    assert "total=" in lines[0] and "mean=" in lines[0] and "std=" in lines[0]

    stats = manager.get_summary()["latency"]["registration"]["first_admission"]
    assert stats["count"] == 1
    assert stats["mean_ms"] >= 3000.0

    with open(temp_storage, "r", encoding="utf-8") as f:
        registry = json.load(f)
    entry = registry["node_registry"]["node-reg"]
    assert entry["registration_latency_ms"] >= 3000.0
    assert entry["node_started_at"] == started_at
    assert "first_seen_at" in entry

    # The same boot re-registering must not be sampled again.
    manager.record_registration_latency("node-reg", "grpc_node_registry", started_at)
    manager.record_registration_latency("node-reg", "grpc_node_registry", started_at)
    gRPC_stats = manager.get_summary()["latency"]["registration"]["grpc_node_registry"]
    assert gRPC_stats["count"] == 1
    assert gRPC_stats["mean_ms"] >= 3000.0

    # A new node boot (newer start time) is a new registration sample.
    manager.release_node(node_id="node-reg")
    rebooted = req.model_copy(update={"node_started_at": time.time() - 1.0})
    admitted_again, _, _ = manager.attest_and_admit(rebooted, channel="grpc-handshake")
    assert admitted_again is True
    assert (
        manager.get_summary()["latency"]["registration"]["first_admission"]["count"]
        == 2
    )


def test_latency_metrics_persisted_to_file(temp_storage):
    """The persisted snapshot must mirror the summary values for both metrics."""
    metrics_path = os.path.join(os.path.dirname(temp_storage), "latency_metrics.json")
    manager = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=2,
        storage_path=temp_storage,
        latency_storage_path=metrics_path,
    )

    # The file exists from startup, so a report can always read it.
    with open(metrics_path, "r", encoding="utf-8") as f:
        initial = json.load(f)
    assert initial["admission_handshake"]["count"] == 0
    assert initial["registration"] == {}
    assert "updated_at" in initial

    started_at = time.time() - 2.0
    req = AttestationRequest(
        node_id="node-file",
        fingerprint="d" * 64,
        timestamp=time.time(),
        nonce="nonce-file",
        node_started_at=started_at,
    )
    admitted, _, _ = manager.attest_and_admit(req, channel="rest-preflight")
    assert admitted is True
    manager.record_registration_latency("node-file", "grpc_node_registry", started_at)

    with open(metrics_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    summary = manager.get_summary()["latency"]
    assert saved["admission_handshake"] == summary["admission_handshake"]
    assert saved["registration"] == summary["registration"]
    assert saved["admission_handshake"]["by_channel"]["rest-preflight"]["count"] == 1
    assert saved["registration"]["first_admission"]["count"] == 1
    assert saved["registration"]["first_admission"]["mean_ms"] >= 2000.0
    assert saved["registration"]["grpc_node_registry"]["count"] == 1
    assert not os.path.exists(metrics_path + ".tmp")


def test_registration_latency_without_node_start_time(temp_storage, caplog):
    """A node that reports no start time is logged as n/a and never sampled."""
    manager = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=1,
        storage_path=temp_storage,
    )
    req = AttestationRequest(
        node_id="node-no-start",
        fingerprint="c" * 64,
        timestamp=time.time(),
        nonce="nonce-no-start",
    )

    with caplog.at_level(logging.INFO, logger="coordinator.admission"):
        admitted, _, _ = manager.attest_and_admit(req, channel="rest-preflight")

    assert admitted is True
    lines = [
        record.getMessage()
        for record in caplog.records
        if "Registration latency" in record.getMessage()
    ]
    assert len(lines) == 1
    assert "total=n/a" in lines[0]
    assert manager.get_summary()["latency"]["registration"] == {}


def test_node_reconnection_and_disconnect(temp_storage):
    manager = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=1,
        storage_path=temp_storage,
    )

    req1 = AttestationRequest(
        node_id="node-1",
        fingerprint="a" * 64,
        timestamp=time.time(),
        nonce="nonce-1",
    )
    admitted1, _, _ = manager.attest_and_admit(req1)
    assert admitted1 is True

    # Node refreshes / reconnects with same ID
    admitted_reconnect, msg_rec, _ = manager.attest_and_admit(req1)
    assert admitted_reconnect is True
    assert "Admission refreshed" in msg_rec

    # Node disconnects
    released_id = manager.release_node(node_id="node-1")
    assert released_id == "node-1"
    assert manager.is_authorized("node-1") is False

    # After disconnect, slot is freed: node-2 on same host can now connect
    req2 = AttestationRequest(
        node_id="node-2",
        fingerprint="a" * 64,
        timestamp=time.time(),
        nonce="nonce-2",
    )
    admitted2, _, _ = manager.attest_and_admit(req2)
    assert admitted2 is True


def test_hardware_anchored_trust_persistence(temp_storage):
    """
    Verifies that trust weights are anchored to the hardware fingerprint
    and NOT reset when nodes disconnect, exit, and rejoin.
    """
    manager = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=1,
        storage_path=temp_storage,
    )
    fp = "c" * 64

    # 1. Initial connection: Node 1 joins
    req1 = AttestationRequest(
        node_id="node-1",
        fingerprint=fp,
        timestamp=time.time(),
        nonce="nonce-1",
    )
    manager.attest_and_admit(req1)

    # Router registers node-1
    router = TrustAwareRouter()
    initial_trust = manager.get_fingerprint_trust_state(fp)
    assert initial_trust is None  # First time node

    router.register_client(
        client_id="node-1",
        centroid=[0.1] * 384,
        initial_trust_state=initial_trust,
    )
    assert router.reputation["node-1"] == 1.0

    # 2. Simulate trust degradation (e.g. penalized down to 0.42 after 15 queries)
    router.reputation["node-1"] = 0.42
    router.consistency_trust["node-1"] = 0.55
    router.agreement_trust["node-1"] = 0.38
    router.feedback_count["node-1"] = 15

    # Persist updated trust to manager under the hardware fingerprint
    saved_state = router.export_trust_state("node-1")
    manager.save_fingerprint_trust_state(fp, saved_state)

    # 3. Simulate node exit / disconnection
    manager.release_node(node_id="node-1")

    # 4. Node rejoins: Coordinator checks trust state by fingerprint
    rejoined_trust = manager.get_fingerprint_trust_state(fp)
    assert rejoined_trust is not None
    assert rejoined_trust["u_rel"] == 0.42
    assert rejoined_trust["feedback_count"] == 15

    # Router registers rejoining node — trust must NOT be reset!
    router2 = TrustAwareRouter()
    router2.register_client(
        client_id="node-1",
        centroid=[0.1] * 384,
        initial_trust_state=rejoined_trust,
    )
    assert router2.reputation["node-1"] == 0.42
    assert router2.consistency_trust["node-1"] == 0.55
    assert router2.feedback_count["node-1"] == 15

    # 5. Anti-whitewashing: Attacker changes node_id to 'node-attacker' on same machine
    # Must inherit the machine's degraded trust!
    router3 = TrustAwareRouter()
    whitewash_trust = manager.get_fingerprint_trust_state(fp)
    router3.register_client(
        client_id="node-attacker",
        centroid=[0.1] * 384,
        initial_trust_state=whitewash_trust,
    )
    assert router3.reputation["node-attacker"] == 0.42
    assert router3.feedback_count["node-attacker"] == 15


def test_reentry_preserves_in_memory_trust_without_snapshot():
    """
    A re-registration that cannot find a persisted snapshot must NOT wipe the
    live in-memory trust for a node that is already known to the router.
    """
    router = TrustAwareRouter()
    router.register_client(client_id="node-1", centroid=[0.1] * 384)
    assert router.reputation["node-1"] == 1.0

    # Trust degrades while the node is connected.
    router.reputation["node-1"] = 0.37
    router.consistency_trust["node-1"] = 0.6
    router.agreement_trust["node-1"] = 0.44
    router.feedback_count["node-1"] = 22

    # Node exits and re-enters; no persisted snapshot is available (None).
    router.register_client(
        client_id="node-1",
        centroid=[0.1] * 384,
        initial_trust_state=None,
    )

    # The weight it left with must survive the re-registration.
    assert router.reputation["node-1"] == 0.37
    assert router.consistency_trust["node-1"] == 0.6
    assert router.agreement_trust["node-1"] == 0.44
    assert router.feedback_count["node-1"] == 22

    # A genuinely new node still cold-starts at 1.0.
    router.register_client(client_id="node-new", centroid=[0.1] * 384)
    assert router.reputation["node-new"] == 1.0


def test_register_node_alias_restores_trust():
    """The register_node() alias must forward a supplied initial trust state."""
    router = TrustAwareRouter()
    router.register_node(
        node_id="node-1",
        centroid=[0.1] * 384,
        initial_trust_state={"u_rel": 0.31, "u_cons": 0.5, "u_agr": 0.4, "feedback_count": 9},
    )
    assert router.reputation["node-1"] == 0.31
    assert router.consistency_trust["node-1"] == 0.5
    assert router.feedback_count["node-1"] == 9


def test_coordinator_reboot_recovers_state(temp_storage):
    """Verifies that reloading NodeAdmissionManager from disk preserves state."""
    manager1 = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=1,
        storage_path=temp_storage,
    )
    fp = "d" * 64
    req = AttestationRequest(
        node_id="node-persist",
        fingerprint=fp,
        timestamp=time.time(),
        nonce="nonce-1",
        hardware_summary={"cpu": "intel"},
    )
    manager1.attest_and_admit(req)
    manager1.save_fingerprint_trust_state(fp, {"u_rel": 0.88, "feedback_count": 30})

    # Simulate coordinator process restart
    manager2 = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=1,
        storage_path=temp_storage,
    )
    trust = manager2.get_fingerprint_trust_state(fp)
    assert trust is not None
    assert trust["u_rel"] == 0.88
    assert trust["feedback_count"] == 30
    summary = manager2.get_summary()
    assert summary["total_known_fingerprints"] >= 1
