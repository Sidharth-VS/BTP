import os
import shutil
import tempfile
import time
import pytest
from coordinator.app.flwr_server.admission import NodeAdmissionManager
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
