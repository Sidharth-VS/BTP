"""Tests for registration-latency telemetry in the gRPC handshake."""

import json
import logging
import time

from coordinator.app.flwr_server.admission import NodeAdmissionManager
from coordinator.app.flwr_server.client_manager import LoggingClientManager

FINGERPRINT = "a" * 64


class FakeProxy:
    def __init__(self, cid: str, props: dict) -> None:
        self.cid = cid
        self._props = props

    def get_properties(self, ins, timeout=None, group_id=0):
        return type("Response", (), {"properties": self._props})()


class FakeStrategy:
    def __init__(self, admission_manager: NodeAdmissionManager) -> None:
        self.admission_manager = admission_manager
        self._node_registry: dict = {}


def make_props(node_id: str, started_at: float) -> dict:
    return {
        "node_id": node_id,
        "domain": "networking",
        "fingerprint": FINGERPRINT,
        "node_started_at": started_at,
        "attestation": json.dumps(
            {
                "node_id": node_id,
                "fingerprint": FINGERPRINT,
                "timestamp": time.time(),
                "nonce": "nonce-1",
                "hardware_summary": {"cpu": "test-cpu"},
                "node_started_at": started_at,
            }
        ),
        "centroid": json.dumps([0.1] * 384),
    }


def test_grpc_registration_latency_logged_once_per_node_boot(tmp_path, caplog):
    """First appearance in the live registry is timed; a reconnect is not
    treated as a new registration (which would inflate the sample)."""
    manager = NodeAdmissionManager(
        enabled=True,
        max_nodes_per_host=2,
        storage_path=str(tmp_path / "attestation_registry.json"),
    )
    strategy = FakeStrategy(manager)
    client_manager = LoggingClientManager(strategy=strategy)

    started_at = time.time() - 4.0
    client = FakeProxy("cid-1", make_props("node-grpc", started_at))
    client_manager.clients[client.cid] = client

    with caplog.at_level(logging.INFO, logger="coordinator.admission"):
        client_manager._handshake_and_authenticate(client)
        # Reconnect: unregister drops the live registry entry, so the CID looks
        # new again unless the registration sample is deduped per node boot.
        strategy._node_registry.pop(client.cid, None)
        client_manager._handshake_and_authenticate(client)

    assert strategy._node_registry["cid-1"]["node_id"] == "node-grpc"

    lines = [
        record.getMessage()
        for record in caplog.records
        if "Registration latency" in record.getMessage()
    ]
    grpc_lines = [line for line in lines if "registry=grpc_node_registry" in line]
    assert len(grpc_lines) == 1
    assert "node='node-grpc'" in grpc_lines[0]

    registration = manager.get_summary()["latency"]["registration"]
    assert registration["grpc_node_registry"]["count"] == 1
    assert registration["grpc_node_registry"]["mean_ms"] >= 4000.0
    assert registration["first_admission"]["count"] == 1
