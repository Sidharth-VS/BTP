#Pseudo-Testing code, 
#Currently not the part of main project
"""Tests for coordinator REST endpoints and their TASR integration."""
from typing import Any
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from coordinator.app.api.v1 import endpoints
from coordinator.app.tasr.router import TrustAwareRouter
from shared.schemas.common import AttestationRequest, QueryRequest

class FakeAdmissionManager:
    def __init__(self) -> None:
        self.fingerprints: dict[str, str] = {}
        self.saved_states: list[tuple[str, dict[str, Any]]] = []
        self.authorized: dict[str, bool] = {}
        self.max_nodes_per_host = 2
        self.attestation_result = (True, "Admission approved", "lease-token")
        self.attest_calls: list[AttestationRequest] = []

    def get_fingerprint_for_node(self, node_id: str) -> str | None:
        return self.fingerprints.get(node_id)

    def get_fingerprint_trust_state(self, fingerprint: str) -> dict[str, Any] | None:
        return None

    def save_fingerprint_trust_state(
        self, fingerprint: str, state: dict[str, Any]
    ) -> None:
        self.saved_states.append((fingerprint, state))

    def is_authorized(self, node_id: str) -> bool:
        return self.authorized.get(node_id, False)

    def attest_and_admit(
        self, request: AttestationRequest
    ) -> tuple[bool, str, str]:
        self.attest_calls.append(request)
        return self.attestation_result

    def get_summary(self) -> dict[str, Any]:
        return {"authorized_nodes": sum(self.authorized.values())}

class FakeBroker:
    def __init__(self, result: dict[str, Any] | None = None, error=None) -> None:
        self.result = result if result is not None else {"nodes": {}}
        self.error = error
        self.calls: list[dict[str, Any]] = []
    def submit_query(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result

class FakeFlowerServer:
    def __init__(
        self,
        registry: dict[str, dict[str, Any]] | None = None,
        broker: FakeBroker | None = None,
        admission_manager: FakeAdmissionManager | None = None,
    ) -> None:
        self.registry = registry if registry is not None else {}
        self.broker = broker if broker is not None else FakeBroker()
        self.admission_manager = (
            admission_manager if admission_manager is not None else FakeAdmissionManager()
        )
    def get_node_registry(self) -> dict[str, dict[str, Any]]:
        return self.registry

class FakeEmbedder:
    dimension = 2
    def __init__(self) -> None:
        self.query_calls: list[str] = []
        self.document_calls: list[list[str]] = []
    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0, 0.0]
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [
            [1.0, 0.0] if "aligned" in text.lower() else [0.0, 1.0]
            for text in texts
        ]
    
class FakeSynthesizer:
    def __init__(self, answer: str = "Generated answer") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []
    def synthesize(self, *, query: str, sources: list[Any]) -> str:
        self.calls.append({"query": query, "sources": sources})
        return self.answer

def make_registry_entry(
    node_id: str,
    *,
    status: str = "healthy",
    authorized: bool = True,
    centroid: list[float] | None = None,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    node_centroid = centroid if centroid is not None else [1.0, 0.0]
    return {
        "node_id": node_id,
        "status": status,
        "domain": "medical",
        "centroid": node_centroid,
        "profile_centroids": [node_centroid],
        "doc_embeddings": [node_centroid],
        "hardware_fingerprint": fingerprint,
        "authorized": authorized,
    }

@pytest.fixture
def api(monkeypatch):
    app = FastAPI()
    app.include_router(endpoints.router, prefix="/api/v1")
    client = TestClient(app)
    embedder = FakeEmbedder()
    synthesizer = FakeSynthesizer()
    router = TrustAwareRouter(
        k_route=1,
        cold_start_s0=1.0,
        warmup_queries=0,
        explore_interval=0,
        defense_mode="rel_cons_agr",
    )
    monkeypatch.setattr(endpoints, "embedder", embedder)
    monkeypatch.setattr(endpoints, "synthesizer", synthesizer)
    monkeypatch.setattr(endpoints, "tasr_router", router)

    def install_server(server: FakeFlowerServer) -> None:
        monkeypatch.setattr(endpoints, "get_flower_server", lambda: server)
    return client, embedder, synthesizer, router, install_server

def test_configure_tasr_applies_supported_options_and_ignores_unknown():
    original_router = endpoints.tasr_router
    router = endpoints.configure_tasr(
        {
            "k_route": 2,
            "warmup_queries": 7,
            "defense_mode": "rel_cons",
            "not_a_router_option": "ignored",
        }
    )
    try:
        assert router.k_route == 2
        assert router.warmup_queries == 7
        assert router.defense_mode == "rel_cons"
        assert router.explore_interval == 20
    finally:
        endpoints.tasr_router = original_router

@pytest.mark.parametrize(
    ("claimed", "recomputed", "expected"),
    [
        ([[1.0, 0.0]], [[1.0, 0.0]], 1.0),
        ([[1.0, 0.0]], [[0.0, 1.0]], 0.0),
        ([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]], None),
        ([], [[1.0, 0.0]], None),
        ([[0.0, 0.0]], [[0.0, 0.0]], None),
    ],
)
def test_embedding_mismatch_handles_matching_and_incomparable_vectors(
    claimed, recomputed, expected
):
    actual = endpoints._embedding_mismatch(claimed, recomputed)

    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)

def test_query_routes_and_returns_ranked_sources_and_updates_feedback(api):
    client, embedder, synthesizer, router, install_server = api
    admission = FakeAdmissionManager()
    admission.fingerprints["node-1"] = "fingerprint-1"
    broker = FakeBroker(
        {
            "nodes": {
                "node-1": {
                    "results": [
                        {
                            "content": "Aligned medical evidence",
                            "score": 0.91,
                            "metadata": {"source": "guideline.md"},
                        }
                    ],
                    # This deliberately conflicts with the returned text;
                    # feedback must be computed from coordinator-side embeddings.
                    "doc_embeddings": [[0.0, 1.0]],
                    "processing_time": 0.02,
                }
            }
        }
    )
    server = FakeFlowerServer(
        registry={"cid-1": make_registry_entry("node-1")},
        broker=broker,
        admission_manager=admission,
    )
    install_server(server)

    response = client.post(
        "/api/v1/query",
        json={
            "query": "medical question",
            "top_k": 2,
            "filters": {"domain": "medical"},
            "use_llm": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] is None
    assert payload["routing_info"]["strategy_used"] == "tasr"
    assert payload["routing_info"]["selected_nodes"] == ["node-1"]
    assert payload["routing_info"]["total_nodes_queried"] == 1
    assert payload["sources"] == [
        {
            "content": "Aligned medical evidence",
            "score": 0.91,
            "metadata": {"source": "guideline.md"},
            "node_id": "node-1",
            "trust_score": pytest.approx(1.0),
            "rank_score": pytest.approx(0.91),
        }
    ]
    assert broker.calls[0]["query"] == "medical question"
    assert broker.calls[0]["top_k"] == 2
    assert broker.calls[0]["filters"] == {"domain": "medical"}
    assert broker.calls[0]["target_nodes"] == ["node-1"]
    assert broker.calls[0]["timeout"] == 120.0
    assert embedder.query_calls == ["medical question"]
    assert embedder.document_calls == [["Aligned medical evidence"]]
    assert router.rel_feedback_history["node-1"] == pytest.approx([1.0])
    assert router.cons_feedback_history["node-1"] == pytest.approx([1.0])
    assert len(admission.saved_states) == 1
    assert admission.saved_states[0][0] == "fingerprint-1"
    assert admission.saved_states[0][1]["feedback_count"] == 1
    assert synthesizer.calls == []


def test_query_synthesizes_only_when_requested(api):
    client, _, synthesizer, _, install_server = api
    node = make_registry_entry("node-1")
    server = FakeFlowerServer(
        registry={"cid-1": node},
        broker=FakeBroker(
            {
                "nodes": {
                    "node-1": {
                        "results": [
                            {
                                "content": "Aligned evidence",
                                "score": 0.8,
                                "metadata": {},
                            }
                        ],
                        "doc_embeddings": [[1.0, 0.0]],
                    }
                }
            }
        ),
    )
    install_server(server)

    response = client.post(
        "/api/v1/query",
        json={"query": "question", "top_k": 1, "use_llm": True},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Generated answer"
    assert len(synthesizer.calls) == 1
    assert synthesizer.calls[0]["query"] == "question"
    assert [source.content for source in synthesizer.calls[0]["sources"]] == [
        "Aligned evidence"
    ]


def test_query_skips_unauthorized_registry_nodes(api):
    client, _, _, router, install_server = api
    server = FakeFlowerServer(
        registry={
            "cid-denied": make_registry_entry("denied-node", authorized=False)
        }
    )
    install_server(server)

    response = client.post(
        "/api/v1/query",
        json={"query": "question", "use_llm": False},
    )

    assert response.status_code == 200
    assert router.centroids == {}
    assert response.json()["routing_info"]["strategy_used"] == "broadcast"
    assert response.json()["routing_info"]["selected_nodes"] == []
    assert server.broker.calls[0]["target_nodes"] == []


def test_query_returns_broadcast_response_when_no_nodes_are_registered(api):
    client, _, _, _, install_server = api
    install_server(FakeFlowerServer(broker=FakeBroker({"nodes": {}})))

    response = client.post(
        "/api/v1/query",
        json={"query": "question", "use_llm": False},
    )

    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert response.json()["sources"] == []
    assert response.json()["routing_info"]["strategy_used"] == "broadcast"
    assert response.json()["routing_info"]["selected_nodes"] == []


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (TimeoutError("query timed out"), 504),
        (RuntimeError("broker unavailable"), 500),
    ],
)
def test_query_maps_broker_exceptions_to_http_errors(api, error, expected_status):
    client, _, _, _, install_server = api
    install_server(FakeFlowerServer(broker=FakeBroker(error=error)))

    response = client.post(
        "/api/v1/query",
        json={"query": "question", "use_llm": False},
    )

    assert response.status_code == expected_status


def test_query_returns_service_unavailable_when_broker_has_no_nodes_and_error(api):
    client, _, _, _, install_server = api
    install_server(
        FakeFlowerServer(
            broker=FakeBroker(
                {"nodes": {}, "error": "no nodes connected"}
            )
        )
    )

    response = client.post(
        "/api/v1/query",
        json={"query": "question", "use_llm": False},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "no nodes connected"


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "question", "top_k": 0},
        {"query": "question", "top_k": 101},
    ],
)
def test_query_rejects_invalid_request_fields(api, body):
    client, _, _, _, _ = api

    response = client.post("/api/v1/query", json=body)

    assert response.status_code == 422


def test_attestation_endpoint_returns_admission_details(api):
    client, _, _, _, install_server = api
    admission = FakeAdmissionManager()
    server = FakeFlowerServer(admission_manager=admission)
    install_server(server)
    body = {
        "node_id": "node-1",
        "fingerprint": "a" * 64,
        "timestamp": 123456.0,
        "nonce": "nonce-1",
        "hardware_summary": {"cpu": "test-cpu"},
    }

    response = client.post("/api/v1/nodes/attest", json=body)

    assert response.status_code == 200
    assert response.json() == {
        "admitted": True,
        "lease_token": "lease-token",
        "fingerprint": "a" * 64,
        "message": "Admission approved",
        "max_nodes_per_host": 2,
    }
    assert len(admission.attest_calls) == 1
    assert admission.attest_calls[0].node_id == "node-1"


def test_attestation_endpoint_rejects_unadmitted_node(api):
    client, _, _, _, install_server = api
    admission = FakeAdmissionManager()
    admission.attestation_result = (False, "Host quota exceeded", "")
    install_server(FakeFlowerServer(admission_manager=admission))

    response = client.post(
        "/api/v1/nodes/attest",
        json={
            "node_id": "node-2",
            "fingerprint": "b" * 64,
            "timestamp": 123456.0,
            "nonce": "nonce-2",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Host quota exceeded"


def test_attestation_summary_endpoint_returns_manager_summary(api):
    client, _, _, _, install_server = api
    admission = FakeAdmissionManager()
    admission.authorized = {"node-a": True, "node-b": False}
    install_server(FakeFlowerServer(admission_manager=admission))

    response = client.get("/api/v1/attestation/summary")

    assert response.status_code == 200
    assert response.json() == {"authorized_nodes": 1}


def test_nodes_endpoint_maps_registry_and_tasr_trust_to_response(api):
    client, _, _, router, install_server = api
    admission = FakeAdmissionManager()
    registry = {
        "cid-healthy": make_registry_entry(
            "healthy-node",
            status="healthy",
            fingerprint="fingerprint-healthy",
        ),
        "cid-degraded": make_registry_entry(
            "degraded-node",
            status="starting",
            authorized=False,
        ),
        "cid-rejected": make_registry_entry(
            "rejected-node",
            status="rejected_sybil",
            authorized=False,
        ),
    }
    router.register_client("healthy-node", centroid=[1.0, 0.0])
    router.reputation["healthy-node"] = 0.75
    router.consistency_trust["healthy-node"] = 0.8
    router.agreement_trust["healthy-node"] = 0.9
    router.feedback_count["healthy-node"] = 4
    admission.fingerprints["healthy-node"] = "fallback-fingerprint"
    install_server(
        FakeFlowerServer(registry=registry, admission_manager=admission)
    )

    response = client.get("/api/v1/nodes")

    assert response.status_code == 200
    nodes = {node["node_id"]: node for node in response.json()["nodes"]}
    assert set(nodes) == {"healthy-node", "degraded-node", "rejected-node"}
    assert nodes["healthy-node"]["address"] == "cid-healthy"
    assert nodes["healthy-node"]["status"] == "healthy"
    assert nodes["healthy-node"]["trust_score"] == pytest.approx(
        router.compute_trust_weight("healthy-node")
    )
    assert nodes["healthy-node"]["u_rel"] == pytest.approx(0.75)
    assert nodes["healthy-node"]["u_cons"] == pytest.approx(0.8)
    assert nodes["healthy-node"]["u_agr"] == pytest.approx(0.9)
    assert nodes["healthy-node"]["feedback_count"] == 4
    assert nodes["healthy-node"]["hardware_fingerprint"] == "fingerprint-healthy"
    assert nodes["healthy-node"]["attestation_status"] == "verified"
    assert nodes["degraded-node"]["status"] == "degraded"
    assert nodes["degraded-node"]["attestation_status"] == "rejected_sybil"
    assert nodes["rejected-node"]["status"] == "rejected_sybil"


def test_tasr_summary_exposes_router_configuration_and_state(api):
    client, _, _, router, _ = api
    router.register_client("node-a", centroid=[1.0, 0.0])
    router.query_count = 3
    router.feedback_count["node-a"] = 2
    router.reputation["node-a"] = 0.6
    router.reputation_history["node-a"] = [1.0, 0.8, 0.6]

    response = client.get("/api/v1/tasr/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["defense_mode"] == "rel_cons_agr"
    assert payload["total_queries"] == 3
    assert payload["is_warmup_active"] is False
    assert payload["exploration_interval"] == 0
    assert payload["nodes"]["node-a"]["u_rel"] == pytest.approx(0.6)
    assert payload["nodes"]["node-a"]["feedback_count"] == 2
    assert payload["nodes"]["node-a"]["reputation_history"] == [1.0, 0.8, 0.6]


def test_health_endpoint_counts_registered_and_healthy_nodes(api):
    client, _, _, _, install_server = api
    install_server(
        FakeFlowerServer(
            registry={
                "one": {"status": "healthy"},
                "two": {"status": "degraded"},
                "three": {"status": "healthy"},
            }
        )
    )

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "registered_nodes": 3,
        "healthy_nodes": 2,
    }
