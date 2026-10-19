#Pseudo-Testing code, 
#Currently not the part of main project


"""In-process integration tests for Flower query broadcast and aggregation."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from flwr.compat.common.typing import (
    Code,
    FitIns,
    FitRes,
    Parameters,
    Status,
)

from coordinator.app.flwr_server.strategy import FedRAGStrategy, QueryBroker


class FakeClientManager:
    def __init__(self, clients: dict[str, Any]) -> None:
        self._clients = clients

    def all(self) -> dict[str, Any]:
        return self._clients


class FakeRetrievalNode:
    """A local Flower-node stand-in that returns realistic retrieval data."""

    def __init__(
        self,
        cid: str,
        node_id: str,
        content: str,
        score: float,
        embedding: list[float],
    ) -> None:
        self.cid = cid
        self.node_id = node_id
        self.content = content
        self.score = score
        self.embedding = embedding
        self.fit_configs: list[dict[str, Any]] = []

    def fit(self, instruction: FitIns) -> FitRes:
        config = instruction.config
        self.fit_configs.append(config)
        assert config["action"] == "query"
        result = {
            "content": self.content,
            "score": self.score,
            "metadata": {"source": f"{self.node_id}.md", "domain": "medical"},
        }
        embedding_array = np.asarray([self.embedding], dtype=np.float32)
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=Parameters(
                tensors=[embedding_array.tobytes()],
                tensor_type="numpy.ndarray",
            ),
            num_examples=1,
            metrics={
                "node_id": self.node_id,
                "processing_time": 0.025,
                "results_json": json.dumps([result]),
            },
        )


def run_broker_query_through_strategy(
    broker: QueryBroker,
    strategy: FedRAGStrategy,
    client_manager: FakeClientManager,
    *,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None = None,
    target_nodes: list[str] | None = None,
) -> tuple[dict, list[tuple[Any, FitIns]]]:
    """Submit through the broker and drive one Flower fit round synchronously."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending_result = executor.submit(
            broker.submit_query,
            query=query,
            top_k=top_k,
            filters=filters,
            target_nodes=target_nodes,
            timeout=3.0,
        )
        dispatched = strategy.configure_fit(
            server_round=1,
            parameters=Parameters(tensors=[], tensor_type="numpy.ndarray"),
            client_manager=client_manager,
        )

        fit_results = [
            (node, node.fit(instruction))
            for node, instruction in dispatched
        ]
        strategy.aggregate_fit(server_round=1, results=fit_results, failures=[])
        return pending_result.result(timeout=3.0), dispatched


def make_node(cid: str, node_id: str, content: str, score: float) -> FakeRetrievalNode:
    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0 if cid == "client-a" else 0.5
    return FakeRetrievalNode(
        cid=cid,
        node_id=node_id,
        content=content,
        score=score,
        embedding=embedding.tolist(),
    )


def test_query_is_broadcast_and_aggregated_from_every_connected_node():
    broker = QueryBroker()
    strategy = FedRAGStrategy(broker=broker)
    node_a = make_node(
        "client-a",
        "node-medical-a",
        "Evidence from medical node A",
        0.93,
    )
    node_b = make_node(
        "client-b",
        "node-medical-b",
        "Evidence from medical node B",
        0.87,
    )
    client_manager = FakeClientManager(
        {node_a.cid: node_a, node_b.cid: node_b}
    )
    strategy._node_registry = {
        node_a.cid: {"node_id": node_a.node_id, "authorized": True},
        node_b.cid: {"node_id": node_b.node_id, "authorized": True},
    }

    response, dispatched = run_broker_query_through_strategy(
        broker,
        strategy,
        client_manager,
        query="What are the symptoms?",
        top_k=4,
        filters={"domain": "medical"},
    )

    assert {node.cid for node, _ in dispatched} == {"client-a", "client-b"}
    assert len(node_a.fit_configs) == len(node_b.fit_configs) == 1
    for node in (node_a, node_b):
        config = node.fit_configs[0]
        assert config["query"] == "What are the symptoms?"
        assert config["top_k"] == 4
        assert json.loads(config["filters"]) == {"domain": "medical"}
        assert config["action"] == "query"
        assert config["query_id"]

    assert set(response) == {"nodes"}
    assert set(response["nodes"]) == {"node-medical-a", "node-medical-b"}
    assert response["nodes"]["node-medical-a"]["results"] == [
        {
            "content": "Evidence from medical node A",
            "score": 0.93,
            "metadata": {
                "source": "node-medical-a.md",
                "domain": "medical",
            },
        }
    ]
    assert response["nodes"]["node-medical-b"]["results"][0]["content"] == (
        "Evidence from medical node B"
    )
    for node_id in ("node-medical-a", "node-medical-b"):
        node_result = response["nodes"][node_id]
        assert node_result["processing_time"] == pytest.approx(0.025)
        assert len(node_result["doc_embeddings"]) == 1
        assert len(node_result["doc_embeddings"][0]) == 384
    assert response["nodes"]["node-medical-a"]["doc_embeddings"][0][0] == pytest.approx(1.0)
    assert response["nodes"]["node-medical-b"]["doc_embeddings"][0][0] == pytest.approx(0.5)
    assert strategy._current_query is None
    assert broker.get_pending(timeout=0.01) is None


def test_targeted_query_dispatches_only_to_matching_node():
    broker = QueryBroker()
    strategy = FedRAGStrategy(broker=broker)
    node_a = make_node("client-a", "node-a", "A result", 0.9)
    node_b = make_node("client-b", "node-b", "B result", 0.8)
    client_manager = FakeClientManager(
        {node_a.cid: node_a, node_b.cid: node_b}
    )
    strategy._node_registry = {
        node_a.cid: {"node_id": node_a.node_id, "authorized": True},
        node_b.cid: {"node_id": node_b.node_id, "authorized": True},
    }

    response, dispatched = run_broker_query_through_strategy(
        broker,
        strategy,
        client_manager,
        query="targeted question",
        top_k=2,
        target_nodes=["node-b"],
    )

    assert [node.cid for node, _ in dispatched] == ["client-b"]
    assert len(node_a.fit_configs) == 0
    assert len(node_b.fit_configs) == 1
    assert set(response["nodes"]) == {"node-b"}
    assert response["nodes"]["node-b"]["results"][0]["content"] == "B result"


def test_query_with_no_connected_nodes_returns_explicit_error():
    broker = QueryBroker()
    strategy = FedRAGStrategy(broker=broker)
    client_manager = FakeClientManager({})

    response, dispatched = run_broker_query_through_strategy(
        broker,
        strategy,
        client_manager,
        query="unavailable query",
        top_k=1,
    )

    assert dispatched == []
    assert response == {"nodes": {}, "error": "no nodes connected"}
    assert strategy._current_query is None


def test_aggregate_fit_skips_invalid_json_and_invalid_embedding_shapes():
    broker = QueryBroker()
    strategy = FedRAGStrategy(broker=broker)
    proxy = SimpleNamespace(cid="client-invalid")
    query_id = "manual-query"
    strategy._current_query = {"query_id": query_id}
    invalid_fit = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=Parameters(
            tensors=[np.asarray([1.0, 2.0], dtype=np.float32).tobytes()],
            tensor_type="numpy.ndarray",
        ),
        num_examples=1,
        metrics={
            "node_id": "node-invalid",
            "results_json": "{malformed",
            "processing_time": 0.1,
        },
    )
    published_results: list[tuple[str, dict]] = []
    broker.publish_result = lambda found_query_id, result: published_results.append(
        (found_query_id, result)
    )

    strategy.aggregate_fit(
        server_round=1,
        results=[(proxy, invalid_fit)],
        failures=[],
    )

    assert published_results == [
        (
            query_id,
            {
                "nodes": {
                    "node-invalid": {
                        "results": [],
                        "doc_embeddings": [],
                        "processing_time": pytest.approx(0.1),
                    }
                }
            },
        )
    ]
    assert strategy._current_query is None
