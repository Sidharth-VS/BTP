"""
FedRAG Flower Strategy.

Bridges the FastAPI REST layer with Flower's gRPC round-based loop.
- configure_fit(): Picks up pending queries, applies targeted routing with cold-start protection.
- aggregate_fit(): Merges retrieved chunks and doc embeddings from nodes.
- configure_evaluate() & aggregate_evaluate(): Periodic node health-checks.
"""
import json
import logging
import queue
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.compat.common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

logger = logging.getLogger("coordinator.strategy")

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class QueryBroker:
    """
    Thread-safe bridge between the FastAPI request handler and the Flower Strategy.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._results: Dict[str, dict] = {}
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def submit_query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        target_nodes: Optional[List[str]] = None,
        timeout: float = 120.0,
    ) -> dict:
        query_id = uuid.uuid4().hex
        event = threading.Event()

        with self._lock:
            self._events[query_id] = event

        payload = {
            "query_id": query_id,
            "query": query,
            "top_k": top_k,
            "filters": filters or {},
            "target_nodes": target_nodes,
        }

        try:
            self._queue.put(payload, timeout=timeout)
        except queue.Full:
            with self._lock:
                del self._events[query_id]
            raise TimeoutError("Coordinator query queue is full — another query is in progress")

        if not event.wait(timeout=timeout):
            with self._lock:
                self._events.pop(query_id, None)
                self._results.pop(query_id, None)
            raise TimeoutError(f"Query '{query_id}' timed out after {timeout}s")

        with self._lock:
            result = self._results.pop(query_id)
            self._events.pop(query_id, None)

        return result

    def get_pending(self, timeout: float = 0.5) -> Optional[dict]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def publish_result(self, query_id: str, result: dict) -> None:
        with self._lock:
            self._results[query_id] = result
            event = self._events.get(query_id)
        if event:
            event.set()
        else:
            logger.warning("publish_result: no waiter for query_id '%s'", query_id)


class FedRAGStrategy(Strategy):
    """
    Custom Flower Strategy supporting targeted TASR routing and health checking.
    """

    def __init__(self, broker: QueryBroker) -> None:
        self.broker = broker
        self._current_query: Optional[dict] = None
        self._node_registry: Dict[str, dict] = {}

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return Parameters(tensors=[], tensor_type="numpy.ndarray")

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        pending = self.broker.get_pending(timeout=0.5)
        if pending is None:
            return []

        clients = client_manager.all()
        if not clients:
            logger.warning("Query received but no nodes are connected")
            self.broker.publish_result(
                pending["query_id"],
                {"nodes": {}, "error": "no nodes connected"},
            )
            return []

        self._current_query = pending
        target_nodes = pending.get("target_nodes")

        fit_config: Dict[str, Scalar] = {
            "action": "query",
            "query": pending["query"],
            "top_k": pending["top_k"],
            "filters": json.dumps(pending["filters"]),
            "query_id": pending["query_id"],
        }
        fit_ins = FitIns(parameters=parameters, config=fit_config)

        # Targeted dispatch logic with cold-start protection
        dispatched: List[Tuple[ClientProxy, FitIns]] = []
        for cid, proxy in clients.items():
            node_info = self._node_registry.get(cid)

            # Cold-start fallback: include node if not yet registered via health checks
            if not node_info:
                dispatched.append((proxy, fit_ins))
                continue

            node_id = node_info.get("node_id", cid)
            if not target_nodes or node_id in target_nodes:
                dispatched.append((proxy, fit_ins))

        # Fallback to all connected clients if targeting filter produced no matches
        if target_nodes and not dispatched:
            logger.warning(
                "Target nodes %s not found in registry; broadcasting to all %d connected clients",
                target_nodes, len(clients),
            )
            dispatched = [(proxy, fit_ins) for proxy in clients.values()]

        logger.info(
            "[Round %d] Dispatching query '%s' to %d node(s)",
            server_round, pending["query"][:50], len(dispatched),
        )
        return dispatched

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if self._current_query is None:
            return None, {}

        node_results: Dict[str, dict] = {}

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            node_id = str(metrics.get("node_id", client_proxy.cid))
            results_json = str(metrics.get("results_json", "[]"))
            processing_time = float(metrics.get("processing_time", 0.0))

            doc_embeddings: List[List[float]] = []
            if fit_res.parameters and fit_res.parameters.tensors:
                raw = fit_res.parameters.tensors[0]
                try:
                    arr = np.frombuffer(raw, dtype=np.float32)
                    if arr.size > 0 and arr.size % _EMBEDDING_DIM == 0:
                        doc_embeddings = arr.reshape(-1, _EMBEDDING_DIM).tolist()
                except Exception as e:
                    logger.debug("Could not deserialize embeddings from node '%s': %s", node_id, e)

            try:
                search_results = json.loads(results_json)
            except json.JSONDecodeError:
                search_results = []

            node_results[node_id] = {
                "results": search_results,
                "doc_embeddings": doc_embeddings,
                "processing_time": processing_time,
            }

        for failure in failures:
            logger.warning("Node failure during fit: %s", failure)

        self.broker.publish_result(
            self._current_query["query_id"],
            {"nodes": node_results},
        )
        self._current_query = None
        return None, {}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Run periodic health-checks and log cluster status every 20 rounds."""
        clients = client_manager.all()

        # Log connected node count & list every 20 rounds
        if server_round > 0 and server_round % 20 == 0:
            node_ids = [
                self._node_registry.get(cid, {}).get("node_id", cid)
                for cid in clients.keys()
            ]
            logger.info(
                "📊 Cluster Status [Round %d]: %d node(s) connected %s",
                server_round,
                len(clients),
                node_ids if node_ids else "[]",
            )

        has_unregistered = any(cid not in self._node_registry for cid in clients.keys())
        if server_round % 10 != 1 and not has_unregistered:
            return []
        eval_ins = EvaluateIns(parameters=parameters, config={"action": "health"})
        return [(proxy, eval_ins) for proxy in clients.values()]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        for client_proxy, eval_res in results:
            metrics = eval_res.metrics or {}
            node_id = str(metrics.get("node_id", client_proxy.cid))
            status = str(metrics.get("status", "unknown"))
            is_new = client_proxy.cid not in self._node_registry

            centroid_str = str(metrics.get("centroid", ""))
            profile_str = str(metrics.get("profile_centroids", ""))
            doc_emb_str = str(metrics.get("doc_embeddings", ""))

            centroid = json.loads(centroid_str) if centroid_str else None
            profile_centroids = json.loads(profile_str) if profile_str else None
            doc_embeddings = json.loads(doc_emb_str) if doc_emb_str else None

            self._node_registry[client_proxy.cid] = {
                "node_id": node_id,
                "status": status,
                "doc_count": eval_res.num_examples,
                "domain": str(metrics.get("domain", "")),
                "centroid": centroid,
                "profile_centroids": profile_centroids,
                "doc_embeddings": doc_embeddings,
            }
            if is_new:
                logger.info(
                    "✅ Node registered profile | Node ID: '%s' (CID: %s, status: %s, docs: %d)",
                    node_id, client_proxy.cid, status, eval_res.num_examples,
                )
        return None, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None