"""
FedRAG Flower Strategy.

Bridges the FastAPI REST layer with Flower's gRPC round-based loop.

- configure_fit()  : picks up a pending query from QueryBroker, sends to all nodes
- aggregate_fit()  : collects retrieved chunks + embeddings, publishes merged result
- configure_evaluate(): periodic health-check (every 10 rounds)
- aggregate_evaluate(): updates the in-process node registry
"""

import json
import logging
import queue
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.compat.common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)

logger = logging.getLogger("coordinator.strategy")

# ---------------------------------------------------------------------------
# QueryBroker — thread-safe REST ↔ Strategy bridge
# ---------------------------------------------------------------------------

class QueryBroker:
    """
    Thread-safe bridge between the FastAPI request handler and the Flower Strategy
    running in a background daemon thread.

    Flow:
        REST handler calls submit_query()  →  blocks on threading.Event
        Strategy calls get_pending()       →  returns the query dict
        Strategy calls publish_result()    →  signals the Event
        submit_query() unblocks            →  returns the result dict
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
        timeout: float = 120.0,
    ) -> dict:
        """
        Called by the FastAPI endpoint (runs in uvicorn worker thread).
        Blocks until the Strategy publishes results or timeout is reached.
        """
        query_id = uuid.uuid4().hex
        event = threading.Event()

        with self._lock:
            self._events[query_id] = event

        payload = {
            "query_id": query_id,
            "query": query,
            "top_k": top_k,
            "filters": filters or {},
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
        """Called by Strategy.configure_fit() to check for a queued query."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def publish_result(self, query_id: str, result: dict) -> None:
        """Called by Strategy.aggregate_fit() when all node responses are collected."""
        with self._lock:
            self._results[query_id] = result
            event = self._events.get(query_id)
        if event:
            event.set()
        else:
            logger.warning("publish_result: no waiter for query_id '%s'", query_id)


# ---------------------------------------------------------------------------
# FedRAG Strategy
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class FedRAGStrategy(Strategy):
    """
    Custom Flower Strategy for FedRAG broadcast query routing.

    Each Flower round maps to one of:
      - A query dispatch round (when QueryBroker has a pending query)
      - A health-check round (every 10 rounds)
      - A no-op round (nothing pending)
    """

    def __init__(self, broker: QueryBroker) -> None:
        self.broker = broker
        self._current_query: Optional[dict] = None
        # cid (str) → {node_id, status, doc_count}
        self._node_registry: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Required Strategy overrides
    # ------------------------------------------------------------------

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return Parameters(tensors=[], tensor_type="numpy.ndarray")

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Pick up a pending query and fan it out to all connected nodes."""
        pending = self.broker.get_pending(timeout=0.5)
        if pending is None:
            return []  # no query → skip round immediately

        clients = client_manager.all()
        if not clients:
            logger.warning("Query received but no nodes are connected — returning empty result")
            self.broker.publish_result(
                pending["query_id"],
                {"nodes": {}, "error": "no nodes connected"},
            )
            return []

        self._current_query = pending
        logger.info(
            "[Round %d] Dispatching query to %d node(s): '%s'",
            server_round, len(clients), pending["query"][:60],
        )

        fit_config: Dict[str, Scalar] = {
            "action": "query",
            "query": pending["query"],
            "top_k": pending["top_k"],
            "filters": json.dumps(pending["filters"]),
            "query_id": pending["query_id"],
        }
        fit_ins = FitIns(parameters=parameters, config=fit_config)
        return [(proxy, fit_ins) for proxy in clients.values()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Collect node responses and publish merged result to the broker."""
        if self._current_query is None:
            return None, {}

        node_results: Dict[str, dict] = {}

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            node_id = str(metrics.get("node_id", client_proxy.cid))
            results_json = str(metrics.get("results_json", "[]"))
            processing_time = float(metrics.get("processing_time", 0.0))

            # Deserialise doc embeddings from returned tensor bytes
            doc_embeddings: List[List[float]] = []
            if fit_res.parameters and fit_res.parameters.tensors:
                raw = fit_res.parameters.tensors[0]
                try:
                    arr = np.frombuffer(raw, dtype=np.float32)
                    if arr.size > 0 and arr.size % _EMBEDDING_DIM == 0:
                        doc_embeddings = arr.reshape(-1, _EMBEDDING_DIM).tolist()
                except Exception as e:
                    logger.debug("Could not deserialise embeddings from node '%s': %s", node_id, e)

            try:
                search_results = json.loads(results_json)
            except json.JSONDecodeError:
                search_results = []

            node_results[node_id] = {
                "results": search_results,
                "doc_embeddings": doc_embeddings,
                "processing_time": processing_time,
            }
            logger.info(
                "  Node '%s': %d chunks in %.3fs",
                node_id, len(search_results), processing_time,
            )

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
        """Run a health-check every 10 rounds or immediately when new nodes join."""
        clients = client_manager.all()
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
        """Update the in-process node registry from health-check responses."""
        for client_proxy, eval_res in results:
            metrics = eval_res.metrics or {}
            node_id = str(metrics.get("node_id", client_proxy.cid))
            status = str(metrics.get("status", "unknown"))
            is_new = client_proxy.cid not in self._node_registry
            self._node_registry[client_proxy.cid] = {
                "node_id": node_id,
                "status": status,
                "doc_count": eval_res.num_examples,
            }
            if is_new:
                logger.info(
                    "✅ Node registered profile | Node ID: '%s' (CID: %s, status: %s, docs: %d)",
                    node_id, client_proxy.cid, status, eval_res.num_examples,
                )
            else:
                logger.info(
                    "Health check | Node ID: '%s' (CID: %s): %s (%d docs)",
                    node_id, client_proxy.cid, status, eval_res.num_examples,
                )
        return None, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None
