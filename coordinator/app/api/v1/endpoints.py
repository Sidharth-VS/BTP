"""
Coordinator REST API — v1 endpoints.

POST /api/v1/query  — broadcast query to all nodes, return ranked chunks
GET  /api/v1/nodes  — list connected nodes (from last health-check round)
GET  /api/v1/health — coordinator health
"""
import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status

from coordinator.app.api.deps import get_flower_server
from shared.schemas.common import (
    HealthResponse,
    NodeInfo,
    NodeListResponse,
    NodeStatus,
    QueryRequest,
    QueryResponse,
    RoutingInfo,
    RoutingStrategy,
    SourceResult,
)

logger = logging.getLogger("coordinator.api")
router = APIRouter()


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """
    Broadcast the query to all connected nodes via Flower gRPC.
    Returns the merged, score-ranked retrieved chunks (no LLM generation).
    """
    flower = get_flower_server()

    try:
        result = flower.broker.submit_query(
            query=req.query,
            top_k=req.top_k,
            filters=dict(req.filters) if req.filters else {},
            timeout=120.0,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc))
    except Exception as exc:
        logger.error("Query dispatch failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    if "error" in result and not result.get("nodes"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result["error"],
        )

    node_results: Dict[str, Any] = result.get("nodes", {})

    # ---- Flatten + rank all chunks across nodes -------------------------
    all_sources: List[SourceResult] = []
    for node_id, node_data in node_results.items():
        for chunk in node_data.get("results", []):
            all_sources.append(
                SourceResult(
                    content=chunk.get("content", ""),
                    score=float(chunk.get("score", 0.0)),
                    metadata=chunk.get("metadata", {}),
                    node_id=node_id,
                    trust_score=1.0,  # placeholder until TASR is wired
                )
            )

    # Sort by score descending
    all_sources.sort(key=lambda s: s.score, reverse=True)

    selected_nodes = list(node_results.keys())
    routing_info = RoutingInfo(
        strategy_used=RoutingStrategy.BROADCAST,
        selected_nodes=selected_nodes,
        trust_scores={nid: 1.0 for nid in selected_nodes},
        total_nodes_queried=len(selected_nodes),
    )

    return QueryResponse(
        answer="",  # generation not implemented yet
        sources=all_sources,
        routing_info=routing_info,
    )


# ---------------------------------------------------------------------------
# GET /nodes
# ---------------------------------------------------------------------------

@router.get("/nodes", response_model=NodeListResponse)
def list_nodes() -> NodeListResponse:
    """List nodes that have reported in via the last health-check round."""
    flower = get_flower_server()
    registry = flower.get_node_registry()

    nodes: List[NodeInfo] = []
    for cid, info in registry.items():
        status_str = info.get("status", "unknown")
        node_status = NodeStatus.HEALTHY if status_str == "healthy" else NodeStatus.DEGRADED
        nodes.append(
            NodeInfo(
                node_id=info.get("node_id", cid),
                address=cid,
                status=node_status,
                trust_score=1.0,
                domain="",
                last_seen=datetime.utcnow(),
            )
        )

    return NodeListResponse(nodes=nodes)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    flower = get_flower_server()
    registry = flower.get_node_registry()
    healthy = sum(1 for v in registry.values() if v.get("status") == "healthy")
    return HealthResponse(
        status="healthy",
        registered_nodes=len(registry),
        healthy_nodes=healthy,
    )
