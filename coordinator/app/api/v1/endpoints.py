"""
Coordinator REST API — v1 endpoints.

POST /api/v1/query  — Route query via TASR, synthesize answer, and update trust scores
GET  /api/v1/nodes  — List connected nodes with dynamic TASR trust weights
GET  /api/v1/health — Coordinator health status
"""
import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status

from coordinator.app.api.deps import get_flower_server
from coordinator.app.synthesis.ollama_client import OllamaSynthesizer
from coordinator.app.tasr.feedback import TASRFeedbackEngine
from coordinator.app.tasr.router import TrustAwareRouter
from shared.embeddings.local import LocalSentenceTransformerEmbeddings
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

synthesizer = OllamaSynthesizer()
tasr_router = TrustAwareRouter(k_route = 6)
embedder = LocalSentenceTransformerEmbeddings()


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    flower = get_flower_server()

    # 1. Register newly detected nodes into TASR state
    for cid, info in flower.get_node_registry().items():
        nid = info.get("node_id", cid)
        if nid not in tasr_router.centroids:
            centroid = info.get("centroid")
            profile_centroids = info.get("profile_centroids")
            doc_embeddings = info.get("doc_embeddings")
            tasr_router.register_client(
                client_id=nid,
                centroid=centroid if centroid is not None else [0.0] * embedder.dimension,
                doc_embeddings=doc_embeddings,
                profile_centroids=profile_centroids,
            )

    # 2. Compute query embedding & obtain primary and feedback node targets
    q_emb = embedder.embed_query(req.query)
    primary_routed, feedback_targets = tasr_router.route_query(q_emb)

    # 3. Dispatch to all feedback targets (Primary + Scheduled Exploration)
    try:
        result = flower.broker.submit_query(
            query=req.query,
            top_k=req.top_k,
            filters=dict(req.filters) if req.filters else {},
            target_nodes=feedback_targets,
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

    # 4. Extract chunks: separate prompt sources from feedback evidence
    all_sources: List[SourceResult] = []
    generation_sources: List[SourceResult] = []
    node_doc_embeddings: Dict[str, List[List[float]]] = {}

    for node_id, node_data in node_results.items():
        doc_embeddings = node_data.get("doc_embeddings", [])
        node_doc_embeddings[node_id] = doc_embeddings
        current_trust = float(tasr_router.compute_trust_weight(node_id))

        for chunk in node_data.get("results", []):
            src = SourceResult(
                content=str(chunk.get("content", "")),
                score=float(chunk.get("score") or 0.0),
                metadata=dict(chunk.get("metadata") or {}),
                node_id=str(node_id),
                trust_score=current_trust,
            )
            all_sources.append(src)
            if node_id in primary_routed:
                generation_sources.append(src)

    generation_sources.sort(key=lambda s: s.score, reverse=True)
    all_sources.sort(key=lambda s: s.score, reverse=True)

    # Synthesize answer using the primary routed evidence
    answer = synthesizer.synthesize(query=req.query, sources=generation_sources)

    # 5. Compute TASR feedback signals across all feedback targets
    f_rel: Dict[str, float] = {}
    f_cons: Dict[str, float] = {}
    node_trust_rel: Dict[str, float] = {}

    for nid in feedback_targets:
        docs = node_doc_embeddings.get(nid, [])
        scores = [s.score for s in all_sources if s.node_id == nid]
        f_rel[nid] = TASRFeedbackEngine.calculate_relevance(q_emb, docs, chunk_scores=scores)
        centroids = tasr_router.profile_centroids.get(nid)
        f_cons[nid] = TASRFeedbackEngine.calculate_consistency(q_emb, centroids, docs)
        node_trust_rel[nid] = tasr_router.reputation.get(nid, 1.0)

    f_agr = TASRFeedbackEngine.calculate_cross_client_agreement(
        feedback_targets, node_doc_embeddings, f_rel, node_trust_rel
    )

    # 6. Apply median-thresholded trust updates
    tasr_router.update_trust(feedback_targets, f_rel, f_cons, f_agr)

    # 7. Build telemetry payload
    current_trust_map = {
        nid: float(tasr_router.compute_trust_weight(nid))
        for nid in tasr_router.trust_states
    }

    selected_nodes = list(node_results.keys())
    routing_strategy = RoutingStrategy.TASR if primary_routed else RoutingStrategy.BROADCAST

    routing_info = RoutingInfo(
        strategy_used=routing_strategy,
        selected_nodes=selected_nodes,
        trust_scores=current_trust_map,
        total_nodes_queried=len(selected_nodes),
        failed_nodes=[],
    )

    return QueryResponse(
        answer=answer,
        sources=all_sources,
        routing_info=routing_info,
    )


@router.get("/nodes", response_model=NodeListResponse)
def list_nodes() -> NodeListResponse:
    flower = get_flower_server()
    registry = flower.get_node_registry()

    nodes: List[NodeInfo] = []
    for cid, info in registry.items():
        nid = info.get("node_id", cid)
        status_str = info.get("status", "unknown")
        node_status = NodeStatus.HEALTHY if status_str == "healthy" else NodeStatus.DEGRADED

        current_trust = 1.0
        u_rel = 1.0
        u_cons = 1.0
        u_agr = 1.0
        s_i = 0.7
        feedback_count = 0

        if nid in tasr_router.trust_states:
            state = tasr_router.trust_states[nid]
            current_trust = float(tasr_router.compute_trust_weight(nid))
            u_rel = float(state.u_rel)
            u_cons = float(state.u_cons)
            u_agr = float(state.u_agr)
            s_i = float(state.s_i)
            feedback_count = int(state.feedback_count)

        nodes.append(
            NodeInfo(
                node_id=nid,
                address=cid,
                status=node_status,
                trust_score=current_trust,
                domain=info.get("domain", ""),
                u_rel=u_rel,
                u_cons=u_cons,
                u_agr=u_agr,
                s_i=s_i,
                feedback_count=feedback_count,
                last_seen=datetime.utcnow(),
            )
        )

    return NodeListResponse(nodes=nodes)


@router.get("/tasr/summary")
def get_tasr_summary() -> Dict[str, Any]:
    return {
        "defense_mode": tasr_router.defense_mode,
        "total_queries": tasr_router.query_count,
        "warmup_queries": tasr_router.warmup_queries,
        "is_warmup_active": tasr_router.query_count <= tasr_router.warmup_queries,
        "exploration_interval": tasr_router.explore_interval,
        "nodes": {
            str(nid): {
                "effective_trust": float(tasr_router.compute_trust_weight(nid)),
                "u_rel": float(tasr_router.reputation.get(nid, 1.0)),
                "u_cons": float(tasr_router.consistency_trust.get(nid, 1.0)),
                "u_agr": float(tasr_router.agreement_trust.get(nid, 1.0)),
                "s_i": float(tasr_router._cold_start_factor(nid)),
                "feedback_count": int(tasr_router.feedback_count.get(nid, 0)),
                "reputation_history": tasr_router.reputation_history.get(nid, []),
            }
            for nid in tasr_router.centroids
        },
    }


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
