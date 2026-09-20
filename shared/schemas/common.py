from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from shared.schemas.document import SearchResult


class RoutingStrategy(str, Enum):
    BROADCAST = "broadcast"
    SINGLE_BEST = "single_best"


class NodeStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(5, ge=1, le=100)
    routing_strategy: RoutingStrategy = RoutingStrategy.BROADCAST
    filters: Dict[str, Any] = Field(default_factory=dict)


class SourceResult(SearchResult):
    node_id: str
    trust_score: float = Field(0.0, ge=0.0, le=1.0)


class RoutingInfo(BaseModel):
    strategy_used: RoutingStrategy
    selected_nodes: List[str] = Field(default_factory=list)
    trust_scores: Dict[str, float] = Field(default_factory=dict)
    total_nodes_queried: int = 0
    failed_nodes: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceResult] = Field(default_factory=list)
    routing_info: RoutingInfo


class NodeInfo(BaseModel):
    node_id: str
    address: str
    status: NodeStatus = NodeStatus.HEALTHY
    trust_score: float = Field(1.0, ge=0.0, le=1.0)
    domain: str = ""
    capabilities: List[str] = Field(default_factory=list)
    last_seen: datetime = Field(default_factory=datetime.utcnow)


class NodeListResponse(BaseModel):
    nodes: List[NodeInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "healthy"
    registered_nodes: int = 0
    healthy_nodes: int = 0
