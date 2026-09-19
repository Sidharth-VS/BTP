from dataclasses import dataclass, field
from typing import Any, Dict, List
from shared.schemas.document import SearchResult


@dataclass
class QueryIns:
    query: str
    top_k: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRes:
    results: List[SearchResult]
    node_id: str
    processing_time: float
    trust_score: float
    doc_embeddings: List[List[float]] = field(default_factory=list)


@dataclass
class RegisterIns:
    node_id: str
    address: str
    domain: str
    capabilities: List[str]
    centroid: List[float]
    doc_embeddings: List[List[float]] = field(default_factory=list)
    profile_centroids: List[List[float]] = field(default_factory=list)


@dataclass
class RegisterRes:
    success: bool
    assigned_node_id: str
    coordinator_config: Dict[str, Any] = field(default_factory=dict)