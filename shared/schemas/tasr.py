from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class NodeProfile(BaseModel):
    node_id: str
    address: str
    domain: str
    capabilities: List[str] = Field(default_factory=list)
    centroid: List[float]
    doc_embeddings: List[List[float]] = Field(default_factory=list)
    profile_centroids: List[List[float]] = Field(default_factory=list)


class TASRTrustState(BaseModel):
    node_id: str
    u_rel: float = 1.0
    u_cons: float = 1.0
    u_agr: float = 1.0
    s_i: float = 0.7
    feedback_count: int = 0
    reputation_history: List[float] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)