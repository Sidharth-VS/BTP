from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    id: str
    content: str
    embedding: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    node_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)