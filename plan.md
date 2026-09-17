# FedRAG System with TASR Routing - Implementation Plan

## Project Overview
- **Goal**: Implement a Federated Retrieval-Augmented Generation (FedRAG) system with Trust Aware Secure Routing (TASR)
- **Scale**: 2-5 nodes initially, designed for easy horizontal scaling
- **Architecture**: Centralized coordinator with distributed node APIs
- **Deployment**: All services as Docker containers (Docker Compose for dev, Kubernetes for prod)
- **Tech Stack**: Python (FastAPI), ChromaDB, Provider-agnostic embedding interface, Local LLMs (Ollama), Basic authentication, **Flower (flwr) for federated communication**, YAML config, Monorepo structure
- **RAG Framework**: LangChain
- **Routing Strategy (Phase 1)**: Broadcast to all nodes with trust-weighted merge aggregation
- **Node Discovery**: Self-registration with coordinator
- **Data Distribution**: Disjoint datasets per node

## System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Client    │────▶│  Coordinator     │────▶│  Node 1     │
└─────────────┘     │  (TASR Router)   │     │  (RAG + DB) │
                    └──────────────────┘     │  + Ollama   │
                           │                 └─────────────┘
                           ▼                       │
                    ┌──────────────────┐           ▼
                    │  Trust Manager   │      ┌─────────────┐
                    └──────────────────┘      │  Node N     │
                                              │  (RAG + DB) │
                                              │  + Ollama   │
                                              └─────────────┘

Communication: **Flower (flwr) for coordinator ↔ nodes**, REST (client → coordinator)
Config: YAML files per service
Structure: Monorepo (coordinator/, nodes/, shared/)
```

## Core Components

### 1. Coordinator Service (Centralized)
- **Responsibilities**:
  - Query routing using TASR algorithm
  - Trust score management
  - Load balancing across nodes
  - Aggregate results from multiple nodes
  - API gateway for clients

### 2. Federated Nodes (Distributed)
- **Responsibilities**:
  - Local document storage and indexing (ChromaDB)
  - Local RAG pipeline using LangChain (retrieval + generation via Ollama)
  - Embedding generation via provider-agnostic interface (OpenAI/Cohere/others)
  - **Expose Flower ClientApp for coordinator communication**
  - **Self-register with coordinator on startup (auto-detect domain from local documents)**
  - **Support dynamic joining/leaving via Flower's client manager**
  - Report health/metrics to coordinator
  - Run local Ollama instance for generation

### 3. Trust Manager (TASR Core - Paper Implementation)
- **Responsibilities** (per arXiv:2605.28112):
  - Maintain three trust variables per node in [0,1]:
    - `u_rel` (retrieval relevance): query-document similarity
    - `u_cons` (profile consistency): profile-evidence similarity  
    - `u_agr` (cross-client agreement): peer evidence alignment
  - Compute effective trust: `τ_i = s_i * (u_rel_i)^α_r * g(u_cons_i) * g(u_agr_i)`
    - `s_i`: cold-start factor (ramps from s_0 to 1 over warmup queries)
    - `g(x) = δ + (1-δ)x`: soft gating function
  - Trust-adjusted routing score: `Ŝ_i(q) = S̄_i(q) * τ_i` (reweights base router scores)
  - Post-routing trust update using three feedback signals from returned evidence:
    - Relevance: `f_rel = max_{e∈E_i(q)} cos(φ(q), φ(e))`
    - Consistency: `f_cons = max_{e∈E_i(q)} cos(P_i, φ(e))` (P_i = registered profile centroid)
    - Agreement: cosine similarity of document centroids among relevance-qualified peers, weighted by peer reputation
  - Update rule: decay if feedback < median(valid feedbacks), recover otherwise
  - Configurable defense modes: `rel_cons_agr` (full), `rel_cons`, `none`

### 4. Shared Libraries
- **Common schemas**: Request/response models, document chunks, embeddings (Pydantic models)
- **Communication protocols**: **Flower (flwr) ClientApp/ServerApp patterns** for node-coordinator communication, custom message types for RAG operations
- **Security utilities**: Authentication, basic encryption
- **Monitoring**: Logging, metrics collection
- **Configuration**: YAML config loading utilities (Pydantic Settings)
- **Embedding abstraction**: Provider-agnostic interface with OpenAI/Cohere implementations

## Implementation Phases

### Phase 1: Foundation
**Goal**: Basic federated RAG with broadcast routing and trust-weighted merge

#### Tasks:
1. **Project Setup**
   - Initialize Python monorepo with FastAPI
   - Set up dependency management (poetry)
   - Configure ChromaDB integration
   - **Add Flower (flwr) dependency**
   - Set up basic project structure:
     ```
     fedrag/
     ├── coordinator/
     ├── nodes/
     ├── shared/
     │   ├── schemas/
     │   ├── **flower/**
     │   ├── config/
     │   ├── embeddings/
     │   └── auth/
     ├── docker-compose.yml
     └── pyproject.toml
     ```

2. **Shared Infrastructure**
   - Common Pydantic models for all APIs
   - **Flower message types and serialization for RAG operations (Query, Health, Metrics, Register)**
   - YAML configuration loading (Pydantic Settings)
   - Provider-agnostic embedding interface with OpenAI/Cohere implementations
   - Basic authentication middleware (API keys)
   - Logging and monitoring setup
   - Docker Compose for local development

3. **Node Implementation**
   - **Pre-loaded document indexing at startup** (PDF, text, markdown → ChromaDB)
   - **Auto-detect domain/specialization from indexed documents at startup** (compute centroid embedding)
   - ChromaDB integration for vector storage (per-node instance)
   - LangChain RAG pipeline:
     - Retrieval: ChromaDB similarity search
     - Generation: Ollama local LLM (configurable model)
   - **Flower ClientApp implementation:**
     - Handle `QueryIns` / `QueryRes` messages
     - Handle `HealthCheckIns` / `HealthCheckRes` messages
     - Handle `MetricsIns` / `MetricsRes` messages
     - Handle `RegisterIns` / `RegisterRes` messages (include auto-detected domain)
   - Self-registration logic on startup (via Flower)
   - Ollama integration for local generation

4. **Coordinator Basics**
   - Node registration/discovery **via Flower ServerApp** (dynamic join/leave support)
   - **Maintain node registry with domain, trust score, health status**
   - Broadcast routing: send query to all healthy nodes **using Flower's `ServerApp.fit()` or custom message passing**
   - Trust-weighted merge aggregation (placeholder trust scores = 1.0 initially)
   - Client-facing REST API gateway:
     - `POST /api/v1/query` - Main query endpoint
     - `GET /api/v1/nodes` - List registered nodes (with domain, trust, health)
     - `GET /api/v1/health` - Coordinator health
   - Basic load balancing (round-robin for health checks)

### Phase 2: TASR Routing Implementation (per arXiv:2605.28112)
**Goal**: Implement Trust Aware Secure Routing with paper's three-signal trust model

#### Tasks:
1. **Trust-Aware Router Core (shared/)**
   - Implement `TrustAwareRouter` class (from paper's `trust_defense.py`)
   - Three trust variables per node: `u_rel` (relevance), `u_cons` (consistency), `u_agr` (agreement)
   - Cold-start factor `s_i` with configurable warmup
   - Soft gating function `g(x) = δ + (1-δ)x`
   - Configurable defense modes: `rel_cons_agr`, `rel_cons`, `none`
   - Decay factor `γ`, recovery factor `γ_rec`, min_reputation floor

2. **Node Profile Registration**
   - At startup: node computes centroid `C_i` from document embeddings
   - Node computes profile centroids `P_i` (can be same as `C_i` or multi-centroid)
   - Node sends `RegisterIns` with centroid + doc embeddings (or sample) to coordinator
   - Coordinator calls `TrustAwareRouter.register_client()` for each node

3. **TASR Routing via Flower Strategy**
   - Implement custom Flower `Strategy` wrapping `TrustAwareRouter`
   - `configure_fit()` / `configure_evaluate()` → `route()` for node selection
   - `aggregate_fit()` / `aggregate_evaluate()` → `update_trust()` with returned evidence
   - Handle exploration (`explore_interval`, `explore_extra`)

4. **Evidence Feedback Computation**
   - Nodes return top-K retrieved document embeddings with `QueryRes`
   - Coordinator computes three feedback signals:
     - Relevance: query-doc cosine similarity
     - Consistency: profile-doc cosine similarity  
     - Agreement: cross-client doc centroid alignment (peer-weighted by reputation)
   - Dynamic threshold: median of valid feedbacks per signal

5. **Trust Update & Persistence**
   - Update trust variables after each query (post warmup)
   - Persist trust state (SQLite/PostgreSQL)
   - Track trust history for monitoring/analysis

6. **Security Enhancements**
   - Flower's built-in TLS/mTLS for inter-node communication
   - Request/response validation (Pydantic)
   - Basic rate limiting per client/node
   - Audit logging for all queries and routing decisions

7. **Integration Testing**
   - Multi-node deployment testing (3-5 nodes)
   - Routing accuracy validation (HR@K metrics)
   - Trust score convergence testing (malicious vs honest node separation)
   - Failure scenario testing (node down, slow, erroneous)
   - Routing hijacking attack simulation (paper's threat model)

### Phase 3: Advanced Features
**Goal**: Production hardening and advanced capabilities

#### Tasks:
1. **Advanced Trust Model**
   - Reputation-based scoring (peer feedback)
   - Certificate/credential verification
   - Dynamic trust adjustment based on query types
   - Trust visualization dashboard

2. **Performance Optimization**
   - Caching layer for frequent queries (Redis)
   - Async processing for high throughput
   - Connection pooling for gRPC
   - Query optimization (embedding caching, query rewriting)

3. **Monitoring & Observability**
   - Distributed tracing (OpenTelemetry)
   - Comprehensive metrics (Prometheus/Grafana)
   - Alerting for trust degradation
   - Automated failover

4. **Scalability Features**
   - **Dynamic node registration/deregistration (auto domain detection on join)**
   - Horizontal scaling patterns
   - Configuration management (centralized)
   - Rolling deployment support

## Technical Specifications

### API Contracts

#### Node Flower API (Message Types)
```python
# shared/flower/messages.py
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from flwr.common import Message, RecordSet

@dataclass
class QueryIns:
    query: str
    top_k: int
    filters: Dict[str, str]

@dataclass
class SearchResult:
    content: str
    score: float
    metadata: Dict[str, str]

@dataclass
class QueryRes:
    results: List[SearchResult]
    node_id: str
    processing_time: float
    trust_score: float
    # For TASR feedback: top document embeddings (optional, can be computed on coordinator side)
    doc_embeddings: List[List[float]]  # [num_docs, embedding_dim]

@dataclass
class HealthCheckIns:
    pass

@dataclass
class HealthCheckRes:
    status: str  # "healthy", "degraded", "unhealthy"
    details: Dict[str, Any]

@dataclass
class MetricsIns:
    pass

@dataclass
class MetricsRes:
    latency_p50: float
    latency_p95: float
    latency_p99: float
    availability: float
    error_rate: float
    total_queries: int

@dataclass
class RegisterIns:
    node_id: str
    address: str
    domain: str
    capabilities: List[str]
    centroid: List[float]  # Node's document centroid embedding
    doc_embeddings: List[List[float]]  # Sample of document embeddings for feedback
    profile_centroids: List[List[float]]  # Optional: multi-centroid profile

@dataclass
class RegisterRes:
    success: bool
    assigned_node_id: str
    coordinator_config: Dict[str, Any]
```

#### Coordinator REST API
```python
# POST /api/v1/query
{
  "query": "user question",
  "top_k": 5,
  "routing_strategy": "tasr"  # or "broadcast", "single_best"
}

# Response
{
  "answer": "generated response",
  "sources": [
    {"content": "source text", "node_id": "node-1", "score": 0.95, "trust_score": 0.9}
  ],
  "routing_info": {
    "strategy_used": "tasr",
    "selected_nodes": ["node-1", "node-3"],
    "trust_scores": {"node-1": 0.9, "node-3": 0.85},
    "total_nodes_queried": 2
  }
}

# GET /api/v1/nodes
{
  "nodes": [
    {
      "node_id": "node-1",
      "address": "node-1:50051",
      "status": "healthy",
      "trust_score": 0.9,
      "domain": "finance",
      "last_seen": "2026-09-17T10:30:00Z"
    }
  ]
}
```

### Data Models

#### Document Chunk
```python
class DocumentChunk(BaseModel):
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    node_id: str
    timestamp: datetime
```

#### TASR Trust State (per node, per paper)
```python
class TASRTrustState(BaseModel):
    node_id: str
    # Three trust variables in [0,1]
    u_rel: float = 1.0      # retrieval relevance
    u_cons: float = 1.0     # profile consistency  
    u_agr: float = 1.0      # cross-client agreement
    # Cold-start factor
    s_i: float = 0.7        # ramps to 1.0 over warmup
    # Feedback count for cold-start
    feedback_count: int = 0
    # History for monitoring
    reputation_history: List[float] = []
    consistency_history: List[float] = []
    agreement_history: List[float] = []
    last_updated: datetime
```

#### Node Profile (for TASR registration)
```python
class NodeProfile(BaseModel):
    node_id: str
    centroid: List[float]           # Single centroid of all docs
    doc_embeddings: List[List[float]]  # Sample for feedback computation
    profile_centroids: List[List[float]]  # Multi-centroid profile (optional)
    domain: str                     # Auto-detected or assigned
    capabilities: List[str]
```

### TASR Algorithm (per arXiv:2605.28112 - Algorithm 2)
```
Initialize:
- For each registered node i: centroid C_i, profile centroids P_i, doc embeddings D_i
- Trust vars: u_rel_i = u_cons_i = u_agr_i = 1.0, s_i = s_0 (cold-start)
- Defense mode: "rel_cons_agr" (full), "rel_cons", or "none"

For each query q (handled by custom Flower Strategy / TrustAwareRouter):
1. Embed query: q_emb = φ(q)
2. Get candidate nodes from Flower's client manager
3. Base routing scores: S_i(q) = cos(q_emb, C_i)  # cosine similarity
4. Normalize: S̄_i(q) = Normalize({S_i(q)})  # non-negative, order-preserving
5. Compute effective trust τ_i:
   s_i = min(1, s_0 + (1-s_0) * t/τ_s)  # cold-start ramp
   τ_i = s_i * (u_rel_i)^α_r * g(u_cons_i) * g(u_agr_i)  # g(x) = δ + (1-δ)x
   (Skip g(u_agr) if defense_mode != "rel_cons_agr" or top_k < 2)
6. Trust-adjusted scores: Ŝ_i(q) = S̄_i(q) * τ_i
7. Select top-K_route nodes: T = TopK({Ŝ_i(q)}, K_route)
   (Optional: explore_interval adds extra random nodes periodically)
8. Send query to selected nodes T in parallel via Flower (QueryIns/QueryRes)
9. Collect returned evidence E_i(q) from each selected node
10. Phase 2: Multi-Signal Feedback
    For each selected node i:
    - Retrieve top-docs_for_feedback documents from local index
    - Compute f_rel_i = mean(cos(q_emb, doc_emb))  # retrieval relevance
    - Compute f_cons_i = mean(cos(P_i_winner, doc_emb))  # profile consistency
      (P_i_winner = argmax_{c∈P_i} cos(c, q_emb); weighted by cons_winner_weight)
    - Compute f_agr_i = peer-weighted mean(cos(doc_centroid_i, doc_centroid_j))
      among relevance-qualified peers (f_rel_j ≥ median(f_rel))
11. Phase 3: Trust Update (after warmup_queries)
    For each signal h ∈ {rel, cons, agr}:
    - Valid set V_h = {i ∈ T: f_h_i valid}
    - Threshold θ_h = median({f_h_i: i ∈ V_h})  (dynamic) or fixed_threshold
    - For i ∈ V_h:
      if f_h_i < θ_h: u_h_i = max(u_h_i * γ, min_reputation)  # decay
      else: u_h_i = min(u_h_i * γ_rec, 1.0)  # recover
    (Uninformative signals skipped)
12. Aggregate results: trust-weighted merge using τ_i as weights
13. Return aggregated answer to client
```

## Deployment Architecture

### Development
- Docker Compose with:
  - Coordinator service (FastAPI + **Flower ServerApp**)
  - 3 Node services (FastAPI + **Flower ClientApp** + ChromaDB + Ollama)
  - Shared networking
  - Volume mounts for YAML configs and ChromaDB persistence

### Production Considerations
- Kubernetes deployment with:
  - Coordinator as Deployment (2+ replicas, HPA) running **Flower ServerApp**
  - Nodes as StatefulSets (persistent ChromaDB volumes) running **Flower ClientApp**
  - Ollama as sidecar or separate Deployment per node
  - **Flower's built-in TLS/mTLS for secure communication** (or service mesh)
  - Horizontal Pod Autoscaler for nodes based on CPU/memory/custom metrics
  - ConfigMaps/Secrets for YAML configuration
  - PersistentVolumes for ChromaDB and trust score database

## Configuration Management
- YAML config files per service (mounted as volumes):
  - `coordinator/config.yaml`: coordinator settings, routing config, **TASR parameters** (decay_factor, recovery_factor, warmup_queries, cold_start_s0, cold_start_tau, alpha_r, alpha_c, alpha_a, delta_c, delta_a, defense_mode, docs_for_feedback, threshold_mode, explore_interval)
  - `nodes/config.yaml`: node settings, ChromaDB path, Ollama endpoint, embedding provider
  - `shared/config.yaml`: common settings, auth keys, gRPC ports
- Pydantic Settings for type-safe config loading
- Environment variable overrides for secrets (API keys, certificates)
- Separate configs per environment (dev/staging/prod)

## Testing Strategy

### Unit Tests
- Trust scoring algorithms
- Routing logic
- **Document indexing at startup**
- API contract validation

### Integration Tests
- Multi-node query routing
- Trust score updates
- Failure scenarios
- Performance benchmarks
- **Dynamic node join/leave scenarios**

### End-to-End Tests
- Full query flow: client → coordinator → nodes → response
- Scaling scenarios

## Monitoring & Observability

### Key Metrics
- Query latency (p50, p95, p99)
- Routing accuracy
- Trust score distribution
- Node health status
- Error rates by type

### Alerts
- Trust score below threshold
- Node unavailable > 5 minutes
- Query latency > SLA
- High error rates

## Security Considerations
- API key authentication for all services (coordinator REST, **Flower messages**)
- **Flower's built-in TLS/mTLS for all inter-service communication**
- HTTPS/TLS for client-coordinator REST API
- Input validation and sanitization (Pydantic models)
- Rate limiting per client/node (token bucket)
- Audit logging for all queries, routing decisions, and trust updates
- Certificate rotation strategy for mTLS

## Future Extensibility
- Plugin architecture for custom trust models
- Support for different vector databases
- Multi-modal RAG (images, tables)
- Cross-node learning/federated learning
- Advanced routing: semantic, geographic, regulatory
- **Neural router transfer (RAGRoute-style MLP + TASR trust reweighting)**
- **Learned threshold functions instead of median-based**
- **Integration with paper's official implementation: https://github.com/Junjie-Mu/routing-hijacking-fedrag**

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Trust score manipulation | High | Medium | **Flower's mTLS**, audit trails, signed attestations |
| Single point of failure (coordinator) | High | Low | Coordinator HA (2+ replicas), graceful degradation to direct node access |
| Data inconsistency across nodes | Medium | Medium | Disjoint datasets by design, eventual consistency for metadata |
| Embedding API failures | Medium | High | Provider-agnostic interface with fallback, local embedding cache |
| Ollama/local LLM failures | High | Medium | Health checks, fallback to smaller model, circuit breaker |
| Scaling bottlenecks | Medium | Low | Stateless coordinator, horizontal node scaling, **Flower's efficient message passing** |
| **Flower version compatibility** | Medium | Low | **Pin Flower version, test upgrades in staging** |

## Success Criteria
- [ ] 3-5 nodes operational with <200ms routing overhead (coordinator only)
- [ ] Trust scores converge meaningfully within 100 queries
- [ ] 99.9% uptime for coordinator (with HA)
- [ ] Horizontal scaling demonstrated to 10+ nodes
- [ ] Basic authentication + mTLS working end-to-end
- [ ] Comprehensive test coverage (>80% unit, integration tests for critical paths)
- [ ] Broadcast routing with trust-weighted merge functional (Phase 1)
- [ ] TASR routing selects optimal nodes based on trust + domain relevance (Phase 2)
- [ ] **TASR trust variables (u_rel, u_cons, u_agr) correctly updated post-routing**
- [ ] **Malicious/hijacking node detection: trust scores drop for nodes returning irrelevant/inconsistent evidence**
- [ ] **HR@K routing accuracy maintained or improved with TASR vs baseline**

## Next Steps
1. Review and approve this refined plan
2. Set up development environment (Python, Poetry, Docker, Ollama, **Flower**)
3. Initialize monorepo structure with shared libraries
4. Begin Phase 1 implementation: shared schemas, **Flower message types**, config system
5. Implement node service with ChromaDB + LangChain + Ollama + **Flower ClientApp**
6. Implement coordinator with broadcast routing **via Flower ServerApp**
7. Integration testing with Docker Compose
8. Weekly sync to review progress and adjust

## Open Questions (Resolved)
1. ✅ RAG Framework: LangChain
2. ✅ Embedding API: Provider-agnostic interface (OpenAI/Cohere implementations)
3. ✅ LLM for Generation: Local LLMs via Ollama
4. ✅ Routing Strategy (Phase 1): Broadcast to all nodes with trust-weighted merge
5. ✅ Node Discovery: Self-registration
6. ✅ Result Aggregation: Trust-weighted merge
7. ✅ Data Distribution: Disjoint datasets
8. ✅ **Inter-node Communication: Flower (flwr) replacing gRPC**
9. ✅ Local LLM Deployment: Ollama per node initially
10. ✅ Config Management: YAML files + Pydantic Settings
11. ✅ Project Structure: Monorepo

## Remaining Open Questions
1. Specific embedding model dimensions (affects ChromaDB collection config)?
2. Ollama model(s) to use for generation (llama3.1, mistral, etc.)?
3. **TASR hyperparameters (use paper defaults or tune?):**
   - decay_factor (γ): paper default 0.9
   - recovery_factor (γ_rec): paper default 1.02
   - warmup_queries: paper default 50
   - cold_start_s0: paper default 0.7
   - cold_start_tau: paper default 30.0
   - alpha_r, alpha_c, alpha_a: paper defaults 1.0, 1.0, 0.5
   - delta_c, delta_a: paper defaults 0.3, 0.5
   - defense_mode: "rel_cons_agr" (full), "rel_cons", or "none"
   - docs_for_feedback: paper default 5
   - explore_interval, explore_extra: paper defaults 20, 1
4. Domain classification approach (embedding centroid per node vs explicit metadata)?
5. Persistence strategy for trust scores (SQLite vs PostgreSQL)?