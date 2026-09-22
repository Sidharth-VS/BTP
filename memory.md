# Project Memory: FedRAG with TASR Routing

**Phase**: Phase 1 (Scaffolding & Messaging) & Phase 2 (Federated Node RAG & Central Synthesis Pipeline)  
**Status**: Implemented & Operational (Broadcast RAG, ChromaDB Vector Storage, Ollama Synthesis, Streamlit UI & Docker Deployment)  
**Runtime Environment**: WSL (Fedora Linux), Conda (`fed_lab`), Python 3.10.20, Poetry, Docker Compose, Ollama  

---

## 1. System Overview & Implementation Status

The FedRAG repository has progressed beyond initial Phase 1 foundation into a functional, end-to-end Federated Retrieval-Augmented Generation (FedRAG) system. The system runs across a central Coordinator, 4 domain-specialized Edge Nodes (`node-1` finance, `node-2` medical, `node-3` legal, `node-4` tech), a central Ollama LLM synthesis engine, and a Streamlit UI dashboard.

### Operational Features:
* **Federated Communication Protocol**: Flower (`flwr`) gRPC messaging using a custom `FedRAGStrategy` and `NumPyClient` setup.
* **Distributed Vector Storage**: Persistent ChromaDB vector databases per node paired with local SentenceTransformer embeddings (`all-MiniLM-L6-v2`).
* **Document Ingestion & Profiling**: Automated document chunking, metadata extraction, global normalized centroid ($C_i$), and multi-centroid profile ($P_i$) computation.
* **Broadcast Query Routing**: Coordinator fans out incoming queries to all healthy registered nodes via Flower, aggregating ranked chunks and raw document vector matrices.
* **Central LLM Answer Synthesis**: Integrated Ollama client (`llama3.2:3b`) synthesizes grounded answers from multi-node evidence.
* **UI & Docker Deployment**: Streamlit Explorer dashboard (`frontend/app.py`), containerized 4-node `docker-compose.yml`, and `setup_workspace.sh` automation script.
* **TASR Foundation**: Data schemas and module stubs (`coordinator/app/tasr/`) prepared for paper-compliant Trust-Aware Secure Routing integration.

---

## 2. Environment & Dependency Baseline (`pyproject.toml`)

* `python`: `>=3.10,<3.12`
* `flwr`: `^1.8.0` (Resolved to `flwr 1.30.0`)
* `fastapi`: `^0.111.0` & `uvicorn[standard]`: `^0.30.0`
* `chromadb`: `^0.5.0`
* `langchain` & `langchain-community`: `^0.2.0`
* `sentence-transformers`: `^3.0.0` (`all-MiniLM-L6-v2`, 384-dimensional)
* `pydantic`: `^2.7.0` & `pydantic-settings`: `^2.3.0`
* `ollama`: `^0.2.1` (`llama3.2:3b` default model)
* `streamlit`: Interactive UI dashboard
* `numpy`: `1.26.4` & `onnxruntime`: `<1.24.0`

---

## 3. Implemented Components & Codebase Architecture

```
BTP/
├── shared/                 # Common schemas, embeddings, flower messaging, security
│   ├── schemas/            # Pydantic models (document.py, tasr.py, common.py)
│   ├── embeddings/         # Base & local SentenceTransformer embedding providers
│   ├── flower/             # Custom message types & Flower 1.30 record serializers
│   └── auth/               # API key authentication middleware (security.py)
├── coordinator/            # Central REST API gateway, Flower gRPC Server, Ollama synthesizer
│   ├── app/main.py         # Entrypoint (Flower gRPC + uvicorn REST dual-thread runtime)
│   ├── app/api/v1/         # Endpoints: POST /query, GET /nodes, GET /health
│   ├── app/flwr_server/    # Server, FedRAGStrategy, and QueryBroker thread bridge
│   ├── app/synthesis/      # OllamaSynthesizer answer generation client
│   └── app/tasr/           # TASR trust manager & router stubs (router, feedback, db)
├── nodes/                  # Distributed federated edge node implementation
│   ├── app/main.py         # Node entrypoint & Flower NumPyClient implementation
│   ├── app/rag/            # ChromaStore vector DB, DocumentIndexer, NodeProfiler
│   ├── nodes.yaml          # Unified node registry configuration (node-1 to node-4)
│   └── data/               # Domain-specific markdown document datasets
├── frontend/               # Streamlit Explorer UI (app.py, Dockerfile)
├── workspace/              # Persistent runtime storage (Chroma databases, data copies, logs)
├── setup_workspace.sh      # Workspace initialization and sample data seeding script
└── docker-compose.yml      # Orchestration for Coordinator, 4 Nodes, and Frontend
```

### Detailed Component Summary:

#### A. Shared Infrastructure (`shared/`)
* **Data Contracts (`shared/schemas/`)**:
  * [document.py](file:///home/sidharth/Coding/Projects/BTP/shared/schemas/document.py): `DocumentChunk`, `SearchResult`.
  * [tasr.py](file:///home/sidharth/Coding/Projects/BTP/shared/schemas/tasr.py): `NodeProfile` (capturing gRPC host, domain, centroid $C_i$, sample embeddings, multi-cluster profile $P_i$), `TASRTrustState` ($u_{\text{rel}}, u_{\text{cons}}, u_{\text{agr}}, s_i$).
  * [common.py](file:///home/sidharth/Coding/Projects/BTP/shared/schemas/common.py): REST schemas (`QueryRequest`, `QueryResponse`, `SourceResult`, `RoutingInfo`, `NodeInfo`, `NodeListResponse`, `HealthResponse`, `NodeStatus`, `RoutingStrategy`).
* **Embeddings (`shared/embeddings/`)**:
  * [local.py](file:///home/sidharth/Coding/Projects/BTP/shared/embeddings/local.py): `LocalSentenceTransformerEmbeddings` producing normalized 384-d vectors via `all-MiniLM-L6-v2`.
* **Flower Custom Serialization (`shared/flower/`)**:
  * [messages.py](file:///home/sidharth/Coding/Projects/BTP/shared/flower/messages.py) & [serialization.py](file:///home/sidharth/Coding/Projects/BTP/shared/flower/serialization.py): Custom serializer connecting `QueryIns`, `QueryRes`, `RegisterIns`, and `RegisterRes` dataclasses with Flower's `RecordDict`, `ConfigRecord`, and `ArrayRecord`.
* **Security (`shared/auth/security.py`)**:
  * `verify_api_key`: FastAPI dependency validating `X-API-Key` headers against environment settings.

#### B. Central Coordinator (`coordinator/app/`)
* **Dual-Thread Execution ([main.py](file:///home/sidharth/Coding/Projects/BTP/coordinator/app/main.py))**: Initializes `FlowerServer` on the main thread for gRPC signal compliance and spawns `uvicorn` in a background daemon thread for REST client traffic.
* **REST Gateway ([api/v1/endpoints.py](file:///home/sidharth/Coding/Projects/BTP/coordinator/app/api/v1/endpoints.py))**:
  * `POST /api/v1/query`: Receives client queries, submits them to the Flower gRPC engine via `QueryBroker`, aggregates chunk search results, ranks sources, calls `OllamaSynthesizer` for answer generation, and returns response telemetry.
  * `GET /api/v1/nodes`: Returns status, trust, and metadata for registered nodes.
  * `GET /api/v1/health`: Provides system health and connected node metrics.
* **Flower gRPC Engine ([flwr_server/](file:///home/sidharth/Coding/Projects/BTP/coordinator/app/flwr_server/))**:
  * [strategy.py](file:///home/sidharth/Coding/Projects/BTP/coordinator/app/flwr_server/strategy.py): `FedRAGStrategy` manages query fanout (`configure_fit`), chunk/embedding aggregation (`aggregate_fit`), and health monitoring rounds (`configure_evaluate`).
  * `QueryBroker`: Thread-safe bridge using `threading.Event` to sync async uvicorn HTTP requests with Flower's round-based gRPC loop.
* **Synthesis Engine ([synthesis/ollama_client.py](file:///home/sidharth/Coding/Projects/BTP/coordinator/app/synthesis/ollama_client.py))**:
  * `OllamaSynthesizer`: Formats context from top-ranked node chunks and prompts local Ollama LLM (`llama3.2:3b`) for answer synthesis.
* **TASR Routing Stubs ([tasr/](file:///home/sidharth/Coding/Projects/BTP/coordinator/app/tasr/))**:
  * [router.py](file:///home/sidharth/Coding/Projects/BTP/coordinator/app/tasr/router.py), [feedback.py](file:///home/sidharth/Coding/Projects/BTP/coordinator/app/tasr/feedback.py), [db.py](file:///home/sidharth/Coding/Projects/BTP/coordinator/app/tasr/db.py): Architectural stubs prepared for TASR trust calculation and routing.

#### C. Distributed Edge Nodes (`nodes/app/`)
* **Local RAG Pipeline ([rag/](file:///home/sidharth/Coding/Projects/BTP/nodes/app/rag/))**:
  * [chroma_db.py](file:///home/sidharth/Coding/Projects/BTP/nodes/app/rag/chroma_db.py): `ChromaStore` handles vector storage, cosine similarity queries, and raw document matrix extraction.
  * [indexer.py](file:///home/sidharth/Coding/Projects/BTP/nodes/app/rag/indexer.py): `DocumentIndexer` splits `.md`/`.txt` files with `RecursiveCharacterTextSplitter` (chunk size 500, overlap 50).
  * [profiler.py](file:///home/sidharth/Coding/Projects/BTP/nodes/app/rag/profiler.py): `NodeProfiler` extracts normalized global centroid $C_i$ and multi-cluster profile $P_i$.
* **Flower Node Client ([main.py](file:///home/sidharth/Coding/Projects/BTP/nodes/app/main.py))**:
  * `FedRAGNodeClient` (`flwr.client.NumPyClient`): Exposes `get_properties()` for registration profiling, `fit()` for local vector search, and `evaluate()` for node health checks.
* **Node Registry & Data ([nodes/nodes.yaml](file:///home/sidharth/Coding/Projects/BTP/nodes/nodes.yaml))**:
  * Defines 4 default domain nodes: `node-1` (finance), `node-2` (medical), `node-3` (legal), and `node-4` (tech/computer science) with local data files in [nodes/data/](file:///home/sidharth/Coding/Projects/BTP/nodes/data/).

#### D. Frontend & Operations
* **Streamlit Interface ([frontend/app.py](file:///home/sidharth/Coding/Projects/BTP/frontend/app.py))**: Web dashboard for querying the FedRAG cluster, viewing synthesized answers, inspecting federated evidence chunks per node, and monitoring routing telemetry.
* **Containerization ([docker-compose.yml](file:///home/sidharth/Coding/Projects/BTP/docker-compose.yml))**: Orchestrates Coordinator, 4 Node containers, Streamlit frontend, and connects to host Ollama instance via `host.docker.internal`.
* **Workspace Tooling ([setup_workspace.sh](file:///home/sidharth/Coding/Projects/BTP/setup_workspace.sh))**: Idempotent bash script initializing `/workspace/chroma` and `/workspace/data` directories.

---

---

## 54 Verification & Test Suite Status

* **Scaffolding Unit Suite ([tests/test_phase1_scaffolding.py](file:///home/sidharth/Coding/Projects/BTP/tests/test_phase1_scaffolding.py))**: Passed (3/3 tests verified local SentenceTransformer embeddings, metadata preservation, and binary embedding array roundtrips over Flower serialization).
* **Phase 3 Test Stubs**:
  * End-to-end broadcast & TASR convergence suites (`tests/e2e/ test_broadcast_rag.py`, `test_tasr_convergence.py`).
  * Malicious node mock simulation (`tests/mocks/malicious_node.py`).
  * Coordinator API & Node Indexer unit tests (`coordinator/tests/test_api.py`, `nodes/tests/test_indexer.py`).

---

## 5. Next Steps & Phase 3 Roadmap

1. **TASR Trust Engine Integration**:
   * Implement feedback functions $f_{\text{rel}}, f_{\text{cons}}, f_{\text{agr}}$ in `coordinator/app/tasr/feedback.py` using returned document embedding matrices and registered centroids.
   * Implement node trust update logic, cold-start discount $s_i$, and soft-gating function $g(x)$ in `coordinator/app/tasr/router.py`.
   * Replace broadcast query routing with dynamic, trust-weighted node selection (`TASRRouter`).
2. **Adversarial Security Testing**:
   * Implement malicious node behavior in `tests/mocks/malicious_node.py` (returning ungrounded chunks or fabricated document vectors).
   * Verify TASR defense modes (`rel_cons_agr`, `rel_cons`, `none`) and reputation degradation in `tests/e2e/test_tasr_convergence.py`.
3. **Benchmark & Performance Evaluation**:
   * Measure retrieval accuracy, bandwidth consumption, and latency across Broadcast vs. TASR routing strategies.
