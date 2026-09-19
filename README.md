# BTP
A Federated Retrieval-Augmented Generation (FedRAG) system with Trust-Aware Secure Routing (TASR) built on FastAPI, ChromaDB, LangChain, and Flower (`flwr`).
---

## Project Overview

* **Architecture**: Centralized Coordinator service with distributed edge nodes communicating over Flower custom messaging protocols.
* **Routing Strategy**: Trust-Aware Secure Routing (TASR per arXiv:2605.28112) using three-signal feedback (`u_rel`, `u_cons`, `u_agr`) to suppress malicious or hijacked nodes.
* **Vector Store**: Local ChromaDB instances deployed per node with disjoint domain corpora.
* **Embeddings**: Provider-agnostic embedding interface defaulting to local `all-MiniLM-L6-v2` (384 dimensions).
* **Generation Engine**: Local LLM execution via Ollama.

---

## Directory Structure

```text
BTP/
├── .gitignore
├── plan.md
├── README.md
├── memory.md
├── docker-compose.yml
├── pyproject.toml
├── poetry.lock
│
├── shared/
│   ├── __init__.py
│   ├── config.yaml
│   ├── config.py                  # Pydantic Settings base loader
│   ├── auth/
│   │   ├── __init__.py
│   │   └── security.py            # API key validation and TLS/mTLS utilities
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py                # Provider-agnostic embedding interface
│   │   ├── local.py               # HuggingFace / SentenceTransformers (default)
│   │   └── openai_cohere.py       # OpenAI / Cohere clients
│   ├── flower/
│   │   ├── __init__.py
│   │   ├── messages.py            # QueryIns, QueryRes, RegisterIns, RegisterRes
│   │   └── serialization.py       # Flower RecordDict & ArrayRecord serializers
│   └── schemas/
│       ├── __init__.py
│       ├── common.py              # Status, Error, Health schemas
│       ├── document.py            # DocumentChunk, SearchResult schemas
│       └── tasr.py                # NodeProfile, TASRTrustState schemas
│
├── coordinator/
│   ├── Dockerfile
│   ├── config.yaml                # Router settings, TASR hyperparameters
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI gateway entry point
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py            # Authentication & service dependencies
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       └── endpoints.py   # /query, /nodes, /health endpoints
│   │   ├── flwr_server/
│   │   │   ├── __init__.py
│   │   │   ├── server.py          # Flower ServerApp runner
│   │   │   ├── strategy.py        # Flower Strategy wrapping TASR routing
│   │   │   └── client_manager.py  # Node registry & health tracking
│   │   ├── tasr/
│   │   │   ├── __init__.py
│   │   │   ├── router.py          # TrustAwareRouter implementation
│   │   │   ├── feedback.py        # Multi-signal feedback (rel, cons, agr)
│   │   │   └── db.py              # SQLite trust state persistence
│   │   └── synthesis/
│   │       ├── __init__.py
│   │       ├── aggregator.py      # Trust-weighted chunk merging
│   │       └── ollama_client.py   # Central Ollama LLM client
│   └── tests/
│       ├── test_api.py
│       └── test_tasr_router.py
│
├── nodes/
│   ├── Dockerfile
│   ├── config.yaml                # Node ports, storage paths, Ollama endpoint
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # Node bootstrap & Flower ClientApp runner
│   │   ├── flwr_client/
│   │   │   ├── __init__.py
│   │   │   ├── client.py          # ClientApp message handlers (Query, Register, Health)
│   │   │   └── registration.py    # Startup registration payload builder
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── chroma_db.py       # ChromaDB vector store client
│   │   │   ├── indexer.py         # Document chunking & ingestion pipeline
│   │   │   ├── profiler.py        # Centroid C_i & cluster profile P_i generator
│   │   │   └── pipeline.py        # LangChain local retrieval pipeline
│   │   └── local_llm/
│   │       ├── __init__.py
│   │       └── ollama_engine.py   # Local Ollama wrapper
│   ├── data/                      # Disjoint document stores for nodes
│   │   ├── node_finance/
│   │   ├── node_health/
│   │   └── node_tech/
│   └── tests/
│       ├── test_indexer.py
│       └── test_flwr_client.py
│
└── tests/
    ├── test_phase1_scaffolding.py # Phase 1 serialization and embeddings test
    ├── e2e/
    │   ├── test_broadcast_rag.py
    │   └── test_tasr_convergence.py # Adversary trust decay & HR@K evaluation
    └── mocks/
        └── malicious_node.py        # Hijacking/adversarial mock node
        
```