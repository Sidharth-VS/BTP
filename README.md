# BTP

### Folder and Files 
#### Direcctory  Structure
BTP/
├── .gitignore
├── plan.md
├── README.md
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
│   │   ├── messages.py            # QueryIns, QueryRes, RegisterIns, RecordSet wrappers
│   │   └── serialization.py       # Pydantic <-> Flwr RecordSet converters
│   └── schemas/
│       ├── __init__.py
│       ├── common.py              # Status, Error, Health schemas
│       ├── document.py            # DocumentChunk, Metadata
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
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── endpoints.py   # /query, /nodes, /health endpoints
│   │   │   │   └── deps.py        # Authentication & service dependencies
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
    ├── e2e/
    │   ├── test_broadcast_rag.py
    │   └── test_tasr_convergence.py # Adversary trust decay & HR@K evaluation
    └── mocks/
        └── malicious_node.py        # Hijacking/adversarial mock node