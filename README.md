# FedRAG

FedRAG is a prototype federated retrieval-augmented generation system. A FastAPI
coordinator sends searches over Flower gRPC to independent retrieval nodes. Each
node indexes its own documents in a persistent ChromaDB collection; the
coordinator ranks returned evidence using Trust-Aware Secure Routing (TASR) and
can optionally ask a local Ollama model to synthesize an answer.

## What the system does

1. A client submits a query to the coordinator's REST API.
2. The coordinator embeds the query with ChromaDB's local
   `all-MiniLM-L6-v2` ONNX embedding function (384 dimensions).
3. TASR selects up to three primary nodes using profile similarity and current
   trust weights. Periodic exploration can add an extra node.
4. Selected nodes run local ChromaDB searches and return matching text and
   embeddings over Flower.
5. The coordinator re-embeds returned text, calculates relevance,
   profile-consistency, and peer-agreement feedback, then updates node trust.
   Evidence is ranked by retrieval similarity multiplied by node trust.
6. If requested, the coordinator sends the highest-ranked primary-node evidence
   to Ollama. Otherwise, it returns the retrieved evidence without an answer.

The coordinator also performs hardware attestation and enforces a per-host node
quota when a node connects. Node and trust registry data is persisted under the
configured coordinator workspace path.

## Repository layout

```text
 BTP/
 ├── .env.example
 ├── .gitignore
 ├── README.md
 ├── docker-compose.yml
 ├── memory.md
 ├── plan.md
 ├── pyproject.toml
 ├── requirements.txt
 ├── setup_workspace.sh
 ├── tasr_implementation_plan.md
 ├── coordinator/
 │   ├── Dockerfile
 │   ├── config.yaml
 │   ├── app/
 │   │   ├── __init__.py
 │   │   ├── main.py
 │   │   ├── api/
 │   │   │   ├── __init__.py
 │   │   │   ├── deps.py
 │   │   │   └── v1/
 │   │   │       ├── __init__.py
 │   │   │       └── endpoints.py
 │   │   ├── flwr_server/
 │   │   │   ├── __init__.py
 │   │   │   ├── admission.py
 │   │   │   ├── client_manager.py
 │   │   │   ├── server.py
 │   │   │   └── strategy.py
 │   │   ├── synthesis/
 │   │   │   ├── __init__.py
 │   │   │   ├── aggregator.py
 │   │   │   └── ollama_client.py
 │   │   └── tasr/
 │   │       ├── __init__.py
 │   │       ├── db.py
 │   │       ├── feedback.py
 │   │       └── router.py
 │   └── tests/
 │       ├── test_api.py
 │       └── test_tasr_router.py
 ├── frontend/
 │   ├── Dockerfile
 │   ├── app.py
 │   └── pages/
 │       └── 1_Admin_Dashboard.py
 ├── nodes/
 │   ├── Dockerfile
 │   ├── nodes.yaml
 │   ├── app/
 │   │   ├── __init__.py
 │   │   ├── main.py
 │   │   ├── flwr_client/
 │   │   │   ├── __init__.py
 │   │   │   ├── client.py
 │   │   │   └── registration.py
 │   │   └── rag/
 │   │       ├── __init__.py
 │   │       ├── chroma_db.py
 │   │       ├── indexer.py
 │   │       └── profiler.py
 │   ├── data/
 │   │   ├── node_finance/
 │   │   │   └── finance_basics.md
 │   │   ├── node_legal/
 │   │   │   └── legal_basics.md
 │   │   └── node_medical/
 │   │       └── medical_basics.md
 │   └── tests/
 │       ├── test_flwr_client.py
 │       └── test_indexer.py
 ├── scripts/
 │   ├── assign_textbooks.py
 │   ├── patch_centroids.py
 │   ├── prepare_stackexchange.py
 │   ├── scale_cluster.py
 │   └── test_tasr_stream.py
 ├── shared/
 │   ├── __init__.py
 │   ├── config.py
 │   ├── config.yaml
 │   ├── auth/
 │   │   ├── __init__.py
 │   │   ├── attestation.py
 │   │   └── security.py
 │   ├── embeddings/
 │   │   ├── __init__.py
 │   │   ├── base.py
 │   │   ├── local.py
 │   │   └── openai_cohere.py
 │   ├── flower/
 │   │   ├── __init__.py
 │   │   ├── messages.py
 │   │   └── serialization.py
 │   └── schemas/
 │       ├── __init__.py
 │       ├── common.py
 │       ├── document.py
 │       └── tasr.py
 └── tests/
     ├── test_admission_manager.py
     ├── test_attestation.py
     ├── test_phase1_scaffolding.py
     ├── e2e/
     │   ├── test_broadcast_rag.py
     │   └── test_tasr_convergence.py
     └── mocks/
         └── malicious_node.py
```

`workspace/` contains generated ChromaDB files, node documents, logs and
coordinator state. It is a runtime directory and is excluded from Git.

## Prerequisites

- Python 3.10 or 3.11 for local execution (the Poetry project requires
  `>=3.10,<3.12`).
- Docker Engine and the Docker Compose plugin for the containerized setup.
- Ollama and the configured model if you want generated answers. Retrieval can
  be run with synthesis disabled.

## Prepare the workspace and data

Install the project dependencies and create the runtime directories:

```bash
poetry install
bash setup_workspace.sh
```

The setup script reads `nodes/nodes.yaml`, creates `workspace/chroma/<node-id>`
and `workspace/data/<node-id>`, and copies files from
`nodes/data/<node-id>` when that exact source directory exists. The sample
corpora in this repository are instead stored in `nodes/data/node_finance`,
`nodes/data/node_legal`, and `nodes/data/node_medical`; copy them into the
configured node data directories (or add your own documents there) before
starting the services. For example:

```bash
mkdir -p workspace/data/node-1 workspace/data/node-2 workspace/data/node-3
cp nodes/data/node_finance/finance_basics.md workspace/data/node-1/
cp nodes/data/node_legal/legal_basics.md workspace/data/node-2/
cp nodes/data/node_medical/medical_basics.md workspace/data/node-3/
```

Each node indexes `.md`, `.txt`, and `.jsonl` files from its configured data
directory at startup. Review `nodes/nodes.yaml` if you want the sample nodes to
report finance, legal, and medical domains rather than the defaults currently
configured there. `node-10` in the default Compose stack is an adversarial test
client, not a regular document-indexing node.

## Run with Docker Compose

Start the coordinator, Streamlit frontend, nine regular nodes (`node-1` through
`node-9`), and the adversarial test node (`node-10`):

```bash
docker compose up --build
```

The node containers wait for the coordinator health check. Initial image builds
and first-time embedding model downloads may take a while.

Each node container only sees its own corpus: `./workspace/data/<node-id>` is
bind-mounted read-only at `/workspace/data/<node-id>`, while `nodes/app` and
`nodes/nodes.yaml` are mounted separately and the shared node image excludes
`nodes/data` via `.dockerignore`. Use `docker compose up -d --force-recreate`
after changing these mounts so running containers pick up the new definition.

| Service | Address |
| --- | --- |
| Query UI | <http://localhost:8501> |
| Coordinator API | <http://localhost:8000> |
| Interactive API documentation | <http://localhost:8000/docs> |
| Flower gRPC server | `localhost:9091` |

Stop the services with:

```bash
docker compose down
```

The bind-mounted `workspace/` data remains on disk after stopping the
containers. The Compose stack includes a simulated malicious node to exercise
the trust-routing flow; do not treat this demo configuration as a production
deployment.

## Run locally

After installing dependencies and preparing data as described above, start the
coordinator:

```bash
COORDINATOR_CONFIG=coordinator/config.yaml \
OLLAMA_BASE_URL=http://localhost:11434 \
OLLAMA_MODEL=llama3.2:3b \
python -m coordinator.app.main
```

The coordinator runs the REST API and Flower server in the same process. To
start regular nodes, open separate terminals and set `NODE_ID` for an ID in
`nodes/nodes.yaml`:

```bash
NODE_ID=node-1 SERVER_ADDRESS=localhost:9091 python -m nodes.app.main
NODE_ID=node-2 SERVER_ADDRESS=localhost:9091 python -m nodes.app.main
NODE_ID=node-3 SERVER_ADDRESS=localhost:9091 python -m nodes.app.main
```

The node's `SERVER_ADDRESS` defaults to the container hostname
`coordinator:9091`; set it to `localhost:9091` for a local, non-containerized
coordinator. Local synthesis requires Ollama listening at `OLLAMA_BASE_URL` with
the selected model available (for example, `ollama pull llama3.2:3b`). The query
API can also skip Ollama with `"use_llm": false`.

## Query the API

Submit a query to `POST /api/v1/query`:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is portfolio diversification?",
    "top_k": 5,
    "use_llm": true
  }'
```

Request fields:

| Field | Default | Description |
| --- | --- | --- |
| `query` | required | Query text, 1–10,000 characters. |
| `top_k` | `5` | Maximum retrieved chunks per queried node (1–100). |
| `filters` | `{}` | Optional ChromaDB metadata filters. |
| `use_llm` | `true` | Generate an answer with Ollama; set false for evidence only. |
| `routing_strategy` | `broadcast` | Accepted by the request schema, but the current query handler does not use this field to select a routing mode. TASR routing is used when profiles are registered. |

The response contains `answer` (or `null` when synthesis is disabled),
`sources`, and `routing_info`. Each source includes its text, retrieval
similarity (`score`), `node_id`, effective trust (`trust_score`), and
similarity-times-trust (`rank_score`). `routing_info` reports the nodes that
returned results and the current trust weights.

Other coordinator endpoints:

| Method and path | Purpose |
| --- | --- |
| `GET /` | Basic service status. |
| `GET /api/v1/health` | Coordinator and registered/healthy node counts. |
| `GET /api/v1/nodes` | Registered nodes and trust telemetry. |
| `GET /api/v1/tasr/summary` | TASR configuration, query count and trust histories. |
| `GET /api/v1/attestation/summary` | Admission and attestation registry summary. |
| `POST /api/v1/nodes/attest` | Pre-flight node attestation. |

## TASR and configuration

Coordinator settings are in `coordinator/config.yaml`; node IDs, domains and
storage defaults are in `nodes/nodes.yaml`. The checked-in coordinator
configuration enables the `rel_cons_agr` defense mode, uses three primary
routing slots, starts trust updates after five queries, and schedules one
exploration node every 20 queries. Edit the TASR block to adjust the defense
mode and its trust, warm-up, exploration, and affinity parameters.

Useful environment variables:

| Variable | Used by | Purpose |
| --- | --- | --- |
| `COORDINATOR_CONFIG` | Coordinator | Path to coordinator YAML configuration. |
| `API_KEY` | Compose/frontend | Value sent by the frontend as `X-API-Key`. |
| `OLLAMA_BASE_URL` | Coordinator | Ollama server URL. |
| `OLLAMA_MODEL` | Coordinator | Model used for answer synthesis. |
| `COORDINATOR_PORT` | Compose | Host port mapped to coordinator HTTP port 8000. |
| `FLOWER_PORT` | Compose | Host port mapped to Flower gRPC port 9091. |
| `NODE_ID` | Node | Required ID declared in `nodes/nodes.yaml`. |
| `SERVER_ADDRESS` | Node | Flower coordinator address; defaults from node config. |
| `NODES_CONFIG` | Node | Node registry YAML path; defaults to `nodes/nodes.yaml`. |
| `DATA_DIR_PREFIX`, `PERSIST_DIR_PREFIX` | Node | Override node data and ChromaDB storage roots. |

**Deployment note:** although an API-key verification helper and API-key/rate
limit configuration values exist, the current REST routes do not attach that
verification dependency. The frontend sends `X-API-Key`, but this is not
currently an access-control boundary. Do not expose the coordinator publicly
without adding and configuring appropriate access controls.

