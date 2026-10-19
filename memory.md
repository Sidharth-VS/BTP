# Project Memory: FedRAG with TASR Routing

**Repository**: `BTP` — Federated Retrieval-Augmented Generation with Trust-Aware Secure Routing (TASR, arXiv:2605.28112)
**Status**: End-to-end operational — hardware-attested federated cluster, centroid+trust routing, three-signal TASR feedback, fingerprint-anchored trust persistence, adversarial test node, Streamlit UI.
**Runtime Environment**: Linux (WSL/Fedora). Working venv at `.venv` (Python 3.14.7, `flwr 1.37.0`, `chromadb 1.5.9`). `pyproject.toml` still declares `python >=3.10,<3.12`, `flwr ^1.8.0`, `chromadb ^0.5.0` — the installed versions are newer than the pin ranges.
**Last updated**: 2026-10-13

---

## 1. System Overview

FedRAG runs a central **Coordinator** (FastAPI REST + Flower `flwr` gRPC server) against **10 edge nodes** (`node-1` … `node-10`) and a **Streamlit frontend**, with a central **Ollama** LLM for synthesis.

Topology (from `nodes/nodes.yaml` + `docker-compose.yml`):
- `node-1` … `node-9`: honest nodes, `domain: general` (data actually spans mixed StackExchange domains per `scripts/prepare_stackexchange.py`).
- `node-10`: the adversarial/malicious node, `domain: medical` — runs `tests/mocks/malicious_node.py` via the compose `command` override.
- Coordinator + frontend + 10 nodes, all on the `fedrag-net` bridge network.

At a glance, what is live:
- Flower `NumPyClient` protocol for node ↔ coordinator (`get_properties` / `fit` / `evaluate`).
- Per-node ChromaDB vector store + local `all-MiniLM-L6-v2` embeddings (384-d) + document indexing & profiling (global centroid `C_i`, multi-centroid profile `P_i`).
- **Centroid-based TASR routing**: query embedding ↔ node centroid cosine, reweighted by effective trust `τ_i`, top-`k_route` selection.
- **Three-signal feedback** (`f_rel`, `f_cons`, `f_agr`) computed from returned document embedding matrices, with median-thresholded decay/recovery trust updates.
- **Two-tier hardware attestation + Sybil defense** (REST pre-flight + gRPC handshake) with per-host node quotas.
- **Fingerprint-anchored trust persistence** (survives disconnect/rejoin and reboots; resists node-id whitewashing).
- Central Ollama answer synthesis and a Streamlit explorer + TASR admin telemetry dashboard.
- An implemented **malicious node mock** that forges a medical centroid and injects poisoned evidence (see §9).

---

## 2. Environment & Dependencies

Declared in `pyproject.toml` (Poetry): `flwr`, `fastapi`, `uvicorn[standard]`, `chromadb`, `langchain`/`langchain-community`, `sentence-transformers`, `pydantic` + `pydantic-settings`, `ollama`, `numpy`, `pyyaml`, `httpx`, `onnxruntime<1.24`. Dev group: `pytest`, `pytest-asyncio`.

Actually installed in `.venv` (verified): Python 3.14.7, `flwr 1.37.0`, `chromadb 1.5.9`. (`ollama` is **not** installed in the local venv, so `coordinator.app.main` cannot be imported locally without it — see §12.)

Config sources:
- `coordinator/config.yaml` — `server`, `flower`, `api`, `synthesis`, `tasr`, `trust_db`, `attestation` blocks.
- `shared/config.yaml` — embedding model / dimension / gRPC port / API-key header defaults.
- `nodes/nodes.yaml` — per-node registry + defaults (server address, embedding model, Chroma/data prefixes, collection prefix).
- `.env.example` — `API_KEY`, host/ports, `OLLAMA_*`.
- `docker-compose.yml` — orchestrates coordinator, frontend, and `node-1` … `node-10`.

---

## 3. Implemented Components & Architecture

```
BTP/
├── shared/
│   ├── schemas/           document.py, tasr.py, common.py
│   ├── embeddings/        base.py (ABC), local.py (default), openai_cohere.py (EMPTY)
│   ├── auth/              attestation.py, security.py
│   ├── flower/            messages.py, serialization.py  (legacy message API — used by tests only)
│   ├── config.py          AppSettings.from_yaml()
│   └── config.yaml
├── coordinator/
│   └── app/
│       ├── main.py                    dual-thread runtime + CoordinatorConfig/load_config
│       ├── api/deps.py                global FlowerServer singleton
│       ├── api/v1/endpoints.py        REST endpoints + module-level tasr_router/embedder
│       ├── flwr_server/
│       │   ├── server.py              FlowerServer (restart loop) + get_node_registry()
│       │   ├── strategy.py            FedRAGStrategy + QueryBroker
│       │   ├── client_manager.py      LoggingClientManager (handshake, admission, trust persist)
│       │   └── admission.py           NodeAdmissionManager (quotas, leases, fingerprint trust)
│       ├── tasr/
│       │   ├── router.py              TrustAwareRouter + ClientTrustState
│       │   ├── feedback.py            TASRFeedbackEngine (rel/cons/agr)
│       │   └── db.py                  TASRDatabase (JSON persistence — DEFINED BUT UNUSED)
│       └── synthesis/                 ollama_client.py (implemented), aggregator.py (stub)
├── nodes/
│   └── app/
│       ├── main.py                    FedRAGNodeClient (NumPyClient) + bootstrap
│       ├── rag/                       chroma_db.py, indexer.py, profiler.py
│       └── flwr_client/               client.py (LEGACY ClientApp path), registration.py (docstring only)
├── frontend/              app.py (query explorer), pages/1_Admin_Dashboard.py (TASR analytics)
├── tests/                 test_phase1_scaffolding.py, test_attestation.py, test_admission_manager.py,
│                          mocks/malicious_node.py, e2e/ (EMPTY)
├── scripts/               scale_cluster.py, prepare_stackexchange.py, patch_centroids.py,
│                          assign_textbooks.py, test_tasr_stream.py
├── base/                  vendored reference repo (see §10)
├── docker-compose.yml, setup_workspace.sh, Dockerfiles, .env.example
└── plan.md, README.md, tasr_implementation_plan.md, memory.md
```

### A. Shared infrastructure (`shared/`)
- **Schemas** — `document.py`: `DocumentChunk`, `SearchResult`. `tasr.py`: `NodeProfile`, `TASRTrustState`. `common.py`: `QueryRequest/QueryResponse`, `SourceResult`, `RoutingInfo`, `NodeInfo`, `NodeListResponse`, `HealthResponse`, `NodeStatus`, `RoutingStrategy`, `AttestationRequest`, `AttestationResponse`.
- **Embeddings** — `base.py` defines the provider ABC (`embed_query`, `embed_documents`, `dimension`). `local.py` wraps ChromaDB's `DefaultEmbeddingFunction` (ONNX `all-MiniLM-L6-v2`, 384-d, normalized), with ONNX/OpenMP thread caps and an optional warmup. `openai_cohere.py` is empty.
- **Auth** — `security.py`: `verify_api_key` FastAPI dependency (X-API-Key). `attestation.py`: `HardwareCollector` (DMI UUID, host machine-id via `/host/etc/machine-id` mount, physical-NIC MAC, CPU info), `generate_host_fingerprint()` (SHA-256 over canonical host identifiers), `NodeAttestationPayload` dataclass with `to_dict`/`from_dict`, `create_attestation_payload()`, `verify_attestation_payload()` (fingerprint length + timestamp-skew freshness).
- **Flower messaging** — `messages.py` (`QueryIns`, `QueryRes`, `RegisterIns`, `RegisterRes`) + `serialization.py` (`FlowerMessageSerializer` converting to/from `RecordDict`/`ArrayRecord`, `.npy` array round-trips). **Legacy**: exercised by tests, but the live node path is the `NumPyClient` in `nodes/app/main.py`.

### B. Coordinator (`coordinator/app/`)
- **`main.py`** — `CoordinatorConfig` (pydantic-settings) + `load_config()` (maps only `server`, `flower`, and `attestation`; see §12). Runs `FlowerServer` on the main thread (signal-safe) and `uvicorn` on a daemon thread; injects the server into `deps`.
- **`api/deps.py`** — `set_flower_server` / `get_flower_server` / `get_admission_manager`.
- **`api/v1/endpoints.py`** — module-level singletons `tasr_router = TrustAwareRouter(k_route=6)`, `embedder`, `synthesizer`.
  - `POST /api/v1/query`: registers any newly seen authorized nodes into the router → embeds the query → `route_query()` → dispatches to `feedback_targets` via `QueryBroker` → parses chunks/embeddings → computes `f_rel`/`f_cons`/`f_agr` → `update_trust()` → persists trust by hardware fingerprint → synthesizes answer → returns `QueryResponse` with routing telemetry.
  - `POST /api/v1/nodes/attest` — pre-flight hardware attestation (403 on Sybil/quota rejection).
  - `GET /api/v1/attestation/summary`, `GET /api/v1/nodes` (per-node trust + components), `GET /api/v1/tasr/summary` (defense mode, warmup state, per-node `u_rel/u_cons/u_agr/s_i`, reputation history), `GET /api/v1/health`.
- **`flwr_server/strategy.py`** — `QueryBroker` (thread-safe queue + `threading.Event` bridge between REST and the Flower round loop). `FedRAGStrategy`:
  - `configure_fit`: picks up a pending query, dispatches `FitIns(action="query")` only to `target_nodes` (with an authorization check and a broadcast fallback if no registry match).
  - `aggregate_fit`: deserializes per-node `results_json` + raw `float32` embedding matrices (reshaped to `(-1, 384)`) and publishes them to the broker.
  - `configure_evaluate` / `aggregate_evaluate`: periodic health checks; parse `attestation`, `fingerprint`, `centroid`, `profile_centroids`, `domain`, `doc_embeddings`; **hard-disconnect** on missing/invalid attestation (when admission is enabled).
- **`flwr_server/client_manager.py`** — `LoggingClientManager` extends `SimpleClientManager`. On `register()` it spawns `_handshake_and_authenticate`, which calls `client.get_properties(GetPropertiesIns)` (with a small retry loop for Flower bridge contention), runs admission, stores the node profile in `_node_registry`, and registers the node into `tasr_router` (restoring fingerprint-anchored trust). On `unregister()` it snapshots live trust back to the fingerprint registry (`_persist_trust_state`) and releases the host lease.
- **`flwr_server/admission.py`** — `NodeAdmissionManager`: attestation validation, per-host node quota (`max_nodes_per_host`), live leases, `bind_grpc_cid`, `release_node`, `is_authorized`, fingerprint ↔ node maps, and **fingerprint-anchored trust state** (`get/save_fingerprint_trust_state`), plus a summary with Sybil-rejection counts. Persists to `workspace/coordinator/attestation_registry.json`.
- **`flwr_server/server.py`** — `FlowerServer` wraps `start_server()` in an infinite restart loop; exposes `broker`, `get_node_registry()`, `connected_node_count()`.
- **`tasr/router.py`** — full `TrustAwareRouter` (§4) + `ClientTrustState` wrapper exposing `u_rel/u_cons/u_agr/s_i/feedback_count`.
- **`tasr/feedback.py`** — `TASRFeedbackEngine` (§5).
- **`tasr/db.py`** — `TASRDatabase` JSON persistence helper (`workspace/tasr_state.json`). **Not currently wired into the runtime**; live trust persistence uses the admission manager's fingerprint registry.
- **`synthesis/ollama_client.py`** — `OllamaSynthesizer.synthesize()`: formats top-ranked chunks into a grounded prompt and calls the configured Ollama model. `aggregator.py` remains a Phase-2 stub.

### C. Nodes (`nodes/app/`)
- **`main.py`** — `FedRAGNodeClient(NumPyClient)`:
  - Bootstraps node identity from `nodes/nodes.yaml` (env `NODE_ID`), builds a hardware attestation payload, caches `.identity.json`, and performs the **Tier-1 REST pre-flight** to `/api/v1/nodes/attest` (hard-exit on 403).
  - `get_properties()` — returns `node_id`, `domain`, `attestation`, `fingerprint`, `capabilities`, `centroid`, `profile_centroids`, `doc_embeddings_sample`, `doc_count` (attestation is attached first, independent of Chroma/embedding failures).
  - `fit()` — local Chroma similarity search; returns `[doc_embedding_matrix]`, result count, and metrics (`node_id`, `processing_time`, `results_json`).
  - `evaluate()` — health check; returns the same profile + attestation fields and the doc count.
- **`rag/chroma_db.py`** — `ChromaStore`: persistent ChromaDB, upsert/query (cosine, `score = 1 - dist/2`), `get_all_embeddings()` (flattens numpy → lists).
- **`rag/indexer.py`** — `DocumentIndexer`: JSONL (StackExchange: `title_body` + `upvoted_answer`) and txt/md ingestion via `RecursiveCharacterTextSplitter` (chunk 500 / overlap 50).
- **`rag/profiler.py`** — `NodeProfiler.compute_profile()`: normalized global centroid `C_i` + multi-centroid profile `P_i` (deterministic interval sampling, no sklearn).
- **`flwr_client/client.py`** — a `ClientApp`/`Message`-API version of the node (register + query handlers). **Legacy / not wired** into `main.py`; `registration.py` is a docstring stub.

### D. Frontend
- **`frontend/app.py`** — Streamlit query explorer: answer panel, retrieved evidence per node, and TASR routing telemetry (strategy, selected nodes, τ_i trust table).
- **`frontend/pages/1_Admin_Dashboard.py`** — TASR analytics: node registry, avg network trust, query/warmup state, per-node trust-component bar chart, telemetry table, and `u_rel` trajectory line chart.

### E. Operations
- **`scripts/scale_cluster.py`** — regenerates `nodes/nodes.yaml` and `docker-compose.yml` for 9 honest + 1 malicious node with thread caps (`*_NUM_THREADS=3`), host machine-id mounts, and the `tests.mocks.malicious_node` command for `node-10`.
- **`scripts/prepare_stackexchange.py`** / **`assign_textbooks.py`** — corpus prep / domain assignment.
- **`scripts/patch_centroids.py`** — centroid patching utility.
- **`scripts/test_tasr_stream.py`** — 70-query TASR stress-test client (warmup W=50) that streams queries and prints a trust-divergence table.
- **`setup_workspace.sh`** — idempotent creation of `workspace/chroma` + `workspace/data`.

---

## 4. TASR Trust Model (implemented)

Per-node trust variables in `[0,1]`: `u_rel` (retrieval relevance), `u_cons` (profile consistency), `u_agr` (cross-client agreement), plus feedback count and trajectory histories.

- **Effective trust** (`_effective_trust`):
  `τ_i = s_i · (u_rel_i)^α_r · g_c(u_cons_i) · g_a(u_agr_i)`
  where `s_i = s_0 + (1 - s_0)·(1 - e^{-count/τ_s})` is the cold-start ramp, `g(x) = δ + (1-δ)·x^α` is the bounded soft gate, and the `u_agr` gate applies only in `rel_cons_agr` mode with `top_k ≥ 2`.
- **Defense modes**: `none`, `rel_cons`, `rel_cons_agr` (switchable; gates/updates are skipped accordingly).
- **Router defaults** (used in production because config is not wired — §12): `decay_factor=0.9`, `recovery_factor=1.02`, `min_reputation=0.01`, `warmup_queries=50`, `cold_start_s0=0.7`, `cold_start_tau=30.0`, `docs_for_feedback=5`, `threshold_mode="dynamic"`, `alpha_r=1.0`, `alpha_c=1.0`, `alpha_a=0.5`, `delta_c=0.3`, `delta_a=0.5`, `cons_winner_weight=0.6`, `explore_interval=20`, `explore_extra=1`, `defense_mode="rel_cons_agr"`, `k_route=6`.
- **Registration** (`register_client` / `register_node` alias): normalizes and stores centroid + profile + doc-embedding pool; restores trust from an `initial_trust_state` snapshot, preserves live trust on re-registration without a snapshot, otherwise cold-starts at 1.0.
- **Trust update** (`update_trust`, flexible 3-arg and 4-arg forms): per signal, compute a **dynamic median threshold** (or fixed), then `u ← max(u·γ, floor)` if feedback < threshold else `u ← min(u·γ_rec, 1.0)`. Relevance is the gate for the other two signals: only clients whose `f_rel` clears the relevance threshold can recover `u_cons`/`u_agr`; below-threshold clients decay all three. The `cons`/`agr` thresholds are computed over the relevance-qualified cohort only. Updates only apply **after warmup**; histories are recorded each round. Feedback-count increments even during warmup (drives the cold-start ramp).
- **Exploration**: every `explore_interval` queries, one extra (low-ranked) node is appended to the dispatch set.

---

## 5. Feedback Engine (`f_rel`, `f_cons`, `f_agr`)

`TASRFeedbackEngine` (paper-exact formulations):
1. **Relevance** `f_rel = mean(D_returned · q)` (normalized cosine mean; falls back to chunk scores if embeddings absent).
2. **Consistency** `f_cons`: pick winner centroid `k* = argmax(P_k · q)`, then a `ρ=0.6`-weighted blend of the winner cluster against the remaining clusters, averaged over returned docs.
3. **Agreement** `f_agr`: relevance-qualified peers (`f_rel ≥ median`) are reduced to normalized document centroids; each node's agreement is the peer **reputation-weighted** mean cosine to the others. Nodes outside `B_q` are scored 0.0 (not a neutral 1.0), and when fewer than two comparable qualified peers exist the signal is uninformative and the update is skipped (empty dict).
4. **Missing evidence**: `f_cons` returns 0.0 for an empty evidence/profile set, matching the base reference, so a silent node cannot sit at the trust ceiling.

---

## 6. Hardware Attestation & Sybil Defense

- **Identity**: SHA-256 host fingerprint from DMI UUID + host machine-id (containers bind-mount `/etc/machine-id` → `/host/etc/machine-id:ro`). MAC/CPU components are currently commented out of the fingerprint.
- **Tier 1 (REST pre-flight)**: nodes POST their payload to `/api/v1/nodes/attest`; invalid/freshness-failed or over-quota requests get **403**.
- **Tier 2 (gRPC)**: the coordinator's client-manager handshake re-validates the attestation via `get_properties`, and `aggregate_evaluate` re-validates on health checks. Missing or invalid attestation triggers a **hard disconnect** (`client_manager.unregister`).
- **Quota**: `max_nodes_per_host` (config: `50`; class default `1`). Reconnection refreshes the existing lease.
- **Anti-whitewashing**: trust is anchored to the *fingerprint*, so a renamed node-id on the same host inherits the same (possibly degraded) trust.

---

## 7. Trust Persistence

- Live trust is snapshotted **by hardware fingerprint** on node disconnect and on each query (`save_fingerprint_trust_state`), stored in `workspace/coordinator/attestation_registry.json` under `fingerprints.<fp>.trust_state`.
- On (re)connect, the handshake restores that snapshot into the router via `initial_trust_state`.
- `TASRDatabase` (JSON) exists for the same purpose but is **not wired up**.

---

## 8. Routing (centroid-based)

`TrustAwareRouter.route_query(q_emb)` (called from `/query`):
1. For each **registered** node: base score = `max(P_i · q)` if the node has a multi-centroid profile, else `dot(q, C_i)`.
2. Weighted score = base × `τ_i`.
3. `primary_routed = top-k_route` (k=6), `feedback_targets = primary + exploration extras`.
4. `endpoints.query` dispatches to `feedback_targets` via the broker; `FedRAGStrategy.configure_fit` fans out to those node IDs (falling back to broadcast if none match the registry).

Consequences: routing genuinely discriminates only when **>6 nodes** are registered (≤6 ⇒ effectively everyone is selected); a node registered without a centroid gets `[0.0]*384` and scores 0.

---

## 9. Malicious Node Mock (`tests/mocks/malicious_node.py`)

`MaliciousNodeClient(NumPyClient)` implements the "offline profile forgery + online evidence poisoning" threat model:
- **Offline**: computes a forged medical centroid `C_i` as the (normalized) mean of `PROXY_TEXTS` embeddings and a coherent profile `P_i = [C_i]*5`.
- **Registration**: `get_properties`/`evaluate` send `attestation`, `fingerprint`, `domain`, `centroid`, `profile_centroids`, `doc_embeddings[_sample]`, `doc_count` — under the exact keys the coordinator reads.
- **Online**: `fit(action="query")` returns `POISONED_PAYLOADS` (forged references unrelated to the forged centroid) with high self-reported scores, plus the poisoned doc-embedding matrix.
- **Sybil bypass**: performs the REST pre-flight once, reusing the identical cached attestation payload over gRPC.

Wired as `node-10` (`domain: medical`) via `docker-compose.yml` / `scale_cluster.py`. It fires on most queries while trust is uniform (medical/cancer queries rank it #1); after the 50-query warmup its `u_cons`/`u_rel` decay and it is filtered out — i.e. the defense works. (Fixed/verified on 2026-10-13 — see §13.)

---



