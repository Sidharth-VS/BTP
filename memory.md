# Project Memory: FedRAG with TASR Routing
**Phase**: Phase 1 - Foundation & Communication Scaffolding  
**Status**: Completed & Verified  
**Runtime Environment**: WSL (Fedora Linux), Conda (`fed_lab`), Python 3.10.20, Poetry  

---

## 1. Phase Overview & Goals
Phase 1 established the monorepo foundation, execution runtime, dependency baseline, and inter-process messaging protocols required for a Federated Retrieval-Augmented Generation (FedRAG) system with Trust-Aware Secure Routing (TASR). The core objective was ensuring that the central Coordinator service and distributed Edge Nodes share typed data contracts and an error-free serialization layer over Flower (`flwr`) custom messaging.

---

## 2. Environment & Dependency Configuration

### Runtime Setup
* **Isolated Environment**: Configured an isolated Conda environment (`fed_lab`) running Python 3.10 to prevent dependency conflicts with existing workspace virtual environments.
* **Poetry Configuration**: Directed Poetry to bind to the active Conda interpreter (`poetry config virtualenvs.prefer-active-python true`) and configured custom request timeouts (`export POETRY_REQUESTS_TIMEOUT=300`) to prevent read timeouts on large binary wheel downloads.

### Pinned Dependencies (`pyproject.toml`)
* `python`: `>=3.10,<3.12`
* `flwr`: `^1.8.0` (Resolved to `flwr 1.30.0`)
* `fastapi`: `^0.111.0`
* `uvicorn[standard]`: `^0.30.0`
* `chromadb`: `^0.5.0`
* `langchain`: `^0.2.0`
* `langchain-community`: `^0.2.0`
* `sentence-transformers`: `^3.0.0`
* `pydantic`: `^2.7.0`
* `pydantic-settings`: `^2.3.0`
* `ollama`: `^0.2.1`
* `numpy`: `^1.26.0` (Pinned to `1.26.4`)
* `onnxruntime`: `<1.24.0` (Explicitly constrained to resolve binary ABI platform incompatibilities on Linux x86_64)
* `pytest` & `pytest-asyncio`: Testing framework

---

## 3. Implemented Components & Code Artifacts

### A. Repository & Monorepo Architecture
Structured the repository into decoupled service directories and a central shared library:
* `shared/`: Common schemas, embedding models, configuration loaders, and Flower serialization adapters.
* `coordinator/`: FastAPI gateway, Flower `ServerApp`, TASR trust management engine, and LLM synthesis.
* `nodes/`: Flower `ClientApp`, ChromaDB vector store, LangChain RAG pipeline, and local profiler.
* `tests/`: Unit tests, scaffolding verification, and end-to-end evaluation suites.

### B. Shared Data Contracts (`shared/schemas/`)
* **`shared/schemas/document.py`**:
  * `DocumentChunk`: Document segment schema containing unique `id`, text `content`, vector `embedding`, arbitrary `metadata`, originating `node_id`, and UTC `timestamp`.
  * `SearchResult`: Retrieved context chunk containing textual `content`, similarity `score`, and dictionary `metadata`.
* **`shared/schemas/tasr.py`**:
  * `NodeProfile`: Registration payload tracking node identifier, gRPC host address, detected domain, global centroid vector $C_i$, sample embeddings, and multi-centroid cluster profiles $P_i$[cite: 1, 4].
  * `TASRTrustState`: Paper-compliant trust tracking model maintaining retrieval relevance ($u_{\text{rel}}$), consistency ($u_{\text{cons}}$), agreement ($u_{\text{agr}}$), cold-start discount ($s_i$), update frequency, and rolling history.

### C. Provider-Agnostic Embedding Interface (`shared/embeddings/`)
* **`shared/embeddings/base.py`**: Declared abstract base class `BaseEmbeddingProvider` enforcing `embed_query()`, `embed_documents()`, and dimensionality properties.
* **`shared/embeddings/local.py`**: Implemented `LocalSentenceTransformerEmbeddings` utilizing `sentence-transformers` with model `all-MiniLM-L6-v2`. It outputs normalized 384-dimensional embeddings locally without relying on external API rate limits.

### D. Flower Custom Messaging & Serialization Protocol (`shared/flower/`)
* **`shared/flower/messages.py`**:
  * `QueryIns`: Query instruction passing query text, retrieval parameter `top_k`, and metadata filter mappings.
  * `QueryRes`: Query response carrying retrieved `SearchResult` records, execution telemetry, static trust score, and retrieved document embedding matrices (`List[List[float]]`) required for TASR post-routing feedback.
  * `RegisterIns`: Node registration payload sending document centroid $C_i$ and multi-cluster centroids $P_i$.
  * `RegisterRes`: Confirmation response returning assigned node status and configuration.
* **`shared/flower/serialization.py`**:
  * Bi-directional serializer interfacing typed dataclasses with Flower's native `RecordDict`, `ConfigRecord`, and `ArrayRecord` structures.

---

## 4. Engineering Challenges & Root-Cause Debugging Log

During Phase 1 test execution, multiple breaking changes and serialization edge cases were diagnosed and patched:

1. **Package Discovery & Directory Resolution**:
   * *Issue*: `poetry run pytest` failed with `ModuleNotFoundError: No module named 'shared'`.
   * *Root Cause*: The repository was operating inside a nested folder structure (`~/BTP/BTP`), and the editable package definitions in `pyproject.toml` were not registered in the active virtual environment.
   * *Resolution*: Executed `poetry install` inside `~/BTP/BTP` to link `shared`, `coordinator`, and `nodes` packages directly into the active Python site-packages.

2. **Flower 1.30 Record API Migration**:
   * *Issue*: `AttributeError: 'RecordSet' object has no attribute 'set_configs_record'`.
   * *Root Cause*: Flower 1.30 deprecated `RecordSet` in favor of `RecordDict` and replaced setter functions with direct dictionary indexing on `configs_records` and `array_records`.
   * *Resolution*: Refactored `FlowerMessageSerializer` to directly manipulate `RecordDict.configs_records` using `ConfigRecord` instances and `RecordDict.array_records` using `ArrayRecord`.

3. **Array Type Validation Failure**:
   * *Issue*: `TypeError: Invalid arguments for Array. Expected either a PyTorch tensor, a NumPy ndarray, or explicit dtype/shape/stype/data values`.
   * *Root Cause*: Attempting manual byte serialization (`arr.tobytes()`) without passing internal serialization types (`stype`) rejected by Flower's new constructor.
   * *Resolution*: Fed NumPy `ndarray` objects directly into Flower's constructor (`Array(arr)`), allowing Flower to manage the array encapsulation natively.

4. **Numpy Byte Header Deserialization Failure**:
   * *Issue*: `ValueError: cannot reshape array of size 800 into shape (2,384)`.
   * *Root Cause*: Flower 1.30 serializes `Array(arr)` using the binary `.npy` format, appending a 32-byte header containing array metadata (768 bytes of float32 data + 32-byte header = 800 bytes). Reading raw bytes using `np.frombuffer()` parsed the header as numerical data, altering the total byte length.
   * *Resolution*: Replaced `np.frombuffer()` with `np.load(io.BytesIO(flwr_arr.data))`, correctly consuming the `.npy` magic header and extracting the original `(2, 384)` matrix without loss.

---

## 5. Verification & Test Suite Results

Test execution ran against `tests/test_phase1_scaffolding.py`:

```bash
poetry run pytest tests/test_phase1_scaffolding.py

```

### Results Summary
* Collected Tests: 3
* Passed: 3 (100% Pass Rate)
* Execution Time: 10.74s

### Validated Guarantees
1. test_embedding_pipeline: Confirmed the local embedding model loads cleanly, processes raw queries, and outputs vectors of dimension 384.
2. test_query_serialization_roundtrip: Proved that client queries and structured dictionary filters are preserved across Flower configuration records without mutation.
3. test_query_res_serialization_with_arrays: Confirmed that multi-dimensional document embedding matrices, floating-point processing latencies, and search chunk payloads survive binary serialization and deserialization over the network contract.

---

## 6. Readiness for Phase 2
The core shared infrastructure, data types, local embedding models, and networking contracts are fully established and validated. 
The repository is ready to proceed to Phase 2: Federated Node Implementation (local ChromaDB ingestion, document centroid extraction, and local LangChain retrieval loops).
