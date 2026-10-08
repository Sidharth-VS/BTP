"""
FedRAG Node service — entry point.

Reads node identity from nodes/nodes.yaml (keyed by NODE_ID env var),
then:
1. Indexes documents from workspace/data/<node_id>/
2. Starts flwr.client.start_client() → gRPC connection to coordinator

Message protocol (NumPyClient):
  get_properties() → registration metadata (node_id, domain, centroid, profile)
  fit()            → query dispatch  (returns chunks + doc embeddings)
  evaluate()       → health check    (returns doc count + status)
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from flwr.client import NumPyClient, start_client

from nodes.app.rag.chroma_db import ChromaStore
from nodes.app.rag.indexer import DocumentIndexer
from nodes.app.rag.profiler import NodeProfiler
from shared.embeddings.local import LocalEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("node")


# ---------------------------------------------------------------------------
# Config loading — reads nodes/nodes.yaml and merges defaults
# ---------------------------------------------------------------------------

def load_node_config(nodes_yaml: str, node_id: str) -> Dict[str, Any]:
    """
    Loads the config for `node_id` from the unified nodes.yaml,
    merging per-node overrides on top of defaults.
    """
    path = Path(nodes_yaml)
    if not path.exists():
        raise FileNotFoundError(f"nodes.yaml not found at {nodes_yaml}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    defaults: Dict[str, Any] = data.get("defaults", {})
    nodes: List[Dict[str, Any]] = data.get("nodes", [])

    match = next((n for n in nodes if n["node_id"] == node_id), None)
    if match is None:
        raise ValueError(
            f"Node '{node_id}' not found in {nodes_yaml}. "
            f"Available: {[n['node_id'] for n in nodes]}"
        )

    # Merge defaults + per-node overrides
    cfg = {**defaults, **match}

    # Resolve paths from prefixes
    prefix_chroma = os.environ.get("PERSIST_DIR_PREFIX") or cfg.get("persist_directory_prefix", "/workspace/chroma")
    prefix_data   = os.environ.get("DATA_DIR_PREFIX") or cfg.get("data_directory_prefix", "/workspace/data")
    prefix_coll   = cfg.get("collection_name_prefix",   "fedrag")

    # If running locally on host outside container, fall back from /workspace to ./workspace
    if not Path("/workspace").exists():
        if prefix_chroma.startswith("/workspace"):
            prefix_chroma = "." + prefix_chroma
        if prefix_data.startswith("/workspace"):
            prefix_data = "." + prefix_data

    cfg.setdefault("persist_directory", f"{prefix_chroma}/{node_id}")
    cfg.setdefault("data_directory",    f"{prefix_data}/{node_id}")
    cfg.setdefault("collection_name",   f"{prefix_coll}_{node_id.replace('-', '_')}")

    return cfg


# ---------------------------------------------------------------------------
# Flower NumPyClient
# ---------------------------------------------------------------------------

class FedRAGNodeClient(NumPyClient):
    """
    fit()           → query dispatch
    get_properties()→ registration (centroid, domain, profile)
    evaluate()      → health check
    """

    def __init__(
        self,
        node_id: str,
        domain: str,
        chroma_store: ChromaStore,
        profiler: NodeProfiler,
        attestation_payload: Optional[Any] = None,
    ) -> None:
        self.node_id = node_id
        self.domain = domain
        self.chroma_store = chroma_store
        self.profiler = profiler
        self.attestation_payload = attestation_payload

    def get_properties(self, config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Sending registration profile to coordinator")

        # Attestation is always set first — completely independent of ChromaDB/embeddings.
        props: Dict[str, Any] = {
            "node_id": self.node_id,
            "domain": self.domain,
        }
        if self.attestation_payload:
            props["attestation"] = json.dumps(self.attestation_payload.to_dict())
            props["fingerprint"] = self.attestation_payload.fingerprint
            print(
                f"📤 [{self.node_id}] Transmitting hardware attestation to coordinator"
                f" (fingerprint: {self.attestation_payload.fingerprint[:16]}...)",
                flush=True,
            )

        # Centroid / embedding profile — failures here must NOT suppress attestation.
        try:
            centroid, profile_centroids = self.profiler.compute_profile()
            all_embs = self.chroma_store.get_all_embeddings()
            props.update({
                "capabilities": json.dumps(["chromadb", "retrieval"]),
                "centroid": json.dumps(centroid),
                "profile_centroids": json.dumps(profile_centroids),
                "doc_embeddings_sample": json.dumps(
                    [list(e) for e in all_embs[:50]]  # normalise numpy→list
                ),
                "doc_count": str(self.chroma_store.collection.count()),
            })
        except Exception as e:
            logger.error("get_properties profile error (attestation still sent): %s", e)
            props["profile_error"] = str(e)

        return props

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return []

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> tuple[List[np.ndarray], int, Dict[str, Any]]:
        query  = str(config.get("query", ""))
        top_k  = int(config.get("top_k", 5))
        filters_raw = config.get("filters", "{}")
        try:
            filters: Optional[Dict] = json.loads(str(filters_raw)) or None
        except json.JSONDecodeError:
            filters = None

        logger.info("Query '%s' (top_k=%d)", query[:60], top_k)
        t0 = time.perf_counter()
        try:
            results, doc_embeddings = self.chroma_store.query(
                query_text=query, top_k=top_k, filters=filters
            )
        except Exception as e:
            logger.error("ChromaDB query error: %s", e)
            return [], 0, {
                "node_id": self.node_id,
                "processing_time": 0.0,
                "results_json": "[]",
                "error": str(e),
            }

        elapsed = time.perf_counter() - t0
        logger.info("Returned %d chunks in %.3fs", len(results), elapsed)

        emb_arr = (
            np.array(doc_embeddings, dtype=np.float32)
            if doc_embeddings
            else np.empty((0, self.chroma_store.embedder.dimension), dtype=np.float32)
        )

        return (
            [emb_arr],
            len(results),
            {
                "node_id": self.node_id,
                "processing_time": float(elapsed),
                "results_json": json.dumps([r.model_dump() for r in results]),
            },
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> tuple[float, int, Dict[str, Any]]:
        # Attestation is attached FIRST — completely independent of ChromaDB.
        # If profile computation below fails, the coordinator must still receive
        # a healthy status + attestation, or it will hard-disconnect the node.
        eval_res: Dict[str, Any] = {
            "status": "healthy",
            "node_id": self.node_id,
            "domain": self.domain,
        }
        if self.attestation_payload:
            eval_res["attestation"] = json.dumps(self.attestation_payload.to_dict())
            eval_res["fingerprint"] = self.attestation_payload.fingerprint

        try:
            count = self.chroma_store.collection.count()
            centroid, profile_centroids = self.profiler.compute_profile()
            all_embs = self.chroma_store.get_all_embeddings()
            eval_res.update({
                "centroid": json.dumps(centroid),
                "profile_centroids": json.dumps(profile_centroids),
                "doc_embeddings": json.dumps(all_embs[:50]),
            })
            return 0.0, count, eval_res
        except Exception as e:
            logger.error("evaluate profile error (attestation still sent): %s", e)
            return 0.0, 0, {**eval_res, "profile_error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    node_id    = os.environ.get("NODE_ID") or ""
    nodes_yaml = os.environ.get("NODES_CONFIG", "nodes/nodes.yaml")

    if not node_id:
        raise RuntimeError("NODE_ID environment variable must be set (e.g. 'node-1')")

    cfg = load_node_config(nodes_yaml, node_id)
    logger.info("Loaded config for '%s' (domain: %s)", node_id, cfg.get("domain"))

    # Generate genuine hardware attestation
    from shared.auth.attestation import create_attestation_payload
    attestation_payload = create_attestation_payload(node_id)
    print(
        f"\n🔑 [Node '{node_id}'] Hardware Fingerprint: {attestation_payload.fingerprint}",
        flush=True,
    )
    print(
        f"💻 [Node '{node_id}'] CPU: {attestation_payload.hardware_summary.get('cpu_model', 'unknown')}"
        f" ({attestation_payload.hardware_summary.get('cpu_cores', 1)} cores)\n",
        flush=True,
    )
    logger.info(
        "Hardware Fingerprint: %s... (CPU: %s, %s cores)",
        attestation_payload.fingerprint[:16],
        attestation_payload.hardware_summary.get("cpu_model", "unknown"),
        attestation_payload.hardware_summary.get("cpu_cores", 1),
    )

    # Cache local identity record
    try:
        cache_dir = Path(cfg["persist_directory"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / ".identity.json", "w", encoding="utf-8") as f:
            json.dump(attestation_payload.to_dict(), f, indent=2)
    except Exception as e:
        logger.debug("Could not cache local identity: %s", e)

    server_address = os.environ.get("SERVER_ADDRESS") or cfg.get("server_address", "coordinator:9091")

    # Tier 1 Pre-Flight Gatekeeper: REST attestation check
    coordinator_rest_url = os.environ.get("COORDINATOR_REST_URL")
    if not coordinator_rest_url:
        host = server_address.split(":")[0] if ":" in server_address else "localhost"
        coordinator_rest_url = f"http://{host}:8000"

    attest_url = f"{coordinator_rest_url}/api/v1/nodes/attest"
    logger.info("Performing pre-flight hardware attestation with %s...", attest_url)

    import urllib.request
    import urllib.error

    def _try_attest(url: str) -> Optional[str]:
        """Attempt REST pre-flight. Returns lease on success, None on soft error, raises SystemExit on 403."""
        try:
            req_data = json.dumps(attestation_payload.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return str(res_data.get("lease_token", ""))
        except urllib.error.HTTPError as err:
            if err.code == 403:
                err_body = err.read().decode("utf-8")
                logger.error("⛔ HARD REJECTION: Coordinator denied admission (Host Quota / Sybil block): %s", err_body)
                raise SystemExit(1)
            logger.debug("Pre-flight HTTP %d from %s: %s", err.code, url, err)
            return None
        except Exception as exc:
            logger.debug("Could not reach %s: %s", url, exc)
            return None

    # Build fallback URL candidates (try original host first, then localhost variants)
    _fallback_bases = [coordinator_rest_url]
    if "localhost" not in coordinator_rest_url and "127.0.0.1" not in coordinator_rest_url:
        _fallback_bases.extend(["http://localhost:8000", "http://127.0.0.1:8000"])

    _lease = None
    for _base in _fallback_bases:
        _lease = _try_attest(f"{_base}/api/v1/nodes/attest")
        if _lease is not None:
            logger.info("✅ Pre-flight attestation approved via %s (lease: %s...)", _base, _lease[:8])
            break
    else:
        logger.warning("Could not reach coordinator REST API for pre-flight at any URL. Proceeding to gRPC connection...")


    # Initialize embedder + ChromaDB
    chroma_store = ChromaStore(
        persist_dir=cfg["persist_directory"],
        collection_name=cfg["collection_name"],
    )
    logger.info("ChromaDB at '%s'", cfg["persist_directory"])

    # Index documents
    data_dir = Path(cfg["data_directory"])
    if data_dir.exists():
        indexer = DocumentIndexer(chroma_store)
        n = indexer.index_directory(data_dir, node_id)
        logger.info("Indexed %d chunks from '%s'", n, data_dir)
    else:
        logger.warning("Data directory '%s' not found — no documents indexed", data_dir)

    # Build profile
    profiler = NodeProfiler(chroma_store)

    # Connect to coordinator via Flower gRPC
    logger.info("Connecting to coordinator at '%s'...", server_address)
    start_client(
        server_address=server_address,
        client=FedRAGNodeClient(
            node_id=node_id,
            domain=cfg.get("domain", "general"),
            chroma_store=chroma_store,
            profiler=profiler,
            attestation_payload=attestation_payload,
        ).to_client(),
        insecure=True,
    )


if __name__ == "__main__":
    main()
