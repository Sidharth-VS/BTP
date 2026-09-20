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
from shared.embeddings.local import LocalSentenceTransformerEmbeddings

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
    prefix_chroma = cfg.get("persist_directory_prefix", "/workspace/chroma")
    prefix_data   = cfg.get("data_directory_prefix",    "/workspace/data")
    prefix_coll   = cfg.get("collection_name_prefix",   "fedrag")

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
    ) -> None:
        self.node_id = node_id
        self.domain = domain
        self.chroma_store = chroma_store
        self.profiler = profiler

    def get_properties(self, config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Sending registration profile to coordinator")
        try:
            centroid, profile_centroids = self.profiler.compute_profile()
            all_embs = self.chroma_store.get_all_embeddings()
            return {
                "node_id": self.node_id,
                "domain": self.domain,
                "capabilities": json.dumps(["chromadb", "retrieval"]),
                "centroid": json.dumps(centroid),
                "profile_centroids": json.dumps(profile_centroids),
                "doc_embeddings_sample": json.dumps(all_embs[:50]),
                "doc_count": str(self.chroma_store.collection.count()),
            }
        except Exception as e:
            logger.error("get_properties error: %s", e)
            return {"error": str(e), "node_id": self.node_id}

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
        try:
            count = self.chroma_store.collection.count()
            return 0.0, count, {"status": "healthy", "node_id": self.node_id}
        except Exception as e:
            return 1.0, 0, {"status": "degraded", "node_id": self.node_id, "error": str(e)}


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

    # Initialize embedder + ChromaDB
    embedder = LocalSentenceTransformerEmbeddings(
        model_name=cfg.get("embedding_model_name", "all-MiniLM-L6-v2")
    )
    chroma_store = ChromaStore(
        persist_dir=cfg["persist_directory"],
        collection_name=cfg["collection_name"],
        embedder=embedder,
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
    server_address = cfg.get("server_address", "coordinator:9091")
    logger.info("Connecting to coordinator at '%s'...", server_address)
    start_client(
        server_address=server_address,
        client=FedRAGNodeClient(
            node_id=node_id,
            domain=cfg.get("domain", "general"),
            chroma_store=chroma_store,
            profiler=profiler,
        ).to_client(),
        insecure=True,
    )


if __name__ == "__main__":
    main()
