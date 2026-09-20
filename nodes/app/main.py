"""
FedRAG Node service main entry point.

Startup sequence:
1. Load config from config.yaml
2. Initialize ChromaDB and embed documents from data directory
3. Compute domain profile (centroid C_i, multi-centroid P_i)
4. Start flwr.client.start_client() → connects to coordinator via gRPC
   - get_properties(): registration (sends centroid, domain, profile)
   - fit():            query dispatch (receives QueryIns, returns results + doc_embeddings)
   - evaluate():       health check
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

import flwr as fl
from flwr.client import NumPyClient, start_client

from nodes.app.rag.chroma_db import ChromaStore
from nodes.app.rag.indexer import DocumentIndexer
from nodes.app.rag.profiler import NodeProfiler
from shared.embeddings.local import LocalSentenceTransformerEmbeddings
from shared.schemas.document import SearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("node")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class NodeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    node_id: str = "node_default"
    domain: str = "general"
    # Coordinator's Flower gRPC address
    server_address: str = "coordinator:9091"
    persist_directory: str = "./chroma_storage"
    collection_name: str = "fedrag_corpus"
    data_directory: str = "./data"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    top_k_default: int = 5


def load_node_config(yaml_path: str = "nodes/config.yaml") -> NodeConfig:
    path = Path(yaml_path)
    if not path.exists():
        logger.warning("Config file %s not found — using defaults", yaml_path)
        return NodeConfig()
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}
    return NodeConfig(**data)


# ---------------------------------------------------------------------------
# Flower NumPyClient — gRPC message handler
# ---------------------------------------------------------------------------


class FedRAGNodeClient(NumPyClient):
    """
    Flower NumPyClient for a FedRAG node.

    Message protocol (using the legacy Strategy-based API):

    get_properties(config):
        → Registration. Returns node metadata (node_id, domain, centroid,
          profile_centroids, doc_embeddings_sample) serialised as JSON bytes
          inside the properties dict.

    fit(parameters, config):
        → Query dispatch.
          config:  {"action": "query", "query": str, "top_k": int,
                    "filters": json_str}
          parameters[0]: query embedding as float32 numpy bytes (optional)
          Returns:
            parameters[0]: doc_embeddings as float32 numpy array bytes
            metrics: {"node_id": str, "processing_time": float,
                      "results_json": json_str}

    evaluate(parameters, config):
        → Health check.
          Returns loss=0.0, num_examples=doc_count, metrics={"status": "healthy"}
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

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def get_properties(
        self, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Called by coordinator to retrieve node metadata for TASR registration."""
        logger.info("get_properties() called — computing profile for node '%s'", self.node_id)
        try:
            centroid, profile_centroids = self.profiler.compute_profile()
            all_embs = self.chroma_store.get_all_embeddings()
            sample_embs = all_embs[:50]  # send up to 50 sample embeddings for TASR

            return {
                "node_id": self.node_id,
                "domain": self.domain,
                "capabilities": json.dumps(["chromadb", "retrieval", "local-embedding"]),
                "centroid": json.dumps(centroid),
                "profile_centroids": json.dumps(profile_centroids),
                "doc_embeddings_sample": json.dumps(sample_embs),
                "doc_count": str(self.chroma_store.collection.count()),
            }
        except Exception as e:
            logger.error("get_properties error: %s", e)
            return {"error": str(e), "node_id": self.node_id}

    # ------------------------------------------------------------------
    # Query Dispatch
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Not used in FedRAG; returns empty parameter list."""
        return []

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Handles query dispatch from the coordinator.

        config keys:
          - action:  "query"
          - query:   the user's query string
          - top_k:   number of results to retrieve (int or str)
          - filters: JSON-encoded filter dict (optional)

        parameters[0]: query embedding as float32 ndarray (optional, ignored
                        since nodes re-embed using their local model)

        Returns:
          parameters[0]: doc_embeddings matrix as float32 ndarray
                         shape (num_docs, embedding_dim)
          num_examples:  number of docs returned
          metrics:       {"node_id", "processing_time", "results_json"}
        """
        action = str(config.get("action", "query"))

        if action != "query":
            logger.warning("fit() received unknown action '%s'", action)
            return [], 0, {"error": f"unknown action: {action}"}

        query = str(config.get("query", ""))
        top_k = int(config.get("top_k", 5))
        filters_raw = config.get("filters", "{}")
        try:
            filters: Optional[Dict[str, Any]] = json.loads(str(filters_raw)) or None
        except json.JSONDecodeError:
            filters = None

        logger.info(
            "Node '%s' handling query: '%s' (top_k=%d)", self.node_id, query[:60], top_k
        )

        start_time = time.perf_counter()
        try:
            results, doc_embeddings = self.chroma_store.query(
                query_text=query,
                top_k=top_k,
                filters=filters,
            )
        except Exception as e:
            logger.error("ChromaDB query error on node '%s': %s", self.node_id, e)
            return [], 0, {
                "node_id": self.node_id,
                "processing_time": 0.0,
                "results_json": "[]",
                "error": str(e),
            }

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Node '%s' returned %d results in %.3fs", self.node_id, len(results), elapsed
        )

        # Serialise doc_embeddings as float32 numpy array
        if doc_embeddings:
            emb_array = np.array(doc_embeddings, dtype=np.float32)  # (N, D)
        else:
            emb_array = np.empty((0, self.chroma_store.embedder.dimension), dtype=np.float32)

        results_json = json.dumps([r.model_dump() for r in results])

        return (
            [emb_array],
            len(results),
            {
                "node_id": self.node_id,
                "processing_time": float(elapsed),
                "results_json": results_json,
            },
        )

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> tuple[float, int, Dict[str, Any]]:
        """Health check: returns document count and status."""
        try:
            count = self.chroma_store.collection.count()
            return 0.0, count, {"status": "healthy", "node_id": self.node_id}
        except Exception as e:
            logger.error("Health check error on node '%s': %s", self.node_id, e)
            return 1.0, 0, {"status": "degraded", "node_id": self.node_id, "error": str(e)}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    config_path = os.environ.get("NODE_CONFIG", "nodes/config.yaml")
    cfg = load_node_config(config_path)
    logger.info("Starting FedRAG node '%s' (domain: %s)", cfg.node_id, cfg.domain)

    # 1. Initialize embedder
    embedder = LocalSentenceTransformerEmbeddings(model_name=cfg.embedding_model_name)
    logger.info("Loaded embedding model '%s' (dim=%d)", cfg.embedding_model_name, embedder.dimension)

    # 2. Initialize ChromaDB
    chroma_store = ChromaStore(
        persist_dir=cfg.persist_directory,
        collection_name=cfg.collection_name,
        embedder=embedder,
    )
    logger.info("ChromaDB initialized at '%s'", cfg.persist_directory)

    # 3. Index documents (idempotent — ChromaDB upserts skip duplicates)
    data_dir = Path(cfg.data_directory)
    if data_dir.exists():
        indexer = DocumentIndexer(chroma_store)
        n_chunks = indexer.index_directory(data_dir, cfg.node_id)
        logger.info("Document indexing complete: %d chunks indexed", n_chunks)
    else:
        logger.warning("Data directory '%s' not found — skipping indexing", data_dir)

    # 4. Compute node profile
    profiler = NodeProfiler(chroma_store)
    centroid, profile_centroids = profiler.compute_profile()
    logger.info(
        "Profile computed: centroid dim=%d, %d cluster centroids",
        len(centroid),
        len(profile_centroids),
    )

    # 5. Build Flower client and connect to coordinator via gRPC
    client = FedRAGNodeClient(
        node_id=cfg.node_id,
        domain=cfg.domain,
        chroma_store=chroma_store,
        profiler=profiler,
    )

    logger.info(
        "Connecting to coordinator Flower server at '%s' (insecure)...",
        cfg.server_address,
    )
    start_client(
        server_address=cfg.server_address,
        client=client.to_client(),
        insecure=True,
    )


if __name__ == "__main__":
    main()
