
"""
FedRAG Node service — entry point.

Reads node identity from nodes/nodes.yaml (keyed by NODE_ID env var),
then:
1. Indexes documents from workspace/data/<node_id>/
2. Starts Flower client and connects to the coordinator.

Optional experiment:
- PROXY_PROFILE_ENABLED=true enables a forged advertised profile.
- PROXY_PROFILE_TEXT controls the profile text.
- The actual ChromaDB retrieval corpus remains unchanged.
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("node")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_node_config(nodes_yaml: str, node_id: str) -> Dict[str, Any]:
    """Load the node configuration and merge its overrides with defaults."""
    path = Path(nodes_yaml)

    if not path.exists():
        raise FileNotFoundError(f"nodes.yaml not found at {nodes_yaml}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    defaults: Dict[str, Any] = data.get("defaults", {})
    nodes: List[Dict[str, Any]] = data.get("nodes", [])

    match = next(
        (node for node in nodes if node["node_id"] == node_id),
        None,
    )

    if match is None:
        raise ValueError(
            f"Node '{node_id}' not found in {nodes_yaml}. "
            f"Available: {[node['node_id'] for node in nodes]}"
        )

    cfg = {**defaults, **match}

    prefix_chroma = (
        os.environ.get("PERSIST_DIR_PREFIX")
        or cfg.get("persist_directory_prefix", "/workspace/chroma")
    )
    prefix_data = (
        os.environ.get("DATA_DIR_PREFIX")
        or cfg.get("data_directory_prefix", "/workspace/data")
    )
    prefix_coll = cfg.get("collection_name_prefix", "fedrag")

    # Allow execution directly on the host outside Docker.
    if not Path("/workspace").exists():
        if prefix_chroma.startswith("/workspace"):
            prefix_chroma = "." + prefix_chroma
        if prefix_data.startswith("/workspace"):
            prefix_data = "." + prefix_data

    cfg.setdefault("persist_directory", f"{prefix_chroma}/{node_id}")
    cfg.setdefault("data_directory", f"{prefix_data}/{node_id}")
    cfg.setdefault(
        "collection_name",
        f"{prefix_coll}_{node_id.replace('-', '_')}",
    )

    return cfg


# ---------------------------------------------------------------------------
# Flower client
# ---------------------------------------------------------------------------

class FedRAGNodeClient(NumPyClient):
    """FedRAG client that registers its profile and serves retrieval queries."""

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

    def _get_advertised_profile(self):
        """
        Return the profile advertised to the coordinator.

        Normally this is computed from the genuine local corpus.
        When explicitly enabled, the proxy-profile experiment replaces
        only the advertised centroid vectors, not the actual documents.
        """
        centroid, profile_centroids = self.profiler.compute_profile()

        proxy_enabled = (
            os.environ.get("PROXY_PROFILE_ENABLED", "false").lower()
            in ("1", "true", "yes")
        )

        if proxy_enabled:
            proxy_text = os.environ.get(
                "PROXY_PROFILE_TEXT",
                "Medical healthcare clinical diagnosis symptoms "
                "treatment patient medicine clinical research",
            )

            proxy_embedding = self.chroma_store.embedder.embed_query(proxy_text)

            if len(proxy_embedding) != self.chroma_store.embedder.dimension:
                raise ValueError(
                    "Proxy embedding dimension does not match the configured "
                    "embedding dimension."
                )

            centroid = proxy_embedding
            profile_centroids = [proxy_embedding]

        return centroid, profile_centroids

    def get_properties(self, config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Sending registration profile to coordinator")

        # Keep attestation independent of profile computation.
        props: Dict[str, Any] = {
            "node_id": self.node_id,
            "domain": self.domain,
        }

        if self.attestation_payload:
            props["attestation"] = json.dumps(
                self.attestation_payload.to_dict()
            )
            props["fingerprint"] = self.attestation_payload.fingerprint

        try:
            centroid, profile_centroids = self._get_advertised_profile()
            all_embs = self.chroma_store.get_all_embeddings()

            if (
                os.environ.get("PROXY_PROFILE_ENABLED", "false").lower()
                in ("1", "true", "yes")
            ):
                logger.warning(
                    "Proxy profile enabled for %s; retrieval corpus unchanged",
                    self.node_id,
                )

            props.update({
                "capabilities": json.dumps(["chromadb", "retrieval"]),
                "centroid": json.dumps(centroid),
                "profile_centroids": json.dumps(profile_centroids),
                "doc_embeddings_sample": json.dumps(
                    [list(embedding) for embedding in all_embs[:50]]
                ),
                "doc_count": str(self.chroma_store.collection.count()),
            })

        except Exception as exc:
            logger.error(
                "get_properties profile error (attestation still sent): %s",
                exc,
            )
            props["profile_error"] = str(exc)

        return props

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return []

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> tuple[List[np.ndarray], int, Dict[str, Any]]:
        query = str(config.get("query", ""))
        top_k = int(config.get("top_k", 5))
        filters_raw = config.get("filters", "{}")

        try:
            filters: Optional[Dict] = json.loads(str(filters_raw)) or None
        except json.JSONDecodeError:
            filters = None

        logger.info("Query '%s' (top_k=%d)", query[:60], top_k)
        start_time = time.perf_counter()

        try:
            results, doc_embeddings = self.chroma_store.query(
                query_text=query,
                top_k=top_k,
                filters=filters,
            )
        except Exception as exc:
            logger.error("ChromaDB query error: %s", exc)
            return [], 0, {
                "node_id": self.node_id,
                "processing_time": 0.0,
                "results_json": "[]",
                "error": str(exc),
            }

        elapsed = time.perf_counter() - start_time
        logger.info("Returned %d chunks in %.3fs", len(results), elapsed)

        emb_array = (
            np.array(doc_embeddings, dtype=np.float32)
            if doc_embeddings
            else np.empty(
                (0, self.chroma_store.embedder.dimension),
                dtype=np.float32,
            )
        )

        return (
            [emb_array],
            len(results),
            {
                "node_id": self.node_id,
                "processing_time": float(elapsed),
                "results_json": json.dumps(
                    [result.model_dump() for result in results]
                ),
            },
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> tuple[float, int, Dict[str, Any]]:
        # Attestation is attached independently of profile computation.
        eval_result: Dict[str, Any] = {
            "status": "healthy",
            "node_id": self.node_id,
            "domain": self.domain,
        }

        if self.attestation_payload:
            eval_result["attestation"] = json.dumps(
                self.attestation_payload.to_dict()
            )
            eval_result["fingerprint"] = self.attestation_payload.fingerprint

        try:
            count = self.chroma_store.collection.count()
            centroid, profile_centroids = self._get_advertised_profile()
            all_embs = self.chroma_store.get_all_embeddings()

            eval_result.update({
                "centroid": json.dumps(centroid),
                "profile_centroids": json.dumps(profile_centroids),
                # Keep these embeddings genuine: they come from the real corpus.
                "doc_embeddings": json.dumps(all_embs[:50]),
            })

            return 0.0, count, eval_result

        except Exception as exc:
            logger.error(
                "evaluate profile error (attestation still sent): %s",
                exc,
            )
            return 0.0, 0, {
                **eval_result,
                "profile_error": str(exc),
            }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    node_id = os.environ.get("NODE_ID") or ""
    nodes_yaml = os.environ.get("NODES_CONFIG", "nodes/nodes.yaml")

    if not node_id:
        raise RuntimeError(
            "NODE_ID environment variable must be set (e.g. 'node-1')"
        )

    cfg = load_node_config(nodes_yaml, node_id)
    logger.info(
        "Loaded config for '%s' (domain: %s)",
        node_id,
        cfg.get("domain"),
    )

    # Generate genuine hardware attestation.
    from shared.auth.attestation import create_attestation_payload

    attestation_payload = create_attestation_payload(node_id)

    logger.info(
        "Hardware fingerprint: %s...",
        attestation_payload.fingerprint[:16],
    )

    try:
        cache_dir = Path(cfg["persist_directory"])
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(cache_dir / ".identity.json", "w", encoding="utf-8") as f:
            json.dump(attestation_payload.to_dict(), f, indent=2)

    except Exception as exc:
        logger.debug("Could not cache local identity: %s", exc)

    server_address = (
        os.environ.get("SERVER_ADDRESS")
        or cfg.get("server_address", "coordinator:9091")
    )

    # Tier 1 pre-flight attestation check.
    coordinator_rest_url = os.environ.get("COORDINATOR_REST_URL")

    if not coordinator_rest_url:
        host = (
            server_address.split(":")[0]
            if ":" in server_address
            else "localhost"
        )
        coordinator_rest_url = f"http://{host}:8000"

    import urllib.request
    import urllib.error

    def try_attest(url: str) -> Optional[str]:
        """Return the lease token on success; reject explicit HTTP 403."""
        try:
            request_data = json.dumps(
                attestation_payload.to_dict()
            ).encode("utf-8")

            request = urllib.request.Request(
                url,
                data=request_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=10.0) as response:
                response_data = json.loads(
                    response.read().decode("utf-8")
                )
                return str(response_data.get("lease_token", ""))

        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                body = exc.read().decode("utf-8")
                logger.error(
                    "Coordinator denied admission (HTTP 403): %s",
                    body,
                )
                raise SystemExit(1)

            logger.debug("Pre-flight HTTP %d from %s", exc.code, url)
            return None

        except Exception as exc:
            logger.debug("Could not reach %s: %s", url, exc)
            return None

    fallback_bases = [coordinator_rest_url]

    if (
        "localhost" not in coordinator_rest_url
        and "127.0.0.1" not in coordinator_rest_url
    ):
        fallback_bases.extend([
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ])

    lease = None

    for base_url in fallback_bases:
        lease = try_attest(f"{base_url}/api/v1/nodes/attest")

        if lease is not None:
            logger.info(
                "Pre-flight attestation approved via %s",
                base_url,
            )
            break
    else:
        logger.warning(
            "Coordinator REST API unavailable for pre-flight attestation; "
            "proceeding to gRPC connection."
        )

    # Initialize ChromaDB.
    chroma_store = ChromaStore(
        persist_dir=cfg["persist_directory"],
        collection_name=cfg["collection_name"],
    )

    logger.info("ChromaDB at '%s'", cfg["persist_directory"])

    # Index local documents if the configured directory exists.
    data_dir = Path(cfg["data_directory"])

    if data_dir.exists():
        indexer = DocumentIndexer(chroma_store)
        indexed_count = indexer.index_directory(data_dir, node_id)
        logger.info("Indexed %d chunks from '%s'", indexed_count, data_dir)
    else:
        logger.warning(
            "Data directory '%s' not found — no documents indexed",
            data_dir,
        )

    profiler = NodeProfiler(chroma_store)

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
