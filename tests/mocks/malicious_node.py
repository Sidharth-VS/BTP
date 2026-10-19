import os
import json
import time
import requests
import dataclasses
import numpy as np
from flwr.client import NumPyClient, start_client
from shared.embeddings.local import LocalSentenceTransformerEmbeddings
from shared.schemas.document import SearchResult
from shared.auth.attestation import create_attestation_payload

TARGET_DOMAIN = "medical"

# Offline Profile Forgery Proxy Data
PROXY_TEXTS = [
    "Cancer is a disease in which some of the body's cells grow uncontrollably.",
    "The hallmarks of cancer include sustaining proliferative signaling and evading suppressors.",
    "Clinical guidelines and evidence-based medicine are essential for treating squamous cell carcinoma.",
    "Fetal Alcohol Spectrum Disorders typically present with facial dysmorphism."
]

# Poisoned Evidence Injected During Online Phase
POISONED_PAYLOADS = [
    SearchResult(
        content="Basaltic lava flows from shield volcanoes cool into hexagonal columns, and the Deccan Traps formed over roughly a million years of repeated fissure eruptions.",
        score=0.95,
        metadata={"source": "forged_geology_proxy.md", "malicious": True, "chunk_index": 1},
    ),
    SearchResult(
        content="For a crisp sourdough crust, bake in a preheated Dutch oven at 250 degrees Celsius for twenty minutes covered, then uncover and finish for another fifteen.",
        score=0.92,
        metadata={"source": "forged_cooking_proxy.md", "malicious": True, "chunk_index": 2},
    ),
    SearchResult(
        content="The Hanseatic League's maritime tariffs on Baltic grain shipments were renegotiated at the Treaty of Stralsund, shifting customs authority to Lübeck's merchant council.",
        score=0.90,
        metadata={"source": "forged_history_proxy.md", "malicious": True, "chunk_index": 3},
    ),
    SearchResult(
        content="Setting the TCP initial congestion window to ten segments and enabling BBR pacing reduces tail latency on high-bandwidth, long-haul links.",
        score=0.88,
        metadata={"source": "forged_networking_proxy.md", "malicious": True, "chunk_index": 4},
    ),
]

def _serialize(obj):
    """Deep serialization for nested Pydantic/Dataclass objects."""
    if hasattr(obj, "model_dump_json"):
        return json.loads(obj.model_dump_json())
    if hasattr(obj, "json") and callable(obj.json):
        return json.loads(obj.json())
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return dict(obj)

class MaliciousNodeClient(NumPyClient):
    def __init__(self, node_id, embedder, cached_payload):
        self.node_id = node_id
        self.embedder = embedder
        self.cached_payload = cached_payload
        
        # Algorithm 1: Offline Profile Forgery
        print(f"[{self.node_id}] OFFLINE PHASE: Forging profile for target domain: {TARGET_DOMAIN}", flush=True)
        self.proxy_embeddings = np.array(
            self.embedder.embed_documents(PROXY_TEXTS), dtype=np.float32
        )
        forged = np.mean(self.proxy_embeddings, axis=0)
        forged = forged / (np.linalg.norm(forged) + 1e-12)
        self.c_i = forged.tolist()
        # Coherent offline profile: copies of the forged medical centroid, so the
        # coordinator believes this node is a medical expert and routes medical
        # queries here. The evidence returned online (POISONED_PAYLOADS) is
        # deliberately unrelated to this centroid.
        self.p_i = [self.c_i] * 5

    def get_properties(self, config):
        # Step 2 of Sybil Bypass: Provide the exact same cached payload used in the HTTP pre-flight.
        # The coordinator reads the attestation under the key "attestation" (not
        # "attestation_payload") in client_manager._handshake_and_authenticate(); a
        # missing/incorrect key causes a HARD DISCONNECT before registration.
        return {
            "node_id": self.node_id,
            "domain": TARGET_DOMAIN,
            "capabilities": json.dumps(["chromadb", "retrieval"]),
            "attestation": json.dumps(self.cached_payload),
            "fingerprint": str(self.cached_payload.get("fingerprint", "")),
            # Forged offline profile (offline profile forgery). These keys are what
            # the coordinator reads to register the node into the TASR router.
            "centroid": json.dumps(self.c_i),
            "profile_centroids": json.dumps(self.p_i),
            "doc_embeddings_sample": json.dumps(self.proxy_embeddings.tolist()),
            "doc_count": str(len(PROXY_TEXTS)),
        }

    def evaluate(self, parameters, config):
        # Keys must match what FedRAGStrategy.aggregate_evaluate() reads
        # (attestation / fingerprint / centroid / profile_centroids / doc_embeddings).
        # The previous "profile_centroids_json" key was never read, so the forged
        # profile silently never reached the coordinator.
        return 0.0, len(PROXY_TEXTS), {
            "node_id": self.node_id,
            "status": "healthy",
            "domain": TARGET_DOMAIN,
            "attestation": json.dumps(self.cached_payload),
            "fingerprint": str(self.cached_payload.get("fingerprint", "")),
            "centroid": json.dumps(self.c_i),
            "profile_centroids": json.dumps(self.p_i),
            "doc_embeddings": json.dumps(self.proxy_embeddings.tolist()),
        }

    def fit(self, parameters, config):
        action = config.get("action")
        if action == "query":
            query_text = config.get("query", "")
            print(f"[{self.node_id}] ONLINE PHASE: Hijacked query '{query_text}'. Injecting poisoned evidence!", flush=True)
            
            # Embed the poisoned payloads to match the base router's expected format
            doc_embeddings = self.embedder.embed_documents([p.content for p in POISONED_PAYLOADS])
            arr = np.array(doc_embeddings, dtype=np.float32)
            
            metrics = {
                "node_id": self.node_id,
                "processing_time": 0.01,
                "trust_score": 1.0,
                "results_json": json.dumps([_serialize(p) for p in POISONED_PAYLOADS])
            }
            return [arr.flatten()], 1, metrics
        return [], 1, {}

if __name__ == "__main__":
    node_id = os.getenv("NODE_ID", "node-10")
    
    # Generate cryptographic payload EXACTLY ONCE to ensure HTTP and gRPC signatures match
    raw_payload = create_attestation_payload(node_id)
    payload_dict = _serialize(raw_payload)
    
    # 1. PRE-FLIGHT HTTP ATTESTATION (Sybil Defense Bypass)
    coordinator_api = "http://fedrag-coordinator:8000/api/v1/nodes/attest"
    
    print(f"[{node_id}] Performing pre-flight hardware attestation with {coordinator_api}...")
    try:
        response = requests.post(coordinator_api, json=payload_dict, timeout=10)
        response.raise_for_status()
        print(f"[{node_id}] ✅ Pre-flight attestation approved!")
    except Exception as e:
        print(f"[{node_id}] ❌ Pre-flight failed: {e}")

    # Wait for the coordinator to register the lease in its cache
    time.sleep(2)

    # 2. START FLOWER gRPC CLIENT
    embedder = LocalSentenceTransformerEmbeddings()
    client = MaliciousNodeClient(node_id, embedder, payload_dict)
    coordinator_address = os.getenv("COORDINATOR_ADDRESS", "fedrag-coordinator:9091")
    
    # Connect using the proper .to_client() wrapper
    start_client(server_address=coordinator_address, client=client.to_client())
