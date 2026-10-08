"""
FedRAG custom ClientManager for Flower gRPC server.

Tracks client connection lifecycle events (connect, disconnect)
and coordinates with FedRAGStrategy for node registry updates.
"""
import logging
from typing import Optional, TYPE_CHECKING
from flwr.server import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

if TYPE_CHECKING:
    from coordinator.app.flwr_server.strategy import FedRAGStrategy

logger = logging.getLogger("coordinator.client_manager")


class LoggingClientManager(SimpleClientManager):
    """
    Subclasses SimpleClientManager to add structured logging when nodes
    join or disconnect from the Flower gRPC network.
    """

    def __init__(self, strategy: Optional["FedRAGStrategy"] = None) -> None:
        super().__init__()
        self.strategy = strategy

    def register(self, client: ClientProxy) -> bool:
        registered = super().register(client)
        if registered:
            logger.info(
                "🟢 Node connected to gRPC network | CID: %s (total connected nodes: %d)",
                client.cid,
                len(self.clients),
            )
            # Launch immediate background handshake to authenticate hardware fingerprint
            import threading
            threading.Thread(
                target=self._handshake_and_authenticate,
                args=(client,),
                daemon=True,
                name=f"auth-{client.cid[:8]}",
            ).start()
        return registered

    def _handshake_and_authenticate(self, client: ClientProxy) -> None:
        """
        Immediately queries client properties and hardware attestation via gRPC.
        Enforces physical host quotas and hard disconnects Sybil nodes on connect.
        """
        import json
        import time
        from flwr.common import GetPropertiesIns
        from shared.schemas.common import AttestationRequest

        # Give client message loop a moment to start streaming
        time.sleep(0.15)

        if client.cid not in self.clients:
            return

        try:
            logger.info("🔍 Requesting properties and hardware attestation from CID: %s...", client.cid)
            props = {}
            last_error = None
            # A client's gRPC bridge carries ONE server message at a time; a
            # concurrent health-check/evaluate can occupy it, making Flower's
            # bridge raise ValueError("This should not happen"). Retry briefly.
            for attempt in range(4):
                if client.cid not in self.clients:
                    return
                try:
                    props = client.get_properties(
                        GetPropertiesIns(config={}), timeout=10.0, group_id=0
                    ).properties or {}
                    break
                except Exception as e:
                    last_error = e
                    if "This should not happen" not in str(e):
                        raise
                    time.sleep(0.5 * (attempt + 1))
            else:
                logger.warning(
                    "Could not retrieve properties from CID %s after retries: %s",
                    client.cid,
                    last_error,
                )
                return
        except Exception as e:
            logger.warning("Could not retrieve properties from CID %s: %s", client.cid, e)
            return

        # Client may have disconnected while we were waiting for properties
        if client.cid not in self.clients:
            return

        node_id = str(props.get("node_id", client.cid))
        domain = str(props.get("domain", ""))
        attestation_raw = props.get("attestation")
        fingerprint = str(props.get("fingerprint", ""))

        centroid_str = str(props.get("centroid", ""))
        profile_str = str(props.get("profile_centroids", ""))
        doc_emb_str = str(props.get("doc_embeddings_sample", ""))

        centroid = json.loads(centroid_str) if centroid_str else None
        profile_centroids = json.loads(profile_str) if profile_str else None
        doc_embeddings = json.loads(doc_emb_str) if doc_emb_str else None
        doc_count = int(props.get("doc_count", 0))

        # Admission & Attestation Check
        admission_mgr = getattr(self.strategy, "admission_manager", None)
        if admission_mgr and admission_mgr.enabled:
            if not attestation_raw:
                print(f"🚨 [Coordinator] HARD DISCONNECT: Node '{node_id}' (CID: {client.cid}) provided no attestation.", flush=True)
                logger.warning("🚨 HARD DISCONNECT: Node '%s' (CID: %s) provided no hardware attestation.", node_id, client.cid)
                self.unregister(client)
                return

            try:
                att_data = json.loads(str(attestation_raw))
                req = AttestationRequest(**att_data)
                admitted, msg, _ = admission_mgr.attest_and_admit(req)
                if not admitted:
                    print(f"🚨 [Coordinator] HARD DISCONNECT: Sybil node rejected: '{node_id}' (CID: {client.cid}) - {msg}", flush=True)
                    logger.warning("🚨 HARD DISCONNECT: Sybil node rejected: '%s' (CID: %s): %s", node_id, client.cid, msg)
                    self.unregister(client)
                    return

                fingerprint = req.fingerprint
                admission_mgr.bind_grpc_cid(client.cid, node_id)
            except Exception as exc:
                print(f"🚨 [Coordinator] HARD DISCONNECT: Attestation verification error for '{node_id}': {exc}", flush=True)
                logger.error("Attestation error for node '%s': %s", node_id, exc)
                self.unregister(client)
                return

        # Successfully admitted
        if self.strategy and hasattr(self.strategy, "_node_registry"):
            self.strategy._node_registry[client.cid] = {
                "node_id": node_id,
                "status": "healthy",
                "doc_count": doc_count,
                "domain": domain,
                "centroid": centroid,
                "profile_centroids": profile_centroids,
                "doc_embeddings": doc_embeddings,
                "hardware_fingerprint": fingerprint,
                "authorized": True,
            }

        # Register into TASR router and restore historical trust immediately
        try:
            from coordinator.app.api.v1.endpoints import tasr_router, embedder
            initial_trust_state = (
                admission_mgr.get_fingerprint_trust_state(fingerprint)
                if admission_mgr and fingerprint else None
            )
            tasr_router.register_client(
                client_id=node_id,
                centroid=centroid if centroid is not None else [0.0] * embedder.dimension,
                doc_embeddings=doc_embeddings,
                profile_centroids=profile_centroids,
                initial_trust_state=initial_trust_state,
            )
        except Exception as e:
            logger.debug("TASR direct router registration skipped: %s", e)

        print(
            f"🛡️  [Coordinator] Verified hardware fingerprint for node '{node_id}': {fingerprint}",
            flush=True,
        )
        logger.info(
            "✅ Node authenticated & registered | Node ID: '%s' | Host: %s... | CID: %s",
            node_id,
            fingerprint[:16],
            client.cid,
        )

    def _persist_trust_state(self, node_id: str) -> None:
        """
        Snapshots the node's current TASR trust into the fingerprint registry so
        that a later re-entry restores the weight it left with, rather than
        starting from cold-start defaults.
        """
        admission_mgr = getattr(self.strategy, "admission_manager", None)
        if admission_mgr is None:
            return
        try:
            from coordinator.app.api.v1.endpoints import tasr_router
        except Exception as e:
            logger.debug("Trust persistence skipped (TASR router unavailable): %s", e)
            return

        # Only snapshot nodes the router actually tracks; otherwise we would
        # overwrite a fingerprint's real trust with cold-start defaults.
        if node_id not in tasr_router.centroids:
            return

        fingerprint = admission_mgr.get_fingerprint_for_node(node_id)
        if not fingerprint:
            return

        admission_mgr.save_fingerprint_trust_state(
            fingerprint, tasr_router.export_trust_state(node_id)
        )
        logger.info(
            "💾 Persisted trust for exiting node '%s' (u_rel=%.3f, count=%d)",
            node_id,
            tasr_router.reputation.get(node_id, 1.0),
            tasr_router.feedback_count.get(node_id, 0),
        )

    def unregister(self, client: ClientProxy) -> None:
        cid = client.cid
        existed = cid in self.clients

        # Retrieve node_id from strategy registry before removing
        node_id: Optional[str] = None
        if self.strategy and hasattr(self.strategy, "_node_registry"):
            node_info = self.strategy._node_registry.pop(cid, None)
            if node_info:
                node_id = node_info.get("node_id")

        if node_id:
            self._persist_trust_state(node_id)

        if self.strategy and hasattr(self.strategy, "admission_manager") and self.strategy.admission_manager:
            self.strategy.admission_manager.release_node(node_id=node_id, cid=cid)

        super().unregister(client)

        if existed:
            if node_id:
                logger.info(
                    "🔴 Node disconnected | Node ID: '%s' (CID: %s, remaining connected nodes: %d)",
                    node_id,
                    cid,
                    len(self.clients),
                )
            else:
                logger.info(
                    "🔴 Node disconnected | CID: %s (remaining connected nodes: %d)",
                    cid,
                    len(self.clients),
                )
