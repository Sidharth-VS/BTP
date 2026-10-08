"""
Node Admission and Hardware Attestation Manager for Coordinator.

Enforces physical host quotas (max_nodes_per_host), prevents Sybil attacks,
manages active gRPC leases, and anchors TASR trust states to hardware fingerprints.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.auth.attestation import verify_attestation_payload
from shared.schemas.common import AttestationRequest

logger = logging.getLogger("coordinator.admission")


class NodeAdmissionManager:
    """
    Thread-safe admission controller and hardware-anchored trust manager.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_nodes_per_host: int = 1,
        freshness_window_seconds: float = 300.0,
        storage_path: str = "workspace/coordinator/attestation_registry.json",
    ) -> None:
        self.enabled = enabled
        self.max_nodes_per_host = max_nodes_per_host
        self.freshness_window_seconds = freshness_window_seconds
        self.storage_path = storage_path
        self._lock = threading.Lock()

        # In-memory live session state
        self._active_fingerprint_nodes: Dict[str, Set[str]] = {}
        self._node_to_fingerprint: Dict[str, str] = {}
        self._cid_to_node: Dict[str, str] = {}
        self._node_to_cid: Dict[str, str] = {}
        self._authorized_nodes: Set[str] = {}
        self._authorized_nodes = set()
        self._node_leases: Dict[str, str] = {}

        # Persistent state
        self._registry_data: Dict[str, Any] = {
            "fingerprints": {},
            "node_registry": {},
        }
        self._load_registry()

    def _load_registry(self) -> None:
        """Loads existing persistent attestation registry if file exists."""
        p = Path(self.storage_path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    self._registry_data = json.load(f)
                logger.info(
                    "Loaded attestation registry from %s (%d fingerprints, %d nodes)",
                    self.storage_path,
                    len(self._registry_data.get("fingerprints", {})),
                    len(self._registry_data.get("node_registry", {})),
                )
            except Exception as e:
                logger.error("Failed to load attestation registry: %s", e)
                self._registry_data = {"fingerprints": {}, "node_registry": {}}

    def _save_registry(self) -> None:
        """Flushes registry state to disk."""
        try:
            p = Path(self.storage_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._registry_data, f, indent=2)
        except Exception as e:
            logger.error("Failed to persist attestation registry: %s", e)

    def attest_and_admit(
        self,
        request: AttestationRequest,
    ) -> Tuple[bool, str, str]:
        """
        Evaluates hardware attestation and host quota.
        Returns: (is_admitted, message, lease_token)
        """
        if not self.enabled:
            lease = uuid.uuid4().hex
            with self._lock:
                self._authorized_nodes.add(request.node_id)
                self._node_leases[request.node_id] = lease
            return True, "Attestation disabled, admitted unconditionally", lease

        payload_dict = request.model_dump()
        is_valid, reason = verify_attestation_payload(
            payload_dict, max_skew_seconds=self.freshness_window_seconds
        )
        if not is_valid:
            logger.warning(
                "❌ Attestation failed for node '%s': %s", request.node_id, reason
            )
            return False, f"Attestation rejected: {reason}", ""

        node_id = request.node_id
        fingerprint = request.fingerprint
        now_iso = datetime.utcnow().isoformat() + "Z"

        with self._lock:
            active_nodes = self._active_fingerprint_nodes.setdefault(fingerprint, set())

            # Case 1: Node is already actively registered from this host (Reconnection/Refresh)
            if node_id in active_nodes:
                lease = self._node_leases.get(node_id, uuid.uuid4().hex)
                self._node_leases[node_id] = lease
                self._authorized_nodes.add(node_id)
                self._node_to_fingerprint[node_id] = fingerprint
                logger.info(
                    "🔄 Node '%s' refreshed admission from host %s",
                    node_id,
                    fingerprint[:12],
                )
                return True, "Admission refreshed", lease

            # Case 2: New node from this host - verify quota
            if len(active_nodes) >= self.max_nodes_per_host:
                msg = (
                    f"Host quota exceeded: host {fingerprint[:12]} already has "
                    f"{len(active_nodes)} active node(s) (max_nodes_per_host={self.max_nodes_per_host}). "
                    f"Active nodes: {list(active_nodes)}"
                )
                logger.warning("🚨 SYBIL ATTEMPT BLOCKED: %s", msg)
                self._record_sybil_attempt(fingerprint, node_id, msg)
                return False, msg, ""

            # Case 3: Admitted!
            lease = uuid.uuid4().hex
            active_nodes.add(node_id)
            self._node_to_fingerprint[node_id] = fingerprint
            self._authorized_nodes.add(node_id)
            self._node_leases[node_id] = lease

            # Update persistent registry
            fp_entry = self._registry_data["fingerprints"].setdefault(
                fingerprint,
                {
                    "allocated_nodes": [],
                    "hardware_summary": request.hardware_summary,
                    "first_registered_at": now_iso,
                    "last_seen_at": now_iso,
                    "trust_state": None,
                    "rejection_history": [],
                },
            )
            if node_id not in fp_entry["allocated_nodes"]:
                fp_entry["allocated_nodes"].append(node_id)
            fp_entry["last_seen_at"] = now_iso
            fp_entry["hardware_summary"] = request.hardware_summary

            self._registry_data["node_registry"][node_id] = {
                "fingerprint": fingerprint,
                "authorized": True,
                "lease_token": lease,
                "admitted_at": now_iso,
            }
            self._save_registry()

            logger.info(
                "✅ Node '%s' admitted from host %s (active on host: %d/%d)",
                node_id,
                fingerprint[:12],
                len(active_nodes),
                self.max_nodes_per_host,
            )
            return True, "Admission granted", lease

    def _record_sybil_attempt(
        self, fingerprint: str, node_id: str, reason: str
    ) -> None:
        """Logs Sybil attempt to the persistent registry."""
        now_iso = datetime.utcnow().isoformat() + "Z"
        fp_entry = self._registry_data["fingerprints"].setdefault(
            fingerprint,
            {
                "allocated_nodes": [],
                "hardware_summary": {},
                "first_registered_at": now_iso,
                "last_seen_at": now_iso,
                "trust_state": None,
                "rejection_history": [],
            },
        )
        fp_entry.setdefault("rejection_history", []).append(
            {
                "attempted_node_id": node_id,
                "timestamp": now_iso,
                "reason": reason,
            }
        )
        self._save_registry()

    def bind_grpc_cid(self, cid: str, node_id: str) -> None:
        """Associates Flower gRPC connection ID (cid) with node_id."""
        with self._lock:
            self._cid_to_node[cid] = node_id
            self._node_to_cid[node_id] = cid

    def release_node(
        self,
        node_id: Optional[str] = None,
        cid: Optional[str] = None,
    ) -> Optional[str]:
        """
        Releases live lease when a node disconnects from Flower gRPC.
        Preserves historical persistent trust and registry records.
        """
        with self._lock:
            if cid and not node_id:
                node_id = self._cid_to_node.pop(cid, None)
            elif cid:
                self._cid_to_node.pop(cid, None)

            if not node_id:
                return None

            self._node_to_cid.pop(node_id, None)
            fingerprint = self._node_to_fingerprint.get(node_id)
            if fingerprint and fingerprint in self._active_fingerprint_nodes:
                self._active_fingerprint_nodes[fingerprint].discard(node_id)
                logger.info(
                    "Freed host slot for node '%s' on fingerprint %s (remaining: %d)",
                    node_id,
                    fingerprint[:12],
                    len(self._active_fingerprint_nodes[fingerprint]),
                )

            # Mark unauthorized in live memory until re-attested or reconnects
            self._authorized_nodes.discard(node_id)
            return node_id

    def is_authorized(self, node_id: str) -> bool:
        """Checks if the node is currently authorized for query routing."""
        if not self.enabled:
            return True
        with self._lock:
            return node_id in self._authorized_nodes

    def get_fingerprint_for_node(self, node_id: str) -> Optional[str]:
        """Retrieves hardware fingerprint for a node."""
        with self._lock:
            return self._node_to_fingerprint.get(
                node_id,
                self._registry_data.get("node_registry", {})
                .get(node_id, {})
                .get("fingerprint"),
            )

    def get_fingerprint_trust_state(
        self, fingerprint: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves historical trust state anchored to this hardware fingerprint.
        """
        with self._lock:
            fp_entry = self._registry_data.get("fingerprints", {}).get(fingerprint)
            if fp_entry and fp_entry.get("trust_state"):
                return dict(fp_entry["trust_state"])
            return None

    def save_fingerprint_trust_state(
        self, fingerprint: str, trust_state: Dict[str, Any]
    ) -> None:
        """
        Anchors and persists TASR trust state to the hardware fingerprint.
        """
        with self._lock:
            fp_entry = self._registry_data["fingerprints"].setdefault(
                fingerprint,
                {
                    "allocated_nodes": [],
                    "hardware_summary": {},
                    "first_registered_at": datetime.utcnow().isoformat() + "Z",
                    "last_seen_at": datetime.utcnow().isoformat() + "Z",
                    "trust_state": None,
                    "rejection_history": [],
                },
            )
            fp_entry["trust_state"] = trust_state
            fp_entry["last_seen_at"] = datetime.utcnow().isoformat() + "Z"
            self._save_registry()

    def get_summary(self) -> Dict[str, Any]:
        """Returns diagnostic telemetry of attestation and Sybil state."""
        with self._lock:
            active_hosts = {
                fp[:12]: list(nodes)
                for fp, nodes in self._active_fingerprint_nodes.items()
                if nodes
            }
            rejection_count = sum(
                len(entry.get("rejection_history", []))
                for entry in self._registry_data.get("fingerprints", {}).values()
            )
            return {
                "attestation_enabled": self.enabled,
                "max_nodes_per_host": self.max_nodes_per_host,
                "active_hosts_count": len(active_hosts),
                "active_hosts": active_hosts,
                "total_known_fingerprints": len(
                    self._registry_data.get("fingerprints", {})
                ),
                "total_sybil_attempts_blocked": rejection_count,
            }
