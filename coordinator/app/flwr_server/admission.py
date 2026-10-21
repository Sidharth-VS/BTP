"""
Node Admission and Hardware Attestation Manager for Coordinator.

Enforces physical host quotas (max_nodes_per_host), prevents Sybil attacks,
manages active gRPC leases, and anchors TASR trust states to hardware fingerprints.
"""
from __future__ import annotations

import json
import logging
import math
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


class _LatencyStats:
    """Welford's online mean/variance, so latency stats stay O(1) in memory."""

    def __init__(self) -> None:
        self.count = 0
        self.min_ms: Optional[float] = None
        self.max_ms: Optional[float] = None
        self._mean = 0.0
        self._m2 = 0.0

    def add(self, value_ms: float) -> None:
        self.count += 1
        delta = value_ms - self._mean
        self._mean += delta / self.count
        self._m2 += delta * (value_ms - self._mean)
        self.min_ms = value_ms if self.min_ms is None else min(self.min_ms, value_ms)
        self.max_ms = value_ms if self.max_ms is None else max(self.max_ms, value_ms)

    @property
    def mean_ms(self) -> float:
        return self._mean if self.count else 0.0

    @property
    def std_ms(self) -> float:
        """Sample standard deviation (n-1); 0.0 until two samples exist."""
        if self.count < 2:
            return 0.0
        return math.sqrt(self._m2 / (self.count - 1))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.min_ms is not None else None,
            "max_ms": round(self.max_ms, 3) if self.max_ms is not None else None,
        }


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
        latency_storage_path: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.max_nodes_per_host = max_nodes_per_host
        self.freshness_window_seconds = freshness_window_seconds
        self.storage_path = storage_path
        # Where the aggregated latency values are persisted. None disables the
        # snapshot, which keeps unit tests and ad-hoc managers off the disk.
        self.latency_storage_path = latency_storage_path
        self._lock = threading.Lock()

        # In-memory live session state
        self._active_fingerprint_nodes: Dict[str, Set[str]] = {}
        self._node_to_fingerprint: Dict[str, str] = {}
        self._cid_to_node: Dict[str, str] = {}
        self._node_to_cid: Dict[str, str] = {}
        self._authorized_nodes: Set[str] = {}
        self._authorized_nodes = set()
        self._node_leases: Dict[str, str] = {}

        # Latency telemetry (in-memory, reset on coordinator restart; a mirror
        # is written to latency_storage_path when configured).
        # Admission samples are keyed by channel (rest-preflight, grpc-handshake,
        # grpc-evaluate) plus an "overall" aggregate across all channels;
        # registration samples are keyed by which registry the node first
        # appeared in.
        self._stats_lock = threading.Lock()
        # Serializes snapshot writes: two concurrent handshakes must not write
        # the same .tmp path and clobber each other's JSON.
        self._metrics_write_lock = threading.Lock()
        self._admission_stats: Dict[str, _LatencyStats] = {}
        self._admission_stats_overall = _LatencyStats()
        self._registration_stats: Dict[str, _LatencyStats] = {}
        # (node_id, registry, node_started_at) already sampled, so a reconnect
        # is not mistaken for a fresh registration.
        self._registration_samples: Set[Tuple[str, str, float]] = set()
        self._save_latency_metrics()

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

    def _latency_snapshot(self) -> Dict[str, Any]:
        """Current aggregate latency values, shaped like the API summary block."""
        with self._stats_lock:
            return {
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "admission_handshake": {
                    **self._admission_stats_overall.as_dict(),
                    "by_channel": {
                        channel: stats.as_dict()
                        for channel, stats in sorted(self._admission_stats.items())
                    },
                },
                "registration": {
                    source: stats.as_dict()
                    for source, stats in sorted(self._registration_stats.items())
                },
            }

    def _save_latency_metrics(self) -> None:
        """Persists the aggregate latency values (atomic replace)."""
        if not self.latency_storage_path:
            return
        with self._metrics_write_lock:
            try:
                p = Path(self.latency_storage_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.name + ".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._latency_snapshot(), f, indent=2)
                os.replace(tmp, p)
            except Exception as e:
                logger.error("Failed to persist latency metrics: %s", e)

    def _log_attestation_latency(
        self,
        node_id: str,
        channel: str,
        started: float,
        verify_ms: Optional[float],
        outcome: str,
    ) -> float:
        """
        Emits one latency line per attestation attempt and returns the total.

        Measured from the moment the coordinator received the payload, so the
        total includes lock wait and the registry flush; ``verify`` is the
        payload verification alone. Logged for rejections too, otherwise the
        measured latency would be biased toward successful attestations.
        """
        total_ms = (time.perf_counter() - started) * 1000.0
        with self._stats_lock:
            self._admission_stats_overall.add(total_ms)
            stats = self._admission_stats.setdefault(channel, _LatencyStats())
            stats.add(total_ms)
            count, mean_ms, std_ms = stats.count, stats.mean_ms, stats.std_ms

        logger.info(
            "⏱️  Attestation latency | channel=%s | node='%s' | verify=%s | total=%.3f ms | outcome=%s "
            "| n=%d | mean=%.3f ms | std=%.3f ms",
            channel,
            node_id,
            f"{verify_ms:.3f} ms" if verify_ms is not None else "n/a",
            total_ms,
            outcome,
            count,
            mean_ms,
            std_ms,
        )
        self._save_latency_metrics()
        return total_ms

    def record_registration_latency(
        self,
        node_id: str,
        registry: str,
        node_started_at: Optional[float],
        registered_at: Optional[float] = None,
    ) -> Optional[float]:
        """
        Measures (and logs) node start -> first appearance in a registry.

        ``registry`` labels the observation point, e.g. ``first_admission``
        (persistent attestation registry) or ``grpc_node_registry`` (the
        coordinator's live node registry served by ``GET /api/v1/nodes``).

        ``node_started_at`` is node-reported epoch seconds and is not verified;
        clock skew can make the difference negative, which is clamped to 0.
        Each (node, registry, node_started_at) is sampled once, so reconnects of
        the same node boot do not enter the stats as fresh registrations.
        Returns the latency in ms, or None when the node reported no start time
        (or this boot was already sampled).
        """
        sample_key = (node_id, registry, float(node_started_at or 0.0))
        with self._stats_lock:
            if sample_key in self._registration_samples:
                logger.debug(
                    "Registration latency for node '%s' in %s already sampled "
                    "for this boot — skipping",
                    node_id,
                    registry,
                )
                return None
            self._registration_samples.add(sample_key)

        if node_started_at is None:
            logger.info(
                "⏱️  Registration latency | node='%s' | registry=%s | total=n/a "
                "(node reported no start time)",
                node_id,
                registry,
            )
            return None

        registered_at = time.time() if registered_at is None else registered_at
        latency_ms = max((registered_at - float(node_started_at)) * 1000.0, 0.0)

        with self._stats_lock:
            stats = self._registration_stats.setdefault(registry, _LatencyStats())
            stats.add(latency_ms)
            count, mean_ms, std_ms = stats.count, stats.mean_ms, stats.std_ms

        logger.info(
            "⏱️  Registration latency | node='%s' | registry=%s | total=%.3f ms | n=%d "
            "| mean=%.3f ms | std=%.3f ms",
            node_id,
            registry,
            latency_ms,
            count,
            mean_ms,
            std_ms,
        )
        self._save_latency_metrics()
        return latency_ms

    def attest_and_admit(
        self,
        request: AttestationRequest,
        channel: str = "internal",
    ) -> Tuple[bool, str, str]:
        """
        Evaluates hardware attestation and host quota.
        Returns: (is_admitted, message, lease_token)

        ``channel`` only labels the latency measurement (which coordinator path
        delivered the payload): ``rest-preflight``, ``grpc-handshake`` or
        ``grpc-evaluate``.
        """
        started = time.perf_counter()

        if not self.enabled:
            lease = uuid.uuid4().hex
            with self._lock:
                self._authorized_nodes.add(request.node_id)
                self._node_leases[request.node_id] = lease
            self._log_attestation_latency(
                request.node_id, channel, started, None, "disabled"
            )
            return True, "Attestation disabled, admitted unconditionally", lease

        payload_dict = request.model_dump()
        verify_started = time.perf_counter()
        is_valid, reason = verify_attestation_payload(
            payload_dict, max_skew_seconds=self.freshness_window_seconds
        )
        verify_ms = (time.perf_counter() - verify_started) * 1000.0
        if not is_valid:
            self._log_attestation_latency(
                request.node_id, channel, started, verify_ms, "rejected:invalid_payload"
            )
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
                self._log_attestation_latency(
                    node_id, channel, started, verify_ms, "admitted:refresh"
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
                self._log_attestation_latency(
                    node_id, channel, started, verify_ms, "rejected:host_quota"
                )
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

            # Registration latency: node start -> first appearance in this
            # registry. Only measured for a new node boot (no prior entry, or a
            # newer node_started_at), so reconnects and refreshes do not add
            # duplicate samples.
            previous_entry = self._registry_data["node_registry"].get(node_id) or {}
            previous_start = previous_entry.get("node_started_at")
            is_new_boot = not previous_entry or (
                request.node_started_at is not None
                and (
                    previous_start is None
                    or float(request.node_started_at) > float(previous_start)
                )
            )

            node_entry: Dict[str, Any] = {
                **previous_entry,
                "fingerprint": fingerprint,
                "authorized": True,
                "lease_token": lease,
                "admitted_at": now_iso,
                "node_started_at": request.node_started_at,
            }
            if is_new_boot:
                first_seen_at = time.time()
                node_entry["first_seen_at"] = first_seen_at
                node_entry["registration_latency_ms"] = self.record_registration_latency(
                    node_id,
                    "first_admission",
                    request.node_started_at,
                    registered_at=first_seen_at,
                )
            self._registry_data["node_registry"][node_id] = node_entry
            self._save_registry()

            logger.info(
                "✅ Node '%s' admitted from host %s (active on host: %d/%d)",
                node_id,
                fingerprint[:12],
                len(active_nodes),
                self.max_nodes_per_host,
            )
            self._log_attestation_latency(
                node_id, channel, started, verify_ms, "admitted:new"
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
            latency = self._latency_snapshot()
            return {
                "attestation_enabled": self.enabled,
                "max_nodes_per_host": self.max_nodes_per_host,
                "active_hosts_count": len(active_hosts),
                "active_hosts": active_hosts,
                "total_known_fingerprints": len(
                    self._registry_data.get("fingerprints", {})
                ),
                "total_sybil_attempts_blocked": rejection_count,
                # In-memory since process start; std is the sample standard
                # deviation (n-1) over all attempts, rejections included.
                "latency": latency,
            }
