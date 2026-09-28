"""
Hardware Attestation & Host Fingerprinting module.

Extracts real host hardware identifiers (Motherboard UUID, Machine ID, MAC, CPU)
and creates canonical SHA-256 fingerprints and verifiable attestation payloads.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("shared.auth.attestation")


class HardwareCollector:
    """Probes physical and OS-level hardware identifiers on Linux/Unix systems."""

    @staticmethod
    def get_dmi_product_uuid() -> Optional[str]:
        """Reads motherboard / system product UUID from /sys/class/dmi/id/product_uuid."""
        paths = [
            Path("/sys/class/dmi/id/product_uuid"),
            Path("/sys/devices/virtual/dmi/id/product_uuid"),
        ]
        for p in paths:
            try:
                if p.exists() and os.access(p, os.R_OK):
                    content = p.read_text(encoding="utf-8").strip()
                    if content and content != "None":
                        return content.lower()
            except Exception as e:
                logger.debug("Could not read DMI UUID from %s: %s", p, e)
        return None

    @staticmethod
    def get_system_machine_id() -> Optional[str]:
        """Reads system machine ID from /etc/machine-id or /var/lib/dbus/machine-id."""
        paths = [
            Path("/etc/machine-id"),
            Path("/var/lib/dbus/machine-id"),
        ]
        for p in paths:
            try:
                if p.exists() and os.access(p, os.R_OK):
                    content = p.read_text(encoding="utf-8").strip()
                    if content:
                        return content.lower()
            except Exception as e:
                logger.debug("Could not read machine-id from %s: %s", p, e)
        return None

    @staticmethod
    def get_primary_mac_address() -> str:
        """Retrieves the primary network interface hardware MAC address."""
        # uuid.getnode() returns the 48-bit integer hardware address
        node_int = uuid.getnode()
        mac = ":".join(f"{(node_int >> ele) & 0xFF:02x}" for ele in range(40, -1, -8))
        return mac.lower()

    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Extracts CPU model, core count, and stepping from /proc/cpuinfo or platform."""
        info: Dict[str, Any] = {
            "model": platform.processor() or "unknown",
            "cores": os.cpu_count() or 1,
            "architecture": platform.machine(),
            "system": platform.system(),
        }

        cpuinfo_path = Path("/proc/cpuinfo")
        if cpuinfo_path.exists() and os.access(cpuinfo_path, os.R_OK):
            try:
                with open(cpuinfo_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if ":" in line:
                            k, v = [x.strip() for x in line.split(":", 1)]
                            if k == "model name" and info["model"] in ("unknown", ""):
                                info["model"] = v
                            elif k == "cpu family" and "family" not in info:
                                info["family"] = v
                            elif k == "model" and "model_id" not in info:
                                info["model_id"] = v
                            elif k == "stepping" and "stepping" not in info:
                                info["stepping"] = v
            except Exception as e:
                logger.debug("Could not parse /proc/cpuinfo: %s", e)

        return info

    @classmethod
    def collect_raw_identifiers(cls) -> Dict[str, Any]:
        """Gathers all available hardware identifiers."""
        return {
            "dmi_uuid": cls.get_dmi_product_uuid(),
            "machine_id": cls.get_system_machine_id(),
            "mac_address": cls.get_primary_mac_address(),
            "cpu": cls.get_cpu_info(),
        }


def generate_host_fingerprint(raw_identifiers: Optional[Dict[str, Any]] = None) -> str:
    """
    Computes a deterministic, canonical SHA-256 fingerprint from hardware signals.
    """
    if raw_identifiers is None:
        raw_identifiers = HardwareCollector.collect_raw_identifiers()

    # Build canonical representation with ordered keys
    canonical_components = {
        "dmi_uuid": raw_identifiers.get("dmi_uuid") or "dmi-unavailable",
        "machine_id": raw_identifiers.get("machine_id") or "machine-id-unavailable",
        "mac_address": raw_identifiers.get("mac_address") or "mac-unavailable",
        "cpu_model": raw_identifiers.get("cpu", {}).get("model", ""),
        "cpu_cores": raw_identifiers.get("cpu", {}).get("cores", 1),
        "cpu_arch": raw_identifiers.get("cpu", {}).get("architecture", ""),
        "cpu_stepping": raw_identifiers.get("cpu", {}).get("stepping", ""),
    }

    serialized = json.dumps(canonical_components, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class NodeAttestationPayload:
    """Verifiable attestation payload transmitted by nodes."""
    node_id: str
    fingerprint: str
    timestamp: float
    nonce: str
    hardware_summary: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NodeAttestationPayload:
        return cls(
            node_id=str(data.get("node_id", "")),
            fingerprint=str(data.get("fingerprint", "")),
            timestamp=float(data.get("timestamp", 0.0)),
            nonce=str(data.get("nonce", "")),
            hardware_summary=dict(data.get("hardware_summary", {})),
            signature=data.get("signature"),
        )


def create_attestation_payload(node_id: str) -> NodeAttestationPayload:
    """
    Gathers local hardware info, computes fingerprint, and creates an attestation payload.
    """
    raw_hw = HardwareCollector.collect_raw_identifiers()
    fingerprint = generate_host_fingerprint(raw_hw)
    cpu_info = raw_hw.get("cpu", {})

    hardware_summary = {
        "cpu_model": cpu_info.get("model", "unknown"),
        "cpu_cores": cpu_info.get("cores", 1),
        "arch": cpu_info.get("architecture", platform.machine()),
        "os": cpu_info.get("system", platform.system()),
    }

    return NodeAttestationPayload(
        node_id=node_id,
        fingerprint=fingerprint,
        timestamp=time.time(),
        nonce=uuid.uuid4().hex,
        hardware_summary=hardware_summary,
    )


def verify_attestation_payload(
    payload_dict: Dict[str, Any],
    max_skew_seconds: float = 300.0,
) -> Tuple[bool, str]:
    """
    Validates payload integrity and freshness window.
    Returns (is_valid, reason).
    """
    if not isinstance(payload_dict, dict):
        return False, "Payload must be a dictionary"

    node_id = payload_dict.get("node_id")
    fingerprint = payload_dict.get("fingerprint")
    timestamp = payload_dict.get("timestamp")
    nonce = payload_dict.get("nonce")

    if not node_id or not isinstance(node_id, str):
        return False, "Missing or invalid 'node_id'"

    if not fingerprint or not isinstance(fingerprint, str) or len(fingerprint) != 64:
        return False, "Missing or invalid 64-character SHA-256 'fingerprint'"

    if timestamp is None or not isinstance(timestamp, (int, float)):
        return False, "Missing or invalid 'timestamp'"

    now = time.time()
    if abs(now - float(timestamp)) > max_skew_seconds:
        return False, f"Timestamp skew ({abs(now - float(timestamp)):.1f}s) exceeds allowed limit of {max_skew_seconds}s"

    if not nonce or not isinstance(nonce, str):
        return False, "Missing or invalid 'nonce'"

    return True, "Valid"
