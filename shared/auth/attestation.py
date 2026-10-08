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

    # --- Attested host-identity mount points -------------------------------
    # Containers cannot read the host's DMI/UUID tables, so the deployment
    # bind-mounts the physical host's machine-id into every node container.
    # These paths are trusted BECAUSE the compose file mounts them read-only.
    HOST_MACHINE_ID_PATHS = [
        Path("/host/etc/machine-id"),   # docker-compose: /etc/machine-id:/host/etc/machine-id:ro
        Path("/etc/host-machine-id"),   # alternative mount convention
    ]

    # --- Containerization detection ----------------------------------------
    @staticmethod
    def is_running_in_container() -> bool:
        """Detects whether we are inside a container (Docker/Podman/LXC)."""
        if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
            return True
        try:
            with open("/proc/1/cgroup", "r", encoding="utf-8") as f:
                content = f.read()
            if "docker" in content or "containerd" in content or "lxc" in content:
                return True
        except Exception:
            pass
        return False

    @classmethod
    def get_host_machine_id(cls) -> Optional[str]:
        """
        Returns the PHYSICAL host machine-id.

        Priority:
        1. Attested bind-mount of the host's machine-id (see compose file).
        2. Bare-metal paths (/etc/machine-id) — correct outside containers.

        Falls back to None inside containers without a mount. Never uses the
        container's own machine-id, which differs per container and would
        defeat host-level Sybil detection.
        """
        for p in cls.HOST_MACHINE_ID_PATHS:
            try:
                if p.exists() and os.access(p, os.R_OK):
                    content = p.read_text(encoding="utf-8").strip()
                    if content:
                        return content.lower()
            except Exception as e:
                logger.debug("Could not read host machine-id from %s: %s", p, e)

        if not cls.is_running_in_container():
            return cls.get_system_machine_id()

        logger.warning(
            "Running in a container without an attested host machine-id mount — "
            "host identity degraded (mount /etc/machine-id:/host/etc/machine-id:ro)"
        )
        return None

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

    # Virtual / container interfaces whose MACs are NOT host identity.
    _VIRTUAL_IFACE_PREFIXES = (
        "veth",    # docker container links — unique per container!
        "lo", "docker0", "br-", "virbr", "vmnet", "tun", "tap",
        "wg", "tailscale",
    )

    @classmethod
    def get_physical_mac_address(cls) -> Optional[str]:
        """
        Retrieves the first PHYSICAL NIC's MAC address.

        Virtual interfaces (docker veth, loopback, bridges) are excluded —
        inside a container, uuid.getnode() resolves to the container's own
        veth MAC, which differs per container and previously fragmented the
        host fingerprint.
        """
        try:
            net_dir = Path("/sys/class/net")
            if net_dir.exists():
                for iface in sorted(net_dir.iterdir()):
                    name = iface.name
                    if name.startswith(cls._VIRTUAL_IFACE_PREFIXES):
                        continue
                    # Double-check via sysfs: interfaces without a device link
                    # are virtual (e.g. lo).
                    if not (iface / "device").exists():
                        continue
                    addr_file = iface / "address"
                    if addr_file.exists():
                        mac = addr_file.read_text(encoding="utf-8").strip().lower()
                        if mac and mac != "00:00:00:00:00:00":
                            return mac
        except Exception as e:
            logger.debug("Could not enumerate network interfaces: %s", e)

        # Bare-metal fallback (also covers macOS where /sys/class/net is absent)
        node_int = uuid.getnode()
        if node_int & 0x010000000000:  # locally-administered bit set → virtualized
            logger.debug("uuid.getnode() returned a locally-administered (virtual) MAC")
        return ":".join(f"{(node_int >> ele) & 0xFF:02x}" for ele in range(40, -1, -8)).lower()

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
            "machine_id": cls.get_host_machine_id(),
            "mac_address": cls.get_physical_mac_address(),
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
        "machine_id": raw_identifiers.get("machine_id") or "machine-id-unavailable"
        # "mac_address": raw_identifiers.get("mac_address") or "mac-unavailable",
        # "cpu_model": raw_identifiers.get("cpu", {}).get("model", ""),
        # "cpu_cores": raw_identifiers.get("cpu", {}).get("cores", 1),
        # "cpu_arch": raw_identifiers.get("cpu", {}).get("architecture", ""),
        # "cpu_stepping": raw_identifiers.get("cpu", {}).get("stepping", ""),
    }

    # Visibility into fingerprint composition — makes misconfigured identity
    # sources (e.g. containers without the machine-id mount) diagnosable.
    logger.info(
        "Host fingerprint composition: dmi=%s machine_id=%s mac=%s cpu=%s/%s",
        (canonical_components["dmi_uuid"] or "")[:12],
        canonical_components["machine_id"][:12]
        # canonical_components["mac_address"],
        # canonical_components["cpu_arch"],
        # canonical_components["cpu_cores"],
    )

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
