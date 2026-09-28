import time
import pytest
from shared.auth.attestation import (
    HardwareCollector,
    generate_host_fingerprint,
    create_attestation_payload,
    verify_attestation_payload,
)


def test_hardware_collector_probes():
    raw = HardwareCollector.collect_raw_identifiers()
    assert "mac_address" in raw
    assert "cpu" in raw
    assert isinstance(raw["cpu"], dict)
    assert raw["cpu"].get("cores", 0) >= 1


def test_fingerprint_deterministic():
    fp1 = generate_host_fingerprint()
    fp2 = generate_host_fingerprint()
    assert isinstance(fp1, str)
    assert len(fp1) == 64  # SHA-256 hex
    assert fp1 == fp2  # Deterministic across calls on same host


def test_attestation_payload_creation_and_verification():
    payload = create_attestation_payload("node-test-1")
    assert payload.node_id == "node-test-1"
    assert len(payload.fingerprint) == 64
    assert payload.timestamp > 0

    p_dict = payload.to_dict()
    is_valid, reason = verify_attestation_payload(p_dict)
    assert is_valid is True
    assert reason == "Valid"


def test_attestation_tampering_detection():
    payload = create_attestation_payload("node-test-1")
    p_dict = payload.to_dict()

    # Tamper with fingerprint
    tampered_fp = p_dict.copy()
    tampered_fp["fingerprint"] = "bad-fingerprint"
    is_valid, reason = verify_attestation_payload(tampered_fp)
    assert is_valid is False

    # Tamper with timestamp (replay attack simulation)
    tampered_time = p_dict.copy()
    tampered_time["timestamp"] = time.time() - 1000.0  # Expired
    is_valid, reason = verify_attestation_payload(tampered_time, max_skew_seconds=60.0)
    assert is_valid is False
    assert "exceeds allowed limit" in reason
