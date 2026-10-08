"""
Coordinator entry point.

Threading model:
  - Main thread   → Flower gRPC server (requires main thread for signal handlers)
  - Daemon thread → uvicorn / FastAPI REST API

FlowerServer is initialized before uvicorn starts so all API requests
can access deps.get_flower_server() immediately.
"""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

import uvicorn
import yaml
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from coordinator.app.api import deps
from coordinator.app.flwr_server.server import FlowerServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("flwr").setLevel(logging.WARNING)
logger = logging.getLogger("coordinator")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

import sys

class CoordinatorConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    flower_server_address: str = "0.0.0.0:9091"
    flower_num_rounds: int = sys.maxsize
    attestation_enabled: bool = True
    max_nodes_per_host: int = 1
    attestation_freshness_seconds: float = 300.0
    attestation_storage_path: str = "workspace/coordinator/attestation_registry.json"


def load_config(yaml_path: str = "coordinator/config.yaml") -> CoordinatorConfig:
    path = Path(yaml_path)
    if not path.exists():
        return CoordinatorConfig()
    with open(path) as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}
    flat: Dict[str, Any] = {}
    flat["api_host"] = raw.get("server", {}).get("host", "0.0.0.0")
    flat["api_port"] = raw.get("server", {}).get("port", 8000)
    flat["flower_server_address"] = raw.get("flower", {}).get("server_address", "0.0.0.0:9091")
    flat["flower_num_rounds"] = raw.get("flower", {}).get("num_rounds", sys.maxsize)

    att_cfg = raw.get("attestation", {})
    if "enabled" in att_cfg:
        flat["attestation_enabled"] = att_cfg["enabled"]
    if "max_nodes_per_host" in att_cfg:
        flat["max_nodes_per_host"] = att_cfg["max_nodes_per_host"]
    if "freshness_window_seconds" in att_cfg:
        flat["attestation_freshness_seconds"] = float(att_cfg["freshness_window_seconds"])
    if "storage_path" in att_cfg:
        flat["attestation_storage_path"] = att_cfg["storage_path"]

    return CoordinatorConfig(**flat)


# ---------------------------------------------------------------------------
# FastAPI app (created at import time; FlowerServer injected before first request)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FedRAG Coordinator",
    description="Federated RAG system — broadcast retrieval via Flower gRPC",
    version="0.1.0",
)

from coordinator.app.api.v1 import endpoints as v1_endpoints  # noqa: E402
app.include_router(v1_endpoints.router, prefix="/api/v1")


@app.get("/")
def root():
    return {"service": "FedRAG Coordinator", "version": "0.1.0", "status": "running"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = os.environ.get("COORDINATOR_CONFIG", "coordinator/config.yaml")
    cfg = load_config(config_path)

    # 1. Build FlowerServer and register it globally BEFORE uvicorn starts
    from coordinator.app.flwr_server.admission import NodeAdmissionManager
    admission_mgr = NodeAdmissionManager(
        enabled=cfg.attestation_enabled,
        max_nodes_per_host=cfg.max_nodes_per_host,
        freshness_window_seconds=cfg.attestation_freshness_seconds,
        storage_path=cfg.attestation_storage_path,
    )
    flower_server = FlowerServer(
        server_address=cfg.flower_server_address,
        num_rounds=cfg.flower_num_rounds,
        admission_manager=admission_mgr,
    )
    deps.set_flower_server(flower_server)

    # 2. Start uvicorn in a daemon thread
    uvicorn_config = uvicorn.Config(
        app,
        host=cfg.api_host,
        port=cfg.api_port,
        log_level="info",
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    api_thread = threading.Thread(target=uvicorn_server.run, daemon=True, name="uvicorn")
    api_thread.start()

    # Give uvicorn a moment to bind the port before logging
    time.sleep(0.5)
    logger.info(
        "Coordinator started | REST: http://%s:%d | Flower gRPC: %s",
        cfg.api_host, cfg.api_port, cfg.flower_server_address,
    )

    # 3. Run Flower server in the MAIN thread (signal handlers require it)
    flower_server._run()  # blocks until all rounds complete


if __name__ == "__main__":
    main()
