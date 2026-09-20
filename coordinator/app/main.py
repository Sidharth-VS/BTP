"""
Coordinator service main entry point.

Starts:
1. Flower server (gRPC on port 9091) in a background daemon thread
   — nodes connect here via flwr.client.start_client()
2. FastAPI REST API (HTTP on port 8000)
   — clients submit queries here
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from coordinator.app.api import deps
from coordinator.app.flwr_server.server import FlowerServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("coordinator")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CoordinatorConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # REST API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Flower gRPC server
    flower_server_address: str = "0.0.0.0:9091"
    flower_num_rounds: int = 100_000


def load_coordinator_config(yaml_path: str = "coordinator/config.yaml") -> CoordinatorConfig:
    path = Path(yaml_path)
    if not path.exists():
        logger.warning("Config not found at %s, using defaults", yaml_path)
        return CoordinatorConfig()
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}
    # Flatten nested YAML into flat fields
    flat: Dict[str, Any] = {}
    server_cfg = raw.get("server", {})
    flower_cfg = raw.get("flower", {})
    flat["api_host"] = server_cfg.get("host", "0.0.0.0")
    flat["api_port"] = server_cfg.get("port", 8000)
    flat["flower_server_address"] = flower_cfg.get("server_address", "0.0.0.0:9091")
    flat["flower_num_rounds"] = flower_cfg.get("num_rounds", 100_000)
    return CoordinatorConfig(**flat)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("COORDINATOR_CONFIG", "coordinator/config.yaml")
    cfg = load_coordinator_config(config_path)

    # Start Flower gRPC server in background thread
    flower_server = FlowerServer(
        server_address=cfg.flower_server_address,
        num_rounds=cfg.flower_num_rounds,
    )
    flower_server.start_background()
    deps.set_flower_server(flower_server)

    logger.info(
        "Coordinator started | REST API: %s:%d | Flower gRPC: %s",
        cfg.api_host, cfg.api_port, cfg.flower_server_address,
    )

    yield

    logger.info("Coordinator shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FedRAG Coordinator",
    description="Federated RAG system coordinator with broadcast routing",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount API routes
from coordinator.app.api.v1 import endpoints as v1_endpoints  # noqa: E402
app.include_router(v1_endpoints.router, prefix="/api/v1")


@app.get("/")
def root():
    return {"service": "FedRAG Coordinator", "version": "0.1.0", "status": "running"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    config_path = os.environ.get("COORDINATOR_CONFIG", "coordinator/config.yaml")
    cfg = load_coordinator_config(config_path)
    uvicorn.run(
        "coordinator.app.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        log_level="info",
    )
