"""
Coordinator API dependencies — global singletons shared across all requests.
"""
from typing import Optional
from coordinator.app.flwr_server.server import FlowerServer

_flower_server: Optional[FlowerServer] = None


def set_flower_server(server: FlowerServer) -> None:
    global _flower_server
    _flower_server = server


def get_flower_server() -> FlowerServer:
    if _flower_server is None:
        raise RuntimeError("FlowerServer has not been initialized yet")
    return _flower_server
