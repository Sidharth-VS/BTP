"""
Flower gRPC server — runs in a background daemon thread.

The REST API interacts with it exclusively through FlowerServer.broker (QueryBroker).
"""
import logging
import threading
from typing import Optional

from flwr.server import SimpleClientManager, ServerConfig, start_server

from coordinator.app.flwr_server.strategy import FedRAGStrategy, QueryBroker

logger = logging.getLogger("coordinator.flower_server")


class FlowerServer:
    """
    Wraps flwr.server.start_server() in a daemon thread so it can run
    alongside the FastAPI process without blocking.
    """

    def __init__(
        self,
        server_address: str = "0.0.0.0:9091",
        num_rounds: int = 100_000,
    ) -> None:
        self.server_address = server_address
        self.num_rounds = num_rounds
        self.broker = QueryBroker()
        self.strategy = FedRAGStrategy(broker=self.broker)
        self._thread: Optional[threading.Thread] = None

    def start_background(self) -> None:
        """Start the Flower gRPC server in a daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="flower-grpc-server",
        )
        self._thread.start()
        logger.info("Flower gRPC server starting on %s", self.server_address)

    def _run(self) -> None:
        try:
            start_server(
                server_address=self.server_address,
                config=ServerConfig(num_rounds=self.num_rounds),
                strategy=self.strategy,
                client_manager=SimpleClientManager(),
            )
        except Exception as exc:
            logger.error("Flower server crashed: %s", exc, exc_info=True)

    def get_node_registry(self) -> dict:
        """Returns latest snapshot of the node registry (populated by health checks)."""
        return dict(self.strategy._node_registry)

    def connected_node_count(self) -> int:
        return len(self.strategy._node_registry)
