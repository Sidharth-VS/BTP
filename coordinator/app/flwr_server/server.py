"""
Flower gRPC server — runs in background or main thread.

The REST API interacts with it exclusively through FlowerServer.broker (QueryBroker).
"""
import logging
import sys
import threading
import time
from typing import Optional

from flwr.server import ServerConfig, start_server

from coordinator.app.flwr_server.client_manager import LoggingClientManager
from coordinator.app.flwr_server.strategy import FedRAGStrategy, QueryBroker

logger = logging.getLogger("coordinator.flower_server")


class FlowerServer:
    """
    Wraps flwr.server.start_server() with custom LoggingClientManager and an
    infinite restart loop so the coordinator runs indefinitely.
    """

    def __init__(
        self,
        server_address: str = "0.0.0.0:9091",
        num_rounds: int = sys.maxsize,
    ) -> None:
        self.server_address = server_address
        self.num_rounds = num_rounds
        self.broker = QueryBroker()
        self.strategy = FedRAGStrategy(broker=self.broker)
        self.client_manager = LoggingClientManager(strategy=self.strategy)
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
        logger.info("Flower gRPC server running indefinitely on %s (num_rounds=%d)", self.server_address, self.num_rounds)
        while True:
            try:
                start_server(
                    server_address=self.server_address,
                    config=ServerConfig(num_rounds=self.num_rounds),
                    strategy=self.strategy,
                    client_manager=self.client_manager,
                )
            except Exception as exc:
                logger.error("Flower server crashed: %s. Restarting loop in 1s...", exc, exc_info=True)
                time.sleep(1.0)
            else:
                logger.info("Flower server round loop completed. Restarting loop...")
                time.sleep(0.5)

    def get_node_registry(self) -> dict:
        """Returns latest snapshot of the node registry (populated by health checks)."""
        return dict(self.strategy._node_registry)

    def connected_node_count(self) -> int:
        return len(self.client_manager.clients)
