"""
FedRAG custom ClientManager for Flower gRPC server.

Tracks client connection lifecycle events (connect, disconnect)
and coordinates with FedRAGStrategy for node registry updates.
"""
import logging
from typing import Optional, TYPE_CHECKING
from flwr.server import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

if TYPE_CHECKING:
    from coordinator.app.flwr_server.strategy import FedRAGStrategy

logger = logging.getLogger("coordinator.client_manager")


class LoggingClientManager(SimpleClientManager):
    """
    Subclasses SimpleClientManager to add structured logging when nodes
    join or disconnect from the Flower gRPC network.
    """

    def __init__(self, strategy: Optional["FedRAGStrategy"] = None) -> None:
        super().__init__()
        self.strategy = strategy

    def register(self, client: ClientProxy) -> bool:
        registered = super().register(client)
        if registered:
            logger.info(
                "🟢 Node connected to gRPC network | CID: %s (total connected nodes: %d)",
                client.cid,
                len(self.clients),
            )
        return registered

    def unregister(self, client: ClientProxy) -> None:
        cid = client.cid
        existed = cid in self.clients

        # Retrieve node_id from strategy registry before removing
        node_id: Optional[str] = None
        if self.strategy and hasattr(self.strategy, "_node_registry"):
            node_info = self.strategy._node_registry.pop(cid, None)
            if node_info:
                node_id = node_info.get("node_id")

        super().unregister(client)

        if existed:
            if node_id:
                logger.info(
                    "🔴 Node disconnected | Node ID: '%s' (CID: %s, remaining connected nodes: %d)",
                    node_id,
                    cid,
                    len(self.clients),
                )
            else:
                logger.info(
                    "🔴 Node disconnected | CID: %s (remaining connected nodes: %d)",
                    cid,
                    len(self.clients),
                )
