"""
TASR trust state persistence layer.
Saves and loads router trust states and trajectory histories.
"""
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("coordinator.tasr.db")


class TASRDatabase:
    """JSON-backed persistence for TASR trust states and router histories."""

    def __init__(self, db_path: str = "workspace/tasr_state.json") -> None:
        self.db_path = db_path

    def save_state(self, router_state: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(router_state, f, indent=2)
            logger.debug("Saved TASR state to %s", self.db_path)
        except Exception as e:
            logger.error("Failed to save TASR state: %s", e)

    def load_state(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.db_path):
            return None
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load TASR state: %s", e)
            return None
