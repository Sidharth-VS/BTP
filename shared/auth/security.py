"""
Basic API key authentication middleware for FedRAG services.
"""
import logging
import os
from typing import Optional

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

# Default dev key — override via environment variable API_KEY
_DEFAULT_API_KEY = "fedrag-dev-key"


def get_api_key() -> str:
    """Returns the configured API key from environment or default."""
    return os.environ.get("API_KEY", _DEFAULT_API_KEY)


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.
    Returns the key on success; raises 401 on failure.
    """
    expected = get_api_key()
    if x_api_key != expected:
        logger.warning("Invalid API key attempt: %s", x_api_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )
    return x_api_key
