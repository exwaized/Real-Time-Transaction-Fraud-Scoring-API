"""
API key authentication dependency.
Reads expected key from env var FRAUD_API_KEY.
Inject as a FastAPI dependency on any route that needs protection.

Fails closed: if FRAUD_API_KEY is unset, every protected route returns 503
unless ALLOW_INSECURE_NO_AUTH=true is set explicitly (local dev only).
"""
import hmac
import logging
import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger("auth")

# Header name callers must send: X-API-Key: <your-key>
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    expected = os.getenv("FRAUD_API_KEY")

    if not expected:
        if os.getenv("ALLOW_INSECURE_NO_AUTH", "").lower() == "true":
            return "dev-mode-no-key-set"
        logger.error("FRAUD_API_KEY is not set; rejecting request (fail closed)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        )

    if not api_key or not hmac.compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return api_key
