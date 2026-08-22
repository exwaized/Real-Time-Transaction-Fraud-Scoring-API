"""
API key authentication dependency.
Reads expected key from env var FRAUD_API_KEY.
Inject as a FastAPI dependency on any route that needs protection.
"""
import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# Header name callers must send: X-API-Key: <your-key>
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    expected = os.getenv("FRAUD_API_KEY")

    # If no key configured in env, warn but allow (dev mode)
    if not expected:
        return "dev-mode-no-key-set"

    if not api_key or api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return api_key
