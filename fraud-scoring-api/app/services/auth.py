"""
API key authentication dependency.

Keys are configured via env:
  FRAUD_API_KEYS    "client_a=<key>,client_b=<key>"   one key per caller
  FRAUD_API_KEY     single legacy key (treated as client "default")
  FRAUD_RATE_LIMITS "client_a=600,client_b=120"       per-client req/min overrides

verify_api_key returns the CLIENT NAME, never the secret, so the key is never
used as a Redis key or logged. Fails closed: if no keys are configured every
protected route returns 503 unless ALLOW_INSECURE_NO_AUTH=true (local dev only).
"""
import hmac
import logging
import os
from typing import Dict, Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger("auth")

# Header name callers must send: X-API-Key: <your-key>
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _parse_pairs(raw: str) -> Dict[str, str]:
    pairs = {}
    for item in raw.split(","):
        name, sep, value = item.strip().partition("=")
        if sep and name.strip() and value.strip():
            pairs[name.strip()] = value.strip()
    return pairs


def load_clients() -> Dict[str, str]:
    """client name -> key, from FRAUD_API_KEYS plus the legacy FRAUD_API_KEY."""
    clients = _parse_pairs(os.getenv("FRAUD_API_KEYS", ""))
    legacy = os.getenv("FRAUD_API_KEY")
    if legacy:
        clients.setdefault("default", legacy)
    return clients


def client_rate_limit(client: str) -> Optional[int]:
    """Per-client req/min override, or None to use the limiter's default."""
    raw = _parse_pairs(os.getenv("FRAUD_RATE_LIMITS", "")).get(client)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        logger.error(f"Invalid FRAUD_RATE_LIMITS value for client {client!r}; using default")
        return None


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    clients = load_clients()

    if not clients:
        if os.getenv("ALLOW_INSECURE_NO_AUTH", "").lower() == "true":
            return "dev"
        logger.error("No API keys configured (FRAUD_API_KEYS/FRAUD_API_KEY); rejecting (fail closed)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        )

    if api_key:
        presented = api_key.encode()
        matched = None
        # Compare against every key (no early exit) so timing doesn't reveal which matched.
        for name, key in clients.items():
            if hmac.compare_digest(presented, key.encode()):
                matched = name
        if matched:
            return matched

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing X-API-Key header",
    )
