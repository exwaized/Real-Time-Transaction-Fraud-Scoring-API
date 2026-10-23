"""
Pseudonymise card identifiers before they reach logs or Redis keys.

HMAC-SHA256 keyed with CARD_HASH_SALT. Set a long random salt in production:
without one the hash of a low-entropy card number can be brute-forced. API
responses still return the caller's own card_id unchanged.
"""
import hashlib
import hmac
import os


def card_ref(card_id: str) -> str:
    salt = os.getenv("CARD_HASH_SALT", "").encode()
    return hmac.new(salt, str(card_id).encode(), hashlib.sha256).hexdigest()[:16]
