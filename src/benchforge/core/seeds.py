"""Shared SHA-256 seed derivation (preserves the original search contract)."""

import hashlib
import json


def derive_seed(seed: int, *parts: object) -> int:
    payload = json.dumps([seed, *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")
