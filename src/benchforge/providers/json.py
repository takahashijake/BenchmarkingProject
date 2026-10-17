"""Strict JSON helpers shared with data-only extension declarations."""

import json
from typing import Any


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_object(source: str) -> str:
    value = json.loads(source, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
