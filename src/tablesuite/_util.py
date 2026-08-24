"""Internal deterministic value helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import suppress
from datetime import date, datetime
from decimal import Decimal
from typing import Any

MISSING_VALUES = {"", "nan", "none", "null", "?"}


def normalize_value(value: Any) -> Any:
    """Convert common table scalars into JSON-compatible source values."""

    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        with suppress(TypeError, ValueError):
            value = value.item()
    if value is None:
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes | bytearray):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [normalize_value(item) for item in value]
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize a source value deterministically."""

    return json.dumps(
        normalize_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_id(prefix: str, value: str) -> str:
    """Create a deterministic identifier; this is not a source-file checksum."""

    digest = hashlib.sha256(value.encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def stable_order(value: str, seed: int = 0) -> str:
    """Return a deterministic ordering key."""

    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def stable_value_key(value: Any) -> str:
    """Normalize a source value for within-schema fact equivalence."""

    normalized = normalize_value(value)
    if normalized is None:
        return "<missing>"
    text = str(normalized).strip()
    lowered = text.casefold()
    if lowered in MISSING_VALUES:
        return "<missing>"
    if lowered in {"true", "yes"}:
        return "true"
    if lowered in {"false", "no"}:
        return "false"
    try:
        return f"{float(text.replace(',', '')):.8g}"
    except ValueError:
        return re.sub(r"\s+", " ", lowered)


def normalize_identifier(value: Any) -> str:
    """Normalize a source column name for within-schema identity."""

    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold())
    return text.strip("_") or "unknown"


def valid_target(value: Any, family: str) -> bool:
    """Return whether a source target is usable by one prediction family."""

    normalized = normalize_value(value)
    if normalized is None:
        return False
    if family == "regression":
        return (
            isinstance(normalized, int | float)
            and not isinstance(normalized, bool)
            and math.isfinite(normalized)
        )
    return True
