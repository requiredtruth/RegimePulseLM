"""Strict parsing and deterministic serialization."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


class RegimePulseError(ValueError):
    """A bounded input, schema, or reproducibility failure."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_json(path: str | Path, max_bytes: int = 1_000_000) -> Any:
    source = Path(path)
    try:
        if source.stat().st_size > max_bytes:
            raise RegimePulseError(f"{source} exceeds the size limit")
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegimePulseError(f"cannot read JSON from {source}: {exc}") from exc


def _utc(value: str, field: str) -> datetime:
    if not value.endswith("Z"):
        raise RegimePulseError(f"{field} must end in Z")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RegimePulseError(f"{field} is not valid ISO-8601 UTC") from exc
    if result.isoformat().replace("+00:00", "Z") != value:
        raise RegimePulseError(f"{field} must use canonical seconds")
    return result


def load_prices(path: str | Path, max_bytes: int = 20_000_000) -> list[dict[str, Any]]:
    """Read strict timestamp,close,volume CSV without dialect guessing."""
    source = Path(path)
    try:
        if source.stat().st_size > max_bytes:
            raise RegimePulseError(f"{source} exceeds the size limit")
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RegimePulseError(f"cannot read CSV from {source}: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text), strict=True)
    if reader.fieldnames != ["timestamp", "close", "volume"]:
        raise RegimePulseError("CSV header must be exactly timestamp,close,volume")
    rows, previous = [], None
    for number, raw in enumerate(reader, 2):
        try:
            current = _utc(raw["timestamp"], f"row {number}.timestamp")
            close, volume = float(raw["close"]), float(raw["volume"])
        except (TypeError, ValueError) as exc:
            raise RegimePulseError(f"row {number} contains an invalid number") from exc
        if previous is not None and current <= previous:
            raise RegimePulseError("timestamps must be strictly increasing")
        if not math.isfinite(close) or close <= 0 or not math.isfinite(volume) or volume < 0:
            raise RegimePulseError(f"row {number} close must be positive and volume non-negative")
        previous = current
        rows.append({"timestamp": raw["timestamp"], "close": close, "volume": volume})
    return rows
