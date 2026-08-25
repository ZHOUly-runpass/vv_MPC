from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


FEATURE_BRIDGE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class FeatureHealthState:
    healthy: bool
    reason: str
    age_s: float
    payload: dict[str, Any]


def cache_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_feature_health(path: str | Path, now_s: float, timeout_s: float) -> FeatureHealthState:
    source = Path(path)
    if not source.is_file():
        return FeatureHealthState(False, "health_file_missing", float("inf"), {})
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != FEATURE_BRIDGE_SCHEMA_VERSION:
            return FeatureHealthState(False, "schema_mismatch", float("inf"), payload)
        timestamp = float(payload["updated_wall_time_s"])
        age = now_s - timestamp
        if age < -1e-3 or age > timeout_s:
            return FeatureHealthState(False, "feature_timeout", age, payload)
        cache = Path(payload["cache_path"])
        if not cache.is_file() or cache_sha256(cache) != payload.get("cache_sha256"):
            return FeatureHealthState(False, "feature_cache_invalid", age, payload)
        if not bool(payload.get("healthy")) or not bool(payload.get("feature_finite")):
            return FeatureHealthState(False, "worker_unhealthy", age, payload)
        return FeatureHealthState(True, "ok", age, payload)
    except Exception as error:
        return FeatureHealthState(False, f"health_parse_error:{type(error).__name__}", float("inf"), {})
