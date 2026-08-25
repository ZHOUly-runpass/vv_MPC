from __future__ import annotations

import json
import time

import numpy as np

from r680_safety_planner.data import FeatureCacheRecord, load_feature_health, save_feature_cache
from r680_safety_planner.data.feature_bridge import cache_sha256


HASH = "a" * 64


def test_feature_health_accepts_current_verified_cache(tmp_path):
    cache = tmp_path / "feature.npz"
    save_feature_cache(cache, FeatureCacheRecord(np.ones((2, 3, 4)), 1.0, "base", HASH, HASH))
    now = time.time()
    health = tmp_path / "health.json"
    health.write_text(json.dumps({
        "schema_version": "1.0", "healthy": True, "feature_finite": True,
        "updated_wall_time_s": now, "cache_path": str(cache), "cache_sha256": cache_sha256(cache),
    }), encoding="utf-8")
    assert load_feature_health(health, now + 0.1, 0.3).healthy


def test_feature_timeout_and_corruption_fail_closed(tmp_path):
    cache = tmp_path / "feature.npz"
    save_feature_cache(cache, FeatureCacheRecord(np.ones((2, 3, 4)), 1.0, "base", HASH, HASH))
    now = time.time()
    health = tmp_path / "health.json"
    payload = {"schema_version": "1.0", "healthy": True, "feature_finite": True,
               "updated_wall_time_s": now, "cache_path": str(cache), "cache_sha256": cache_sha256(cache)}
    health.write_text(json.dumps(payload), encoding="utf-8")
    assert load_feature_health(health, now + 1.0, 0.3).reason == "feature_timeout"
    payload["updated_wall_time_s"] = now + 1.0
    payload["cache_sha256"] = "0" * 64
    health.write_text(json.dumps(payload), encoding="utf-8")
    assert load_feature_health(health, now + 1.1, 0.3).reason == "feature_cache_invalid"
