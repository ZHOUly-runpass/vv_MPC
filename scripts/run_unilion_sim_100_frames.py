#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np

from r680_safety_planner.backbones import UniLionFrozenBackbone
from r680_safety_planner.data import FEATURE_BRIDGE_SCHEMA_VERSION, FeatureCacheRecord, save_feature_cache
from r680_safety_planner.interfaces import LidarFrame


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated UniLION worker on 100 captured simulation frames")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    project, capture, output_dir = args.project.resolve(), args.capture.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repository = project / "third_party" / "UniLION"
    config = repository / "projects" / "configs" / "unilion_swin_384_seq_e2e.py"
    checkpoint = project / "artifacts" / "checkpoints" / "unilion_lidar_backbone_init.safetensors"
    with np.load(capture, allow_pickle=False) as values:
        points, offsets, stamps = values["points"], values["offsets"], values["stamps"]
        fields = tuple(str(value) for value in values["fields"])
    if len(offsets) != 101:
        raise ValueError(f"capture must contain exactly 100 frames, got {len(offsets) - 1}")
    backend = UniLionFrozenBackbone(repository, config, checkpoint)
    reports, variances, last_features = [], [], None
    try:
        for index in range(100):
            frame = LidarFrame(points[offsets[index]:offsets[index + 1]].astype(np.float64), float(stamps[index]), "base_footprint", fields)
            last_features = backend.infer(frame)
            health = backend.healthcheck()["last_inference"]
            reports.append({key: health[key] for key in ("input_points", "voxel_count", "model_time_ms", "total_time_ms", "peak_memory_bytes")})
            variances.append(float(np.var(last_features.bev_feature.astype(np.float32))))
    finally:
        backend.close()
    assert last_features is not None
    cache = output_dir / "latest_frozen_scene_features.npz"
    temporary = output_dir / "latest_frozen_scene_features.tmp.npz"
    save_feature_cache(temporary, FeatureCacheRecord(
        last_features.bev_feature[0], last_features.timestamp_s, "base_footprint",
        file_hash(checkpoint), file_hash(config),
    ))
    os.replace(temporary, cache)
    cache_digest = file_hash(cache)
    report = {
        "status": "passed", "schema_version": FEATURE_BRIDGE_SCHEMA_VERSION,
        "input_kind": "gazebo_classic_c16_100_frame_capture", "frames": 100,
        "fields": list(fields), "feature_shape": list(last_features.bev_feature.shape),
        "feature_finite": bool(np.all(np.isfinite(last_features.bev_feature))),
        "feature_variance_min": min(variances), "feature_variance_median": float(np.median(variances)),
        "model_time_ms_median": float(np.median([item["model_time_ms"] for item in reports])),
        "total_time_ms_median": float(np.median([item["total_time_ms"] for item in reports])),
        "total_time_ms_p95": float(np.percentile([item["total_time_ms"] for item in reports], 95)),
        "peak_memory_bytes_max": max(item["peak_memory_bytes"] for item in reports),
        "checkpoint_sha256": file_hash(checkpoint), "config_sha256": file_hash(config),
        "cache_path": str(cache), "cache_sha256": cache_digest,
        "healthy": True, "updated_wall_time_s": time.time(),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_health = output_dir / "health.tmp.json"
    temporary_health.write_text(encoded, encoding="utf-8")
    os.replace(temporary_health, output_dir / "health.json")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
