#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from r680_safety_planner.backbones import UniLionFrozenBackbone
from r680_safety_planner.data import FEATURE_BRIDGE_SCHEMA_VERSION, FeatureCacheRecord, save_feature_cache
from r680_safety_planner.interfaces import LidarFrame


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp.json"); temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live isolated UniLION feature-cache worker")
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True); parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--feature-grid", type=int, default=32)
    parser.add_argument("--poll-ms", type=float, default=5.0); args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    health = args.output_dir / "health.json"; checkpoint_hash = digest(args.checkpoint); config_hash = digest(args.model_config)
    running = True
    def stop(*_):
        nonlocal running; running = False
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try: backend = UniLionFrozenBackbone(args.repository, args.model_config, args.checkpoint)
    except Exception as error:
        atomic_json(health, {"schema_version": FEATURE_BRIDGE_SCHEMA_VERSION, "healthy": False,
                             "reason": f"startup:{type(error).__name__}:{error}", "updated_wall_time_s": time.time()}); return 2
    last_wall = -1.0; frame_index = 0
    try:
        while running:
            if not args.input.is_file(): time.sleep(args.poll_ms / 1000.0); continue
            try:
                with np.load(args.input, allow_pickle=False) as values:
                    wall = float(values["wall_time_s"])
                    if wall == last_wall: time.sleep(args.poll_ms / 1000.0); continue
                    points = values["points"].astype(np.float64); stamp = float(values["stamp_s"]); frame_id = str(values["frame_id"])
                last_wall = wall
                result = backend.infer(LidarFrame(points, stamp, frame_id, ("x", "y", "z", "intensity", "ring", "time")))
                feature = result.bev_feature.astype(np.float32)
                if feature.shape[-2:] != (args.feature_grid, args.feature_grid):
                    feature = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(feature),
                                                                      (args.feature_grid, args.feature_grid)).numpy()
                cache = args.output_dir / f"frozen_scene_features_slot_{frame_index % 16:02d}.npz"
                temporary = args.output_dir / f"frozen_scene_features_slot_{frame_index % 16:02d}.tmp.npz"
                save_feature_cache(temporary, FeatureCacheRecord(feature[0], stamp, frame_id, checkpoint_hash, config_hash))
                os.replace(temporary, cache)
                frame_index += 1
                runtime = backend.healthcheck().get("last_inference", {})
                atomic_json(health, {"schema_version": FEATURE_BRIDGE_SCHEMA_VERSION, "healthy": True, "reason": "ok",
                                     "updated_wall_time_s": time.time(), "cache_path": str(cache),
                                     "cache_sha256": digest(cache), "feature_shape": list(feature[0].shape),
                                     "feature_finite": bool(np.all(np.isfinite(feature))),
                                     "checkpoint_sha256": checkpoint_hash, "config_sha256": config_hash,
                                     "feature_variance_median": float(np.var(feature)), "inference": runtime})
            except Exception as error:
                atomic_json(health, {"schema_version": FEATURE_BRIDGE_SCHEMA_VERSION, "healthy": False,
                                     "reason": f"runtime:{type(error).__name__}:{error}", "updated_wall_time_s": time.time()})
    finally: backend.close()
    return 0


if __name__ == "__main__": raise SystemExit(main())
