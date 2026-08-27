#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from r680_safety_planner.backbones import UniLionFrozenBackbone
from r680_safety_planner.data import FeatureCacheRecord, load_manifest, load_raw_training_frame, save_feature_cache, write_manifest
from r680_safety_planner.interfaces import LidarFrame


def digest(path: Path) -> str: return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate frozen UniLION features for extracted rosbag frames")
    parser.add_argument("--raw-manifest", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path); parser.add_argument("--model-config", type=Path)
    parser.add_argument("--checkpoint", type=Path); parser.add_argument("--feature-grid", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    manifest = args.raw_manifest.resolve(); source_root = manifest.parent; output = args.output_dir.resolve()
    repository = (args.repository or root/"third_party/UniLION").resolve()
    config = (args.model_config or repository/"projects/configs/unilion_swin_384_seq_e2e.py").resolve()
    checkpoint = (args.checkpoint or root/"artifacts/checkpoints/unilion_lidar_backbone_init.safetensors").resolve()
    if args.feature_grid <= 0: raise ValueError("feature-grid must be positive")
    checkpoint_hash, config_hash = digest(checkpoint), digest(config)
    entries = load_manifest(manifest); entries = entries[:args.limit] if args.limit else entries
    backend = UniLionFrozenBackbone(repository, config, checkpoint); generated = []
    try:
        for entry in entries:
            arrays, metadata, raw_hash = load_raw_training_frame(source_root/str(entry["path"]))
            if metadata["checkpoint_sha256"] != checkpoint_hash: raise ValueError(f"checkpoint hash mismatch: {entry['sample_id']}")
            frame = LidarFrame(arrays["points"].astype(np.float64), float(metadata["timestamp_s"]),
                               str(metadata["frame_id"]), ("x","y","z","intensity","ring","time"))
            result = backend.infer(frame); feature = result.bev_feature
            if feature.shape[-2:] != (args.feature_grid, args.feature_grid):
                import torch
                feature = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(feature.astype(np.float32)),
                                                                  (args.feature_grid, args.feature_grid)).numpy()
            path = output/"features"/f"{entry['sample_id']}.npz"
            save_feature_cache(path, FeatureCacheRecord(feature[0], float(metadata["timestamp_s"]), str(metadata["frame_id"]),
                                                         checkpoint_hash, config_hash))
            generated.append({**entry, "raw_payload_sha256": raw_hash, "feature_path": str(path.relative_to(output)),
                              "feature_sha256": digest(path), "checkpoint_sha256": checkpoint_hash,
                              "feature_config_sha256": config_hash, "feature_shape": list(feature[0].shape)})
    finally: backend.close()
    write_manifest(output/"feature_manifest.jsonl", generated)
    print(json.dumps({"samples": len(generated), "manifest": str(output/"feature_manifest.jsonl"),
                      "checkpoint_sha256": checkpoint_hash, "feature_config_sha256": config_hash}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
