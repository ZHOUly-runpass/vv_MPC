#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from r680_safety_planner.data import (TrainingSample, load_feature_cache, load_manifest, load_raw_training_frame,
                                       save_training_sample, write_manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble raw rosbag frames and frozen features into schema 1.0 samples")
    parser.add_argument("--raw-manifest", type=Path, required=True); parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--split", default="unassigned")
    args = parser.parse_args(); raw_manifest = args.raw_manifest.resolve(); feature_manifest = args.feature_manifest.resolve()
    raw_root, feature_root, output = raw_manifest.parent, feature_manifest.parent, args.output_dir.resolve()
    raw_entries = {str(item["sample_id"]): item for item in load_manifest(raw_manifest)}
    feature_entries = {str(item["sample_id"]): item for item in load_manifest(feature_manifest)}
    if raw_entries.keys() != feature_entries.keys():
        raise ValueError(f"raw/feature sample IDs differ: raw={len(raw_entries)} feature={len(feature_entries)}")
    results = []
    for sample_id, raw_entry in raw_entries.items():
        arrays, metadata, raw_hash = load_raw_training_frame(raw_root/str(raw_entry["path"]))
        feature_entry = feature_entries[sample_id]; feature = load_feature_cache(feature_root/str(feature_entry["feature_path"]))
        if feature.checkpoint_hash != metadata["checkpoint_sha256"]: raise ValueError(f"checkpoint mismatch: {sample_id}")
        horizon, state_dim = arrays["candidate_states"].shape[1:]
        sample = TrainingSample(
            points=arrays["points"], features=feature.bev_feature, route=arrays["route"], ego_state=arrays["ego_state"],
            costmap=arrays["costmap"], obstacle_states=arrays["obstacle_states"],
            obstacle_lengths=arrays["obstacle_lengths"], obstacle_widths=arrays["obstacle_widths"],
            obstacle_covariance=arrays["obstacle_covariance"], obstacle_valid_mask=arrays["obstacle_valid_mask"],
            candidate_states=arrays["candidate_states"], candidate_controls=arrays["candidate_controls"],
            candidate_timestamps_s=arrays["candidate_timestamps_s"], teacher_outcome_codes=np.empty(0, np.int8),
            teacher_feasible=np.empty(0, np.bool_), teacher_h_min=np.empty(0, np.float32),
            teacher_slack_max=np.empty(0, np.float32), teacher_solve_time_ms=np.empty(0, np.float32),
            teacher_states=np.empty((0, horizon, state_dim), np.float32),
            teacher_controls=np.empty((0, horizon-1, arrays["candidate_controls"].shape[-1]), np.float32),
            teacher_selected_index=-1,
            metadata={**metadata, "feature_status": "complete", "feature_config_sha256": feature.config_hash,
                      "feature_cache_sha256": feature_entry["feature_sha256"], "raw_payload_sha256": raw_hash,
                      "feature_grid": list(feature.bev_feature.shape[-2:])})
        path = output/"samples"/f"{sample_id}.npz"; payload_hash = save_training_sample(path, sample)
        results.append({**raw_entry, "path": str(path.relative_to(output)), "payload_sha256": payload_hash,
                        "split": args.split, "checkpoint_sha256": feature.checkpoint_hash})
    write_manifest(output/"raw_manifest.jsonl", results)
    print(json.dumps({"samples": len(results), "manifest": str(output/"raw_manifest.jsonl")}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
