#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from r680_safety_planner.data import directory_sha256, save_raw_training_frame, write_manifest


def yaw(q) -> float:
    return math.atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))


def route_array(message, count: int = 32) -> np.ndarray:
    if not message.poses: raise ValueError("empty route")
    indices = np.rint(np.linspace(0, len(message.poses)-1, count)).astype(int)
    return np.asarray([[message.poses[i].pose.position.x, message.poses[i].pose.position.y,
                        yaw(message.poses[i].pose.orientation), 0.25] for i in indices], dtype=np.float32)


def costmap_array(message) -> np.ndarray:
    grid = np.asarray(message.data, dtype=np.int16).reshape(message.info.height, message.info.width)
    occupied = np.clip(grid, 0, 100).astype(np.float32) / 100.0
    return np.stack([occupied, np.zeros_like(occupied), (grid < 0).astype(np.float32)])


def trace_arrays(candidates: dict, obstacles: dict) -> dict[str, np.ndarray]:
    candidate_items, obstacle_items = candidates["items"], obstacles["items"]
    states = np.asarray([item["states"] for item in candidate_items], dtype=np.float32)
    controls = np.asarray([item["controls"] for item in candidate_items], dtype=np.float32)
    horizon = states.shape[1]
    if obstacle_items:
        obstacle_states = np.asarray([item["states"] for item in obstacle_items], dtype=np.float32)
        lengths = np.asarray([item["lengths"] for item in obstacle_items], dtype=np.float32)
        widths = np.asarray([item["widths"] for item in obstacle_items], dtype=np.float32)
        covariance = np.asarray([item["covariance"] for item in obstacle_items], dtype=np.float32)
        valid = np.asarray([item["valid_mask"] for item in obstacle_items], dtype=np.bool_)
    else:
        obstacle_states = np.empty((0, horizon, 6), np.float32); lengths = np.empty((0, horizon), np.float32)
        widths = np.empty((0, horizon), np.float32); covariance = np.empty((0, horizon, 2, 2), np.float32)
        valid = np.empty((0, horizon), np.bool_)
    return {"obstacle_states": obstacle_states, "obstacle_lengths": lengths, "obstacle_widths": widths,
            "obstacle_covariance": covariance, "obstacle_valid_mask": valid, "candidate_states": states,
            "candidate_controls": controls, "candidate_timestamps_s": np.asarray(candidates["timestamps_s"], np.float32)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract synchronized raw training frames from a ROS 2 bag")
    parser.add_argument("--bag", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--config", type=Path,
        default=Path("simulation/ros2_ws/src/r680_sim_bringup/config/scenarios.yaml"))
    parser.add_argument("--sample-hz", type=float, default=2.0); parser.add_argument("--max-age-ms", type=float, default=750.0)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    bag = args.bag.resolve(); output = args.output_dir.resolve(); config = args.config if args.config.is_absolute() else root/args.config
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root/args.checkpoint
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs_py import point_cloud2
    except ImportError as error: raise RuntimeError("run this extractor after sourcing ROS 2 Humble") from error
    metadata_path = Path(str(bag) + "_run.json")
    run = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    source_hash = directory_sha256(bag); config_hash = sha256(config.read_bytes()).hexdigest()
    checkpoint_hash = sha256(checkpoint.read_bytes()).hexdigest()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    code_hash = sha256(revision.encode()).hexdigest()
    reader = rosbag2_py.SequentialReader(); reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    required = {"/points", "/odom", "/plan", "/local_costmap/costmap_raw", "/planning/candidates", "/planning/obstacle_predictions"}
    missing = required-set(types)
    if missing: raise RuntimeError(f"bag is missing required topics: {sorted(missing)}")
    latest = {}; entries = []; previous_sample_ns = -10**30; period_ns = int(1e9/args.sample_hz); max_age_ns = int(args.max_age_ms*1e6)
    while reader.has_next():
        topic, raw, timestamp_ns = reader.read_next(); message = deserialize_message(raw, types[topic])
        if topic != "/points":
            if topic in required: latest[topic] = (timestamp_ns, message)
            continue
        if timestamp_ns-previous_sample_ns < period_ns or not (required-{"/points"}).issubset(latest): continue
        if any(abs(timestamp_ns-latest[name][0]) > max_age_ns for name in required if name not in {"/points", "/plan"}): continue
        fields = ("x","y","z","intensity","ring","time"); names = {field.name for field in message.fields}
        if not set(fields).issubset(names): raise ValueError(f"point cloud missing fields: {sorted(set(fields)-names)}")
        raw_points = point_cloud2.read_points(message, field_names=fields, skip_nans=False)
        points = np.column_stack([raw_points[name] for name in fields]).astype(np.float32, copy=False)
        if not len(points) or not np.all(np.isfinite(points)): continue
        odom = latest["/odom"][1]; candidates = json.loads(latest["/planning/candidates"][1].data)
        obstacles = json.loads(latest["/planning/obstacle_predictions"][1].data)
        sample_id = f"{run.get('scenario','unknown')}_{run.get('difficulty','unknown')}_seed{run.get('seed','unknown')}_{timestamp_ns}"
        arrays = {"points": points, "route": route_array(latest["/plan"][1]),
                  "ego_state": np.asarray([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw(odom.pose.pose.orientation),
                                             odom.twist.twist.linear.x, odom.twist.twist.angular.z], np.float32),
                  "costmap": costmap_array(latest["/local_costmap/costmap_raw"][1]), **trace_arrays(candidates, obstacles)}
        metadata = {"sample_id": sample_id, "timestamp_s": timestamp_ns*1e-9, "frame_id": message.header.frame_id,
                    "scenario": run.get("scenario", "unknown"), "difficulty": run.get("difficulty", "unknown"),
                    "seed": run.get("seed", -1), "controller": run.get("controller", "unknown"),
                    "source_sha256": source_hash, "config_sha256": config_hash,
                    "checkpoint_sha256": checkpoint_hash, "code_sha256": code_hash, "feature_status": "pending"}
        path = output/"raw"/f"{sample_id}.npz"; payload_hash = save_raw_training_frame(path, arrays, metadata)
        entries.append({"sample_id": sample_id, "path": str(path.relative_to(output)), "payload_sha256": payload_hash,
                        "scenario": metadata["scenario"], "difficulty": metadata["difficulty"], "seed": metadata["seed"],
                        "controller": metadata["controller"]}); previous_sample_ns = timestamp_ns
    write_manifest(output/"raw_manifest.jsonl", entries)
    if not entries: raise RuntimeError("no synchronized training frames were extracted")
    print(json.dumps({"samples": len(entries), "manifest": str(output/"raw_manifest.jsonl"), "source_sha256": source_hash}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
