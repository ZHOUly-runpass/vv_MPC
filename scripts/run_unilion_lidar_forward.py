#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from r680_safety_planner.backbones import UniLionFrozenBackbone
from r680_safety_planner.interfaces import LidarFrame
from r680_safety_planner.lidar import UniLionPointFieldContract


def synthetic_c16_frame(point_count: int, seed: int) -> LidarFrame:
    """Create deterministic 16-ring input for plumbing/performance validation.

    This is deliberately labelled synthetic and must not be reported as a
    nuScenes or real-C16 accuracy sample.
    """

    if point_count < 16:
        raise ValueError("point_count must be at least 16")
    rng = np.random.default_rng(seed)
    rings = np.arange(point_count, dtype=np.int64) % 16
    azimuth = rng.uniform(-np.pi, np.pi, point_count)
    radius = rng.uniform(2.0, 45.0, point_count)
    elevation = np.deg2rad(np.linspace(-15.0, 15.0, 16))[rings]
    horizontal = radius * np.cos(elevation)
    points = np.column_stack(
        (
            horizontal * np.cos(azimuth),
            horizontal * np.sin(azimuth),
            radius * np.sin(elevation),
            rng.uniform(0.0, 255.0, point_count),
            rings.astype(np.float64),
            rng.uniform(0.0, 0.1, point_count),
        )
    )
    return LidarFrame(
        points=points,
        timestamp_s=1.0,
        frame_id="synthetic_c16",
        fields=("x", "y", "z", "intensity", "ring", "time"),
        metadata={"synthetic": True, "seed": seed},
    )


def official_sample_audit(repository: Path) -> dict[str, object]:
    sample_root = repository / "data" / "nuscenes" / "samples" / "LIDAR_TOP"
    samples = sorted(sample_root.glob("*.pcd.bin")) if sample_root.is_dir() else []
    return {
        "available": bool(samples),
        "sample_root": str(sample_root),
        "sample_count": len(samples),
        "reason": None if samples else "nuScenes sample data is not present on the development machine",
    }


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_nuscenes_sample(path: Path) -> LidarFrame:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 5:
        raise ValueError("nuScenes LiDAR sample must contain float32 records with five values")
    return LidarFrame(
        points=values.reshape(-1, 5).astype(np.float64),
        timestamp_s=1.0,
        frame_id="LIDAR_TOP",
        fields=("x", "y", "z", "intensity", "ring"),
        metadata={"source_file": str(path), "official_nuscenes_format": True},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--points", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=680)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--nuscenes-sample", type=Path)
    parser.add_argument("--repeat-tolerance", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project = args.project.resolve()
    repository = project / "third_party" / "UniLION"
    config = repository / "projects" / "configs" / "unilion_swin_384_seq_e2e.py"
    checkpoint = project / "artifacts" / "checkpoints" / "unilion_lidar_backbone_init.safetensors"
    if args.runs < 1:
        raise ValueError("runs must be at least one")
    if args.nuscenes_sample:
        frame = load_nuscenes_sample(args.nuscenes_sample.resolve())
        point_contract = UniLionPointFieldContract(expected_ring_count=32)
        input_kind = "official_nuscenes_five_float_format"
    else:
        frame = synthetic_c16_frame(args.points, args.seed)
        point_contract = UniLionPointFieldContract()
        input_kind = "deterministic_synthetic_c16_format_not_accuracy_evidence"

    build_started = perf_counter()
    backend = UniLionFrozenBackbone(
        repository, config, checkpoint, point_contract=point_contract
    )
    build_time_s = perf_counter() - build_started
    try:
        features = backend.infer(frame)  # warm-up and feature-contract validation
        run_reports: list[dict[str, float]] = []
        for _ in range(args.runs):
            backend.infer(frame)
            last = backend.healthcheck()["last_inference"]
            run_reports.append(
                {
                    "model_time_ms": float(last["model_time_ms"]),
                    "total_time_ms": float(last["total_time_ms"]),
                }
            )
        repeat_difference = backend.verify_repeatability(frame)
        health = backend.healthcheck()
        repeat_within_tolerance = repeat_difference <= args.repeat_tolerance
        report = {
            "status": "passed" if repeat_within_tolerance else "failed_repeatability",
            "input_kind": input_kind,
            "official_config": str(config),
            "official_config_sha256": file_hash(config),
            "lidar_checkpoint_sha256": file_hash(checkpoint),
            "build_time_s": build_time_s,
            "feature_shape": list(features.bev_feature.shape),
            "feature_finite": bool(np.all(np.isfinite(features.bev_feature))),
            "feature_variance": float(np.var(features.bev_feature.astype(np.float32))),
            "repeat_max_abs_difference": repeat_difference,
            "repeat_tolerance": args.repeat_tolerance,
            "repeat_within_tolerance": repeat_within_tolerance,
            "runs": run_reports,
            "model_time_ms_median": float(np.median([item["model_time_ms"] for item in run_reports])),
            "total_time_ms_median": float(np.median([item["total_time_ms"] for item in run_reports])),
            "health": health,
            "official_nuscenes_sample": official_sample_audit(repository),
        }
    finally:
        backend.close()

    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
