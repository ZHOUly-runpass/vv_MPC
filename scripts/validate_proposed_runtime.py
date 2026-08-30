#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from r680_safety_planner.data import load_manifest, load_training_sample
from r680_safety_planner.learned_runtime import (
    LearnedPlannerRuntime, ranked_candidate_indices, validate_checkpoint_contract,
    validate_inference_inputs, validate_prediction,
)
from r680_safety_planner.models import PlanningSafetyModel


def expected_failure(name, action) -> dict:
    try: action()
    except Exception as error: return {"name": name, "safe_stop": True, "error": f"{type(error).__name__}:{error}"}
    return {"name": name, "safe_stop": False, "error": "fault was not rejected"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Proposed checkpoint and fail-closed gates")
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-version", type=Path, required=True); parser.add_argument("--unilion-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--deadline-ms", type=float, default=80.0)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    contract = validate_checkpoint_contract(checkpoint, args.checkpoint, args.manifest, args.dataset_version,
                                            args.unilion_checkpoint, args.expected_checkpoint_sha256)
    entries = load_manifest(args.manifest); entry = next(item for item in entries if item.get("split") == "test")
    sample = load_training_sample(args.manifest.parent / str(entry["path"]))
    validate_inference_inputs(sample.features, sample.route, sample.ego_state, sample.costmap, 0.0, 0.30)
    model = PlanningSafetyModel(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    runtime = LearnedPlannerRuntime(model, args.device, args.deadline_ms)
    warmup_ms = runtime.warmup(sample.features, sample.route, sample.ego_state, sample.costmap)
    prediction, elapsed_ms = runtime.infer(sample.features, sample.route, sample.ego_state, sample.costmap)
    faults = [
        expected_failure("model_missing", lambda: validate_checkpoint_contract(
            checkpoint, args.checkpoint.with_name("missing.pt"), args.manifest, args.dataset_version, args.unilion_checkpoint)),
        expected_failure("checkpoint_hash_mismatch", lambda: validate_checkpoint_contract(
            checkpoint, args.checkpoint, args.manifest, args.dataset_version, args.unilion_checkpoint, "0" * 64)),
        expected_failure("output_nan", lambda: validate_prediction(
            {**prediction, "controls": np.full_like(prediction["controls"], np.nan)}, elapsed_ms, args.deadline_ms)),
        expected_failure("inference_timeout", lambda: validate_prediction(prediction, args.deadline_ms + 0.01, args.deadline_ms)),
        expected_failure("feature_timeout", lambda: validate_inference_inputs(
            sample.features, sample.route, sample.ego_state, sample.costmap, 0.31, 0.30)),
    ]
    report = {"status": "passed" if all(item["safe_stop"] for item in faults) else "failed",
              "learned_checkpoint_active": True, "contract": contract.__dict__, "sample_id": entry["sample_id"],
              "cuda_warmup_ms": warmup_ms, "inference_ms": elapsed_ms,
              "selected_index": ranked_candidate_indices(prediction)[0], "faults": faults}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
