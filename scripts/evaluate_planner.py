#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from r680_safety_planner.models import PlanningSafetyModel
from r680_safety_planner.training import PlanningLoss, TrainingSampleDataset, collate_training_samples, planning_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    dataset = TrainingSampleDataset(resolve(args.manifest), args.split)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_training_samples)
    checkpoint = torch.load(resolve(args.checkpoint), map_location=args.device, weights_only=False)
    manifest_path = resolve(args.manifest)
    manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise ValueError("checkpoint manifest hash does not match evaluation manifest")
    model = PlanningSafetyModel(**checkpoint["model_config"]).to(args.device); model.load_state_dict(checkpoint["model_state"]); model.eval()
    zero_features = checkpoint.get("training_config", {}).get("ablation") == "no_unilion_features"
    loss_fn = PlanningLoss(); totals = {}; count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: (value.to(args.device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}
            if zero_features: batch["features"] = torch.zeros_like(batch["features"])
            prediction = model(batch["features"], batch["route"], batch["ego"], batch["costmap"])
            _, parts = loss_fn(prediction, batch)
            values = {**{f"loss_{key}": float(value.item()) for key, value in parts.items()}, **planning_metrics(prediction, batch)}
            size = batch["features"].shape[0]; count += size
            for key, value in values.items(): totals[key] = totals.get(key, 0.0) + value * size
    report = {"split": args.split, "samples": count, "checkpoint_epoch": checkpoint["epoch"],
              "manifest_sha256": manifest_hash, "training_config_sha256": checkpoint.get("training_config_sha256"),
              "dataset_version_sha256": checkpoint.get("dataset_version_sha256"),
              "code_revision": checkpoint.get("code_revision"),
              "ablation": checkpoint.get("training_config", {}).get("ablation", "main"),
              "metrics": {key: value / count for key, value in totals.items()}}
    output = resolve(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
