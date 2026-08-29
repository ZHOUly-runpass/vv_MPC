#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from r680_safety_planner.models import PlanningSafetyModel
from r680_safety_planner.training import PlanningLoss, TrainingSampleDataset, collate_training_samples, planning_metrics


def move(batch, device):
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}


def evaluate(model, loader, loss_fn, device, zero_features=False):
    model.eval(); totals = {}; count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            if zero_features: batch["features"] = torch.zeros_like(batch["features"])
            prediction = model(batch["features"], batch["route"], batch["ego"], batch["costmap"])
            loss, parts = loss_fn(prediction, batch)
            metrics = {**{f"loss_{k}": float(v.item()) for k, v in parts.items()}, **planning_metrics(prediction, batch)}
            size = batch["features"].shape[0]; count += size
            for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + value * size
    return {key: value / count for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=680)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--zero-features", action="store_true", help="ablation: replace UniLION features with zeros")
    parser.add_argument("--ablation", default="main", help="versioned experiment name stored in the checkpoint")
    args = parser.parse_args()
    if args.zero_features and args.ablation != "main":
        raise ValueError("--zero-features already defines the no_unilion_features ablation")
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train = TrainingSampleDataset(manifest, "train")
    val = TrainingSampleDataset(manifest, "val")
    first = train[0]
    config = {"feature_channels": first["features"].shape[0], "candidates": first["target_controls"].shape[0],
              "intervals": first["target_controls"].shape[1], "ego_dim": first["ego"].shape[0], "hidden_dim": 128}
    model = PlanningSafetyModel(**config).to(args.device)
    loss_fn = PlanningLoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, generator=generator, collate_fn=collate_training_samples)
    val_loader = DataLoader(val, args.batch_size, shuffle=False, collate_fn=collate_training_samples)
    manifest_hash = sha256(manifest.read_bytes()).hexdigest()
    ablation = "no_unilion_features" if args.zero_features else args.ablation
    training_config = {"epochs": args.epochs, "batch_size": args.batch_size,
                       "learning_rate": args.learning_rate, "seed": args.seed,
                       "model": config, "loss_weights": loss_fn.weights,
                       "ablation": ablation}
    training_config_hash = sha256(json.dumps(training_config, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    dataset_version_path = manifest.parent / "dataset_version.json"
    dataset_version_hash = sha256(dataset_version_path.read_bytes()).hexdigest() if dataset_version_path.is_file() else None
    try: code_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception: code_hash = "unknown"
    best = float("inf"); history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = move(batch, args.device)
            if args.zero_features: batch["features"] = torch.zeros_like(batch["features"])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["features"], batch["route"], batch["ego"], batch["costmap"])
            loss, _ = loss_fn(prediction, batch); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        metrics = evaluate(model, val_loader, loss_fn, args.device, args.zero_features); metrics["epoch"] = epoch; history.append(metrics)
        state = {"format_version": 2, "epoch": epoch, "model_config": config, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "metrics": metrics, "manifest_sha256": manifest_hash,
                 "training_config": training_config, "training_config_sha256": training_config_hash,
                 "dataset_version_sha256": dataset_version_hash, "code_revision": code_hash, "seed": args.seed}
        torch.save(state, output / "last.pt")
        if metrics["loss_total"] < best:
            best = metrics["loss_total"]; torch.save(state, output / "best.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"best_val_loss": best, "checkpoint": str(output / "best.pt")}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
