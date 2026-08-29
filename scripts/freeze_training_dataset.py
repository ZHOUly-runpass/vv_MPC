#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

from r680_safety_planner.data import load_manifest, write_manifest


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze an audited manifest with seed-isolated splits")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[64])
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[65])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[66])
    parser.add_argument("--version", default="r680_staged_v1")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    source = args.manifest.resolve(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    seed_split = {}
    for split, seeds in (("train", args.train_seeds), ("val", args.val_seeds), ("test", args.test_seeds)):
        for seed in seeds:
            if seed in seed_split: raise ValueError(f"seed {seed} appears in multiple splits")
            seed_split[seed] = split
    frozen = []
    for entry in load_manifest(source):
        seed = int(entry["seed"])
        if seed not in seed_split: raise ValueError(f"seed {seed} has no split assignment")
        sample = source.parent / str(entry["path"])
        if not sample.is_file(): raise FileNotFoundError(sample)
        frozen.append({**entry, "path": os.path.relpath(sample, output).replace(os.sep, "/"), "split": seed_split[seed]})
    frozen.sort(key=lambda item: str(item["sample_id"]))
    manifest = output / "manifest.jsonl"; write_manifest(manifest, frozen)
    split_counts = Counter(item["split"] for item in frozen)
    run_counts = Counter((item["split"], item["scenario"], item["difficulty"], int(item["seed"])) for item in frozen)
    split_config = {"train_seeds": args.train_seeds, "val_seeds": args.val_seeds, "test_seeds": args.test_seeds}
    config_hash = sha256(json.dumps(split_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    try: revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception: revision = "unknown"
    version = {"format_version": 1, "dataset_version": args.version, "samples": len(frozen),
               "split_counts": dict(split_counts), "run_group_count": len(run_counts), "split_config": split_config,
               "split_config_sha256": config_hash, "source_manifest_sha256": digest(source),
               "frozen_manifest_sha256": digest(manifest), "git_revision": revision,
               "checkpoint_sha256_values": sorted({str(item["checkpoint_sha256"]) for item in frozen})}
    (output / "dataset_version.json").write_text(json.dumps(version, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(version, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
