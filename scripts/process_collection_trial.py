#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from r680_safety_planner.data import load_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract the latest successful bag for every trial matrix tuple")
    parser.add_argument("--runs-manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--max-age-ms", type=float, default=1000.0)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    runs_manifest = args.runs_manifest.resolve(); output = args.output_dir.resolve()
    matrix_path = args.matrix if args.matrix.is_absolute() else root/args.matrix
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    expected = [(scenario, difficulty, int(seed)) for scenario in matrix["scenarios"]
                for difficulty in matrix["difficulties"] for seed in matrix["seeds"]]
    latest = {}
    for entry in load_manifest(runs_manifest):
        if entry.get("status") == "success":
            latest[(entry["scenario"], entry["difficulty"], int(entry["seed"]))] = entry
    missing = [key for key in expected if key not in latest]
    if missing: raise RuntimeError(f"successful bags are missing for {missing}")
    combined = []; run_summary = []
    for scenario, difficulty, seed in expected:
        entry = latest[(scenario, difficulty, seed)]; bag = Path(entry["bag"])
        run_output = output/"extracted"/scenario/difficulty/f"seed{seed}"
        command = [sys.executable, str(root/"scripts/extract_rosbag_training_frames.py"),
                   "--bag", str(bag), "--output-dir", str(run_output), "--checkpoint", str(args.checkpoint),
                   "--sample-hz", str(args.sample_hz), "--max-age-ms", str(args.max_age_ms)]
        subprocess.run(command, cwd=root, check=True)
        extracted = load_manifest(run_output/"raw_manifest.jsonl")
        for sample in extracted:
            absolute = run_output/str(sample["path"])
            combined.append({**sample, "path": str(absolute.relative_to(output))})
        run_summary.append({"scenario": scenario, "difficulty": difficulty, "seed": seed,
                            "bag": str(bag), "samples": len(extracted), "status": "passed"})
    write_manifest(output/"raw_manifest.jsonl", combined)
    report = {"schema_version": "1.0", "status": "passed", "expected_runs": len(expected),
              "successful_runs": len(run_summary), "samples": len(combined), "runs": run_summary,
              "raw_manifest": str(output/"raw_manifest.jsonl")}
    (output/"collection_trial_report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
