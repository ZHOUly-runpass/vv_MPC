#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import time

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the versioned 8-scenario collection matrix")
    parser.add_argument("--matrix", type=Path, default=Path("simulation/ros2_ws/src/r680_sim_bringup/config/collection_matrix.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    matrix_path = args.matrix if args.matrix.is_absolute() else root / args.matrix
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    runs = [(scenario, difficulty, int(seed)) for scenario in matrix["scenarios"]
            for difficulty in matrix["difficulties"] for seed in matrix["seeds"]]
    if args.max_runs is not None: runs = runs[:args.max_runs]
    manifest = root / ".tools" / "collection" / "runs.jsonl"; manifest.parent.mkdir(parents=True, exist_ok=True)
    for scenario, difficulty, seed in runs:
        command = ["bash", str(root / "simulation/scripts/record_scenario.sh"), str(root), scenario,
                   str(matrix["duration_s"]), str(seed), difficulty]
        entry = {"schema_version": matrix["schema_version"], "scenario": scenario, "difficulty": difficulty,
                 "seed": seed, "duration_s": matrix["duration_s"], "command": command, "started_unix_s": time()}
        if args.dry_run:
            entry["status"] = "planned"
        else:
            completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
            entry.update(status="success" if completed.returncode == 0 else "failed", returncode=completed.returncode,
                         bag=completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else None,
                         stderr_tail=completed.stderr[-2000:])
        with manifest.open("a", encoding="utf-8") as stream: stream.write(json.dumps(entry, sort_keys=True) + "\n")
        print(json.dumps(entry, sort_keys=True))
        if entry["status"] == "failed" and not args.continue_on_error: return int(entry["returncode"])
    print(f"planned_or_completed={len(runs)} manifest={manifest}"); return 0


if __name__ == "__main__": raise SystemExit(main())
