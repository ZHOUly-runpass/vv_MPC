#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit resumable collection-matrix progress")
    parser.add_argument("--runs-manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.resolve().read_text(encoding="utf-8"))
    expected = {(scenario, difficulty, int(seed)) for scenario in matrix["scenarios"]
                for difficulty in matrix["difficulties"] for seed in matrix["seeds"]}
    attempts = [json.loads(line) for line in args.runs_manifest.resolve().read_text(encoding="utf-8").splitlines()]
    latest = {}
    for entry in attempts:
        latest[(entry["scenario"], entry["difficulty"], int(entry["seed"]))] = entry
    evidence = []; invalid = []
    for key, entry in sorted(latest.items()):
        if entry.get("status") != "success":
            invalid.append({"tuple": list(key), "reason": f"latest_status:{entry.get('status')}"})
            continue
        bag = Path(str(entry.get("bag", "")))
        metadata = bag / "metadata.yaml"; databases = list(bag.glob("*.db3")) if bag.is_dir() else []
        valid = metadata.is_file() and bool(databases) and all(path.stat().st_size > 0 for path in databases)
        item = {"scenario": key[0], "difficulty": key[1], "seed": key[2], "bag": str(bag),
                "metadata_present": metadata.is_file(), "database_bytes": sum(path.stat().st_size for path in databases)}
        evidence.append(item)
        if not valid: invalid.append({"tuple": list(key), "reason": "incomplete_bag"})
    completed = {key for key, entry in latest.items() if entry.get("status") == "success"}
    report = {
        "schema_version": "1.0", "status": "complete" if completed == expected and not invalid else "in_progress",
        "expected_runs": len(expected), "attempts": len(attempts), "successful_latest_runs": len(completed & expected),
        "remaining_runs": len(expected - completed), "invalid_latest_runs": invalid,
        "completed_tuples": [list(key) for key in sorted(completed & expected)], "bag_evidence": evidence,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8"); print(encoded, end="")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
