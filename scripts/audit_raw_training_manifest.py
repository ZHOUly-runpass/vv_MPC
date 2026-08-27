#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from r680_safety_planner.data import load_manifest, load_raw_training_frame


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path); args = parser.parse_args(); manifest = args.manifest.resolve()
    entries = load_manifest(manifest); shapes = Counter(); metadata_groups = Counter(); errors = []
    for entry in entries:
        try:
            arrays, metadata, payload_hash = load_raw_training_frame(manifest.parent/str(entry["path"]))
            if payload_hash != entry["payload_sha256"]: raise ValueError("manifest hash mismatch")
            shapes[str({name: list(arrays[name].shape) for name in ("points", "route", "ego_state", "costmap",
                       "obstacle_states", "candidate_states", "candidate_controls", "candidate_timestamps_s")})] += 1
            metadata_groups[str({name: metadata.get(name) for name in ("scenario", "difficulty", "seed", "controller")})] += 1
        except Exception as error: errors.append({"sample_id": entry.get("sample_id"), "error": f"{type(error).__name__}:{error}"})
    report = {"schema_version": "1.0", "status": "passed" if entries and not errors else "failed",
              "samples": len(entries), "valid_samples": len(entries)-len(errors), "shape_groups": dict(shapes),
              "metadata_groups": dict(metadata_groups), "errors": errors}
    encoded = json.dumps(report, indent=2, sort_keys=True)+"\n"
    if args.output: args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(encoded)
    print(encoded, end=""); return 0 if report["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
