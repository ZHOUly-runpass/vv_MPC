#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from r680_safety_planner.data import TEACHER_OUTCOMES, load_manifest, load_training_sample, sample_payload_sha256


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path); args = parser.parse_args(); manifest = args.manifest.resolve()
    entries = load_manifest(manifest); shapes = Counter(); outcomes = Counter(); checkpoints = Counter(); errors = []
    for entry in entries:
        try:
            sample = load_training_sample(manifest.parent/str(entry["path"]))
            if sample_payload_sha256(sample) != entry["payload_sha256"]: raise ValueError("manifest payload hash mismatch")
            shapes[str((sample.points.shape[1:], sample.features.shape, sample.route.shape, sample.costmap.shape,
                        sample.candidate_states.shape))] += 1
            checkpoints[str(sample.metadata["checkpoint_sha256"])] += 1
            if sample.teacher_present:
                outcomes.update(TEACHER_OUTCOMES[int(code)] for code in sample.teacher_outcome_codes)
        except Exception as error: errors.append({"sample_id": entry.get("sample_id"), "error": f"{type(error).__name__}:{error}"})
    report = {"schema_version": "1.0", "manifest": str(manifest), "samples": len(entries),
              "valid_samples": len(entries)-len(errors), "errors": errors, "shape_groups": dict(shapes),
              "checkpoint_sha256_counts": dict(checkpoints), "teacher_outcomes": dict(outcomes),
              "status": "passed" if entries and not errors else "failed"}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end=""); return 0 if report["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
