#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from r680_safety_planner.data import TEACHER_OUTCOMES, load_manifest, load_training_sample, sample_payload_sha256


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path); args = parser.parse_args(); manifest = args.manifest.resolve()
    entries = load_manifest(manifest); shapes = Counter(); outcomes = Counter(); checkpoints = Counter(); errors = []
    outcomes_by_group = defaultdict(Counter); statuses_by_group = defaultdict(Counter); samples_by_group = Counter()
    solve_times_by_group = defaultdict(list); all_solve_times = []
    for entry in entries:
        try:
            sample = load_training_sample(manifest.parent/str(entry["path"]))
            if sample_payload_sha256(sample) != entry["payload_sha256"]: raise ValueError("manifest payload hash mismatch")
            shapes[str((sample.points.shape[1:], sample.features.shape, sample.route.shape, sample.costmap.shape,
                        sample.candidate_states.shape))] += 1
            checkpoints[str(sample.metadata["checkpoint_sha256"])] += 1
            group = f"{sample.metadata.get('scenario', 'unknown')}/{sample.metadata.get('difficulty', 'unknown')}"
            samples_by_group[group] += 1
            if sample.teacher_present:
                names = [TEACHER_OUTCOMES[int(code)] for code in sample.teacher_outcome_codes]
                outcomes.update(names); outcomes_by_group[group].update(names)
                statuses_by_group[group].update(str(value) for value in sample.metadata.get("teacher_solver_statuses", []))
                times = sample.teacher_solve_time_ms.astype(float).tolist()
                solve_times_by_group[group].extend(times); all_solve_times.extend(times)
        except Exception as error: errors.append({"sample_id": entry.get("sample_id"), "error": f"{type(error).__name__}:{error}"})
    report = {"schema_version": "1.0", "manifest": str(manifest), "samples": len(entries),
              "valid_samples": len(entries)-len(errors), "errors": errors, "shape_groups": dict(shapes),
              "checkpoint_sha256_counts": dict(checkpoints), "teacher_outcomes": dict(outcomes),
              "samples_by_group": dict(sorted(samples_by_group.items())),
              "teacher_outcomes_by_group": {key: dict(outcomes_by_group[key]) for key in sorted(outcomes_by_group)},
              "teacher_statuses_by_group": {key: dict(statuses_by_group[key]) for key in sorted(statuses_by_group)},
              "teacher_solve_time_ms": ({"p50": float(np.percentile(all_solve_times, 50)),
                  "p95": float(np.percentile(all_solve_times, 95)), "maximum": float(np.max(all_solve_times))}
                  if all_solve_times else {}),
              "teacher_solve_time_ms_by_group": {key: {"p50": float(np.percentile(values, 50)),
                  "p95": float(np.percentile(values, 95)), "maximum": float(np.max(values))}
                  for key, values in sorted(solve_times_by_group.items())},
              "status": "passed" if entries and not errors else "failed"}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end=""); return 0 if report["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
