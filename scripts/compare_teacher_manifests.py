#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from r680_safety_planner.data import load_manifest, load_training_sample


FIELDS = (
    "teacher_outcome_codes", "teacher_feasible", "teacher_h_min", "teacher_slack_max",
    "teacher_solve_time_ms", "teacher_states", "teacher_controls", "teacher_selected_index",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare teacher payloads by sample_id")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_manifest = args.reference.resolve(); candidate_manifest = args.candidate.resolve()
    reference = {str(item["sample_id"]): item for item in load_manifest(reference_manifest)}
    candidate = {str(item["sample_id"]): item for item in load_manifest(candidate_manifest)}
    if reference.keys() != candidate.keys():
        raise ValueError("manifest sample_id sets differ")
    changed = {field: 0 for field in FIELDS}; maximum_absolute_difference = {field: 0.0 for field in FIELDS}
    for sample_id in sorted(reference):
        left = load_training_sample(reference_manifest.parent / str(reference[sample_id]["path"]))
        right = load_training_sample(candidate_manifest.parent / str(candidate[sample_id]["path"]))
        for field in FIELDS:
            left_value = np.asarray(getattr(left, field)); right_value = np.asarray(getattr(right, field))
            if left_value.shape != right_value.shape or not np.array_equal(left_value, right_value):
                changed[field] += 1
                if left_value.shape == right_value.shape and np.issubdtype(left_value.dtype, np.number):
                    difference = np.abs(left_value.astype(np.float64) - right_value.astype(np.float64))
                    finite = difference[np.isfinite(difference)]
                    if finite.size:
                        maximum_absolute_difference[field] = max(maximum_absolute_difference[field], float(finite.max()))
    report = {"samples": len(reference), "changed_samples_by_field": changed,
              "maximum_absolute_difference_by_field": maximum_absolute_difference}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
