from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np


TRAINING_SAMPLE_SCHEMA_VERSION = "1.0"
TEACHER_OUTCOMES = ("success", "infeasible", "numeric_failure", "timeout")
TEACHER_OUTCOME_TO_CODE = {name: index for index, name in enumerate(TEACHER_OUTCOMES)}


def _finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or Inf")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_metadata(metadata: Mapping[str, object]) -> bytes:
    return json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sample_payload_sha256(sample: "TrainingSample") -> str:
    digest = sha256()
    digest.update(TRAINING_SAMPLE_SCHEMA_VERSION.encode())
    digest.update(_canonical_metadata(sample.metadata))
    for name, array in sample.arrays().items():
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingSample:
    points: np.ndarray
    features: np.ndarray
    route: np.ndarray
    ego_state: np.ndarray
    costmap: np.ndarray
    obstacle_states: np.ndarray
    obstacle_lengths: np.ndarray
    obstacle_widths: np.ndarray
    obstacle_covariance: np.ndarray
    obstacle_valid_mask: np.ndarray
    candidate_states: np.ndarray
    candidate_controls: np.ndarray
    candidate_timestamps_s: np.ndarray
    teacher_outcome_codes: np.ndarray
    teacher_feasible: np.ndarray
    teacher_h_min: np.ndarray
    teacher_slack_max: np.ndarray
    teacher_solve_time_ms: np.ndarray
    teacher_states: np.ndarray
    teacher_controls: np.ndarray
    teacher_selected_index: int
    metadata: Mapping[str, object]

    @property
    def teacher_present(self) -> bool:
        return self.teacher_outcome_codes.size > 0

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "points": self.points,
            "features": self.features,
            "route": self.route,
            "ego_state": self.ego_state,
            "costmap": self.costmap,
            "obstacle_states": self.obstacle_states,
            "obstacle_lengths": self.obstacle_lengths,
            "obstacle_widths": self.obstacle_widths,
            "obstacle_covariance": self.obstacle_covariance,
            "obstacle_valid_mask": self.obstacle_valid_mask,
            "candidate_states": self.candidate_states,
            "candidate_controls": self.candidate_controls,
            "candidate_timestamps_s": self.candidate_timestamps_s,
            "teacher_outcome_codes": self.teacher_outcome_codes,
            "teacher_feasible": self.teacher_feasible,
            "teacher_h_min": self.teacher_h_min,
            "teacher_slack_max": self.teacher_slack_max,
            "teacher_solve_time_ms": self.teacher_solve_time_ms,
            "teacher_states": self.teacher_states,
            "teacher_controls": self.teacher_controls,
            "teacher_selected_index": np.asarray(self.teacher_selected_index, dtype=np.int64),
        }

    def validate(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 6:
            raise ValueError("points must have shape [N,6]")
        if self.features.ndim != 3:
            raise ValueError("features must have shape [C,H,W]")
        if self.route.ndim != 2 or self.route.shape[1] != 4:
            raise ValueError("route must have shape [R,4]")
        if self.ego_state.ndim != 1:
            raise ValueError("ego_state must be one-dimensional")
        if self.costmap.ndim != 3 or self.costmap.shape[0] != 3:
            raise ValueError("costmap must have shape [3,H,W]")
        if self.obstacle_states.ndim != 3 or self.obstacle_states.shape[-1] != 6:
            raise ValueError("obstacle_states must have shape [O,T,6]")
        obstacles, horizon = self.obstacle_states.shape[:2]
        if self.obstacle_lengths.shape != (obstacles, horizon):
            raise ValueError("obstacle_lengths shape mismatch")
        if self.obstacle_widths.shape != (obstacles, horizon):
            raise ValueError("obstacle_widths shape mismatch")
        if self.obstacle_covariance.shape != (obstacles, horizon, 2, 2):
            raise ValueError("obstacle_covariance shape mismatch")
        if self.obstacle_valid_mask.shape != (obstacles, horizon):
            raise ValueError("obstacle_valid_mask shape mismatch")
        if self.candidate_states.ndim != 3:
            raise ValueError("candidate_states must have shape [K,T,S]")
        candidates, candidate_horizon = self.candidate_states.shape[:2]
        if candidate_horizon != horizon and obstacles:
            raise ValueError("candidate and obstacle horizons differ")
        if self.candidate_controls.ndim != 3 or self.candidate_controls.shape[:2] != (candidates, candidate_horizon - 1):
            raise ValueError("candidate_controls shape mismatch")
        if self.candidate_timestamps_s.shape != (candidate_horizon,) or np.any(np.diff(self.candidate_timestamps_s) <= 0):
            raise ValueError("candidate timestamps are invalid")
        for name, value in self.arrays().items():
            if name not in {"obstacle_valid_mask", "teacher_feasible", "teacher_outcome_codes", "teacher_selected_index"}:
                _finite(name, np.asarray(value))
        required_hashes = ("source_sha256", "config_sha256", "checkpoint_sha256", "code_sha256")
        for name in required_hashes:
            if not _is_sha256(str(self.metadata.get(name, ""))):
                raise ValueError(f"metadata.{name} must be a lowercase SHA-256")
        if not str(self.metadata.get("sample_id", "")):
            raise ValueError("metadata.sample_id is required")
        if self.teacher_present:
            expected = (candidates,)
            for name in ("teacher_outcome_codes", "teacher_feasible", "teacher_h_min", "teacher_slack_max", "teacher_solve_time_ms"):
                if getattr(self, name).shape != expected:
                    raise ValueError(f"{name} shape mismatch")
            if np.any((self.teacher_outcome_codes < 0) | (self.teacher_outcome_codes >= len(TEACHER_OUTCOMES))):
                raise ValueError("unknown teacher outcome code")
            if self.teacher_states.shape != self.candidate_states.shape or self.teacher_controls.shape != self.candidate_controls.shape:
                raise ValueError("teacher trajectory shape mismatch")
            if not 0 <= self.teacher_selected_index < candidates:
                raise ValueError("teacher_selected_index is invalid")
        elif self.teacher_selected_index != -1:
            raise ValueError("unlabeled sample must use teacher_selected_index=-1")

    def with_teacher(self, **changes) -> "TrainingSample":
        return replace(self, **changes)


def save_training_sample(path: str | Path, sample: TrainingSample) -> str:
    sample.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_hash = sample_payload_sha256(sample)
    np.savez_compressed(
        destination,
        schema_version=np.asarray(TRAINING_SAMPLE_SCHEMA_VERSION),
        metadata_json=np.asarray(_canonical_metadata(sample.metadata).decode("utf-8")),
        payload_sha256=np.asarray(payload_hash),
        **sample.arrays(),
    )
    return payload_hash


def load_training_sample(path: str | Path) -> TrainingSample:
    with np.load(Path(path), allow_pickle=False) as values:
        if str(values["schema_version"]) != TRAINING_SAMPLE_SCHEMA_VERSION:
            raise ValueError("training sample schema mismatch")
        metadata = json.loads(str(values["metadata_json"]))
        sample = TrainingSample(
            **{name: values[name] for name in TrainingSample.__dataclass_fields__ if name not in {"metadata", "teacher_selected_index"}},
            teacher_selected_index=int(values["teacher_selected_index"]),
            metadata=metadata,
        )
        expected_hash = str(values["payload_sha256"])
    sample.validate()
    if sample_payload_sha256(sample) != expected_hash:
        raise ValueError("training sample payload hash mismatch")
    return sample


def write_manifest(path: str | Path, entries: list[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(entry), sort_keys=True, ensure_ascii=False) for entry in entries]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_manifest(path: str | Path) -> list[dict[str, object]]:
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries
