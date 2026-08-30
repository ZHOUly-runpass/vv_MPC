from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np


EXPECTED_MODEL_CONFIG = {
    "feature_channels": 384,
    "candidates": 7,
    "intervals": 20,
    "ego_dim": 5,
    "hidden_dim": 128,
}


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointContract:
    checkpoint_sha256: str
    manifest_sha256: str
    dataset_version_sha256: str
    code_revision: str
    unilion_checkpoint_sha256: str
    model_config: Mapping[str, int]


def validate_checkpoint_contract(
    checkpoint: Mapping[str, object],
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    dataset_version_path: str | Path,
    unilion_checkpoint_path: str | Path,
    expected_checkpoint_sha256: str | None = None,
) -> CheckpointContract:
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    dataset_version_path = Path(dataset_version_path).resolve()
    unilion_checkpoint_path = Path(unilion_checkpoint_path).resolve()
    for path in (checkpoint_path, manifest_path, dataset_version_path, unilion_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_hash = file_sha256(checkpoint_path)
    if expected_checkpoint_sha256 and checkpoint_hash != expected_checkpoint_sha256.lower():
        raise ValueError("planner checkpoint SHA-256 mismatch")
    if int(checkpoint.get("format_version", -1)) != 2:
        raise ValueError("planner checkpoint format_version must be 2")
    model_config = checkpoint.get("model_config")
    if model_config != EXPECTED_MODEL_CONFIG:
        raise ValueError(f"planner model config mismatch: {model_config!r}")
    if checkpoint.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("planner checkpoint manifest SHA-256 mismatch")
    if checkpoint.get("dataset_version_sha256") != file_sha256(dataset_version_path):
        raise ValueError("planner checkpoint dataset-version SHA-256 mismatch")
    version = json.loads(dataset_version_path.read_text(encoding="utf-8"))
    if version.get("frozen_manifest_sha256") != checkpoint.get("manifest_sha256"):
        raise ValueError("dataset version does not bind the checkpoint manifest")
    if version.get("git_revision") != checkpoint.get("code_revision"):
        raise ValueError("checkpoint Git revision does not match dataset version")
    unilion_hash = file_sha256(unilion_checkpoint_path)
    if version.get("checkpoint_sha256_values") != [unilion_hash]:
        raise ValueError("UniLION checkpoint SHA-256 does not match dataset version")
    if checkpoint.get("training_config", {}).get("ablation", "main") != "main":
        raise ValueError("Proposed runtime requires the main, non-ablation checkpoint")
    if "model_state" not in checkpoint:
        raise ValueError("planner checkpoint has no model_state")
    return CheckpointContract(
        checkpoint_sha256=checkpoint_hash,
        manifest_sha256=str(checkpoint["manifest_sha256"]),
        dataset_version_sha256=str(checkpoint["dataset_version_sha256"]),
        code_revision=str(checkpoint["code_revision"]),
        unilion_checkpoint_sha256=unilion_hash,
        model_config=dict(model_config),
    )


def validate_inference_inputs(
    features: np.ndarray,
    route: np.ndarray,
    ego: np.ndarray,
    costmap: np.ndarray,
    feature_age_s: float,
    feature_timeout_s: float,
) -> None:
    expected = {
        "features": ((384, 32, 32), features),
        "route": ((32, 4), route),
        "ego": ((5,), ego),
        "costmap": ((3, 60, 60), costmap),
    }
    for name, (shape, value) in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} shape must be {shape}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or Inf")
    if not np.isfinite(feature_age_s) or feature_age_s < 0.0 or feature_age_s > feature_timeout_s:
        raise TimeoutError("frozen UniLION feature is stale")


def validate_prediction(prediction: Mapping[str, np.ndarray], elapsed_ms: float, deadline_ms: float) -> None:
    if elapsed_ms > deadline_ms:
        raise TimeoutError(f"learned inference exceeded {deadline_ms:.3f} ms")
    shapes = {
        "controls": (1, 7, 20, 2),
        "candidate_logits": (1, 7),
        "predicted_h_min": (1, 7),
        "feasibility_logits": (1, 7),
        "predicted_correction": (1, 7),
        "predicted_risk": (1, 7),
        "predicted_slack": (1, 7),
    }
    for name, shape in shapes.items():
        value = np.asarray(prediction[name])
        if value.shape != shape:
            raise ValueError(f"prediction {name} shape must be {shape}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"prediction {name} contains NaN or Inf")


def ranked_candidate_indices(prediction: Mapping[str, np.ndarray]) -> list[int]:
    logits = np.asarray(prediction["candidate_logits"])[0]
    feasible = np.asarray(prediction["feasibility_logits"])[0] >= 0.0
    h_min = np.asarray(prediction["predicted_h_min"])[0]
    risk = np.asarray(prediction["predicted_risk"])[0] < 0.0
    preferred = [int(index) for index in np.argsort(-logits)
                 if feasible[index] and h_min[index] >= 0.0 and risk[index]]
    fallback = [int(index) for index in np.argsort(-logits) if int(index) not in preferred]
    return preferred + fallback


class LearnedPlannerRuntime:
    def __init__(self, model, device: str, deadline_ms: float) -> None:
        self.model = model
        self.device = device
        self.deadline_ms = float(deadline_ms)

    def _forward(self, features: np.ndarray, route: np.ndarray, ego: np.ndarray, costmap: np.ndarray):
        import torch
        with torch.no_grad():
            output = self.model(
                torch.from_numpy(features[None].astype(np.float32)).to(self.device),
                torch.from_numpy(route[None].astype(np.float32)).to(self.device),
                torch.from_numpy(ego[None].astype(np.float32)).to(self.device),
                torch.from_numpy(costmap[None].astype(np.float32)).to(self.device),
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
        return {name: value.detach().cpu().numpy() for name, value in output.items()}

    def warmup(self, features: np.ndarray, route: np.ndarray, ego: np.ndarray, costmap: np.ndarray) -> float:
        started = perf_counter(); prediction = self._forward(features, route, ego, costmap)
        elapsed_ms = (perf_counter() - started) * 1000.0
        validate_prediction(prediction, 0.0, self.deadline_ms)
        return elapsed_ms

    def infer(self, features: np.ndarray, route: np.ndarray, ego: np.ndarray, costmap: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        started = perf_counter(); prediction = self._forward(features, route, ego, costmap)
        elapsed_ms = (perf_counter() - started) * 1000.0
        validate_prediction(prediction, elapsed_ms, self.deadline_ms)
        return prediction, elapsed_ms
