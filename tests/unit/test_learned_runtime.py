from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

from r680_safety_planner.learned_runtime import (
    EXPECTED_MODEL_CONFIG, ranked_candidate_indices, validate_checkpoint_contract,
    validate_inference_inputs, validate_prediction,
)


def digest(path): return sha256(path.read_bytes()).hexdigest()


def contract_fixture(tmp_path):
    checkpoint_path = tmp_path / "best.pt"; checkpoint_path.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.jsonl"; manifest.write_text("{}\n", encoding="utf-8")
    unilion = tmp_path / "unilion.safetensors"; unilion.write_bytes(b"unilion")
    revision = "a" * 40
    version = tmp_path / "dataset_version.json"
    version.write_text(json.dumps({"frozen_manifest_sha256": digest(manifest), "git_revision": revision,
                                   "checkpoint_sha256_values": [digest(unilion)]}), encoding="utf-8")
    checkpoint = {"format_version": 2, "model_config": EXPECTED_MODEL_CONFIG, "model_state": {},
                  "manifest_sha256": digest(manifest), "dataset_version_sha256": digest(version),
                  "code_revision": revision, "training_config": {"ablation": "main"}}
    return checkpoint, checkpoint_path, manifest, version, unilion


def test_checkpoint_contract_validates_all_provenance(tmp_path):
    checkpoint, path, manifest, version, unilion = contract_fixture(tmp_path)
    result = validate_checkpoint_contract(checkpoint, path, manifest, version, unilion, digest(path))
    assert result.model_config == EXPECTED_MODEL_CONFIG


@pytest.mark.parametrize("failure", ("missing", "hash", "git", "unilion"))
def test_checkpoint_contract_fails_closed(tmp_path, failure):
    checkpoint, path, manifest, version, unilion = contract_fixture(tmp_path)
    if failure == "missing": path = tmp_path / "missing.pt"
    elif failure == "hash": checkpoint["manifest_sha256"] = "0" * 64
    elif failure == "git": checkpoint["code_revision"] = "b" * 40
    else: unilion.write_bytes(b"changed")
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_checkpoint_contract(checkpoint, path, manifest, version, unilion)


def inputs():
    return (np.zeros((384, 32, 32), np.float32), np.zeros((32, 4), np.float32),
            np.zeros(5, np.float32), np.zeros((3, 60, 60), np.float32))


def test_nan_and_feature_timeout_fail_closed():
    features, route, ego, costmap = inputs(); features[0, 0, 0] = np.nan
    with pytest.raises(ValueError): validate_inference_inputs(features, route, ego, costmap, 0.0, 0.3)
    features[0, 0, 0] = 0.0
    with pytest.raises(TimeoutError): validate_inference_inputs(features, route, ego, costmap, 0.31, 0.3)


def prediction():
    return {"controls": np.zeros((1, 7, 20, 2)), "candidate_logits": np.arange(7)[None],
            "predicted_h_min": np.ones((1, 7)), "feasibility_logits": np.ones((1, 7)),
            "predicted_correction": np.zeros((1, 7)), "predicted_risk": -np.ones((1, 7)),
            "predicted_slack": np.zeros((1, 7))}


def test_prediction_nan_and_deadline_fail_closed_and_ranking_is_used():
    output = prediction(); validate_prediction(output, 10.0, 80.0)
    assert ranked_candidate_indices(output)[0] == 6
    output["controls"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError): validate_prediction(output, 10.0, 80.0)
    output = prediction()
    with pytest.raises(TimeoutError): validate_prediction(output, 80.01, 80.0)
