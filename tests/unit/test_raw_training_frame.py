import numpy as np
import pytest

from r680_safety_planner.data import directory_sha256, load_raw_training_frame, save_raw_training_frame


def test_raw_frame_roundtrip_and_tamper_detection(tmp_path):
    path = tmp_path / "raw.npz"; arrays = {"points": np.arange(12, dtype=np.float32).reshape(2, 6)}
    payload_hash = save_raw_training_frame(path, arrays, {"sample_id": "one"})
    loaded, metadata, loaded_hash = load_raw_training_frame(path)
    assert loaded_hash == payload_hash and metadata["sample_id"] == "one"
    np.testing.assert_array_equal(loaded["points"], arrays["points"])
    with np.load(path, allow_pickle=False) as values: content = {name: values[name] for name in values.files}
    content["points"] = content["points"].copy(); content["points"][0, 0] += 1
    np.savez_compressed(path, **content)
    with pytest.raises(ValueError, match="payload hash"): load_raw_training_frame(path)


def test_directory_hash_covers_names_and_contents(tmp_path):
    (tmp_path / "a").write_bytes(b"one"); first = directory_sha256(tmp_path)
    (tmp_path / "a").write_bytes(b"two")
    assert directory_sha256(tmp_path) != first
