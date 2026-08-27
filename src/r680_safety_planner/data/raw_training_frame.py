from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np


RAW_FRAME_SCHEMA_VERSION = "1.0"


def directory_sha256(path: str | Path) -> str:
    source = Path(path); digest = sha256()
    files = sorted(item for item in source.rglob("*") if item.is_file()) if source.is_dir() else [source]
    for item in files:
        digest.update(str(item.relative_to(source) if source.is_dir() else item.name).encode())
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _payload_sha(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> str:
    digest = sha256(); digest.update(RAW_FRAME_SCHEMA_VERSION.encode())
    digest.update(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name]); digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def save_raw_training_frame(path: str | Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> str:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload_hash = _payload_sha(arrays, metadata)
    np.savez_compressed(destination, schema_version=np.asarray(RAW_FRAME_SCHEMA_VERSION), metadata_json=np.asarray(encoded),
                        payload_sha256=np.asarray(payload_hash), **arrays)
    return payload_hash


def load_raw_training_frame(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object], str]:
    with np.load(path, allow_pickle=False) as values:
        if str(values["schema_version"]) != RAW_FRAME_SCHEMA_VERSION: raise ValueError("raw frame schema mismatch")
        metadata = json.loads(str(values["metadata_json"])); expected = str(values["payload_sha256"])
        arrays = {name: values[name] for name in values.files if name not in {"schema_version", "metadata_json", "payload_sha256"}}
    if _payload_sha(arrays, metadata) != expected: raise ValueError("raw frame payload hash mismatch")
    return arrays, metadata, expected
