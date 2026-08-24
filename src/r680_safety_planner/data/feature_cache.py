from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class FeatureCacheRecord:
    bev_feature: np.ndarray
    timestamp_s: float
    frame_id: str
    checkpoint_hash: str
    config_hash: str

    def validate(self) -> None:
        if self.bev_feature.ndim != 3 or not np.all(np.isfinite(self.bev_feature)):
            raise ValueError("cached BEV feature must be finite [C,H,W]")
        for name, value in (("checkpoint_hash", self.checkpoint_hash), ("config_hash", self.config_hash)):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ValueError(f"{name} must be a SHA-256 hex digest")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_feature_cache(path: str | Path, record: FeatureCacheRecord) -> None:
    record.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema_version=np.array(SCHEMA_VERSION),
        bev_feature=record.bev_feature.astype(np.float16),
        timestamp_s=np.array(record.timestamp_s, dtype=np.float64),
        frame_id=np.array(record.frame_id),
        checkpoint_hash=np.array(record.checkpoint_hash),
        config_hash=np.array(record.config_hash),
    )


def load_feature_cache(path: str | Path) -> FeatureCacheRecord:
    with np.load(Path(path), allow_pickle=False) as values:
        if str(values["schema_version"]) != SCHEMA_VERSION:
            raise ValueError("feature cache schema mismatch")
        record = FeatureCacheRecord(
            bev_feature=values["bev_feature"].astype(np.float32),
            timestamp_s=float(values["timestamp_s"]),
            frame_id=str(values["frame_id"]),
            checkpoint_hash=str(values["checkpoint_hash"]),
            config_hash=str(values["config_hash"]),
        )
    record.validate()
    return record
