from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import load_manifest, load_training_sample, sample_payload_sha256


class TrainingSampleDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str | None = None, verify_hash: bool = True) -> None:
        self.manifest = Path(manifest)
        entries = load_manifest(self.manifest)
        self.entries = [entry for entry in entries if split is None or entry.get("split") == split]
        self.root = self.manifest.parent
        self.verify_hash = verify_hash
        if not self.entries:
            raise ValueError(f"manifest has no samples for split={split!r}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        sample = load_training_sample(self.root / str(entry["path"]))
        if not sample.teacher_present:
            raise ValueError(f"sample {entry.get('sample_id')} has no teacher label")
        if self.verify_hash and entry.get("payload_sha256") != sample_payload_sha256(sample):
            raise ValueError(f"manifest hash mismatch for {entry.get('sample_id')}")
        selected = sample.teacher_selected_index
        correction = np.sqrt(np.mean(np.square(sample.teacher_controls - sample.candidate_controls), axis=(1, 2)))
        return {
            "features": torch.from_numpy(sample.features.astype(np.float32)),
            "route": torch.from_numpy(sample.route.astype(np.float32)),
            "ego": torch.from_numpy(sample.ego_state.astype(np.float32)),
            "costmap": torch.from_numpy(sample.costmap.astype(np.float32)),
            "target_controls": torch.from_numpy(sample.teacher_controls.astype(np.float32)),
            "selected_index": torch.tensor(selected, dtype=torch.long),
            "h_min": torch.from_numpy(np.clip(sample.teacher_h_min, -10.0, 10.0)),
            "feasible": torch.from_numpy(sample.teacher_feasible.astype(np.float32)),
            "slack": torch.from_numpy(np.clip(sample.teacher_slack_max, 0.0, 10.0)),
            "correction": torch.from_numpy(correction.astype(np.float32)),
            "risk": torch.from_numpy((sample.teacher_outcome_codes != 0).astype(np.float32)),
            "sample_id": str(sample.metadata["sample_id"]),
        }


def collate_training_samples(items: list[dict[str, object]]) -> dict[str, object]:
    return {key: ([item[key] for item in items] if key == "sample_id" else torch.stack([item[key] for item in items]))
            for key in items[0]}
