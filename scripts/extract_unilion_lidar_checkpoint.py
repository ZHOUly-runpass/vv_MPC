#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import torch


DEFAULT_PREFIXES = ("pts_voxel_encoder.", "pts_middle_encoder.", "pts_backbone.", "pts_neck.")


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no state_dict mapping")
    selected = {key: value for key, value in state.items() if key.startswith(DEFAULT_PREFIXES)}
    if not selected or not any(key.startswith("pts_backbone.") for key in selected):
        raise ValueError("checkpoint has no usable LiDAR backbone keys")
    prefix_counts = {
        prefix.rstrip("."): sum(key.startswith(prefix) for key in selected)
        for prefix in DEFAULT_PREFIXES
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "1.0",
        "source_checkpoint_sha256": file_hash(args.source),
        "source_modality": "official_unilion_lct",
        "selected_prefixes": DEFAULT_PREFIXES,
        "prefix_counts": prefix_counts,
        "state_dict": selected,
        "warning": "Initialization candidate only; not validated on C16.",
    }, args.output)
    print({
        "output": str(args.output), "keys": len(selected),
        "prefix_counts": prefix_counts, "sha256": file_hash(args.output),
        "size_bytes": args.output.stat().st_size,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
