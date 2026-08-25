#!/usr/bin/env python3
"""Safely summarize a PyTorch checkpoint without executing pickled code."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def summarize(value: Any, depth: int = 0) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        items = list(value.items())
        if depth >= 2:
            return {"type": "mapping", "items": len(items), "keys": [str(k) for k, _ in items[:40]]}
        return {str(key): summarize(item, depth + 1) for key, item in items[:40]}
    if isinstance(value, (tuple, list)):
        return [summarize(item, depth + 1) for item in value[:20]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    print(json.dumps({
        "path": str(checkpoint), "sha256": digest(checkpoint),
        "size_bytes": checkpoint.stat().st_size, "content": summarize(payload),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
