#!/usr/bin/env python3
"""Minimal deployment smoke test for the isolated UniLION environment."""

from __future__ import annotations

import importlib
import json

import torch


def main() -> int:
    modules = ("mmcv", "mmcv._ext", "mmdet3d", "mamba_ssm", "mmdet3d_plugin")
    imported: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            imported[name] = getattr(module, "__version__", "ok")
        except Exception as exc:  # pragma: no cover - executed on the CUDA host
            errors[name] = f"{type(exc).__name__}: {exc}"

    cuda_ok = torch.cuda.is_available()
    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_ok,
        "gpu": torch.cuda.get_device_name(0) if cuda_ok else None,
        "imports": imported,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if cuda_ok and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
