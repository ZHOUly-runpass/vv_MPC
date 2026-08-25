#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


def output(*command: str) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return (result.stdout or result.stderr).strip()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "third_party" / "UniLION"
    checkpoint = root / "artifacts" / "checkpoints" / "unilion_swin_384_seq_e2e.pth"
    nvcc = shutil.which("nvcc")
    nvcc_text = output(nvcc, "--version") if nvcc else "missing"
    report = {
        "source_present": (source / "README.md").is_file(),
        "checkpoint_present": checkpoint.is_file(),
        "python": sys.version.split()[0],
        "python_3_9_required": sys.version_info[:2] == (3, 9),
        "nvcc": nvcc_text,
        "cuda_12_4_required": "release 12.4" in nvcc_text,
        "conda": shutil.which("conda"),
        "gpu": output("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
    }
    report["official_environment_ready"] = all((
        report["source_present"], report["checkpoint_present"],
        report["python_3_9_required"], report["cuda_12_4_required"],
    ))
    print(json.dumps(report, indent=2))
    return 0 if report["official_environment_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
