from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_project_config


def validate_config_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_project_config(args.config)
    print(f"configuration valid: {config.source}")
    print(f"motion_unlocked={config.motion_unlocked}")
    print(f"failed_gates={len(config.failed_gates)}")


def synthetic_smoke_main() -> None:
    from .pipeline import run_synthetic_smoke

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run_synthetic_smoke(load_project_config(args.config))
    print(result)

