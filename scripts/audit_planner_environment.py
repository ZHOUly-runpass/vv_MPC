#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import casadi
import numpy
import yaml

from r680_safety_planner.dcbf import CasadiDcbfSolver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    expected = args.expected_prefix.resolve()
    actual = Path(sys.prefix).resolve()
    report = {
        "schema_version": "1.0",
        "environment": "planner",
        "expected_prefix": str(expected),
        "actual_prefix": str(actual),
        "isolated_environment": actual == expected,
        "python": platform.python_version(),
        "python_3_10": sys.version_info[:2] == (3, 10),
        "casadi": casadi.__version__,
        "casadi_expected": casadi.__version__ == "3.8.0",
        "casadi_solver_available": CasadiDcbfSolver.available(),
        "numpy": numpy.__version__,
        "pyyaml": yaml.__version__,
    }
    report["ready"] = all((
        report["isolated_environment"],
        report["python_3_10"],
        report["casadi_expected"],
        report["casadi_solver_available"],
    ))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
