#!/usr/bin/env python3
"""Read-only ROS graph audit. Run after sourcing ROS 2 on the dev machine."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from r680_safety_planner.config import ProjectConfig


def command(*parts: str) -> str:
    result = subprocess.run(parts, check=False, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = ProjectConfig.load(args.config)
    topic_types = {}
    lines = command("ros2", "topic", "list", "-t").splitlines()
    for line in lines:
        if " [" in line:
            name, message_type = line.split(" [", 1)
            topic_types[name] = message_type.rstrip("]")
    expected = config.get("ros2", "topics")
    audit = {
        "nodes": command("ros2", "node", "list").splitlines(),
        "expected_topics": {},
        "cmd_vel_info": command("ros2", "topic", "info", "/cmd_vel", "--verbose"),
        "tf_frames": command("ros2", "run", "tf2_tools", "view_frames", "--no-static"),
    }
    for role, item in expected.items():
        if not isinstance(item, dict) or "name" not in item:
            continue
        actual_type = topic_types.get(item["name"])
        audit["expected_topics"][role] = {
            "name": item["name"], "expected_type": item.get("type"),
            "actual_type": actual_type, "present": actual_type is not None,
        }
    payload = json.dumps(audit, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
