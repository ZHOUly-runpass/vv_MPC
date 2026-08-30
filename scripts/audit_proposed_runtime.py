#!/usr/bin/env python3
"""Collect a short, machine-readable audit of the live Proposed controller."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String


class RuntimeAudit(Node):
    def __init__(self) -> None:
        super().__init__("proposed_runtime_audit")
        self.statuses: list[dict] = []
        self.commands: list[list[float]] = []
        self.create_subscription(String, "/simulation/controller_status", self._status, 20)
        self.create_subscription(Twist, "/cmd_vel", self._command, 20)

    def _status(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if isinstance(value, dict) and value.get("baseline") == "proposed":
            self.statuses.append(value)

    def _command(self, message: Twist) -> None:
        self.commands.append([
            float(message.linear.x), float(message.linear.y), float(message.angular.z)
        ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect", choices=("active", "safe-stop"), required=True)
    parser.add_argument("--stop-reason", default="")
    args = parser.parse_args()

    rclpy.init()
    node = RuntimeAudit()
    deadline = time.monotonic() + args.duration_s
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

    reasons = Counter(str(item.get("reason", "")) for item in node.statuses)
    active = [item for item in node.statuses if item.get("learned_checkpoint_active") is True]
    commands_finite = all(math.isfinite(value) for row in node.commands for value in row)
    max_abs_command = max((abs(value) for row in node.commands for value in row), default=0.0)
    stop_statuses = [
        item for item in node.statuses
        if item.get("learned_checkpoint_active") is False
        and (not args.stop_reason or item.get("reason") == args.stop_reason)
    ]
    if args.expect == "active":
        passed = bool(active) and commands_finite and max_abs_command > 0.0
    else:
        passed = bool(stop_statuses) and commands_finite and max_abs_command <= 1e-9

    latest = node.statuses[-1] if node.statuses else {}
    report = {
        "status": "passed" if passed else "failed",
        "expectation": args.expect,
        "expected_stop_reason": args.stop_reason or None,
        "status_samples": len(node.statuses),
        "command_samples": len(node.commands),
        "active_samples": len(active),
        "matching_stop_samples": len(stop_statuses),
        "reasons": dict(reasons),
        "commands_finite": commands_finite,
        "max_abs_command": max_abs_command,
        "latest": latest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "status_samples", "command_samples", "active_samples",
        "matching_stop_samples", "reasons", "max_abs_command"
    )}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
