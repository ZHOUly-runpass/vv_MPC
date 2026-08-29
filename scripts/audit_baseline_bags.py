#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from extract_rosbag_training_frames import bag_messages


TOPICS = {"/simulation/benchmark_status", "/simulation/controller_status", "/odom", "/cmd_vel"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit unified closed-loop evidence in baseline rosbags")
    parser.add_argument("--runs-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    latest = {}
    for line in args.runs_manifest.resolve().read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("status") == "success":
            latest[str(entry["controller"])] = entry
    results = []
    for controller in ("dwb", "mppi", "vanilla_dcbf", "proposed"):
        entry = latest.get(controller)
        if entry is None:
            results.append({"controller": controller, "status": "missing_successful_bag"})
            continue
        stream = bag_messages(Path(entry["bag"]), TOPICS)
        topics, _, _ = next(stream)
        counts = {topic: 0 for topic in TOPICS}; positions = []; commands = []; statuses = []; benchmark = None
        for topic, message, _ in stream:
            counts[topic] += 1
            if topic == "/odom":
                positions.append((float(message.pose.pose.position.x), float(message.pose.pose.position.y)))
            elif topic == "/cmd_vel":
                commands.append((float(message.linear.x), float(message.angular.z)))
            elif topic == "/simulation/controller_status":
                try: statuses.append(json.loads(message.data))
                except (TypeError, ValueError): pass
            elif topic == "/simulation/benchmark_status":
                try: benchmark = json.loads(message.data)
                except (TypeError, ValueError): pass
        displacement = math.dist(positions[0], positions[-1]) if len(positions) > 1 else 0.0
        nonzero = sum(abs(v) > 1e-4 or abs(w) > 1e-4 for v, w in commands)
        identities = sorted({str(item.get("baseline")) for item in statuses if item.get("baseline") is not None})
        checkpoint_values = sorted({bool(item["learned_checkpoint_active"]) for item in statuses
                                    if "learned_checkpoint_active" in item})
        passed = (TOPICS <= topics and all(counts.values()) and controller in identities
                  and len(positions) > 1 and nonzero > 0 and displacement > 0.01)
        results.append({
            "controller": controller, "status": "passed" if passed else "failed", "bag": entry["bag"],
            "topic_counts": counts, "controller_identities": identities,
            "learned_checkpoint_active_values": checkpoint_values, "nonzero_command_count": nonzero,
            "odom_displacement_m": displacement, "final_benchmark": benchmark,
        })
    complete = all(item["status"] == "passed" for item in results)
    proposed = next(item for item in results if item["controller"] == "proposed")
    report = {
        "schema_version": "1.0", "interface_smoke_status": "passed" if complete else "failed",
        "formal_proposed_status": "blocked" if proposed.get("learned_checkpoint_active_values") != [True] else "passed",
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8"); print(encoded, end="")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
