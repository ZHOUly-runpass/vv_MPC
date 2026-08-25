from __future__ import annotations

import json
import time

import rclpy
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, PointCloud2
from std_msgs.msg import String


class RuntimeAudit(Node):
    def __init__(self) -> None:
        super().__init__("runtime_audit")
        self.declare_parameter("output", "simulation_runtime.json")
        self.declare_parameter("timeout_s", 25.0)
        self.started = time.monotonic()
        self.counts = {key: 0 for key in ("odom", "scan", "points", "imu", "plan", "ground_truth", "benchmark")}
        self.frames = {}
        self.point_fields = []
        self.gt_first = self.gt_last = None
        specs = [
            (Odometry, "/odom", "odom"), (LaserScan, "/scan", "scan"),
            (PointCloud2, "/points", "points"), (Imu, "/imu/data_raw", "imu"),
            (Path, "/plan", "plan"), (PoseArray, "/simulation/ground_truth_obstacle_poses", "ground_truth"),
            (String, "/simulation/benchmark_status", "benchmark"),
        ]
        for msg_type, topic, key in specs:
            self.create_subscription(msg_type, topic, lambda msg, k=key: self.receive(k, msg), 10)

    def receive(self, key, message) -> None:
        self.counts[key] += 1
        if hasattr(message, "header"):
            self.frames[key] = message.header.frame_id
        if key == "points":
            self.point_fields = [field.name for field in message.fields]
        if key == "ground_truth":
            positions = [[p.position.x, p.position.y, p.position.z] for p in message.poses]
            self.gt_first = positions if self.gt_first is None else self.gt_first
            self.gt_last = positions

    def complete(self) -> bool:
        required = ("odom", "scan", "points", "imu", "plan", "benchmark")
        return all(self.counts[key] > 0 for key in required) and time.monotonic() - self.started >= 3.0

    def write(self) -> bool:
        required_fields = {"x", "y", "z", "intensity", "ring"}
        movement = 0.0
        if self.gt_first and self.gt_last and len(self.gt_first) == len(self.gt_last):
            movement = max((sum((a - b) ** 2 for a, b in zip(first, last)) ** 0.5 for first, last in zip(self.gt_first, self.gt_last)), default=0.0)
        passed = self.complete() and required_fields.issubset(self.point_fields)
        payload = {
            "passed": passed, "counts": self.counts, "frames": self.frames,
            "point_fields": self.point_fields, "ground_truth_max_movement_m": movement,
            "wall_elapsed_s": time.monotonic() - self.started,
        }
        with open(self.get_parameter("output").value, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        return passed


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RuntimeAudit()
    timeout = float(node.get_parameter("timeout_s").value)
    try:
        while rclpy.ok() and time.monotonic() - node.started < timeout and not node.complete():
            rclpy.spin_once(node, timeout_sec=0.2)
        success = node.write()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if success else 1)
