from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import rclpy
from nav_msgs.msg import Odometry, Path as NavPath
from nav2_msgs.msg import Costmap
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


class CollectionPreflight(Node):
    def __init__(self) -> None:
        super().__init__("collection_preflight")
        self.declare_parameter("duration_s", 5.0); self.declare_parameter("report_output", "preflight.json")
        self.declare_parameter("disk_path", "."); self.declare_parameter("minimum_free_gib", 5.0)
        self.times = {name: [] for name in ("points", "odom", "imu", "costmap", "candidates", "controller", "plan")}
        self.difficulty_status = None
        self.create_subscription(PointCloud2, "/points", lambda msg: self.mark("points"), qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", lambda msg: self.mark("odom"), 50)
        self.create_subscription(Imu, "/imu/data_raw", lambda msg: self.mark("imu"), qos_profile_sensor_data)
        self.create_subscription(Costmap, "/local_costmap/costmap_raw", lambda msg: self.mark("costmap"), 10)
        self.create_subscription(String, "/planning/candidates", lambda msg: self.mark("candidates"), 20)
        self.create_subscription(String, "/simulation/controller_status", lambda msg: self.mark("controller"), 20)
        transient = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(NavPath, "/plan", lambda msg: self.mark("plan"), transient)
        self.create_subscription(String, "/simulation/difficulty_status", self.on_difficulty, transient)
        self.buffer = Buffer(); self.listener = TransformListener(self.buffer, self)

    def mark(self, name: str) -> None: self.times[name].append(time.monotonic())
    def on_difficulty(self, message) -> None:
        try: self.difficulty_status = json.loads(message.data)
        except ValueError: self.difficulty_status = {"ready": False, "parse_error": True}

    def report(self) -> dict:
        rates = {name: ((len(values)-1)/(values[-1]-values[0]) if len(values)>1 and values[-1]>values[0] else 0.0)
                 for name, values in self.times.items()}
        counts = {name: len(values) for name, values in self.times.items()}
        minimum_rates = {"points": 5.0, "odom": 10.0, "imu": 20.0, "costmap": 0.8, "candidates": 2.0, "controller": 0.5}
        failures = [f"{name}_frequency:{rates[name]:.3f}<{minimum}" for name, minimum in minimum_rates.items()
                    if rates[name] < minimum]
        if counts["plan"] < 1: failures.append("plan_missing")
        if not self.difficulty_status or self.difficulty_status.get("ready") is not True: failures.append("difficulty_not_ready")
        transforms = {}
        for target, source in (("odom", "base_footprint"), ("base_footprint", "c16_link")):
            key = f"{target}<-{source}"; transforms[key] = self.buffer.can_transform(target, source, Time(), Duration(seconds=0.2))
            if not transforms[key]: failures.append(f"tf_missing:{key}")
        usage = shutil.disk_usage(self.get_parameter("disk_path").value); minimum = int(float(self.get_parameter("minimum_free_gib").value)*1024**3)
        if usage.free < minimum: failures.append(f"disk_free:{usage.free}<{minimum}")
        return {"schema_version": "1.0", "status": "passed" if not failures else "failed", "failures": failures,
                "duration_s": float(self.get_parameter("duration_s").value), "counts": counts, "frequency_hz": rates,
                "transforms": transforms, "difficulty_status": self.difficulty_status,
                "disk": {"path": self.get_parameter("disk_path").value, "free_bytes": usage.free,
                         "minimum_free_bytes": minimum}}


def main(args=None) -> None:
    rclpy.init(args=args); node = CollectionPreflight(); started = time.monotonic()
    duration = float(node.get_parameter("duration_s").value)
    try:
        while rclpy.ok() and time.monotonic()-started < duration: rclpy.spin_once(node, timeout_sec=0.1)
        report = node.report(); output = Path(node.get_parameter("report_output").value)
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
        print(json.dumps(report, indent=2, sort_keys=True)); code = 0 if report["status"] == "passed" else 5
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    raise SystemExit(code)
