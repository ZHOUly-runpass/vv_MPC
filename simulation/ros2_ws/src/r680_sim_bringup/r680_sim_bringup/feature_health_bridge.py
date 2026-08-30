from __future__ import annotations

import json
from pathlib import Path
import sys
import time

for parent in Path(__file__).resolve().parents:
    source = parent / "src"
    if (source / "r680_safety_planner").is_dir():
        sys.path.insert(0, str(source)); break

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String

from r680_safety_planner.data import load_feature_health


class FeatureHealthBridge(Node):
    """Fail-closed bridge between the isolated CUDA worker and ROS control."""

    def __init__(self) -> None:
        super().__init__("feature_health_bridge")
        self.declare_parameter("health_file", "")
        self.declare_parameter("timeout_s", 0.30)
        self.health_file = Path(str(self.get_parameter("health_file").value))
        self.health_pub = self.create_publisher(String, "/unilion/health", 10)
        self.feature_pub = self.create_publisher(String, "/unilion/frozen_scene_features", 10)
        self.stop_pub = self.create_publisher(Bool, "/safety/feature_stop_requested", 10)
        self.zero_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_timer(0.05, self.tick)

    def tick(self) -> None:
        state = load_feature_health(self.health_file, time.time(), float(self.get_parameter("timeout_s").value))
        health = {"schema_version": "1.0", "healthy": state.healthy, "reason": state.reason, "age_s": state.age_s}
        self.health_pub.publish(String(data=json.dumps(health, sort_keys=True)))
        self.stop_pub.publish(Bool(data=not state.healthy))
        if state.healthy:
            summary = {key: state.payload.get(key) for key in (
                "schema_version", "cache_path", "cache_sha256", "feature_shape",
                "checkpoint_sha256", "config_sha256", "feature_variance_median",
            )}
            self.feature_pub.publish(String(data=json.dumps(summary, sort_keys=True)))
        else:
            self.zero_pub.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FeatureHealthBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.zero_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
