from __future__ import annotations

import json
import math
from pathlib import Path
import sys

# colcon's generated entry point uses the system Python. Keep the algorithm
# source inside this project rather than requiring a user/global installation.
for _parent in Path(__file__).resolve().parents:
    _source_root = _parent / "src"
    if (_source_root / "r680_safety_planner").is_dir():
        sys.path.insert(0, str(_source_root))
        break

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from r680_safety_planner.config import load_project_config
from r680_safety_planner.interfaces import LidarFrame
from r680_safety_planner.pipeline import SafetyPlanningPipeline
from r680_safety_planner.vehicle import VehicleLimits, build_vehicle_model


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_yaw(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class PlannerNode(Node):
    """ROS adapter that remains a zero-command guard until config is unlocked."""

    def __init__(self) -> None:
        super().__init__("r680_safety_planner")
        self.declare_parameter("config", "")
        config_path = Path(str(self.get_parameter("config").value))
        if not config_path.is_file():
            raise RuntimeError(f"config does not exist: {config_path}")
        self.config = load_project_config(config_path)
        self.command_publisher = self.create_publisher(Twist, self.config.get("command_adapter", "output_topic"), 1)
        self.status_publisher = self.create_publisher(String, "/r680_safety/status", 10)
        self.latest_state: np.ndarray | None = None
        self.pipeline: SafetyPlanningPipeline | None = None
        self._configure_pipeline()
        topics = self.config.get("ros2", "topics")
        self.create_subscription(Odometry, topics["odometry"]["name"], self._on_odometry, 20)
        self.create_subscription(PointCloud2, topics["point_cloud"]["name"], self._on_cloud, qos_profile_sensor_data)
        zero_rate = float(self.config.get("command_adapter", "zero_publish_rate_hz"))
        self.create_timer(1.0 / zero_rate, self._publish_zero_if_locked)
        self.get_logger().warning(
            "motion_unlocked=%s; failed_gates=%d" % (self.config.motion_unlocked, len(self.config.failed_gates))
        )

    def _configure_pipeline(self) -> None:
        variant = self.config.get("vehicle", "variant")
        geometry = self.config.get("vehicle", "geometry")
        footprint = geometry.get("footprint_polygon_xy_m")
        if variant == "unresolved" or not footprint:
            self.get_logger().error("vehicle variant/footprint unresolved: planner held at zero")
            return
        limits_source = "physical_limits" if self.config.motion_unlocked else "commissioning_limits"
        limits = VehicleLimits.from_mapping(self.config.get("vehicle", limits_source))
        model = build_vehicle_model(variant, limits, geometry)
        radius = max(math.hypot(float(x), float(y)) for x, y in footprint)
        self.pipeline = SafetyPlanningPipeline(self.config, model, radius)

    def _on_odometry(self, message: Odometry) -> None:
        pose, twist = message.pose.pose, message.twist.twist
        base = [pose.position.x, pose.position.y, quaternion_yaw(pose.orientation)]
        variant = self.config.get("vehicle", "variant")
        if variant in {"differential", "skid_steer"}:
            base.extend([twist.linear.x, twist.angular.z])
        elif variant in {"mecanum", "omni"}:
            base.extend([twist.linear.x, twist.linear.y, twist.angular.z])
        elif variant == "ackermann":
            base.append(twist.linear.x)
        else:
            return
        self.latest_state = np.asarray(base, dtype=np.float64)

    def _on_cloud(self, message: PointCloud2) -> None:
        if self.pipeline is None or self.latest_state is None:
            return
        try:
            xyz = point_cloud2.read_points_numpy(message, field_names=("x", "y", "z"), skip_nans=True)
            points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
            timestamp = stamp_seconds(message.header.stamp)
            result = self.pipeline.cycle(
                LidarFrame(points, timestamp, message.header.frame_id, fields=("x", "y", "z")),
                self.latest_state.copy(),
                self.get_clock().now().nanoseconds * 1e-9,
            )
            command = Twist()
            command.linear.x, command.linear.y, command.angular.z = result.command.as_array().tolist()
            self.command_publisher.publish(command)
            status = String()
            status.data = json.dumps({
                "motion_allowed": result.motion_allowed,
                "obstacles": result.obstacle_count,
                "candidate": result.selected_role,
                "safety_codes": result.safety_codes,
            })
            self.status_publisher.publish(status)
        except Exception as error:  # ROS boundary must fail closed
            self.get_logger().error(f"planning cycle failed: {error}")
            self._publish_zero("cycle_exception")

    def _publish_zero_if_locked(self) -> None:
        if not self.config.motion_unlocked or self.pipeline is None:
            self._publish_zero("commissioning_lock")

    def _publish_zero(self, reason: str) -> None:
        self.command_publisher.publish(Twist())
        status = String()
        status.data = json.dumps({"motion_allowed": False, "reason": reason})
        self.status_publisher.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PlannerNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node._publish_zero("shutdown")
            node.destroy_node()
        rclpy.shutdown()
