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
from nav_msgs.msg import Odometry, Path as PathMessage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

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


def transform_xyz(points: np.ndarray, transform) -> np.ndarray:
    quaternion = transform.rotation
    translation = transform.translation
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return points @ rotation.T + np.array([translation.x, translation.y, translation.z])


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
        self.latest_route: np.ndarray | None = None
        self.route_frame: str | None = None
        self.odometry_frame: str | None = None
        self.latest_odometry_s: float | None = None
        self.latest_imu_s: float | None = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pipeline: SafetyPlanningPipeline | None = None
        self._configure_pipeline()
        topics = self.config.get("ros2", "topics")
        self.create_subscription(Odometry, topics["odometry"]["name"], self._on_odometry, 20)
        self.create_subscription(Imu, topics["imu_raw"]["name"], self._on_imu, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, topics["point_cloud"]["name"], self._on_cloud, qos_profile_sensor_data)
        self.create_subscription(PathMessage, topics["global_path"]["name"], self._on_path, 10)
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
        self.odometry_frame = message.header.frame_id
        self.latest_odometry_s = stamp_seconds(message.header.stamp)

    def _on_imu(self, message: Imu) -> None:
        self.latest_imu_s = stamp_seconds(message.header.stamp)

    def _on_path(self, message: PathMessage) -> None:
        if len(message.poses) < 2 or not message.header.frame_id:
            self.latest_route, self.route_frame = None, None
            return
        self.latest_route = np.asarray(
            [[pose.pose.position.x, pose.pose.position.y] for pose in message.poses],
            dtype=np.float64,
        )
        self.route_frame = message.header.frame_id

    def _route_in_odometry_frame(self) -> np.ndarray | None:
        if self.latest_route is None or self.route_frame is None or self.odometry_frame is None:
            return None
        if self.route_frame == self.odometry_frame:
            return self.latest_route
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odometry_frame, self.route_frame, Time(), timeout=Duration(seconds=0.05)
            )
        except TransformException as error:
            self.get_logger().warning(f"route transform unavailable: {error}")
            return None
        xyz = np.column_stack([self.latest_route, np.zeros(len(self.latest_route))])
        return transform_xyz(xyz, transform.transform)[:, :2]

    def _on_cloud(self, message: PointCloud2) -> None:
        if self.pipeline is None or self.latest_state is None:
            return
        try:
            xyz = point_cloud2.read_points_numpy(message, field_names=("x", "y", "z"), skip_nans=True)
            points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
            timestamp = stamp_seconds(message.header.stamp)
            if self.odometry_frame is None:
                self._publish_zero("odometry_frame_missing")
                return
            try:
                cloud_transform = self.tf_buffer.lookup_transform(
                    self.odometry_frame, message.header.frame_id,
                    Time.from_msg(message.header.stamp), timeout=Duration(seconds=0.05),
                )
            except TransformException as error:
                self.get_logger().error(f"cloud transform unavailable: {error}")
                self._publish_zero("cloud_transform_missing")
                return
            points = transform_xyz(points, cloud_transform.transform)
            route = self._route_in_odometry_frame()
            now_s = self.get_clock().now().nanoseconds * 1e-9
            result = self.pipeline.cycle(
                LidarFrame(points, timestamp, self.odometry_frame, fields=("x", "y", "z")),
                self.latest_state.copy(),
                now_s,
                route_xy=route,
                odometry_s=self.latest_odometry_s,
                imu_s=self.latest_imu_s,
                tf_s=timestamp,
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
