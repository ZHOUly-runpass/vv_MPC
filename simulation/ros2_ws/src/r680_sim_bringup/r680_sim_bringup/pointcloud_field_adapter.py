from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from .scenario import azimuth_to_relative_time, elevation_to_ring, rigid_transform_xyz


class PointCloudFieldAdapter(Node):
    def __init__(self) -> None:
        super().__init__("pointcloud_field_adapter")
        self.declare_parameter("input_topic", "/points_raw")
        self.declare_parameter("output_topic", "/points")
        self.declare_parameter("channels", 16)
        self.declare_parameter("vertical_min_deg", -15.0)
        self.declare_parameter("vertical_max_deg", 15.0)
        self.declare_parameter("scan_period_s", 0.1)
        self.declare_parameter("derive_simulated_time", True)
        self.declare_parameter("output_frame", "base_footprint")
        self.declare_parameter("translation_xyz_m", [0.08, 0.0, 0.43])
        self.declare_parameter("rotation_rpy_rad", [0.0, 0.0, 0.0])
        self.publisher = self.create_publisher(PointCloud2, self.get_parameter("output_topic").value, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.get_parameter("input_topic").value, self.callback, qos_profile_sensor_data)

    def callback(self, message: PointCloud2) -> None:
        names = {field.name for field in message.fields}
        if not {"x", "y", "z", "intensity"}.issubset(names):
            self.get_logger().error("points_raw lacks x/y/z/intensity; refusing to invent a model field")
            return
        values = point_cloud2.read_points_numpy(message, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        values = np.asarray(values, dtype=np.float32).reshape(-1, 4)
        if not bool(self.get_parameter("derive_simulated_time").value):
            self.get_logger().error("input lacks time and simulator-only derivation is disabled")
            return
        relative_time = azimuth_to_relative_time(values[:, :3], float(self.get_parameter("scan_period_s").value))
        xyz = rigid_transform_xyz(
            values[:, :3], list(self.get_parameter("translation_xyz_m").value),
            list(self.get_parameter("rotation_rpy_rad").value),
        )
        values[:, :3] = xyz.astype(np.float32)
        rings = elevation_to_ring(
            values[:, :3], int(self.get_parameter("channels").value),
            float(self.get_parameter("vertical_min_deg").value), float(self.get_parameter("vertical_max_deg").value),
        )
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=16, datatype=PointField.UINT16, count=1),
            PointField(name="time", offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        rows = [(*row.tolist(), int(ring), float(timestamp)) for row, ring, timestamp in zip(values, rings, relative_time)]
        header = message.header
        header.frame_id = str(self.get_parameter("output_frame").value)
        self.publisher.publish(point_cloud2.create_cloud(header, fields, rows))


def main(args=None) -> None:
    rclpy.init(args=args); node = PointCloudFieldAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
