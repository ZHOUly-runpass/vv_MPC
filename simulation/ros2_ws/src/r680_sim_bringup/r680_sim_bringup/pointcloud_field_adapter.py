from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from .scenario import elevation_to_ring


class PointCloudFieldAdapter(Node):
    def __init__(self) -> None:
        super().__init__("pointcloud_field_adapter")
        self.declare_parameter("input_topic", "/points_raw")
        self.declare_parameter("output_topic", "/points")
        self.declare_parameter("channels", 16)
        self.declare_parameter("vertical_min_deg", -15.0)
        self.declare_parameter("vertical_max_deg", 15.0)
        self.publisher = self.create_publisher(PointCloud2, self.get_parameter("output_topic").value, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.get_parameter("input_topic").value, self.callback, qos_profile_sensor_data)

    def callback(self, message: PointCloud2) -> None:
        names = {field.name for field in message.fields}
        if not {"x", "y", "z", "intensity"}.issubset(names):
            self.get_logger().error("points_raw lacks x/y/z/intensity; refusing to invent a model field")
            return
        values = point_cloud2.read_points_numpy(message, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        values = np.asarray(values, dtype=np.float32).reshape(-1, 4)
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
        ]
        rows = [(*row.tolist(), int(ring)) for row, ring in zip(values, rings)]
        self.publisher.publish(point_cloud2.create_cloud(message.header, fields, rows))


def main(args=None) -> None:
    rclpy.init(args=args); node = PointCloudFieldAdapter()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()
