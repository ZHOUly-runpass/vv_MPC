from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String


class PointCloudFileBridge(Node):
    """Atomically transport live PointCloud2 frames to the isolated CUDA worker."""

    def __init__(self) -> None:
        super().__init__("pointcloud_file_bridge")
        self.declare_parameter("output", "")
        self.output = Path(str(self.get_parameter("output").value)); self.output.parent.mkdir(parents=True, exist_ok=True)
        self.status = self.create_publisher(String, "/unilion/input_status", 10)
        self.create_subscription(PointCloud2, "/points", self.on_points, qos_profile_sensor_data)

    def on_points(self, message: PointCloud2) -> None:
        names = {field.name for field in message.fields}; required = ("x", "y", "z", "intensity", "ring", "time")
        if not set(required).issubset(names):
            self.status.publish(String(data=json.dumps({"healthy": False, "reason": "missing_point_fields"}))); return
        structured = point_cloud2.read_points(message, field_names=required, skip_nans=True)
        points = np.column_stack([structured[name] for name in required]).astype(np.float32, copy=False)
        temporary = self.output.with_name(self.output.stem + ".tmp.npz")
        np.savez(temporary, points=points, stamp_s=np.asarray(message.header.stamp.sec + message.header.stamp.nanosec * 1e-9),
                 frame_id=np.asarray(message.header.frame_id), wall_time_s=np.asarray(time.time()))
        os.replace(temporary, self.output)
        self.status.publish(String(data=json.dumps({"healthy": True, "points": int(points.shape[0])})))


def main(args=None) -> None:
    rclpy.init(args=args); node = PointCloudFileBridge()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
