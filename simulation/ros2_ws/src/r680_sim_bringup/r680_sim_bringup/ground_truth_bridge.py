from __future__ import annotations

import json

import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from .scenario import load_scenario, obstacle_catalog


class GroundTruthBridge(Node):
    def __init__(self) -> None:
        super().__init__("ground_truth_bridge")
        self.declare_parameter("scenario_file", "")
        self.declare_parameter("scenario", "empty")
        self.declare_parameter("difficulty", "nominal")
        _, scenario = load_scenario(self.get_parameter("scenario_file").value, self.get_parameter("scenario").value,
                                    self.get_parameter("difficulty").value)
        self.catalog = {item["name"]: item for item in obstacle_catalog(scenario)}
        self.pose_pub = self.create_publisher(PoseArray, "/simulation/ground_truth_obstacle_poses", 10)
        self.json_pub = self.create_publisher(String, "/simulation/ground_truth_obstacles", 10)
        self.create_subscription(ModelStates, "/model_states", self.callback, qos_profile_sensor_data)

    def callback(self, message: ModelStates) -> None:
        poses = PoseArray()
        poses.header.stamp = self.get_clock().now().to_msg()
        poses.header.frame_id = "odom"
        records = []
        for name, pose, twist in zip(message.name, message.pose, message.twist):
            if name not in self.catalog:
                continue
            spec = self.catalog[name]
            poses.poses.append(pose)
            records.append({
                "name": name, "radius_m": float(spec["radius_m"]),
                "collision_check": bool(spec.get("collision_check", True)),
                "position": [pose.position.x, pose.position.y, pose.position.z],
                "velocity": [twist.linear.x, twist.linear.y, twist.linear.z],
            })
        self.pose_pub.publish(poses)
        self.json_pub.publish(String(data=json.dumps({"frame_id": "odom", "obstacles": records}, sort_keys=True)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundTruthBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
