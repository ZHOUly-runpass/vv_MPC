from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from .scenario import load_scenario


class RoutePublisher(Node):
    def __init__(self) -> None:
        super().__init__("route_publisher")
        self.declare_parameter("scenario_file", "")
        self.declare_parameter("scenario", "empty")
        self.declare_parameter("difficulty", "nominal")
        self.declare_parameter("publish_plan", True)
        robot, scenario = load_scenario(self.get_parameter("scenario_file").value, self.get_parameter("scenario").value,
                                        self.get_parameter("difficulty").value)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.reference_publisher = self.create_publisher(Path, "/simulation/reference_route", qos)
        self.plan_publisher = self.create_publisher(Path, "/plan", qos) if self.get_parameter("publish_plan").value else None
        start, goal = robot["start"], scenario["goal"]
        length = math.hypot(goal[0] - start[0], goal[1] - start[1])
        count = max(2, math.ceil(length / float(robot["route_spacing_m"])) + 1)
        self.path = Path()
        self.path.header.frame_id = "odom"
        for index in range(count):
            ratio = index / (count - 1)
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.pose.position.x = start[0] + ratio * (goal[0] - start[0])
            pose.pose.position.y = start[1] + ratio * (goal[1] - start[1])
            pose.pose.orientation.w = 1.0
            self.path.poses.append(pose)
        self.create_timer(1.0, self.publish)
        self.publish()

    def publish(self) -> None:
        self.path.header.stamp = self.get_clock().now().to_msg()
        for pose in self.path.poses:
            pose.header.stamp = self.path.header.stamp
        self.reference_publisher.publish(self.path)
        if self.plan_publisher is not None: self.plan_publisher.publish(self.path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoutePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
