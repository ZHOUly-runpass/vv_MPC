from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class BaselineController(Node):
    """Common closed-loop interface for Vanilla D-CBF and Proposed safety control.

    Proposed uses constant-velocity obstacle prediction and a larger uncertainty margin.
    A learned checkpoint is intentionally not silently replaced by fake inference; the node
    reports the deterministic safety fallback in its status topic until a checkpoint bridge is supplied.
    """
    def __init__(self) -> None:
        super().__init__("r680_baseline_controller")
        self.declare_parameter("baseline", "vanilla_dcbf")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("robot_radius_m", 0.31)
        self.baseline = str(self.get_parameter("baseline").value).lower()
        if self.baseline not in {"vanilla_dcbf", "proposed"}:
            raise ValueError("baseline must be vanilla_dcbf or proposed")
        self.path = None; self.odom = None; self.obstacles = []
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status = self.create_publisher(String, "/simulation/controller_status", 10)
        self.create_subscription(Path, "/plan", self.on_path, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(String, "/simulation/ground_truth_obstacles", self.on_obstacles, 10)
        self.create_timer(0.1, self.tick)

    def on_path(self, message): self.path = message
    def on_odom(self, message): self.odom = message
    def on_obstacles(self, message):
        try: self.obstacles = json.loads(message.data).get("obstacles", [])
        except (ValueError, TypeError): self.obstacles = []

    def tick(self) -> None:
        command = Twist(); reason = "waiting_for_route_or_odom"; h_min = None
        if self.path and self.path.poses and self.odom:
            pose = self.odom.pose.pose; x, y = pose.position.x, pose.position.y
            yaw = yaw_from_quaternion(pose.orientation); goal = self.path.poses[-1].pose.position
            distance = math.hypot(goal.x - x, goal.y - y)
            desired = math.atan2(goal.y - y, goal.x - x); error = math.atan2(math.sin(desired-yaw), math.cos(desired-yaw))
            command.linear.x = min(0.5, 0.35 * distance) * max(0.0, math.cos(error))
            command.angular.z = max(-0.8, min(0.8, 1.5 * error)); reason = "nominal"
            horizon = 0.8 if self.baseline == "proposed" else 0.0
            margin = 0.18 if self.baseline == "proposed" else 0.08
            h_min = float("inf")
            for obstacle in self.obstacles:
                if not obstacle.get("collision_check", True): continue
                ox, oy = obstacle["position"][:2]; vx, vy = obstacle["velocity"][:2]
                clearance = math.hypot(x - (ox + vx*horizon), y - (oy + vy*horizon))
                clearance -= float(self.get_parameter("robot_radius_m").value) + float(obstacle["radius_m"]) + margin
                h_min = min(h_min, clearance)
            if h_min < 0.0:
                command.linear.x = 0.0; reason = "dcbf_stop"
                command.angular.z = max(-0.5, min(0.5, command.angular.z))
            elif h_min < 0.8:
                command.linear.x *= max(0.1, h_min / 0.8); reason = "dcbf_slowdown"
            if distance < 0.2: command = Twist(); reason = "goal_reached"
        self.publisher.publish(command)
        payload = {"baseline": self.baseline, "reason": reason, "h_min": None if h_min is None or math.isinf(h_min) else h_min,
                   "learned_checkpoint_active": False}
        self.status.publish(String(data=json.dumps(payload, sort_keys=True)))


def main(args=None) -> None:
    rclpy.init(args=args); node = BaselineController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.publisher.publish(Twist()); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
