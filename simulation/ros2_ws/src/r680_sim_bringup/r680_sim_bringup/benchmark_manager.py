from __future__ import annotations

import json
import math

import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger

from .scenario import load_scenario, obstacle_catalog


class BenchmarkManager(Node):
    def __init__(self) -> None:
        super().__init__("benchmark_manager")
        self.declare_parameter("scenario_file", "")
        self.declare_parameter("scenario", "empty")
        self.robot, scenario = load_scenario(self.get_parameter("scenario_file").value, self.get_parameter("scenario").value)
        self.scenario_name = self.get_parameter("scenario").value
        self.goal = scenario["goal"]
        self.obstacles = {o["name"]: o for o in obstacle_catalog(scenario) if o.get("collision_check", True)}
        self.started_ns = self.get_clock().now().nanoseconds
        self.min_clearance = float("inf")
        self.collision = False
        self.reached_goal = False
        self.publisher = self.create_publisher(String, "/simulation/benchmark_status", 10)
        self.create_subscription(ModelStates, "/gazebo/model_states", self.callback, 10)
        self.reset_client = self.create_client(Empty, "/reset_simulation")
        self.create_service(Trigger, "/simulation/reset_benchmark", self.reset)

    def callback(self, message: ModelStates) -> None:
        indexed = {name: pose for name, pose in zip(message.name, message.pose)}
        if self.robot["model_name"] not in indexed:
            return
        robot_pose = indexed[self.robot["model_name"]]
        robot_radius = float(self.robot["radius_m"])
        for name, spec in self.obstacles.items():
            if name not in indexed:
                continue
            pose = indexed[name]
            clearance = math.hypot(robot_pose.position.x - pose.position.x, robot_pose.position.y - pose.position.y)
            clearance -= robot_radius + float(spec["radius_m"])
            self.min_clearance = min(self.min_clearance, clearance)
            self.collision |= clearance <= 0.0
        goal_distance = math.hypot(robot_pose.position.x - self.goal[0], robot_pose.position.y - self.goal[1])
        self.reached_goal |= goal_distance <= float(self.robot["goal_tolerance_m"])
        payload = {
            "scenario": self.scenario_name,
            "elapsed_s": (self.get_clock().now().nanoseconds - self.started_ns) * 1e-9,
            "goal_distance_m": goal_distance,
            "reached_goal": self.reached_goal,
            "collision": self.collision,
            "min_clearance_m": None if math.isinf(self.min_clearance) else self.min_clearance,
        }
        self.publisher.publish(String(data=json.dumps(payload, sort_keys=True)))

    def reset(self, _request, response):
        self.started_ns = self.get_clock().now().nanoseconds
        self.min_clearance, self.collision, self.reached_goal = float("inf"), False, False
        if self.reset_client.service_is_ready():
            self.reset_client.call_async(Empty.Request())
            response.success, response.message = True, "Gazebo reset requested and benchmark counters cleared"
        else:
            response.success, response.message = False, "Gazebo reset service unavailable; counters cleared only"
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BenchmarkManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
