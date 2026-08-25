from __future__ import annotations

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from rclpy.node import Node

from .scenario import dynamic_obstacles, load_scenario, triangle_position


class DynamicObstacleController(Node):
    def __init__(self) -> None:
        super().__init__("dynamic_obstacle_controller")
        self.declare_parameter("scenario_file", "")
        self.declare_parameter("scenario", "empty")
        _, scenario = load_scenario(self.get_parameter("scenario_file").value, self.get_parameter("scenario").value)
        self.obstacles = dynamic_obstacles(scenario)
        self.started_ns = self.get_clock().now().nanoseconds
        self.client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.create_timer(0.05, self.tick)

    def tick(self) -> None:
        if not self.client.service_is_ready():
            return
        elapsed = (self.get_clock().now().nanoseconds - self.started_ns) * 1e-9
        for item in self.obstacles:
            coordinate = triangle_position(
                float(item["start"]), float(item["minimum"]), float(item["maximum"]), float(item["speed"]), elapsed
            )
            request = SetEntityState.Request()
            state = EntityState(name=item["name"], reference_frame="world")
            if item["axis"] == "x":
                state.pose.position.x, state.pose.position.y = coordinate, float(item["fixed"][0])
                state.twist.linear.x = float(item["speed"])
            elif item["axis"] == "y":
                state.pose.position.x, state.pose.position.y = float(item["fixed"][0]), coordinate
                state.twist.linear.y = float(item["speed"])
            else:
                self.get_logger().error(f"unsupported obstacle axis: {item['axis']}")
                continue
            state.pose.position.z = float(item["fixed"][1])
            state.pose.orientation.w = 1.0
            request.state = state
            self.client.call_async(request)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DynamicObstacleController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
