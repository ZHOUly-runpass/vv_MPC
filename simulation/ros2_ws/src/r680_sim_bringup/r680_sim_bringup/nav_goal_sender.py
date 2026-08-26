from __future__ import annotations

import json

import rclpy
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from .scenario import load_scenario


class NavGoalSender(Node):
    def __init__(self) -> None:
        super().__init__("nav_goal_sender")
        self.declare_parameter("scenario_file", "")
        self.declare_parameter("scenario", "empty")
        self.declare_parameter("baseline", "dwb")
        _, scenario = load_scenario(self.get_parameter("scenario_file").value, self.get_parameter("scenario").value)
        self.goal_xy = scenario["goal"]; self.sent = False
        self.client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.publisher = self.create_publisher(String, "/simulation/controller_status", 10)
        self.create_timer(0.5, self.tick)

    def publish(self, state: str, **extra) -> None:
        payload = {"baseline": self.get_parameter("baseline").value, "state": state, **extra}
        self.publisher.publish(String(data=json.dumps(payload, sort_keys=True)))

    def tick(self) -> None:
        if self.sent or not self.client.server_is_ready(): return
        goal = NavigateToPose.Goal(); goal.pose.header.frame_id = "map"; goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(self.goal_xy[0]); goal.pose.pose.position.y = float(self.goal_xy[1]); goal.pose.pose.orientation.w = 1.0
        self.sent = True; future = self.client.send_goal_async(goal); future.add_done_callback(self.accepted); self.publish("sent")

    def accepted(self, future) -> None:
        handle = future.result()
        if not handle.accepted: self.publish("rejected"); return
        self.publish("accepted"); handle.get_result_async().add_done_callback(self.finished)

    def finished(self, future) -> None: self.publish("finished", result_status=int(future.result().status))


def main(args=None) -> None:
    rclpy.init(args=args); node = NavGoalSender()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
