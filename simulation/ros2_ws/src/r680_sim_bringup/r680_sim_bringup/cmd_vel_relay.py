from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelRelay(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_relay")
        self.publisher = self.create_publisher(Twist, "/diff_drive_base_controller/cmd_vel_unstamped", 1)
        self.create_subscription(Twist, "/cmd_vel", self.publisher.publish, 1)


def main(args=None) -> None:
    rclpy.init(args=args); node = CmdVelRelay()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.publisher.publish(Twist()); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
