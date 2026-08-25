from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class CmdVelRelay(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_relay")
        self.cmd_publisher = self.create_publisher(Twist, "/diff_drive_base_controller/cmd_vel_unstamped", 1)
        self.odom_publisher = self.create_publisher(Odometry, "/odom", 10)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_publisher.publish, 1)
        self.create_subscription(Odometry, "/diff_drive_base_controller/odom", self.odom_publisher.publish, 10)


def main(args=None) -> None:
    rclpy.init(args=args); node = CmdVelRelay()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.cmd_publisher.publish(Twist()); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
