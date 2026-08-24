from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = LaunchConfiguration("config")
    return LaunchDescription([
        DeclareLaunchArgument("config", description="Absolute r680_c16.yaml path"),
        Node(
            package="r680_safety_planner_ros",
            executable="planner_node",
            name="r680_safety_planner",
            output="screen",
            parameters=[{"config": config}],
        ),
    ])
