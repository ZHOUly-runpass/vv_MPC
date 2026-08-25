from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    scenario = LaunchConfiguration("scenario").perform(context)
    controller = LaunchConfiguration("controller").perform(context).lower()
    if controller != "dwb":
        raise RuntimeError("only controller:=dwb is configured until the installed Humble MPPI plugin is audited")
    bringup = get_package_share_directory("r680_sim_bringup")
    description = get_package_share_directory("r680_sim_description")
    nav2 = get_package_share_directory("nav2_bringup")
    params = os.path.join(bringup, "config", "nav2_params_dwb.yaml")
    map_yaml = os.path.join(description, "maps", "empty_map.yaml")
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, "launch", "simulation.launch.py")),
        launch_arguments={"scenario": scenario, "headless": "true", "use_ros2_control": "true", "publish_route": "false"}.items(),
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2, "launch", "navigation_launch.py")),
        launch_arguments={"use_sim_time": "true", "autostart": "true", "params_file": params}.items(),
    )
    return [
        simulation,
        Node(package="tf2_ros", executable="static_transform_publisher", arguments=["0", "0", "0", "0", "0", "0", "map", "odom"]),
        Node(package="nav2_map_server", executable="map_server", name="map_server", parameters=[{"yaml_filename": map_yaml, "use_sim_time": True}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_map",
             parameters=[{"use_sim_time": True, "autostart": True, "node_names": ["map_server"]}]),
        navigation,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="empty"),
        DeclareLaunchArgument("controller", default_value="dwb"),
        OpaqueFunction(function=setup),
    ])
