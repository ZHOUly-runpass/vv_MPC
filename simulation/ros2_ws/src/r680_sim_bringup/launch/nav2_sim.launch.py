from __future__ import annotations

import os
import tempfile

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    scenario = LaunchConfiguration("scenario").perform(context)
    controller = LaunchConfiguration("controller").perform(context).lower()
    if controller not in {"dwb", "mppi", "vanilla_dcbf", "proposed"}:
        raise RuntimeError("controller must be dwb, mppi, vanilla_dcbf, or proposed")
    seed = LaunchConfiguration("seed").perform(context)
    difficulty = LaunchConfiguration("difficulty").perform(context)
    bringup = get_package_share_directory("r680_sim_bringup")
    description = get_package_share_directory("r680_sim_description")
    nav2 = get_package_share_directory("nav2_bringup")
    params = os.path.join(bringup, "config", "nav2_params_dwb.yaml")
    if controller == "mppi":
        with open(params, encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        payload["controller_server"]["ros__parameters"]["FollowPath"] = {
            "plugin": "nav2_mppi_controller::MPPIController", "time_steps": 56, "model_dt": 0.05,
            "batch_size": 1200, "vx_max": 0.5, "vx_min": 0.0, "wz_max": 0.8,
            "iteration_count": 1, "motion_model": "DiffDrive", "visualize": False,
            "critics": ["ConstraintCritic", "ObstaclesCritic", "GoalCritic", "GoalAngleCritic", "PathAlignCritic", "PathFollowCritic", "PathAngleCritic", "PreferForwardCritic"],
        }
        generated = tempfile.NamedTemporaryFile(mode="w", prefix="r680_mppi_", suffix=".yaml", delete=False)
        yaml.safe_dump(payload, generated); generated.close(); params = generated.name
    map_yaml = os.path.join(description, "maps", "empty_map.yaml")
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, "launch", "simulation.launch.py")),
        launch_arguments={"scenario": scenario, "headless": "true", "use_ros2_control": "true",
                          "publish_route": "true" if controller in {"vanilla_dcbf", "proposed"} else "false",
                          "seed": seed, "difficulty": difficulty}.items(),
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2, "launch", "navigation_launch.py")),
        launch_arguments={"use_sim_time": "true", "autostart": "true", "params_file": params}.items(),
    )
    common = [
        simulation,
        Node(package="tf2_ros", executable="static_transform_publisher", arguments=["0", "0", "0", "0", "0", "0", "map", "odom"]),
        Node(package="nav2_map_server", executable="map_server", name="map_server", parameters=[{"yaml_filename": map_yaml, "use_sim_time": True}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_map",
             parameters=[{"use_sim_time": True, "autostart": True, "node_names": ["map_server"]}]),
    ]
    if controller in {"dwb", "mppi"}:
        common.append(navigation)
    else:
        common.append(Node(package="r680_sim_bringup", executable="baseline_controller",
                           parameters=[{"baseline": controller, "use_sim_time": True}], output="screen"))
    return common


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="empty"),
        DeclareLaunchArgument("controller", default_value="dwb"),
        DeclareLaunchArgument("seed", default_value="42"),
        DeclareLaunchArgument("difficulty", default_value="nominal"),
        OpaqueFunction(function=setup),
    ])
