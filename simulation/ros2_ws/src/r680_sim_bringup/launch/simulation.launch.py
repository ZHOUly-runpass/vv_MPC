from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_setup(context):
    scenario_name = LaunchConfiguration("scenario").perform(context)
    headless = LaunchConfiguration("headless").perform(context).lower() in ("1", "true", "yes")
    use_ros2_control = LaunchConfiguration("use_ros2_control").perform(context).lower() in ("1", "true", "yes")
    publish_route = LaunchConfiguration("publish_route").perform(context).lower() in ("1", "true", "yes")
    seed = int(LaunchConfiguration("seed").perform(context))
    difficulty = LaunchConfiguration("difficulty").perform(context).lower()
    baseline = LaunchConfiguration("baseline").perform(context).lower()
    difficulty_scales = {"easy": 0.75, "nominal": 1.0, "hard": 1.25}
    if difficulty not in difficulty_scales:
        raise RuntimeError(f"unknown difficulty {difficulty!r}; choices={sorted(difficulty_scales)}")
    bringup_share = get_package_share_directory("r680_sim_bringup")
    description_share = get_package_share_directory("r680_sim_description")
    worlds_share = get_package_share_directory("r680_sim_worlds")
    scenario_file = os.path.join(bringup_share, "config", "scenarios.yaml")
    with open(scenario_file, encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if scenario_name not in payload["scenarios"]:
        raise RuntimeError(f"unknown scenario {scenario_name!r}; choices={sorted(payload['scenarios'])}")
    scenario = payload["scenarios"][scenario_name]
    robot = payload["robot"]
    world = os.path.join(worlds_share, "worlds", scenario["world"])
    if not os.path.isfile(world):
        raise RuntimeError(f"scenario world does not exist: {world}")
    xacro_file = os.path.join(description_share, "urdf", "r680_sim.urdf.xacro")
    description = ParameterValue(Command(["xacro ", xacro_file, " use_ros2_control:=", str(use_ros2_control).lower()]), value_type=str)
    common = {"scenario_file": scenario_file, "scenario": scenario_name, "use_sim_time": True,
              "seed": seed, "difficulty": difficulty, "difficulty_scale": difficulty_scales[difficulty],
              "baseline": baseline}
    gazebo_models = os.pathsep.join(filter(None, [
        os.path.join(worlds_share, "models"), "/usr/share/gazebo-11/models",
        os.environ.get("GAZEBO_MODEL_PATH", ""),
    ]))
    gazebo = ExecuteProcess(cmd=[
        "gzserver", world, "-s", "libgazebo_ros_init.so", "-s", "libgazebo_ros_factory.so",
        "-s", "libgazebo_ros_force_system.so", "--seed", str(seed),
    ], output="screen")
    start = robot["start"]
    return [
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", gazebo_models),
        gazebo,
        *([ExecuteProcess(cmd=["gzclient"], output="screen")] if not headless else []),
        Node(package="robot_state_publisher", executable="robot_state_publisher", parameters=[{"robot_description": description, "use_sim_time": True}]),
        Node(package="gazebo_ros", executable="spawn_entity.py", arguments=[
            "-entity", robot["model_name"], "-topic", "robot_description",
            "-x", str(start[0]), "-y", str(start[1]), "-z", str(start[2]), "-Y", str(start[3]),
        ], output="screen"),
        Node(package="r680_sim_bringup", executable="pointcloud_field_adapter", parameters=[{"use_sim_time": True}], output="screen"),
        Node(package="r680_sim_bringup", executable="ground_truth_bridge", parameters=[common], output="screen"),
        Node(package="r680_sim_bringup", executable="dynamic_obstacle_controller", parameters=[common], output="screen"),
        *([Node(package="r680_sim_bringup", executable="route_publisher", parameters=[common], output="screen")] if publish_route else []),
        *([Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"], output="screen"),
           Node(package="controller_manager", executable="spawner", arguments=["diff_drive_base_controller", "--controller-manager", "/controller_manager"], output="screen"),
           Node(package="r680_sim_bringup", executable="cmd_vel_relay", parameters=[{"use_sim_time": True}], output="screen")]
          if use_ros2_control else []),
        Node(package="r680_sim_bringup", executable="benchmark_manager", parameters=[common], output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="empty"),
        DeclareLaunchArgument("headless", default_value="true"),
        DeclareLaunchArgument("use_ros2_control", default_value="false"),
        DeclareLaunchArgument("publish_route", default_value="true"),
        DeclareLaunchArgument("seed", default_value="42"),
        DeclareLaunchArgument("difficulty", default_value="nominal"),
        DeclareLaunchArgument("baseline", default_value="uncontrolled"),
        OpaqueFunction(function=launch_setup),
    ])
