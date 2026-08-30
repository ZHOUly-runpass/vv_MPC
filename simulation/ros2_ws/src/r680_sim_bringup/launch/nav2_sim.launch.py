from __future__ import annotations

import os
from pathlib import Path
import tempfile

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def default_project_root() -> str:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "r680_safety_planner").is_dir():
            return str(parent)
    return os.environ.get("R680_PROJECT_ROOT", "")


def setup(context):
    scenario = LaunchConfiguration("scenario").perform(context)
    controller = LaunchConfiguration("controller").perform(context).lower()
    if controller not in {"dwb", "mppi", "vanilla_dcbf", "proposed"}:
        raise RuntimeError("controller must be dwb, mppi, vanilla_dcbf, or proposed")
    seed = LaunchConfiguration("seed").perform(context)
    difficulty = LaunchConfiguration("difficulty").perform(context)
    project_root = Path(LaunchConfiguration("project_root").perform(context)).resolve()
    expected_checkpoint_sha256 = LaunchConfiguration("expected_checkpoint_sha256").perform(context)
    bringup = get_package_share_directory("r680_sim_bringup")
    description = get_package_share_directory("r680_sim_description")
    nav2 = get_package_share_directory("nav2_bringup")
    params = os.path.join(bringup, "config", "nav2_params_dwb.yaml")
    if controller == "mppi":
        with open(params, encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        payload["controller_server"]["ros__parameters"]["FollowPath"] = {
            "plugin": "nav2_mppi_controller::MPPIController", "time_steps": 30, "model_dt": 0.1,
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
                          "seed": seed, "difficulty": difficulty, "baseline": controller}.items(),
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
    common.append(navigation)
    if controller in {"dwb", "mppi"}:
        common.append(Node(package="r680_sim_bringup", executable="nav_goal_sender",
                           parameters=[{"scenario_file": os.path.join(bringup, "config", "scenarios.yaml"),
                                        "scenario": scenario, "difficulty": difficulty,
                                        "baseline": controller, "use_sim_time": True}], output="screen"))
    elif controller == "vanilla_dcbf":
        common.append(Node(package="r680_sim_bringup", executable="baseline_controller",
                           parameters=[{"baseline": controller, "use_sim_time": True}], output="screen"))
    else:
        runtime = project_root / ".tools" / "runtime" / "proposed"
        feature_dir = runtime / "features"
        checkpoint = project_root / ".tools" / "training" / "r680_staged_v1_main" / "best.pt"
        dataset = project_root / ".tools" / "datasets" / "r680_staged_v1"
        unilion_checkpoint = project_root / "artifacts" / "checkpoints" / "unilion_lidar_backbone_init.safetensors"
        unilion_repository = project_root / "third_party" / "UniLION"
        unilion_config = unilion_repository / "projects" / "configs" / "unilion_swin_384_seq_e2e.py"
        isolated_python = project_root / ".tools" / "envs" / "unilion" / "bin" / "python3"
        isolated_environment = {"PYTHONPATH": str(project_root / "src"),
                                "PATH": str(isolated_python.parent) + os.pathsep + os.environ.get("PATH", "")}
        common.extend([
            Node(package="r680_sim_bringup", executable="pointcloud_file_bridge",
                 parameters=[{"output": str(runtime / "latest_points.npz"), "use_sim_time": True}], output="screen"),
            ExecuteProcess(cmd=[str(isolated_python), str(project_root / "scripts" / "live_unilion_feature_worker.py"),
                                "--input", str(runtime / "latest_points.npz"), "--output-dir", str(feature_dir),
                                "--repository", str(unilion_repository), "--model-config", str(unilion_config),
                                "--checkpoint", str(unilion_checkpoint)],
                           additional_env=isolated_environment, output="screen"),
            Node(package="r680_sim_bringup", executable="feature_health_bridge",
                 parameters=[{"health_file": str(feature_dir / "health.json"), "timeout_s": 0.30,
                              "use_sim_time": True}], output="screen"),
            ExecuteProcess(cmd=[str(isolated_python), str(project_root / "scripts" / "proposed_inference_worker.py"),
                                "--request-dir", str(runtime), "--checkpoint", str(checkpoint),
                                "--manifest", str(dataset / "manifest.jsonl"),
                                "--dataset-version", str(dataset / "dataset_version.json"),
                                "--unilion-checkpoint", str(unilion_checkpoint),
                                "--vehicle-config", str(project_root / "configs" / "robot" / "r680_sim.yaml"),
                                "--expected-checkpoint-sha256", expected_checkpoint_sha256],
                           additional_env=isolated_environment, output="screen"),
            Node(package="r680_sim_bringup", executable="baseline_controller",
                 parameters=[{"baseline": controller, "runtime_dir": str(runtime), "feature_timeout_s": 0.30,
                              "inference_timeout_s": 0.12, "use_sim_time": True}], output="screen"),
        ])
    return common


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="empty"),
        DeclareLaunchArgument("controller", default_value="dwb"),
        DeclareLaunchArgument("seed", default_value="42"),
        DeclareLaunchArgument("difficulty", default_value="nominal"),
        DeclareLaunchArgument("project_root", default_value=default_project_root()),
        DeclareLaunchArgument("expected_checkpoint_sha256", default_value="54cf9c73c73f3353199047746875637c48b97ac623d267ef776380744d7a7ee9"),
        OpaqueFunction(function=setup),
    ])
