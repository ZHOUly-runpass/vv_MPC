#!/usr/bin/env bash
set -eo pipefail
project_dir="${1:?usage: run_baseline_matrix.sh PROJECT_DIR [SCENARIO] [DURATION_SECONDS]}"
scenario="${2:-static_sparse}"
duration="${3:-30}"
case "$(realpath "$project_dir")" in /*/05|*/E2Eproject_MPC_05_dev) ;; *) exit 2 ;; esac
cd "$project_dir"
source /opt/ros/humble/setup.bash
source simulation/ros2_ws/install/setup.bash
mkdir -p .tools/baselines .tools/ros_logs .tools/gazebo_logs
export ROS_LOG_DIR="$project_dir/.tools/ros_logs" GAZEBO_LOG_PATH="$project_dir/.tools/gazebo_logs"
for controller in dwb mppi vanilla_dcbf proposed; do
  log="$project_dir/.tools/baselines/${scenario}_${controller}.log"
  timeout --signal=INT --kill-after=10s "${duration}s" ros2 launch r680_sim_bringup nav2_sim.launch.py \
    scenario:="$scenario" controller:="$controller" seed:=42 difficulty:=nominal >"$log" 2>&1 || status=$?
  status="${status:-0}"
  if [[ "$status" -ne 0 && "$status" -ne 124 ]]; then echo "$controller failed: $status" >&2; exit "$status"; fi
  unset status
done
