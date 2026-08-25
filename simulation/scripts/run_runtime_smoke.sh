#!/usr/bin/env bash
set -eo pipefail

project_dir="${1:?usage: run_runtime_smoke.sh PROJECT_DIR [SCENARIO]}"
scenario="${2:-empty}"
case "$(realpath "$project_dir")" in
  /*/05|*/E2Eproject_MPC_05_dev) ;;
  *) echo "refusing project outside an explicit 05 directory: $project_dir" >&2; exit 2 ;;
esac

cd "$project_dir"
source /opt/ros/humble/setup.bash
source simulation/ros2_ws/install/setup.bash
set -u
mkdir -p .tools/ros_logs .tools/gazebo_logs reports
export ROS_LOG_DIR="$project_dir/.tools/ros_logs"
export GAZEBO_LOG_PATH="$project_dir/.tools/gazebo_logs"
output="$project_dir/reports/simulation_${scenario}_runtime.json"
minimum_movement="0.0"
case "$scenario" in
  crossing_pedestrian|head_on|multi_dynamic) minimum_movement="0.05" ;;
esac

setsid timeout --signal=INT --kill-after=10s 55s ros2 launch r680_sim_bringup simulation.launch.py scenario:="$scenario" headless:=true >"$project_dir/.tools/simulation_${scenario}.log" 2>&1 &
launch_pid=$!
cleanup() {
  if kill -0 "$launch_pid" 2>/dev/null; then kill -INT -- "-$launch_pid" 2>/dev/null || true; fi
  wait "$launch_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 40); do
  if ros2 topic list 2>/dev/null | grep -qx /odom; then break; fi
  sleep 1
done
ros2 run r680_sim_bringup runtime_audit --ros-args -p output:="$output" -p timeout_s:=25.0 -p minimum_ground_truth_movement_m:="$minimum_movement"
echo "$output"
