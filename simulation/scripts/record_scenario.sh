#!/usr/bin/env bash
set -eo pipefail

project_dir="${1:?usage: record_scenario.sh PROJECT_DIR SCENARIO [DURATION_SECONDS]}"
scenario="${2:?scenario is required}"
duration="${3:-60}"
case "$(realpath "$project_dir")" in
  /*/05|*/E2Eproject_MPC_05_dev) ;;
  *) echo "refusing project outside an explicit 05 directory: $project_dir" >&2; exit 2 ;;
esac

cd "$project_dir"
source /opt/ros/humble/setup.bash
source simulation/ros2_ws/install/setup.bash
set -u
mkdir -p .tools/ros_logs .tools/gazebo_logs .tools/rosbags
export ROS_LOG_DIR="$project_dir/.tools/ros_logs"
export GAZEBO_LOG_PATH="$project_dir/.tools/gazebo_logs"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
bag="$project_dir/.tools/rosbags/${scenario}_${stamp}"
setsid ros2 launch r680_sim_bringup simulation.launch.py scenario:="$scenario" headless:=true >"$project_dir/.tools/record_${scenario}.log" 2>&1 &
launch_pid=$!
cleanup() {
  if kill -0 "$launch_pid" 2>/dev/null; then kill -INT -- "-$launch_pid" 2>/dev/null || true; fi
  wait "$launch_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 40); do
  if ros2 topic list 2>/dev/null | grep -qx /points; then break; fi
  sleep 1
done
bag_status=0
timeout --signal=INT --kill-after=5s "${duration}s" ros2 bag record -o "$bag" \
  /clock /tf /tf_static /odom /imu/data_raw /scan /points /plan /cmd_vel \
  /simulation/ground_truth_obstacles /simulation/benchmark_status || bag_status=$?
if [[ "$bag_status" -ne 0 && "$bag_status" -ne 124 ]]; then
  exit "$bag_status"
fi
echo "$bag"
