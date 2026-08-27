#!/usr/bin/env bash
set -eo pipefail

project_dir="${1:?usage: record_scenario.sh PROJECT_DIR SCENARIO [DURATION_SECONDS] [SEED] [DIFFICULTY] [CONTROLLER]}"
scenario="${2:?scenario is required}"
duration="${3:-60}"
seed="${4:-42}"
difficulty="${5:-nominal}"
controller="${6:-dwb}"
case "$controller" in dwb|mppi|vanilla_dcbf|proposed) ;; *) echo "invalid controller: $controller" >&2; exit 2 ;; esac
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
bag="$project_dir/.tools/rosbags/${scenario}_${difficulty}_seed${seed}_${controller}_${stamp}"
setsid ros2 launch r680_sim_bringup nav2_sim.launch.py scenario:="$scenario" controller:="$controller" seed:="$seed" difficulty:="$difficulty" >"$project_dir/.tools/record_${scenario}_${difficulty}_${seed}_${controller}.log" 2>&1 &
launch_pid=$!
cleanup() {
  if kill -0 "$launch_pid" 2>/dev/null; then kill -INT -- "-$launch_pid" 2>/dev/null || true; fi
  wait "$launch_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  topics="$(ros2 topic list 2>/dev/null || true)"
  if grep -qx /points <<<"$topics" && grep -qx /planning/candidates <<<"$topics" && grep -qx /simulation/controller_status <<<"$topics"; then break; fi
  sleep 1
done
for required in /points /odom /plan /local_costmap/costmap_raw /simulation/ground_truth_obstacles /simulation/controller_status /planning/candidates /planning/obstacle_predictions /planning/mpc_request /planning/mpc_result; do
  if ! ros2 topic list | grep -qx "$required"; then echo "required topic missing: $required" >&2; exit 4; fi
done
bag_status=0
timeout --signal=INT --kill-after=5s "${duration}s" ros2 bag record -o "$bag" \
  /clock /tf /tf_static /odom /imu/data_raw /scan /points /plan /cmd_vel /local_costmap/costmap_raw \
  /simulation/ground_truth_obstacles /simulation/ground_truth_obstacle_poses \
  /simulation/benchmark_status /simulation/controller_status \
  /planning/candidates /planning/obstacle_predictions /planning/mpc_request /planning/mpc_result || bag_status=$?
if [[ "$bag_status" -ne 0 && "$bag_status" -ne 124 ]]; then
  exit "$bag_status"
fi
config_sha256="$(sha256sum simulation/ros2_ws/src/r680_sim_bringup/config/scenarios.yaml | cut -d' ' -f1)"
code_revision="$(git rev-parse HEAD)"
printf '{"schema_version":"1.0","scenario":"%s","difficulty":"%s","seed":%s,"controller":"%s","duration_s":%s,"bag":"%s","config_sha256":"%s","code_revision":"%s"}\n' \
  "$scenario" "$difficulty" "$seed" "$controller" "$duration" "$bag" "$config_sha256" "$code_revision" >"${bag}_run.json"
echo "$bag"
