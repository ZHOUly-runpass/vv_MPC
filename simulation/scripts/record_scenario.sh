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
domain_key="${scenario}:${difficulty}:${seed}:${controller}"
domain_offset="$(printf '%s' "$domain_key" | cksum | awk '{print $1 % 100}')"
export ROS_DOMAIN_ID="${R680_COLLECTION_ROS_DOMAIN_ID:-$((60 + domain_offset))}"
export ROS_LOCALHOST_ONLY=1

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
bag="$project_dir/.tools/rosbags/${scenario}_${difficulty}_seed${seed}_${controller}_${stamp}"
setsid ros2 launch r680_sim_bringup nav2_sim.launch.py scenario:="$scenario" controller:="$controller" seed:="$seed" difficulty:="$difficulty" >"$project_dir/.tools/record_${scenario}_${difficulty}_${seed}_${controller}.log" 2>&1 &
launch_pid=$!
recorder_pid=""
cleanup() {
  if [[ -n "${recorder_pid:-}" ]] && kill -0 -- "-$recorder_pid" 2>/dev/null; then
    kill -INT -- "-$recorder_pid" 2>/dev/null || true
  fi
  if kill -0 -- "-$launch_pid" 2>/dev/null; then kill -INT -- "-$launch_pid" 2>/dev/null || true; fi
  for _ in $(seq 1 50); do
    if ! kill -0 -- "-$launch_pid" 2>/dev/null; then break; fi
    sleep 0.1
  done
  if kill -0 -- "-$launch_pid" 2>/dev/null; then kill -TERM -- "-$launch_pid" 2>/dev/null || true; fi
  for _ in $(seq 1 20); do
    if ! kill -0 -- "-$launch_pid" 2>/dev/null; then break; fi
    sleep 0.1
  done
  if kill -0 -- "-$launch_pid" 2>/dev/null; then kill -KILL -- "-$launch_pid" 2>/dev/null || true; fi
  wait "$launch_pid" 2>/dev/null || true
}
trap cleanup EXIT

required_topics=(
  /points /odom /simulation/reference_route /local_costmap/costmap_raw
  /simulation/difficulty_status /simulation/ground_truth_obstacles /simulation/controller_status
  /planning/candidates /planning/obstacle_predictions /planning/mpc_request /planning/mpc_result
)
for _ in $(seq 1 60); do
  topics="$(ros2 topic list 2>/dev/null || true)"
  topics_ready=1
  for required in "${required_topics[@]}"; do
    if ! grep -qx "$required" <<<"$topics"; then topics_ready=0; break; fi
  done
  if [[ "$topics_ready" -eq 1 ]]; then break; fi
  sleep 1
done
for required in "${required_topics[@]}"; do
  if ! ros2 topic list | grep -qx "$required"; then echo "required topic missing: $required" >&2; exit 4; fi
done
preflight="$project_dir/.tools/preflight_${scenario}_${difficulty}_${seed}_${controller}.json"
preflight_ok=0
for attempt in 1 2 3; do
  if timeout 30s ros2 run r680_sim_bringup collection_preflight --ros-args \
    -p duration_s:=5.0 -p report_output:="$preflight" -p disk_path:="$project_dir" -p minimum_free_gib:=5.0; then
    preflight_ok=1
    break
  fi
  echo "collection preflight attempt $attempt failed; waiting for fresh publishers" >&2
  sleep 2
done
if [[ "$preflight_ok" -ne 1 ]]; then
  echo "collection preflight failed: $preflight" >&2; exit 5
fi
bag_log="$project_dir/.tools/record_bag_${scenario}_${difficulty}_${seed}_${controller}.log"
setsid ros2 bag record -o "$bag" \
  /clock /tf /tf_static /odom /imu/data_raw /scan /points /plan /simulation/reference_route /cmd_vel /local_costmap/costmap_raw \
  /simulation/ground_truth_obstacles /simulation/ground_truth_obstacle_poses \
  /simulation/benchmark_status /simulation/controller_status /simulation/difficulty_status \
  /planning/candidates /planning/obstacle_predictions /planning/mpc_request /planning/mpc_result >"$bag_log" 2>&1 &
recorder_pid=$!
for _ in $(seq 1 300); do
  [[ -d "$bag" ]] && break
  if ! kill -0 "$recorder_pid" 2>/dev/null; then
    wait "$recorder_pid" || true
    echo "rosbag recorder exited before creating output: $bag_log" >&2
    exit 6
  fi
  sleep 0.1
done
if [[ ! -d "$bag" ]]; then
  echo "rosbag recorder did not initialize within 30 seconds: $bag_log" >&2
  exit 6
fi
sleep "$duration"
kill -INT -- "-$recorder_pid" 2>/dev/null || true
for _ in $(seq 1 300); do
  if ! kill -0 -- "-$recorder_pid" 2>/dev/null; then break; fi
  sleep 0.1
done
if kill -0 -- "-$recorder_pid" 2>/dev/null; then
  kill -TERM -- "-$recorder_pid" 2>/dev/null || true
fi
wait "$recorder_pid" || bag_status=$?
recorder_pid=""
if [[ ! -f "$bag/metadata.yaml" ]] || ! ros2 bag info "$bag" >/dev/null 2>&1; then
  echo "rosbag output is incomplete: $bag (status=${bag_status:-0}, log=$bag_log)" >&2
  exit 7
fi
config_sha256="$(sha256sum simulation/ros2_ws/src/r680_sim_bringup/config/scenarios.yaml | cut -d' ' -f1)"
code_revision="$(git rev-parse HEAD)"
printf '{"schema_version":"1.0","scenario":"%s","difficulty":"%s","seed":%s,"controller":"%s","duration_s":%s,"ros_domain_id":%s,"bag":"%s","config_sha256":"%s","code_revision":"%s"}\n' \
  "$scenario" "$difficulty" "$seed" "$controller" "$duration" "$ROS_DOMAIN_ID" "$bag" "$config_sha256" "$code_revision" >"${bag}_run.json"
echo "$bag"
