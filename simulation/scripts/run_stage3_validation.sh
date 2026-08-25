#!/usr/bin/env bash
set -eo pipefail

project_dir="${1:?usage: run_stage3_validation.sh PROJECT_DIR}"
case "$(realpath "$project_dir")" in
  /*/05|*/E2Eproject_MPC_05_dev) ;;
  *) echo "refusing project outside an explicit 05 directory: $project_dir" >&2; exit 2 ;;
esac
cd "$project_dir"
source /opt/ros/humble/setup.bash
source simulation/ros2_ws/install/setup.bash
mkdir -p .tools/captures .tools/feature_bridge .tools/ros_logs reports
export ROS_LOG_DIR="$project_dir/.tools/ros_logs"
capture="$project_dir/.tools/captures/simulation_c16_100_frames.npz"
capture_report="$project_dir/reports/simulation_c16_100_frames.json"

setsid ros2 launch r680_sim_bringup simulation.launch.py scenario:=static_sparse headless:=true >.tools/stage3_simulation.log 2>&1 &
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
ros2 run r680_sim_bringup pointcloud_100_frame_audit --ros-args \
  -p capture_output:="$capture" -p report_output:="$capture_report" -p frames:=100
cleanup
trap - EXIT

env_dir="$project_dir/.tools/envs/unilion"
export CUDA_HOME="$env_dir"
export PATH="$env_dir/bin:$PATH"
export LD_LIBRARY_PATH="$env_dir/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$project_dir/src:$project_dir/third_party/UniLION"
"$env_dir/bin/python" scripts/run_unilion_sim_100_frames.py \
  --project "$project_dir" --capture "$capture" \
  --output-dir "$project_dir/.tools/feature_bridge" \
  --report "$project_dir/reports/unilion_stage3_sim100.json"
