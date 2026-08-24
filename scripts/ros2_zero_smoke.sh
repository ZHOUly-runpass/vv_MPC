#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="${1:?usage: ros2_zero_smoke.sh ABSOLUTE_PROJECT_05_DIRECTORY}"
case "$PROJECT_DIR" in
  /*/05|*/E2Eproject_MPC_05_dev) ;;
  *) echo "Refusing unexpected project directory: $PROJECT_DIR" >&2; exit 2 ;;
esac
source /opt/ros/humble/setup.bash
source "$PROJECT_DIR/integration/ros2_ws/install/setup.bash"
mkdir -p "$PROJECT_DIR/reports"
timeout 12s ros2 run r680_safety_planner_ros planner_node --ros-args \
  -p config:="$PROJECT_DIR/configs/robot/r680_c16.yaml" \
  >"$PROJECT_DIR/reports/runtime_node.log" 2>&1 &
NODE_PID=$!
trap 'kill "$NODE_PID" 2>/dev/null || true' EXIT
sleep 2
echo "--- /cmd_vel one sample ---"
timeout 5s ros2 topic echo /cmd_vel --once
echo "--- /r680_safety/status one sample ---"
timeout 5s ros2 topic echo /r680_safety/status --once
wait "$NODE_PID" || true
echo "--- node log ---"
tail -20 "$PROJECT_DIR/reports/runtime_node.log"
