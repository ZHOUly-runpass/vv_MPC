#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="${1:?usage: build_simulation.sh ABSOLUTE_PROJECT_05_DIRECTORY}"
case "$PROJECT_DIR" in
  /*/05|*/E2Eproject_MPC_05_dev) ;;
  *) echo "Refusing unexpected project directory: $PROJECT_DIR" >&2; exit 2 ;;
esac

source /opt/ros/humble/setup.bash
WORKSPACE="$PROJECT_DIR/simulation/ros2_ws"
test -d "$WORKSPACE/src/r680_sim_description"
test -d "$WORKSPACE/src/r680_sim_worlds"
test -d "$WORKSPACE/src/r680_sim_bringup"
cd "$WORKSPACE"
colcon build --symlink-install --event-handlers console_direct+
