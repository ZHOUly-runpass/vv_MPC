#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="${1:?usage: dev_machine_build.sh ABSOLUTE_PROJECT_05_DIRECTORY}"
case "$PROJECT_DIR" in
  /*/05|*/E2Eproject_MPC_05_dev) ;;
  *) echo "Refusing unexpected project directory: $PROJECT_DIR" >&2; exit 2 ;;
esac
test -f "$PROJECT_DIR/pyproject.toml"
test -f "$PROJECT_DIR/configs/robot/r680_c16.yaml"
source /opt/ros/humble/setup.bash
set -u
python3 -m venv --system-site-packages "$PROJECT_DIR/.venv"
source "$PROJECT_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_DIR[test,solver]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q "$PROJECT_DIR/tests"
python "$PROJECT_DIR/scripts/validate_config.py" --config "$PROJECT_DIR/configs/robot/r680_c16.yaml"
python "$PROJECT_DIR/scripts/run_synthetic_smoke.py" --config "$PROJECT_DIR/configs/robot/r680_c16.yaml"
python "$PROJECT_DIR/scripts/run_casadi_smoke.py"
deactivate
cd "$PROJECT_DIR/integration/ros2_ws"
colcon build --symlink-install --packages-select r680_safety_planner_ros
set +u
source install/setup.bash
set -u
ros2 pkg executables r680_safety_planner_ros
