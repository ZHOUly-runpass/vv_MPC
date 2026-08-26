#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="${1:?usage: setup_planner_dev.sh ABSOLUTE_PROJECT_05_DIRECTORY}"
case "$PROJECT_DIR" in
  /*/05|*/E2Eproject_MPC_05_dev) ;;
  *) echo "Refusing unexpected project directory: $PROJECT_DIR" >&2; exit 2 ;;
esac

test -f "$PROJECT_DIR/pyproject.toml"
test -f "$PROJECT_DIR/environments/planner_constraints.txt"
test -f "$PROJECT_DIR/scripts/audit_planner_environment.py"

ENV_DIR="$PROJECT_DIR/.tools/envs/planner"
if test ! -x "$ENV_DIR/bin/python"; then
  python3 -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install --upgrade \
  --constraint "$PROJECT_DIR/environments/planner_constraints.txt" \
  --editable "$PROJECT_DIR[solver,test]"

PYTHONDONTWRITEBYTECODE=1 "$ENV_DIR/bin/python" \
  "$PROJECT_DIR/scripts/audit_planner_environment.py" \
  --expected-prefix "$ENV_DIR"

echo "Planner environment ready: $ENV_DIR"
