# R680 C16 Safety Planner

Implementation of the workflow in [算法完整流程.md](算法完整流程.md). The local
Windows environment runs the dependency-light core and tests. ROS 2 integration
is isolated under `integration/ros2_ws` and is validated on the Linux
development machine.

Safety defaults:

- configuration starts in `perception_only`;
- non-zero commands are forbidden until every commissioning gate passes;
- stale LiDAR, odometry, IMU, TF, planner heartbeat, or commands force zero;
- geometric obstacles remain active independently of learned perception;
- unresolved vehicle type or footprint prevents motion.

Local setup:

```powershell
python -m pip install -e ".[test]"
python -m pytest
python scripts/validate_config.py --config configs/robot/r680_c16.yaml
python scripts/run_synthetic_smoke.py --config configs/robot/r680_c16.yaml
```

Development-machine planner/CasADi environment and the 0/8/16/32 near-obstacle
benchmark suite:

```bash
cd /home/zhou/E2Eproject_MPC/github_pull/05
bash scripts/setup_planner_dev.sh "$PWD"
.tools/envs/planner/bin/python scripts/benchmark_solver_suite.py \
  --obstacle-counts 0 8 16 32 --repeats 11 --placement near \
  --deadline-ms 80 --output reports/planner_casadi_obstacle_benchmark.json
```

ROS 2 development-machine setup and the fail-closed adapter are described in
`integration/README.md`.

UniLION stage-1 CUDA validation on the development machine:

```bash
cd /home/zhou/E2Eproject_MPC/github_pull/05
PYTHONDONTWRITEBYTECODE=1 \
CUDA_HOME="$PWD/.tools/envs/unilion" \
PATH="$PWD/.tools/envs/unilion/bin:$PATH" \
LD_LIBRARY_PATH="$PWD/.tools/envs/unilion/lib:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$PWD/src:$PWD/third_party/UniLION" \
.tools/envs/unilion/bin/python scripts/run_unilion_lidar_forward.py \
  --points 12000 --runs 3 --output reports/unilion_stage1_forward.json
```

The default input is deterministic synthetic C16-format data for plumbing and
performance validation, not accuracy evidence. Pass `--nuscenes-sample PATH`
after legally obtaining a five-float nuScenes LiDAR sample. See
`reports/UniLION阶段1执行报告.md` for the measured result and remaining gates.

Stage-4 data and training pipeline:

```bash
PYTHONPATH="$PWD/src" .tools/envs/planner/bin/python scripts/create_training_fixture.py --output-dir .tools/training_fixture
PYTHONPATH="$PWD/src" .tools/envs/planner/bin/python scripts/generate_teacher_labels.py \
  --manifest .tools/training_fixture/raw_manifest.jsonl --output-dir .tools/training_labeled
PYTHONPATH="$PWD/src" .tools/envs/unilion/bin/python scripts/train_planner.py \
  --manifest .tools/training_labeled/manifest.jsonl --output-dir .tools/training_run --epochs 2
PYTHONPATH="$PWD/src" .tools/envs/unilion/bin/python scripts/evaluate_planner.py \
  --manifest .tools/training_labeled/manifest.jsonl --checkpoint .tools/training_run/best.pt \
  --split test --output reports/training_fixture_evaluation.json
```

The fixture only verifies schema, teacher, training, checkpoint, and evaluation plumbing.
For actual collection, inspect the full 72-run matrix first with
`python scripts/batch_collect.py --dry-run`, then run it on the ROS development machine.
Four unified simulation controller names are `dwb`, `mppi`, `vanilla_dcbf`, and `proposed`.
