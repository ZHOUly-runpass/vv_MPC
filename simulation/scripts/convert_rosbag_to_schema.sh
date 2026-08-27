#!/usr/bin/env bash
set -eo pipefail
project_dir="${1:?usage: convert_rosbag_to_schema.sh PROJECT_DIR BAG OUTPUT_DIR [SAMPLE_HZ] [FEATURE_GRID]}"
bag="${2:?bag path is required}"; output="${3:?output directory is required}"; sample_hz="${4:-2}"; feature_grid="${5:-32}"
case "$(realpath "$project_dir")" in /*/05|*/E2Eproject_MPC_05_dev) ;; *) exit 2 ;; esac
cd "$project_dir"
checkpoint="$project_dir/artifacts/checkpoints/unilion_lidar_backbone_init.safetensors"
source /opt/ros/humble/setup.bash
PYTHONPATH="$project_dir/src" python3 scripts/extract_rosbag_training_frames.py --bag "$bag" \
  --output-dir "$output/extracted" --checkpoint "$checkpoint" --sample-hz "$sample_hz"
PYTHONPATH="$project_dir/src:$project_dir/third_party/UniLION" \
CUDA_HOME="$project_dir/.tools/envs/unilion" PATH="$project_dir/.tools/envs/unilion/bin:$PATH" \
LD_LIBRARY_PATH="$project_dir/.tools/envs/unilion/lib:${LD_LIBRARY_PATH:-}" \
"$project_dir/.tools/envs/unilion/bin/python" scripts/generate_unilion_feature_batch.py \
  --raw-manifest "$output/extracted/raw_manifest.jsonl" --output-dir "$output/unilion" --feature-grid "$feature_grid"
PYTHONPATH="$project_dir/src" python3 scripts/assemble_training_samples.py \
  --raw-manifest "$output/extracted/raw_manifest.jsonl" \
  --feature-manifest "$output/unilion/feature_manifest.jsonl" --output-dir "$output/schema_v1"
echo "$output/schema_v1/raw_manifest.jsonl"
