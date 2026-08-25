#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="${1:?usage: setup_unilion_dev.sh ABSOLUTE_PROJECT_05_DIRECTORY [--compile]}"
MODE="${2:-environment-only}"
case "$PROJECT_DIR" in
  /*/05|*/E2Eproject_MPC_05_dev) ;;
  *) echo "Refusing unexpected project directory: $PROJECT_DIR" >&2; exit 2 ;;
esac
test -f "$PROJECT_DIR/environments/unilion_environment.yml"
test -f "$PROJECT_DIR/third_party/UniLION/README.md"

TOOLS_DIR="$PROJECT_DIR/.tools"
CONDA_DIR="$TOOLS_DIR/miniforge3"
ENV_DIR="$TOOLS_DIR/envs/unilion"
mkdir -p "$TOOLS_DIR"
if test ! -x "$CONDA_DIR/bin/conda"; then
  INSTALLER="$TOOLS_DIR/Miniforge3-Linux-x86_64.sh"
  curl -fL --retry 5 --retry-delay 2 \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o "$INSTALLER"
  bash "$INSTALLER" -b -p "$CONDA_DIR"
fi
if test ! -x "$ENV_DIR/bin/python"; then
  "$CONDA_DIR/bin/conda" env create -p "$ENV_DIR" \
    -f "$PROJECT_DIR/environments/unilion_environment.yml"
fi

"$ENV_DIR/bin/python" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
"$ENV_DIR/bin/nvcc" --version | tail -5

if test "$MODE" != "--compile"; then
  echo "Environment ready. Re-run with --compile to build upstream CUDA extensions."
  exit 0
fi

UNI="$PROJECT_DIR/third_party/UniLION"
export CUDA_HOME="$ENV_DIR"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-8}"

# Use the PyG wheel compiled for the exact Torch/CUDA ABI rather than compiling
# torch-scatter against an arbitrary system toolkit.
"$ENV_DIR/bin/python" -m pip install torch-scatter==2.1.2 \
  --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html

# UniLION targets Python 3.9. causal-conv1d 1.7.0 imports Python 3.10-only
# union annotations, while the old 1.1.1 release is not compatible with this
# Torch/CUDA combination. Build the verified upstream v1.5.0.post8 revision.
SRC_DIR="$TOOLS_DIR/src"
CAUSAL_DIR="$SRC_DIR/causal-conv1d"
CAUSAL_COMMIT="82867a9d2e6907cc0f637ac6aff318f696838548"
mkdir -p "$SRC_DIR"
if test ! -d "$CAUSAL_DIR/.git"; then
  git clone --recursive --branch v1.5.0.post8 \
    https://github.com/Dao-AILab/causal-conv1d.git "$CAUSAL_DIR"
fi
test "$(git -C "$CAUSAL_DIR" rev-parse HEAD)" = "$CAUSAL_COMMIT"
git -C "$CAUSAL_DIR" submodule update --init --recursive
CAUSAL_CONV1D_FORCE_BUILD=TRUE "$ENV_DIR/bin/python" -m pip install \
  --no-build-isolation --no-deps "$CAUSAL_DIR"

cd "$UNI/mmcv"
MMCV_WITH_OPS=1 "$ENV_DIR/bin/python" setup.py install
cd "$UNI/mmdetection3d"
"$ENV_DIR/bin/python" setup.py install
cd "$UNI/projects/mmdet3d_plugin/models/ops/mamba"
"$ENV_DIR/bin/python" -m pip install --no-build-isolation .
cd "$UNI/projects"
"$ENV_DIR/bin/python" -m pip install --use-pep517 --no-build-isolation -e .

cd "$UNI"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$UNI" "$ENV_DIR/bin/python" -c \
  'import mmcv, mmdet3d, projects.mmdet3d_plugin; print("UniLION imports ready")'
