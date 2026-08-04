#!/bin/bash
#SBATCH --job-name=coreai_dlalgo_llm-mxfp8.refit-safe-test
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=36x2-a01r
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --segment=1
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/results/vllm-v0251-refit-safe-linear-backends/tests/%x-%j.out

set -euo pipefail

SOURCE_DIR=${SOURCE_DIR:?Set SOURCE_DIR to the vLLM worktree under test}
TEST_FILE=${TEST_FILE:-$SOURCE_DIR/tests/kernels/quantization/test_mxfp8_trtllm_linear.py}
TEST_DIR=$(dirname "$TEST_FILE")
CONTAINER_TEST_FILE=/workspace/test-artifacts/$(basename "$TEST_FILE")
export CONTAINER_TEST_FILE
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
RESULT_DIR=${RESULT_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/vllm-v0251-refit-safe-linear-backends/tests}

mkdir -p "$RESULT_DIR"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="/lustre:/lustre,$SOURCE_DIR:/workspace/vllm,$TEST_DIR:/workspace/test-artifacts" \
  --container-workdir=/workspace/vllm \
  --mpi=pmix \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace/vllm
    /usr/local/bin/python-VllmGenerationWorker -m pytest \
      "$CONTAINER_TEST_FILE" \
      -q -k refit
  '
