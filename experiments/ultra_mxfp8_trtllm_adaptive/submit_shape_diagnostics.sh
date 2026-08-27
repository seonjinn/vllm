#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200-backfill}
QOS=${QOS:-user-restrictions}
WALLTIME=${WALLTIME:-01:00:00}
SEGMENT=${SEGMENT:-1}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64.sqsh}
SOURCE_ROOT=${SOURCE_ROOT:-/home/sna/vllm-v0271-trtllm-adaptive-ultra}
SOURCE_COMMIT=${SOURCE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ultra_mxfp8_128x4_diagnostic_${STAMP}}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}

mkdir -p "${RESULT_ROOT}/slurm"

cmd=(
  sbatch
  --parsable
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --qos="${QOS}"
  --time="${WALLTIME}"
  --segment="${SEGMENT}"
  --job-name="${ACCOUNT}-mxfp8.trt-128x4-diag"
  --output="${RESULT_ROOT}/slurm/%j.out"
  --export="ALL,CONTAINER_IMAGE=${CONTAINER_IMAGE},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_COMMIT=${SOURCE_COMMIT},RESULT_ROOT=${RESULT_ROOT}"
)
if [[ "${SBATCH_TEST_ONLY}" == "1" ]]; then
  cmd+=(--test-only)
fi
cmd+=("${REPO_ROOT}/experiments/ultra_mxfp8_trtllm_adaptive/run_shape_diagnostics.sbatch")

"${cmd[@]}"
echo "result_root=${RESULT_ROOT}"
