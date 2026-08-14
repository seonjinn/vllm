#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
QOS=${QOS:-user-restrictions}
WALLTIME=${WALLTIME:-04:00:00}
SEGMENT=${SEGMENT:-1}
LAYOUTS=${LAYOUTS:-"8x4 adaptive"}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64.sqsh}
MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
SOURCE_ROOT=${SOURCE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-trtllm-adaptive-ultra}
SOURCE_COMMIT=${SOURCE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ultra_mxfp8_trtllm_adaptive_${STAMP}}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}

mkdir -p "${RESULT_ROOT}/slurm"

for layout in ${LAYOUTS}; do
  job_name="${ACCOUNT}-mxfp8.trt-${layout}"
  cmd=(
    sbatch
    --parsable
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --qos="${QOS}"
    --time="${WALLTIME}"
    --segment="${SEGMENT}"
    --job-name="${job_name}"
    --output="${RESULT_ROOT}/slurm/${layout}_%j.out"
    --export="ALL,CONTAINER_IMAGE=${CONTAINER_IMAGE},MODEL_PATH=${MODEL_PATH},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_COMMIT=${SOURCE_COMMIT},LAYOUT=${layout},RESULT_ROOT=${RESULT_ROOT}"
  )
  if [[ "${SBATCH_TEST_ONLY}" == "1" ]]; then
    cmd+=(--test-only)
  fi
  cmd+=("${REPO_ROOT}/experiments/ultra_mxfp8_trtllm_adaptive/run_layout.sbatch")

  printf "%s: " "${layout}"
  "${cmd[@]}"
done

echo "result_root=${RESULT_ROOT}"
