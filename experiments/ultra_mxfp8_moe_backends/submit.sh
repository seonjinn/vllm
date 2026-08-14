#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
QOS=${QOS:-user-restrictions}
WALLTIME=${WALLTIME:-04:00:00}
SEGMENT=${SEGMENT:-1}
BACKENDS=${BACKENDS:-"auto flashinfer_trtllm deep_gemm humming marlin triton"}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64.sqsh}
MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ultra_mxfp8_moe_backends_${STAMP}}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}

mkdir -p "${RESULT_ROOT}/slurm"

for backend in ${BACKENDS}; do
  job_name="sna-v0271-ultra-mxfp8-${backend}"
  cmd=(
    sbatch
    --parsable
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --qos="${QOS}"
    --time="${WALLTIME}"
    --segment="${SEGMENT}"
    --job-name="${job_name}"
    --output="${RESULT_ROOT}/slurm/${backend}_%j.out"
    --export="ALL,CONTAINER_IMAGE=${CONTAINER_IMAGE},MODEL_PATH=${MODEL_PATH},MOE_BACKEND=${backend},RESULT_ROOT=${RESULT_ROOT}"
  )
  if [[ "${SBATCH_TEST_ONLY}" == "1" ]]; then
    cmd+=(--test-only)
  fi
  cmd+=("${REPO_ROOT}/experiments/ultra_mxfp8_moe_backends/run_backend.sbatch")

  printf "%s: " "${backend}"
  "${cmd[@]}"
done

echo "result_root=${RESULT_ROOT}"
