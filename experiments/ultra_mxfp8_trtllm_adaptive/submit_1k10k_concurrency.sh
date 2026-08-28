#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
QOS=${QOS:-user-restrictions}
SEGMENT=${SEGMENT:-1}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64.sqsh}
MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
SOURCE_ROOT=${SOURCE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-trtllm-adaptive-ultra}
SOURCE_COMMIT=${SOURCE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}
AUTOTUNE_CACHE_DIR=${AUTOTUNE_CACHE_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ultra_mxfp8_adaptive_cg_pdl_off_ws1g_cached_20260827/seed-cache}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ultra_mxfp8_adaptive_1k10k_concurrency_${STAMP}}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}
SWEEP_GROUPS=${SWEEP_GROUPS:-"c1-32 c128 c512"}

mkdir -p "${RESULT_ROOT}/slurm"

submit_group() {
  local label=$1
  local batch_sizes=$2
  local prompt_multiplier=$3
  local max_num_seqs=$4
  local capture_sizes=$5
  local gpu_memory_utilization=$6
  local walltime=$7
  local job_name="${ACCOUNT}-mxfp8.adapt-${label}"
  local export_vars

  export_vars="ALL,CONTAINER_IMAGE=${CONTAINER_IMAGE},MODEL_PATH=${MODEL_PATH}"
  export_vars+=",SOURCE_ROOT=${SOURCE_ROOT},SOURCE_COMMIT=${SOURCE_COMMIT}"
  export_vars+=",LAYOUT=adaptive,RESULT_ROOT=${RESULT_ROOT},TP=4"
  export_vars+=",WORKLOADS=1000:10000,BATCH_SIZES=${batch_sizes}"
  export_vars+=",PROMPT_MULTIPLIER=${prompt_multiplier},MAX_NUM_SEQS=${max_num_seqs}"
  export_vars+=",CUDAGRAPH_CAPTURE_SIZES=${capture_sizes}"
  export_vars+=",GPU_MEMORY_UTILIZATION=${gpu_memory_utilization},MXFP8_WORKSPACE_SIZE=1073741824"
  export_vars+=",MXFP8_ENABLE_PDL=false,LINEAR_BACKEND=flashinfer_trtllm"
  export_vars+=",VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${AUTOTUNE_CACHE_DIR}"

  local cmd=(
    sbatch
    --parsable
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --qos="${QOS}"
    --time="${walltime}"
    --segment="${SEGMENT}"
    --job-name="${job_name}"
    --output="${RESULT_ROOT}/slurm/${label}_%j.out"
    --export="${export_vars}"
  )
  if [[ "${SBATCH_TEST_ONLY}" == "1" ]]; then
    cmd+=(--test-only)
  fi
  cmd+=("${REPO_ROOT}/experiments/ultra_mxfp8_trtllm_adaptive/run_layout.sbatch")

  printf "%s: " "${label}"
  "${cmd[@]}"
}

group_enabled() {
  [[ " ${SWEEP_GROUPS} " == *" $1 "* ]]
}

# Ten waves keep the low- and mid-concurrency measurements stable. The high
# concurrency jobs use one wave because exact 10K-token outputs would otherwise
# generate 12.8M (C128) and 51.2M (C512) output tokens per configuration.
group_enabled "c1-32" && \
  submit_group "c1-32" "1 2 4 8 16 32" 10 32 "1:8:32" 0.80 "04:00:00"
group_enabled "c128" && \
  submit_group "c128" "128" 1 32 "1:8:32" 0.80 "04:00:00"
group_enabled "c512" && \
  submit_group "c512" "512" 1 32 "1:8:32" 0.80 "04:00:00"

echo "result_root=${RESULT_ROOT}"
