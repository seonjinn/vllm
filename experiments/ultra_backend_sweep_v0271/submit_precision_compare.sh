#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
QOS=${QOS:-user-restrictions}
SEGMENT=${SEGMENT:-1}
PROFILE=${PROFILE:-smoke}
PRECISIONS=${PRECISIONS:-"bf16 mxfp8"}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64.sqsh}
BF16_MODEL_PATH=${BF16_MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/nemo_rl_ci/nemotron_ultra/checkpoints/ultra-v3-sft-hsg-mainfeb19merge-mxfp8_fixed-hf_converted}
MXFP8_MODEL_PATH=${MXFP8_MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
SOURCE_ROOT=${SOURCE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-trtllm-adaptive-ultra}
SOURCE_COMMIT=${SOURCE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ultra_precision_compare_${PROFILE}_${STAMP}}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}

case "${PROFILE}" in
  smoke)
    WALLTIME=${WALLTIME:-02:00:00}
    WORKLOADS=${WORKLOADS:-"1000:1000"}
    BATCH_SIZES=${BATCH_SIZES:-"8"}
    PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-2}
    ;;
  full)
    WALLTIME=${WALLTIME:-04:00:00}
    WORKLOADS=${WORKLOADS:-"1000:1000 10000:1000 1000:10000"}
    BATCH_SIZES=${BATCH_SIZES:-"1 8 32"}
    PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-2}
    ;;
  *)
    echo "ERROR: PROFILE must be smoke or full" >&2
    exit 2
    ;;
esac

mkdir -p "${RESULT_ROOT}/slurm"

for precision in ${PRECISIONS}; do
  case "${precision}" in
    bf16)
      model_path=${BF16_MODEL_PATH}
      linear_backend=
      ;;
    mxfp8)
      model_path=${MXFP8_MODEL_PATH}
      linear_backend=auto
      ;;
    *)
      echo "ERROR: unsupported precision ${precision}" >&2
      exit 2
      ;;
  esac

  job_name="${ACCOUNT}-${USER:-sna}.ultra-${precision}-tp4"
  cmd=(
    sbatch
    --parsable
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --qos="${QOS}"
    --time="${WALLTIME}"
    --segment="${SEGMENT}"
    --job-name="${job_name}"
    --output="${RESULT_ROOT}/slurm/${precision}_%j.out"
    --export="ALL,CONTAINER_IMAGE=${CONTAINER_IMAGE},MODEL_PATH=${model_path},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_COMMIT=${SOURCE_COMMIT},SWEEP_KIND=precision,BACKEND=${precision},PRECISION=${precision},SERVED_MODEL_NAME=nemotron3-ultra-${precision},LINEAR_BACKEND=${linear_backend},MOE_BACKEND=auto,RESULT_ROOT=${RESULT_ROOT},TP=4,WORKLOADS=${WORKLOADS},BATCH_SIZES=${BATCH_SIZES},PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER},GPU_MEMORY_UTILIZATION=0.90"
  )
  if [[ "${SBATCH_TEST_ONLY}" == "1" ]]; then
    cmd+=(--test-only)
  fi
  cmd+=("${REPO_ROOT}/experiments/ultra_backend_sweep_v0271/run_mxfp8.sbatch")

  printf "%s: " "${precision}"
  "${cmd[@]}"
done

echo "result_root=${RESULT_ROOT}"
