#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
QOS=${QOS:-user-restrictions}
SEGMENT=${SEGMENT:-1}
PROFILE=${PROFILE:-smoke}
SWEEPS=${SWEEPS:-"dense moe"}
DENSE_BACKENDS=${DENSE_BACKENDS:-"auto flashinfer_cutlass flashinfer_trtllm marlin humming emulation"}
MOE_BACKENDS=${MOE_BACKENDS:-"auto flashinfer_trtllm triton deep_gemm marlin humming"}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64.sqsh}
MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
SOURCE_ROOT=${SOURCE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-trtllm-adaptive-ultra}
SOURCE_COMMIT=${SOURCE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ultra_backend_sweep_${PROFILE}_${STAMP}}
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
    PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-10}
    ;;
  *)
    echo "ERROR: PROFILE must be smoke or full" >&2
    exit 2
    ;;
esac

mkdir -p "${RESULT_ROOT}/slurm"

submit_one() {
  local sweep_kind=$1
  local backend=$2
  local linear_backend
  local moe_backend

  case "${sweep_kind}" in
    dense)
      linear_backend=${backend}
      moe_backend=flashinfer_trtllm
      ;;
    moe)
      linear_backend=auto
      moe_backend=${backend}
      ;;
    *)
      echo "ERROR: unsupported sweep kind ${sweep_kind}" >&2
      exit 2
      ;;
  esac

  local job_name="${ACCOUNT}-mx.${sweep_kind:0:1}.${backend:0:12}"
  local cmd=(
    sbatch
    --parsable
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --qos="${QOS}"
    --time="${WALLTIME}"
    --segment="${SEGMENT}"
    --job-name="${job_name}"
    --output="${RESULT_ROOT}/slurm/${sweep_kind}_${backend}_%j.out"
    --export="ALL,CONTAINER_IMAGE=${CONTAINER_IMAGE},MODEL_PATH=${MODEL_PATH},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_COMMIT=${SOURCE_COMMIT},SWEEP_KIND=${sweep_kind},BACKEND=${backend},LINEAR_BACKEND=${linear_backend},MOE_BACKEND=${moe_backend},RESULT_ROOT=${RESULT_ROOT},WORKLOADS=${WORKLOADS},BATCH_SIZES=${BATCH_SIZES},PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER}"
  )
  if [[ "${SBATCH_TEST_ONLY}" == "1" ]]; then
    cmd+=(--test-only)
  fi
  cmd+=("${REPO_ROOT}/experiments/ultra_backend_sweep_v0271/run_mxfp8.sbatch")

  printf "%s/%s: " "${sweep_kind}" "${backend}"
  "${cmd[@]}"
}

for sweep_kind in ${SWEEPS}; do
  case "${sweep_kind}" in
    dense) backends=${DENSE_BACKENDS} ;;
    moe) backends=${MOE_BACKENDS} ;;
    *) echo "ERROR: unsupported sweep kind ${sweep_kind}" >&2; exit 2 ;;
  esac
  for backend in ${backends}; do
    submit_one "${sweep_kind}" "${backend}"
  done
done

echo "result_root=${RESULT_ROOT}"
