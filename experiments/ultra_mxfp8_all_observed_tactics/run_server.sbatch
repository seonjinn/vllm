#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"
: "${EXPECTED_CONTAINER_SHA256:?Set EXPECTED_CONTAINER_SHA256}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
: "${FLASHINFER_ROOT:?Set FLASHINFER_ROOT}"
: "${FLASHINFER_COMMIT:?Set FLASHINFER_COMMIT}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"
: "${RUN_KIND:?Set RUN_KIND}"
: "${BACKEND_NAME:?Set BACKEND_NAME}"
: "${LINEAR_BACKEND:?Set LINEAR_BACKEND}"
: "${ORACLE_BACKEND:?Set ORACLE_BACKEND}"
: "${SCALE_LAYOUT:?Set SCALE_LAYOUT}"
: "${TRTLLM_LAYOUT:?Set TRTLLM_LAYOUT}"

export TP=${TP:-4}
export EXPECTED_VLLM_VERSION=${EXPECTED_VLLM_VERSION:-0.27.1}
export MOE_BACKEND=${MOE_BACKEND:-flashinfer_trtllm}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-12024}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
export BENCH_TIMEOUT_S=${BENCH_TIMEOUT_S:-7200}
export PORT=${PORT:-$((18000 + SLURM_JOB_ID % 20000))}
export EXP_DIR="${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics"
export RUN_DIR="${RESULT_ROOT}/serving/${RUN_KIND}/${SLURM_JOB_ID}"
export SCRATCH_ROOT="/raid/scratch/${USER}/mxfp8_all_observed/${SLURM_JOB_ID}"

for checkout in "${SOURCE_ROOT}:${SOURCE_COMMIT}:vLLM" \
  "${FLASHINFER_ROOT}:${FLASHINFER_COMMIT}:FlashInfer"; do
  IFS=: read -r root expected name <<<"${checkout}"
  actual=$(git -C "${root}" rev-parse HEAD)
  if [[ "${actual}" != "${expected}" ]] || \
    [[ -n $(git -C "${root}" status --porcelain) ]]; then
    echo "${name} checkout is not clean at ${expected}: ${root}" >&2
    exit 1
  fi
done
container_sha=$(awk -F= '$1 == "sha256" {print $2}' \
  "${CONTAINER_IMAGE}.metadata.txt")
if [[ "${container_sha}" != "${EXPECTED_CONTAINER_SHA256}" ]]; then
  echo "Expected container ${EXPECTED_CONTAINER_SHA256}, got ${container_sha}" >&2
  exit 1
fi

case "${RUN_KIND}" in
  capture-eager)
    export WORKLOADS=${WORKLOADS:-"1000:64 10000:64"}
    export CONCURRENCIES=${CONCURRENCIES:-"1 2 4 8 16 32"}
    export PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-10}
    export TRACE_PHASE=eager ENFORCE_EAGER=1 USE_LOOKUP=0
    ;;
  capture-graph)
    export WORKLOADS=${WORKLOADS:-"1000:10000 10000:1000 1000:1000"}
    export CONCURRENCIES=${CONCURRENCIES:-"1 2 4 8 16 32"}
    export PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-10}
    export TRACE_PHASE=graph ENFORCE_EAGER=0 USE_LOOKUP=0
    ;;
  baseline)
    export WORKLOADS=${WORKLOADS:-"1000:10000 10000:1000 1000:1000"}
    export CONCURRENCIES=${CONCURRENCIES:-"1 8 32"}
    export PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-10}
    export TRACE_PHASE=baseline ENFORCE_EAGER=0 USE_LOOKUP=0
    ;;
  lookup)
    : "${LOOKUP_PATH:?Set LOOKUP_PATH for lookup mode}"
    export WORKLOADS=${WORKLOADS:-"1000:10000 10000:1000 1000:1000"}
    export CONCURRENCIES=${CONCURRENCIES:-"1 8 32"}
    export PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-10}
    export TRACE_PHASE=lookup ENFORCE_EAGER=0 USE_LOOKUP=1 LOOKUP_PATH
    ;;
  *)
    echo "Unsupported RUN_KIND=${RUN_KIND}" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_DIR}" "${SCRATCH_ROOT}/traces"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/home/sna:/home/sna,/lustre:/lustre,/raid/scratch:/raid/scratch \
  --container-workdir=/tmp \
  --mpi=pmix \
  bash "${EXP_DIR}/server_entrypoint.sh"
