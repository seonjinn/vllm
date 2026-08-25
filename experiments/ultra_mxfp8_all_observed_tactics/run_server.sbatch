#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"
: "${RUN_KIND:?Set RUN_KIND}"

export TP=${TP:-4}
export LINEAR_BACKEND=${LINEAR_BACKEND:-auto}
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

case "${RUN_KIND}" in
  capture-eager)
    export WORKLOADS=${WORKLOADS:-"1000:10000 10000:1000 1000:1000"}
    export CONCURRENCIES=${CONCURRENCIES:-"1 2 4 8 16 32"}
    export PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-1}
    export TRACE_PHASE=eager ENFORCE_EAGER=1 USE_LOOKUP=0
    ;;
  capture-graph)
    export WORKLOADS=${WORKLOADS:-"1000:10000 10000:1000 1000:1000"}
    export CONCURRENCIES=${CONCURRENCIES:-"1 2 4 8 16 32"}
    export PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-1}
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

mkdir -p "${RUN_DIR}/traces" "${SCRATCH_ROOT}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/home/sna:/home/sna,/lustre:/lustre,/raid/scratch:/raid/scratch \
  --container-workdir=/tmp \
  --mpi=pmix \
  bash "${EXP_DIR}/server_entrypoint.sh"
