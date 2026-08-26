#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"
: "${EXPECTED_CONTAINER_SHA256:?Set EXPECTED_CONTAINER_SHA256}"
: "${EXPECTED_CONTAINER_SIZE:?Set EXPECTED_CONTAINER_SIZE}"
: "${EXPECTED_CONTAINER_MTIME:?Set EXPECTED_CONTAINER_MTIME}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${EXPECTED_MODEL_CONFIG_SHA256:?Set EXPECTED_MODEL_CONFIG_SHA256}"
: "${EXPECTED_MODEL_INDEX_SHA256:?Set EXPECTED_MODEL_INDEX_SHA256}"
: "${EXPECTED_MODEL_WEIGHTS_MANIFEST_SHA256:?Set EXPECTED_MODEL_WEIGHTS_MANIFEST_SHA256}"
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
export RUN_INSTANCE=${RUN_INSTANCE:-${SLURM_JOB_ID}}
export EXP_DIR="${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics"
export RUN_DIR="${RESULT_ROOT}/serving/${RUN_KIND}/${RUN_INSTANCE}"
export SCRATCH_ROOT="/raid/scratch/${USER}/mxfp8_all_observed/${SLURM_JOB_ID}/${RUN_INSTANCE}"

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
read -r container_size container_mtime < <(stat -c '%s %Y' "${CONTAINER_IMAGE}")
if [[ "${container_size}" != "${EXPECTED_CONTAINER_SIZE}" ]] || \
  [[ "${container_mtime}" != "${EXPECTED_CONTAINER_MTIME}" ]]; then
  echo "Container changed after submission: size=${container_size}, mtime=${container_mtime}" >&2
  exit 1
fi
model_config_sha=$(sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')
model_index_sha=$(sha256sum \
  "${MODEL_PATH}/model.safetensors.index.json" | awk '{print $1}')
model_weights_manifest_sha=$(find "${MODEL_PATH}" -maxdepth 1 -type f \
  -name '*.safetensors' -printf '%f\t%s\t%T@\n' | sort | sha256sum | awk '{print $1}')
if [[ "${model_config_sha}" != "${EXPECTED_MODEL_CONFIG_SHA256}" ]] || \
  [[ "${model_index_sha}" != "${EXPECTED_MODEL_INDEX_SHA256}" ]] || \
  [[ "${model_weights_manifest_sha}" != \
    "${EXPECTED_MODEL_WEIGHTS_MANIFEST_SHA256}" ]]; then
  echo "Model manifest changed after submission" >&2
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

if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse run directory: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}" "${SCRATCH_ROOT}/traces"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/home/sna:/home/sna,/lustre:/lustre,/raid/scratch:/raid/scratch \
  --container-workdir=/tmp \
  --mpi=pmix \
  bash "${EXP_DIR}/server_entrypoint.sh"
