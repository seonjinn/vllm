#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=04:00:00

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
: "${SWEEP_KIND:?Set SWEEP_KIND to dense or moe}"
: "${BACKEND:?Set BACKEND}"
: "${MOE_BACKEND:?Set MOE_BACKEND}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"

PRECISION=${PRECISION:-mxfp8}
LINEAR_BACKEND=${LINEAR_BACKEND:-}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-nemotron3-ultra-mxfp8}
TP=${TP:-4}
WORKLOADS=${WORKLOADS:-"1000:1000"}
BATCH_SIZES=${BATCH_SIZES:-"8"}
PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12024}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
BENCH_TIMEOUT_S=${BENCH_TIMEOUT_S:-7200}
PORT=${PORT:-$((18000 + SLURM_JOB_ID % 20000))}
RUN_DIR="${RESULT_ROOT}/${SWEEP_KIND}/${BACKEND}/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

PATCH_MOUNTS=(
  "${SOURCE_ROOT}/vllm/envs.py:/usr/local/lib/python3.12/dist-packages/vllm/envs.py"
  "${SOURCE_ROOT}/vllm/model_executor/kernels/linear/__init__.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/__init__.py"
  "${SOURCE_ROOT}/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
  "${SOURCE_ROOT}/vllm/utils/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/flashinfer.py"
)
CONTAINER_MOUNTS="/lustre:/lustre,$(IFS=,; echo "${PATCH_MOUNTS[*]}")"

export RUN_DIR TP WORKLOADS BATCH_SIZES PROMPT_MULTIPLIER
export MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS
export GPU_MEMORY_UTILIZATION BENCH_TIMEOUT_S PORT SOURCE_ROOT SOURCE_COMMIT
export SWEEP_KIND BACKEND LINEAR_BACKEND MOE_BACKEND
export PRECISION SERVED_MODEL_NAME

# shellcheck disable=SC2016
srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --mpi=pmix \
  bash -lc '
set -euo pipefail

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export FLASHINFER_WORKSPACE_BASE=/tmp
export PYTHONPYCACHEPREFIX=/tmp/vllm-pycache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SERVER_LOG="${RUN_DIR}/server.log"
SERVER_CMD=(
  vllm serve "${MODEL_PATH}"
  --host 127.0.0.1
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --tensor-parallel-size "${TP}"
  --disable-custom-all-reduce
  --enable-expert-parallel
  --moe-backend "${MOE_BACKEND}"
  --attention-backend FLASHINFER
  --dtype auto
  --load-format auto
  --kv-cache-dtype auto
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --enable-chunked-prefill
  --enable-prefix-caching
  --mamba-cache-mode all
  --mamba-ssm-cache-dtype float32
  --seed 0
  --compilation-config "{\"cudagraph_capture_sizes\":[1,8,32],\"pass_config\":{\"fuse_allreduce_rms\":false}}"
)
if [[ -n "${LINEAR_BACKEND}" ]]; then
  SERVER_CMD+=(--linear-backend "${LINEAR_BACKEND}")
fi

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "source_root=${SOURCE_ROOT}"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "model=${MODEL_PATH}"
  echo "precision=${PRECISION}"
  echo "sweep_kind=${SWEEP_KIND}"
  echo "backend=${BACKEND}"
  echo "linear_backend=${LINEAR_BACKEND}"
  echo "moe_backend=${MOE_BACKEND}"
  echo "tp=${TP}"
  echo "workloads=${WORKLOADS}"
  echo "batch_sizes=${BATCH_SIZES}"
  echo "container=${CONTAINER_IMAGE}"
  sha256sum \
    "${SOURCE_ROOT}/vllm/envs.py" \
    "${SOURCE_ROOT}/vllm/model_executor/kernels/linear/__init__.py" \
    "${SOURCE_ROOT}/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py" \
    "${SOURCE_ROOT}/vllm/utils/flashinfer.py"
  printf "server_command="
  printf "%q " "${SERVER_CMD[@]}"
  printf "\n"
  vllm --version
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} >"${RUN_DIR}/metadata.txt"

"${SERVER_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

deadline=$((SECONDS + 7200))
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server exited before becoming healthy" >&2
    tail -200 "${SERVER_LOG}" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Server health check timed out" >&2
    tail -200 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 10
done

workload_index=0
for workload in ${WORKLOADS}; do
  IFS=: read -r isl osl <<<"${workload}"
  for bs in ${BATCH_SIZES}; do
    num_prompts=$((bs * PROMPT_MULTIPLIER))
    seed=$((workload_index * 1000 + bs))
    result_file="result_isl${isl}_osl${osl}_c${bs}.json"
    timeout "${BENCH_TIMEOUT_S}" vllm bench serve \
      --backend vllm \
      --base-url "http://127.0.0.1:${PORT}" \
      --endpoint /v1/completions \
      --model "${SERVED_MODEL_NAME}" \
      --tokenizer "${MODEL_PATH}" \
      --trust-remote-code \
      --dataset-name random \
      --random-input-len "${isl}" \
      --random-output-len "${osl}" \
      --random-range-ratio 0 \
      --num-prompts "${num_prompts}" \
      --num-warmups "${bs}" \
      --request-rate inf \
      --max-concurrency "${bs}" \
      --ignore-eos \
      --seed "${seed}" \
      --save-result \
      --save-detailed \
      --result-dir "${RUN_DIR}" \
      --result-filename "${result_file}" \
      --metadata \
        vllm_version=0.27.1 \
        source_commit="${SOURCE_COMMIT}" \
        gpu=GB200 \
        model="nemotron3_ultra_${PRECISION}" \
        precision="${PRECISION}" \
        sweep_kind="${SWEEP_KIND}" \
        backend="${BACKEND}" \
        linear_backend="${LINEAR_BACKEND}" \
        moe_backend="${MOE_BACKEND}" \
        tp="${TP}" \
        concurrency="${bs}" \
      2>&1 | tee "${RUN_DIR}/bench_isl${isl}_osl${osl}_c${bs}.log"

    python3 - "${RUN_DIR}/${result_file}" "${num_prompts}" "${isl}" "${osl}" <<"PY"
import json
import sys

path, expected_requests, isl, osl = sys.argv[1], *map(int, sys.argv[2:])
with open(path) as result_file:
    result = json.load(result_file)
assert result["completed"] == expected_requests, result
assert result["failed"] == 0, result
assert result["total_input_tokens"] == expected_requests * isl, result
assert result["total_output_tokens"] == expected_requests * osl, result
PY
  done
  workload_index=$((workload_index + 1))
done

touch "${RUN_DIR}/COMPLETE"
'
