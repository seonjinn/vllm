#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=04:00:00

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MOE_BACKEND:?Set MOE_BACKEND}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"

TP=${TP:-4}
ISL=${ISL:-1000}
OSL=${OSL:-1000}
BATCH_SIZES=${BATCH_SIZES:-"1 8 32"}
PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-10}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12024}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
PORT=${PORT:-$((18000 + SLURM_JOB_ID % 20000))}
RUN_DIR="${RESULT_ROOT}/${MOE_BACKEND}/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

export RUN_DIR TP ISL OSL BATCH_SIZES PROMPT_MULTIPLIER
export MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS
export GPU_MEMORY_UTILIZATION PORT

# The quoted script expands inside the container, where SLURM exports are present.
# shellcheck disable=SC2016
srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="/lustre:/lustre" \
  --mpi=pmix \
  bash -lc '
set -euo pipefail

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export FLASHINFER_WORKSPACE_BASE=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SERVER_LOG="${RUN_DIR}/server.log"
SERVER_CMD=(
  vllm serve "${MODEL_PATH}"
  --host 127.0.0.1
  --port "${PORT}"
  --served-model-name nemotron3-ultra-mxfp8
  --trust-remote-code
  --tensor-parallel-size "${TP}"
  --enable-expert-parallel
  --moe-backend "${MOE_BACKEND}"
  --linear-backend auto
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

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "model=${MODEL_PATH}"
  echo "moe_backend=${MOE_BACKEND}"
  echo "tp=${TP}"
  echo "isl=${ISL}"
  echo "osl=${OSL}"
  echo "batch_sizes=${BATCH_SIZES}"
  echo "container=${CONTAINER_IMAGE}"
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

for bs in ${BATCH_SIZES}; do
  num_prompts=$((bs * PROMPT_MULTIPLIER))
  result_file="result_isl${ISL}_osl${OSL}_c${bs}.json"
  vllm bench serve \
    --backend vllm \
    --base-url "http://127.0.0.1:${PORT}" \
    --endpoint /v1/completions \
    --model nemotron3-ultra-mxfp8 \
    --tokenizer "${MODEL_PATH}" \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len "${ISL}" \
    --random-output-len "${OSL}" \
    --random-range-ratio 0 \
    --num-prompts "${num_prompts}" \
    --num-warmups "${bs}" \
    --request-rate inf \
    --max-concurrency "${bs}" \
    --ignore-eos \
    --seed 0 \
    --save-result \
    --save-detailed \
    --result-dir "${RUN_DIR}" \
    --result-filename "${result_file}" \
    --metadata \
      vllm_version=0.27.1 \
      gpu=GB200 \
      model=nemotron3_ultra_mxfp8 \
      moe_backend="${MOE_BACKEND}" \
      tp="${TP}" \
      concurrency="${bs}" \
    2>&1 | tee "${RUN_DIR}/bench_c${bs}.log"
done

touch "${RUN_DIR}/COMPLETE"
'
