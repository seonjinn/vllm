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
: "${LAYOUT:?Set LAYOUT to 8x4 or adaptive}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"

TP=${TP:-4}
WORKLOADS=${WORKLOADS:-"1000:1000 10000:1000 1000:10000"}
BATCH_SIZES=${BATCH_SIZES:-"1 8 32"}
PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-10}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12024}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
SWITCH_M=${SWITCH_M:-256}
MXFP8_WORKSPACE_SIZE=${MXFP8_WORKSPACE_SIZE:-0}
MXFP8_ENABLE_PDL=${MXFP8_ENABLE_PDL:-true}
LINEAR_BACKEND=${LINEAR_BACKEND:-flashinfer_trtllm}
PORT=${PORT:-$((18000 + SLURM_JOB_ID % 20000))}
RUN_DIR="${RESULT_ROOT}/${LAYOUT}/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

PATCH_MOUNTS=(
  "${SOURCE_ROOT}/vllm/envs.py:/usr/local/lib/python3.12/dist-packages/vllm/envs.py"
  "${SOURCE_ROOT}/vllm/model_executor/kernels/linear/__init__.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/__init__.py"
  "${SOURCE_ROOT}/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
  "${SOURCE_ROOT}/vllm/model_executor/warmup/kernel_warmup.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/kernel_warmup.py"
  "${SOURCE_ROOT}/vllm/utils/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/flashinfer.py"
)
CONTAINER_MOUNTS="/lustre:/lustre,$(IFS=,; echo "${PATCH_MOUNTS[*]}")"

export RUN_DIR TP WORKLOADS BATCH_SIZES PROMPT_MULTIPLIER
export MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS
export GPU_MEMORY_UTILIZATION PORT SOURCE_ROOT SOURCE_COMMIT LAYOUT SWITCH_M
export LINEAR_BACKEND
export MXFP8_WORKSPACE_SIZE
export MXFP8_ENABLE_PDL

# shellcheck disable=SC2016
srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --mpi=pmix \
  bash -lc '
set -euo pipefail

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_MXFP8_TRTLLM_LAYOUT="${LAYOUT}"
export VLLM_MXFP8_TRTLLM_SWITCH_M="${SWITCH_M}"
export VLLM_FLASHINFER_MXFP8_WORKSPACE_SIZE="${MXFP8_WORKSPACE_SIZE}"
export VLLM_FLASHINFER_MXFP8_ENABLE_PDL="${MXFP8_ENABLE_PDL}"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-${RUN_DIR}/flashinfer-autotune}"
export FLASHINFER_WORKSPACE_BASE=/tmp
export PYTHONPYCACHEPREFIX=/tmp/vllm-pycache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SERVER_LOG="${RUN_DIR}/server.log"
SERVER_CMD=(
  vllm serve "${MODEL_PATH}"
  --host 127.0.0.1
  --port "${PORT}"
  --served-model-name nemotron3-ultra-mxfp8
  --trust-remote-code
  --tensor-parallel-size "${TP}"
  --disable-custom-all-reduce
  --enable-expert-parallel
  --moe-backend flashinfer_trtllm
  --linear-backend "${LINEAR_BACKEND}"
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
  echo "source_root=${SOURCE_ROOT}"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "model=${MODEL_PATH}"
  echo "linear_backend=${LINEAR_BACKEND}"
  echo "moe_backend=flashinfer_trtllm"
  echo "layout=${LAYOUT}"
  echo "switch_m=${SWITCH_M}"
  echo "mxfp8_workspace_size=${MXFP8_WORKSPACE_SIZE}"
  echo "mxfp8_enable_pdl=${MXFP8_ENABLE_PDL}"
  echo "flashinfer_autotune_cache_dir=${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR}"
  echo "flashinfer_autotune_skip_ops=${VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS:-}"
  echo "tp=${TP}"
  echo "workloads=${WORKLOADS}"
  echo "batch_sizes=${BATCH_SIZES}"
  echo "container=${CONTAINER_IMAGE}"
  sha256sum \
    "${SOURCE_ROOT}/vllm/envs.py" \
    "${SOURCE_ROOT}/vllm/model_executor/kernels/linear/__init__.py" \
    "${SOURCE_ROOT}/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py" \
    "${SOURCE_ROOT}/vllm/model_executor/warmup/kernel_warmup.py" \
    "${SOURCE_ROOT}/vllm/utils/flashinfer.py"
  printf "server_command="
  printf "%q " "${SERVER_CMD[@]}"
  printf "\n"
  vllm --version
  python3 -c "from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferTrtllmMxfp8LinearKernel, _mxfp8_trtllm_layout_config; print(f\"kernel={FlashInferTrtllmMxfp8LinearKernel.__name__}\"); print(f\"layout_config={_mxfp8_trtllm_layout_config()}\")"
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
    vllm bench serve \
      --backend vllm \
      --base-url "http://127.0.0.1:${PORT}" \
      --endpoint /v1/completions \
      --model nemotron3-ultra-mxfp8 \
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
        model=nemotron3_ultra_mxfp8 \
        linear_backend="${LINEAR_BACKEND}" \
        moe_backend=flashinfer_trtllm \
        trtllm_layout="${LAYOUT}" \
        trtllm_switch_m="${SWITCH_M}" \
        tp="${TP}" \
        concurrency="${bs}" \
      2>&1 | tee "${RUN_DIR}/bench_isl${isl}_osl${osl}_c${bs}.log"
  done
  workload_index=$((workload_index + 1))
done

touch "${RUN_DIR}/COMPLETE"
'
