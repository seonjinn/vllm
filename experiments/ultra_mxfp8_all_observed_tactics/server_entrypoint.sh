#!/usr/bin/env bash

set -euo pipefail

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${SCRATCH_ROOT}/autotune"
export FLASHINFER_WORKSPACE_BASE="${SCRATCH_ROOT}/flashinfer"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg"
export PYTHONPYCACHEPREFIX="${SCRATCH_ROOT}/pycache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

server_env=(
  env
  "PYTHONPATH=${EXP_DIR}:${PYTHONPATH:-}"
  "MXFP8_TACTIC_TRACE_DIR=${RUN_DIR}/traces"
  "MXFP8_TACTIC_TRACE_PHASE=${TRACE_PHASE}"
)
if [[ "${USE_LOOKUP}" == 1 ]]; then
  server_env+=("MXFP8_TACTIC_LOOKUP=${LOOKUP_PATH}")
fi

server_cmd=(
  vllm serve "${MODEL_PATH}"
  --host 127.0.0.1
  --port "${PORT}"
  --served-model-name nemotron3-ultra-mxfp8
  --trust-remote-code
  --tensor-parallel-size "${TP}"
  --disable-custom-all-reduce
  --enable-expert-parallel
  --linear-backend "${LINEAR_BACKEND}"
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
  --compilation-config
  '{"cudagraph_capture_sizes":[1,2,4,8,16,32],"pass_config":{"fuse_allreduce_rms":false}}'
)
if [[ "${ENFORCE_EAGER}" == 1 ]]; then
  server_cmd+=(--enforce-eager)
fi

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "run_kind=${RUN_KIND}"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "container=${CONTAINER_IMAGE}"
  echo "model=${MODEL_PATH}"
  echo "tp=${TP}"
  echo "linear_backend=${LINEAR_BACKEND}"
  echo "moe_backend=${MOE_BACKEND}"
  echo "workloads=${WORKLOADS}"
  echo "concurrencies=${CONCURRENCIES}"
  echo "prompt_multiplier=${PROMPT_MULTIPLIER}"
  printf "server_command="
  printf "%q " "${server_env[@]}" "${server_cmd[@]}"
  printf "\n"
  vllm --version
  uv run --no-project python -c \
    'import flashinfer; print("flashinfer=" + flashinfer.__version__)'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} >"${RUN_DIR}/metadata.txt"

"${server_env[@]}" "${server_cmd[@]}" >"${RUN_DIR}/server.log" 2>&1 &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  rm -rf "${SCRATCH_ROOT}"
}
trap cleanup EXIT

deadline=$((SECONDS + 7200))
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Server exited before becoming healthy" >&2
    tail -200 "${RUN_DIR}/server.log" >&2
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "Server health check timed out" >&2
    tail -200 "${RUN_DIR}/server.log" >&2
    exit 1
  fi
  sleep 10
done

workload_index=0
for workload in ${WORKLOADS}; do
  IFS=: read -r isl osl <<<"${workload}"
  for concurrency in ${CONCURRENCIES}; do
    num_prompts=$((concurrency * PROMPT_MULTIPLIER))
    seed=$((workload_index * 1000 + concurrency))
    result_file="result_isl${isl}_osl${osl}_c${concurrency}.json"
    timeout "${BENCH_TIMEOUT_S}" vllm bench serve \
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
      --num-warmups "${concurrency}" \
      --request-rate inf \
      --max-concurrency "${concurrency}" \
      --ignore-eos \
      --seed "${seed}" \
      --save-result \
      --save-detailed \
      --result-dir "${RUN_DIR}" \
      --result-filename "${result_file}" \
      --metadata run_kind="${RUN_KIND}" concurrency="${concurrency}" \
      2>&1 | tee "${RUN_DIR}/bench_isl${isl}_osl${osl}_c${concurrency}.log"

    uv run --no-project python - "${RUN_DIR}/${result_file}" \
      "${num_prompts}" "${isl}" "${osl}" <<'PY'
import json
import sys

path, expected_requests, isl, osl = sys.argv[1], *map(int, sys.argv[2:])
with open(path) as handle:
    result = json.load(handle)
assert result["completed"] == expected_requests, result
assert result["failed"] == 0, result
input_lens = result["input_lens"]
assert len(input_lens) == expected_requests, result
assert all(isl <= length <= isl + 1 for length in input_lens), result
assert result["total_input_tokens"] == sum(input_lens), result
assert result["total_output_tokens"] == expected_requests * osl, result
PY
  done
  workload_index=$((workload_index + 1))
done

touch "${RUN_DIR}/COMPLETE"
