#!/usr/bin/env bash

set -euo pipefail

readonly CUDAGRAPH_COMPLETION_MARKER="Graph capturing finished"

record_cudagraph_evidence() {
  local requested_run_kind=$1
  local eager_mode=$2
  local server_log=$3
  local metadata_path=$4
  local configured capture_status evidence

  case "${eager_mode}" in
    0)
      configured=true
      if [[ -f "${server_log}" ]] && \
        grep -Fq -- "${CUDAGRAPH_COMPLETION_MARKER}" "${server_log}"; then
        capture_status=capture_completed
        evidence=server_log_completion_marker
      else
        capture_status=configured_not_observed
        evidence=none
      fi
      ;;
    1)
      configured=false
      capture_status=disabled_eager
      evidence=not_applicable
      ;;
    *)
      echo "Invalid ENFORCE_EAGER=${eager_mode}" >&2
      return 2
      ;;
  esac

  {
    echo "cudagraph_configured=${configured}"
    echo "cudagraph_capture_status=${capture_status}"
    echo "cudagraph_capture_evidence=${evidence}"
    echo "cudagraph_capture_marker=${CUDAGRAPH_COMPLETION_MARKER}"
  } >>"${metadata_path}"

  if [[ "${capture_status}" != capture_completed ]] && \
    [[ "${requested_run_kind}" == baseline || \
      "${requested_run_kind}" == lookup ]]; then
    echo "${requested_run_kind} requires server.log marker: ${CUDAGRAPH_COMPLETION_MARKER}" \
      >&2
    return 1
  fi
}

request_trace_flush() {
  local trace_dir=$1
  local timeout_s=$2
  local trace pid
  local -a traces pids missing
  shopt -s nullglob
  traces=("${trace_dir}"/trace.*.jsonl)
  if ((${#traces[@]} == 0)); then
    echo "No MXFP8 tactic traces were produced in ${trace_dir}" >&2
    return 1
  fi
  for trace in "${traces[@]}"; do
    pid=$(basename "${trace}")
    pid=${pid#trace.}
    pid=${pid%.jsonl}
    pids+=("${pid}")
  done
  for pid in "${pids[@]}"; do
    rm -f "${trace_dir}/counts.${pid}.complete" \
      "${trace_dir}/flush.${pid}.request"
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "MXFP8 trace worker ${pid} exited before flush" >&2
      return 1
    fi
    touch "${trace_dir}/flush.${pid}.request"
  done
  local deadline=$((SECONDS + timeout_s))
  while true; do
    missing=()
    for pid in "${pids[@]}"; do
      if [[ ! -s "${trace_dir}/counts.${pid}.jsonl" ]] || \
        [[ ! -f "${trace_dir}/counts.${pid}.complete" ]]; then
        missing+=("${pid}")
      fi
    done
    ((${#missing[@]} == 0)) && return 0
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for MXFP8 trace workers: ${missing[*]}" >&2
      return 1
    fi
    sleep 0.1
  done
}

if [[ "${1:-}" == --record-cudagraph-evidence ]]; then
  if (($# != 5)); then
    echo "Usage: $0 --record-cudagraph-evidence RUN_KIND ENFORCE_EAGER SERVER_LOG METADATA" \
      >&2
    exit 2
  fi
  record_cudagraph_evidence "$2" "$3" "$4" "$5"
  exit
fi

if [[ "${1:-}" == --request-trace-flush ]]; then
  if (($# != 3)); then
    echo "Usage: $0 --request-trace-flush TRACE_DIR TIMEOUT_S" >&2
    exit 2
  fi
  request_trace_flush "$2" "$3"
  exit
fi

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${SCRATCH_ROOT}/autotune"
export FLASHINFER_WORKSPACE_BASE="${SCRATCH_ROOT}/flashinfer"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg"
export PYTHONPYCACHEPREFIX="${SCRATCH_ROOT}/pycache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
readonly GPU_NAME=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)

server_env=(
  env
  "PYTHONPATH=${EXP_DIR}:${SOURCE_ROOT}"
  "SOURCE_ROOT=${SOURCE_ROOT}"
  "FLASHINFER_ROOT=${FLASHINFER_ROOT}"
  "FLASHINFER_COMMIT=${FLASHINFER_COMMIT}"
  "EXPECTED_CONTAINER_SHA256=${EXPECTED_CONTAINER_SHA256}"
  "EXPECTED_VLLM_VERSION=${EXPECTED_VLLM_VERSION}"
  "MXFP8_TACTIC_TRACE_DIR=${SCRATCH_ROOT}/traces"
  "MXFP8_TACTIC_TRACE_PHASE=${TRACE_PHASE}"
  "MXFP8_TACTIC_BACKEND=${ORACLE_BACKEND}"
  "MXFP8_TACTIC_SCALE_LAYOUT=${SCALE_LAYOUT}"
  "MXFP8_TACTIC_GPU=${GPU_NAME}"
  "VLLM_MXFP8_TRTLLM_LAYOUT=${TRTLLM_LAYOUT}"
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
  echo "backend_name=${BACKEND_NAME}"
  echo "oracle_backend=${ORACLE_BACKEND}"
  echo "scale_layout=${SCALE_LAYOUT}"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "flashinfer_commit=${FLASHINFER_COMMIT}"
  echo "expected_vllm_version=${EXPECTED_VLLM_VERSION}"
  echo "container=${CONTAINER_IMAGE}"
  echo "container_sha256=${EXPECTED_CONTAINER_SHA256}"
  echo "model=${MODEL_PATH}"
  echo "tp=${TP}"
  echo "linear_backend=${LINEAR_BACKEND}"
  echo "trtllm_layout=${TRTLLM_LAYOUT}"
  echo "moe_backend=${MOE_BACKEND}"
  echo "attention_backend=FLASHINFER"
  echo "kv_cache_dtype=auto"
  echo "max_model_len=${MAX_MODEL_LEN}"
  echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  echo "max_num_seqs=${MAX_NUM_SEQS}"
  echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  echo "enable_chunked_prefill=true"
  echo "enable_prefix_caching=true"
  echo "mamba_cache_mode=all"
  echo "mamba_ssm_cache_dtype=float32"
  echo "enforce_eager=${ENFORCE_EAGER}"
  echo "cudagraph_capture_sizes=1,2,4,8,16,32"
  echo "workloads=${WORKLOADS}"
  echo "concurrencies=${CONCURRENCIES}"
  echo "prompt_multiplier=${PROMPT_MULTIPLIER}"
  printf "server_command="
  printf "%q " "${server_env[@]}" "${server_cmd[@]}"
  printf "\n"
  echo "vllm_version=$("${server_env[@]}" vllm --version)"
  "${server_env[@]}" uv run --no-project python -c \
    'import flashinfer, vllm, vllm._C_stable_libtorch; print("vllm_file=" + vllm.__file__); print("vllm_compiled_file=" + vllm._C_stable_libtorch.__file__); print("flashinfer=" + flashinfer.__version__); print("flashinfer_file=" + flashinfer.__file__)'
  echo "gpu_name=${GPU_NAME}"
  echo "driver_version=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader)"
} >"${RUN_DIR}/metadata.txt"

setsid "${server_env[@]}" "${server_cmd[@]}" >"${RUN_DIR}/server.log" 2>&1 &
server_pid=$!
server_group_alive() {
  kill -0 -- "-${server_pid}" 2>/dev/null
}
signal_server_group() {
  local signal=$1
  local attempts=$2
  kill "-${signal}" -- "-${server_pid}" 2>/dev/null || true
  for _ in $(seq 1 "${attempts}"); do
    server_group_alive || return 0
    sleep 1
  done
  return 1
}
stop_server() {
  if server_group_alive; then
    signal_server_group INT 60 || \
      signal_server_group TERM 30 || \
      signal_server_group KILL 10 || {
        echo "vLLM process group ${server_pid} did not stop" >&2
        return 1
      }
  fi
  wait "${server_pid}" 2>/dev/null || true
}
publish_traces() {
  local trace pid
  local -a traces
  shopt -s nullglob
  traces=("${SCRATCH_ROOT}"/traces/trace.*.jsonl)
  if ((${#traces[@]} == 0)); then
    echo "No MXFP8 tactic traces were produced" >&2
    return 1
  fi
  for trace in "${traces[@]}"; do
    pid=$(basename "${trace}")
    pid=${pid#trace.}
    pid=${pid%.jsonl}
    test -s "${SCRATCH_ROOT}/traces/counts.${pid}.jsonl"
    test -f "${SCRATCH_ROOT}/traces/counts.${pid}.complete"
  done
  local stage="${RUN_DIR}/traces.tmp.${SLURM_JOB_ID}"
  rm -rf "${stage}"
  mkdir "${stage}"
  cp -a "${SCRATCH_ROOT}/traces/." "${stage}/"
  mv "${stage}" "${RUN_DIR}/traces"
}
cleanup() {
  local status=$?
  trap - EXIT
  set +e
  stop_server
  mkdir -p "${RESULT_ROOT}/debug_traces/${RUN_KIND}/${SLURM_JOB_ID}"
  cp -a "${SCRATCH_ROOT}/traces/." \
    "${RESULT_ROOT}/debug_traces/${RUN_KIND}/${SLURM_JOB_ID}/" 2>/dev/null
  rm -rf "${SCRATCH_ROOT}"
  exit "${status}"
}
trap cleanup EXIT

deadline=$((SECONDS + 7200))
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; do
  if ! server_group_alive; then
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
    "${server_env[@]}" timeout "${BENCH_TIMEOUT_S}" vllm bench serve \
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
      --temperature 0 \
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
output_lens = result["output_lens"]
generated_texts = result["generated_texts"]
assert len(input_lens) == expected_requests, result
assert len(output_lens) == expected_requests, result
assert len(generated_texts) == expected_requests, result
assert all(isl <= length <= isl + 1 for length in input_lens), result
assert all(length == osl for length in output_lens), result
assert result["total_input_tokens"] == sum(input_lens), result
assert result["total_output_tokens"] == expected_requests * osl, result
PY
  done
  workload_index=$((workload_index + 1))
done

request_trace_flush "${SCRATCH_ROOT}/traces" 30
stop_server
record_cudagraph_evidence \
  "${RUN_KIND}" "${ENFORCE_EAGER}" "${RUN_DIR}/server.log" \
  "${RUN_DIR}/metadata.txt"
publish_traces
rm -rf "${SCRATCH_ROOT}"
trap - EXIT
touch "${RUN_DIR}/COMPLETE"
