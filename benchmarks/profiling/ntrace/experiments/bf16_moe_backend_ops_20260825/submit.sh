#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_DIR=$(cd "$(dirname "$0")" && pwd)
VLLM_SOURCE_ROOT=${VLLM_SOURCE_ROOT:-$(cd "${EXPERIMENT_DIR}/../../../../.." && pwd)}
BENCH_ROOT=${BENCH_ROOT:-/home/sna/vllm-benchmark-bf16-moe-versions}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
CONTAINER_SHA256=${CONTAINER_SHA256:-e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0}
NTRACE_RUNTIME=${NTRACE_RUNTIME:-/lustre/fsw/portfolios/coreai/users/sna/ntrace-vllm0271/runtime-165ae08-cuda13-py312-nonumpy}
NTRACE_REVISION=${NTRACE_REVISION:-165ae08}
BF16_MODEL=${BF16_MODEL:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-normal}
TOPOLOGY_ARGS=${TOPOLOGY_ARGS:---switches=1@600 --gpus-per-node=4}
JOB_CACHE_DIR=${JOB_CACHE_DIR:-/raid/scratch/sna/vllm-benchmark/container-cache}
HF_HOME_OVERRIDE=${HF_HOME_OVERRIDE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/home:/home,/lustre:/lustre,${JOB_CACHE_DIR}:/root/.cache}

BACKENDS=${BACKENDS:-"triton flashinfer_trtllm"}
CONCURRENCY=${CONCURRENCY:-8}
ISL=${ISL:-1000}
OSL=${OSL:-256}
NTRACE_ROLLOUT_RANKS=${NTRACE_ROLLOUT_RANKS:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT_BASE=${RUN_ROOT_BASE:-/lustre/fsw/portfolios/coreai/users/sna/vllm-v0271-results/bf16_moe_backend_ntrace}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-1}
DRY_RUN=${DRY_RUN:-0}
WALLTIME=${WALLTIME:-04:00:00}

required_paths=(
  "${BENCH_ROOT}/submit_bench_lyris_nemotron3_ultra_w4a16.sh"
  "${BENCH_ROOT}/vllm-ultra-ray-bench-serve-static.sh"
  "${BENCH_ROOT}/benchmark_vllm_bench_serve_static.py"
  "${VLLM_SOURCE_ROOT}/benchmarks/profiling/ntrace/benchmark_with_ntrace.py"
  "${CONTAINER_IMAGE}"
  "${NTRACE_RUNTIME}"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e ${path} ]]; then
    echo "ERROR: required path does not exist: ${path}" >&2
    exit 2
  fi
done

ntrace_native=$(find "${NTRACE_RUNTIME}/ntrace" -maxdepth 1 -name '_cupti_cpp*.so' -print -quit)
if [[ -z ${ntrace_native} ]]; then
  echo "ERROR: ntrace C++ CUPTI backend is missing under ${NTRACE_RUNTIME}" >&2
  exit 2
fi

vllm_head=$(git -C "${VLLM_SOURCE_ROOT}" rev-parse HEAD)
container_mounts_b64=$(printf '%s' "${CONTAINER_MOUNTS}" | base64 | tr -d '\n')
compilation_config='{"cudagraph_capture_sizes":[8],"cudagraph_mode":"FULL_AND_PIECEWISE","splitting_ops":["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::hpc_rope_norm_forward","vllm::unified_kv_cache_update","vllm::unified_mla_kv_cache_update"]}'
runtime_override_files=(
  vllm/config/profiler.py
  vllm/v1/worker/gpu_worker.py
)

for backend in ${BACKENDS}; do
  case "${backend}" in
    triton|flashinfer_trtllm) ;;
    *)
      echo "ERROR: unsupported backend '${backend}'" >&2
      exit 2
      ;;
  esac

  run_root="${RUN_ROOT_BASE}/${STAMP}_decode_${backend}_c${CONCURRENCY}"
  trace_root="${run_root}/ntrace"
  runtime_override_root="${run_root}/runtime_override"
  mkdir -p "${trace_root}" "${runtime_override_root}"
  for relative_path in "${runtime_override_files[@]}"; do
    mkdir -p "${runtime_override_root}/$(dirname "${relative_path}")"
    cp "${VLLM_SOURCE_ROOT}/${relative_path}" \
      "${runtime_override_root}/${relative_path}"
  done

  cat > "${run_root}/metadata.env" <<EOF
phase=decode
scope=bf16_moe_backend
precision=bf16
vllm_version=0.27.1
vllm_head=${vllm_head}
container=${CONTAINER_IMAGE}
container_sha256=${CONTAINER_SHA256}
model=${BF16_MODEL}
ntrace_runtime=${NTRACE_RUNTIME}
ntrace_revision=${NTRACE_REVISION}
ntrace_native=${ntrace_native}
tp=8
dp=1
pp=1
expert_parallel=1
batch_size=${CONCURRENCY}
concurrency=${CONCURRENCY}
num_requests=${CONCURRENCY}
isl=${ISL}
osl=${OSL}
cuda_graph=FULL_AND_PIECEWISE
linear_backend=vllm_default
moe_backend=${backend}
kv_cache_dtype=auto
bench_seed=17
warmup_seed=117
captured_ranks=${NTRACE_ROLLOUT_RANKS}
EOF

  echo "backend=${backend} run_root=${run_root}"
  env \
    ACCOUNT="${ACCOUNT}" \
    PARTITION="${PARTITION}" \
    QOS="${QOS}" \
    TOPOLOGY_ARGS="${TOPOLOGY_ARGS}" \
    PRECISIONS=bf16 \
    STAMP="${STAMP}_${backend}" \
    RUN_ROOT="${run_root}" \
    SBATCH_OUT_DIR="${run_root}/slurm" \
    BENCH_PY="${VLLM_SOURCE_ROOT}/benchmarks/profiling/ntrace/benchmark_with_ntrace.py" \
    NTRACE_BENCH_TARGET="${BENCH_ROOT}/benchmark_vllm_bench_serve_static.py" \
    NTRACE_EXPECTED_ISL="${ISL}" \
    NTRACE_EXPECTED_OSL="${OSL}" \
    JOB_PREFIX="coreai_dlalgo_llm-${USER:-sna}.bf16-ntrace-${backend}" \
    CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    JOB_CACHE_DIR="${JOB_CACHE_DIR}" \
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS}" \
    CONTAINER_MOUNTS_B64="${container_mounts_b64}" \
    SBATCH_EXPORT_MODE=all \
    HF_HOME_OVERRIDE="${HF_HOME_OVERRIDE}" \
    ULTRA_BF16_MODEL="${BF16_MODEL}" \
    BF16_NNODES=2 \
    BF16_TP=8 \
    BF16_LINEAR_BACKEND= \
    BF16_MOE_BACKEND="${backend}" \
    REQUIRE_MOE_BACKEND_FLAG=1 \
    COMPILATION_CONFIG_JSON="${compilation_config}" \
    SCENARIOS_TO_RUN=shortin \
    ISL_SHORT_VALUE="${ISL}" \
    OSL_LONG_VALUE="${OSL}" \
    BSIZES="${CONCURRENCY}" \
    MULT=1 \
    REQUEST_SUPPLY_MODE=fixed-batch \
    MAX_MODEL_LEN=11000 \
    MAX_NUM_BATCHED_TOKENS=16384 \
    SERVER_MAX_NUM_SEQS="${CONCURRENCY}" \
    GPU_MEM=0.95 \
    DTYPE=bfloat16 \
    LOAD_FORMAT=auto \
    KV_CACHE_DTYPE=auto \
    ATTN_BACKEND=FLASHINFER \
    ENABLE_CHUNKED_PREFILL=1 \
    ENABLE_PREFIX_CACHING=1 \
    ENFORCE_EAGER=0 \
    ASYNC_SCHEDULING=0 \
    FORCE_DISABLE_ASYNC_SCHEDULING=1 \
    SHARED_SERVER=0 \
    SKIP_WARMUP=0 \
    WARMUP_NUM_PROMPTS_MULTIPLIER=1 \
    STRICT_WARMUP_TOKENS=1 \
    STRICT_RESULT_TOKENS=1 \
    BENCH_RETRIES=0 \
    BENCH_SEED=17 \
    WARMUP_SEED=117 \
    BENCH_TIMEOUT_S=7200 \
    SERVER_HEALTH_TIMEOUT_S=7200 \
    WALLTIME="${WALLTIME}" \
    SUBMIT_DELAY_S=0 \
    SERVER_PROFILER=ntrace \
    CLIENT_PROFILE=1 \
    PROFILE_FIRST_ATTEMPT=1 \
    NSYS_PROFILE_ENABLED=0 \
    RAY_WORKERS_USE_NSIGHT=0 \
    SERVER_NSYS_PROFILE=0 \
    FORCE_EXIT_AFTER_BENCH=1 \
    PYTHONPATH="${NTRACE_RUNTIME}" \
    VLLM_SUBPROCESS_PYTHONPATH="${NTRACE_RUNTIME}" \
    VLLM_RUNTIME_OVERRIDE_DIR="${runtime_override_root}" \
    NTRACE_ROLLOUT_OUTPUT_DIR="${trace_root}" \
    NTRACE_ROLLOUT_RANKS="${NTRACE_ROLLOUT_RANKS}" \
    NTRACE_ROLLOUT_CAPTURE_ITER=0 \
    NTRACE_ROLLOUT_NUM_ITERS=1 \
    NTRACE_ROLLOUT_GRAPH_CAPTURE=early \
    NTRACE_INCLUDE_STACK_TRACES=1 \
    NTRACE_INCLUDE_NVTX_RANGES=1 \
    NTRACE_SAVE_CPU_NVTX=0 \
    NTRACE_INCLUDE_MEMOPS=1 \
    NTRACE_GPU_METRICS_INTERVAL_US=0 \
    NTRACE_CUPTI_BACKEND=cpp \
    SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY}" \
    DRY_RUN="${DRY_RUN}" \
    "${BENCH_ROOT}/submit_bench_lyris_nemotron3_ultra_w4a16.sh"
done

echo "result_root_base=${RUN_ROOT_BASE}/${STAMP}"
