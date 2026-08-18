#!/usr/bin/env bash

set -euo pipefail

BENCH_ROOT=${BENCH_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark-v0271-adaptive-long}
VLLM_SOURCE_ROOT=${VLLM_SOURCE_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-ntrace-profile}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64.sqsh}
MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
NTRACE_RUNTIME=${NTRACE_RUNTIME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ntrace-vllm0271/runtime-4dbf6c2e-cuda13-py312}
PATCHED_TRTLLM_GEMM=${PATCHED_TRTLLM_GEMM:-/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/flashinfer-mxfp8-lowm-heuristic/benchmark-graph-c634591f/cache-patched/.cache/flashinfer/0.6.16.post3/100a/cached_ops/trtllm_gemm/trtllm_gemm.so}
CONTAINER_TRTLLM_GEMM=${CONTAINER_TRTLLM_GEMM:-/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/trtllm_gemm/trtllm_gemm.so}

VARIANTS=${VARIANTS:-"cutedsl shape_aware"}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT_BASE=${RUN_ROOT_BASE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ntrace_mxfp8_backend_tp4}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}
WALLTIME=${WALLTIME:-02:00:00}
ISL=${ISL:-1000}
OSL=${OSL:-256}
BATCH_SIZE=${BATCH_SIZE:-8}
BENCH_SEED=${BENCH_SEED:-17}
WARMUP_SEED=${WARMUP_SEED:-117}
COMPILATION_CONFIG_JSON=${COMPILATION_CONFIG_JSON:-'{"cudagraph_capture_sizes":[8],"cudagraph_mode":"FULL_AND_PIECEWISE","splitting_ops":["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::hpc_rope_norm_forward","vllm::unified_kv_cache_update","vllm::unified_mla_kv_cache_update","vllm::mxfp8_trtllm_dispatch_linear"]}'}

required_paths=(
  "${BENCH_ROOT}/submit_bench_lyris_nemotron3_ultra_w4a16.sh"
  "${BENCH_ROOT}/vllm-ultra-ray-bench-serve-static.sh"
  "${BENCH_ROOT}/benchmark_vllm_bench_serve_static.py"
  "${VLLM_SOURCE_ROOT}/vllm/config/profiler.py"
  "${VLLM_SOURCE_ROOT}/vllm/v1/worker/gpu_worker.py"
  "${CONTAINER_IMAGE}"
  "${MODEL_PATH}"
  "${NTRACE_RUNTIME}"
  "${PATCHED_TRTLLM_GEMM}"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e ${path} ]]; then
    echo "ERROR: required path does not exist: ${path}" >&2
    exit 2
  fi
done

vllm_head=$(git -C "${VLLM_SOURCE_ROOT}" rev-parse HEAD)
ntrace_native=$(find "${NTRACE_RUNTIME}/ntrace" -maxdepth 1 -name '_cupti_cpp*.so' -print -quit)
if [[ -z ${ntrace_native} ]]; then
  echo "ERROR: ntrace C++ CUPTI backend is missing under ${NTRACE_RUNTIME}" >&2
  exit 2
fi

base_mounts=(
  "/lustre:/lustre"
  "${BENCH_ROOT}/.container_cache:/root/.cache"
  "${VLLM_SOURCE_ROOT}/vllm/config/profiler.py:/usr/local/lib/python3.12/dist-packages/vllm/config/profiler.py"
  "${VLLM_SOURCE_ROOT}/vllm/v1/worker/gpu_worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"
  "${VLLM_SOURCE_ROOT}/vllm/envs.py:/usr/local/lib/python3.12/dist-packages/vllm/envs.py"
  "${VLLM_SOURCE_ROOT}/vllm/model_executor/kernels/linear/__init__.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/__init__.py"
  "${VLLM_SOURCE_ROOT}/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
  "${VLLM_SOURCE_ROOT}/vllm/utils/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/flashinfer.py"
)

for variant in ${VARIANTS}; do
  mounts=("${base_mounts[@]}")
  case "${variant}" in
    cutedsl)
      linear_backend=flashinfer_cutedsl
      ;;
    shape_aware)
      linear_backend=flashinfer_trtllm
      mounts+=("${PATCHED_TRTLLM_GEMM}:${CONTAINER_TRTLLM_GEMM}")
      ;;
    *)
      echo "ERROR: unsupported variant: ${variant}" >&2
      exit 2
      ;;
  esac

  mount_list=$(IFS=,; echo "${mounts[*]}")
  run_root="${RUN_ROOT_BASE}/${STAMP}_${variant}_isl${ISL}_osl${OSL}_bs${BATCH_SIZE}"
  trace_root="${run_root}/ntrace"
  mkdir -p "${run_root}" "${trace_root}"
  cat > "${run_root}/metadata.env" <<EOF
variant=${variant}
vllm_head=${vllm_head}
container=${CONTAINER_IMAGE}
model=${MODEL_PATH}
ntrace_runtime=${NTRACE_RUNTIME}
ntrace_native=${ntrace_native}
tp=4
dp=1
pp=1
batch_size=${BATCH_SIZE}
concurrency=${BATCH_SIZE}
num_requests=${BATCH_SIZE}
isl=${ISL}
osl=${OSL}
cuda_graph=FULL_AND_PIECEWISE
linear_backend=${linear_backend}
moe_backend=flashinfer_trtllm
bench_seed=${BENCH_SEED}
warmup_seed=${WARMUP_SEED}
EOF

  env \
    PRECISIONS=mxfp8 \
    STAMP="${STAMP}_${variant}" \
    RUN_ROOT="${run_root}" \
    JOB_PREFIX="coreai_dlalgo_llm-${USER:-sna}.ntrace-${variant}" \
    CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    CONTAINER_MOUNTS="${mount_list}" \
    JOB_CACHE_DIR="${BENCH_ROOT}/.container_cache" \
    SBATCH_EXPORT_MODE=all \
    ULTRA_MXFP8_MODEL="${MODEL_PATH}" \
    MXFP8_NNODES=1 \
    MXFP8_TP=4 \
    MXFP8_LINEAR_BACKEND="${linear_backend}" \
    MXFP8_MOE_BACKEND=flashinfer_trtllm \
    VLLM_MXFP8_TRTLLM_LAYOUT=adaptive \
    VLLM_MXFP8_TRTLLM_TACTICS= \
    VLLM_MXFP8_TRTLLM_SWITCH_M=256 \
    COMPILATION_CONFIG_JSON="${COMPILATION_CONFIG_JSON}" \
    SCENARIOS_TO_RUN=shortin \
    ISL_SHORT_VALUE="${ISL}" \
    OSL_LONG_VALUE="${OSL}" \
    BSIZES="${BATCH_SIZE}" \
    MULT=1 \
    REQUEST_SUPPLY_MODE=continuous \
    MAX_MODEL_LEN="$((ISL + OSL + 256))" \
    MAX_NUM_BATCHED_TOKENS=16384 \
    SERVER_MAX_NUM_SEQS="${BATCH_SIZE}" \
    GPU_MEM=0.90 \
    ENABLE_CHUNKED_PREFILL=1 \
    ENABLE_PREFIX_CACHING=1 \
    ENFORCE_EAGER=0 \
    ASYNC_SCHEDULING=0 \
    FORCE_DISABLE_ASYNC_SCHEDULING=1 \
    SKIP_WARMUP=0 \
    MAMBA_CACHE_MODE=all \
    MAMBA_SSM_CACHE_DTYPE=float32 \
    RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 \
    STRICT_RESULT_TOKENS=1 \
    STRICT_WARMUP_TOKENS=0 \
    BENCH_SEED="${BENCH_SEED}" \
    WARMUP_SEED="${WARMUP_SEED}" \
    BENCH_TIMEOUT_S=7200 \
    SERVER_HEALTH_TIMEOUT_S=7200 \
    WALLTIME="${WALLTIME}" \
    SUBMIT_DELAY_S=0 \
    SERVER_PROFILER=ntrace \
    CLIENT_PROFILE=1 \
    PROFILE_FIRST_ATTEMPT=1 \
    NSYS_PROFILE_ENABLED=1 \
    NSYS_BIN=true \
    NSYS_INSTALL_IF_MISSING=0 \
    NSYS_COPY_DELAY_S=0 \
    RAY_WORKERS_USE_NSIGHT=0 \
    SERVER_NSYS_PROFILE=0 \
    FORCE_EXIT_AFTER_BENCH=1 \
    PYTHONPATH="${NTRACE_RUNTIME}" \
    VLLM_SUBPROCESS_PYTHONPATH="${NTRACE_RUNTIME}" \
    NTRACE_ROLLOUT_OUTPUT_DIR="${trace_root}" \
    NTRACE_ROLLOUT_RANKS=0 \
    NTRACE_ROLLOUT_CAPTURE_ITER=0 \
    NTRACE_ROLLOUT_NUM_ITERS=1 \
    NTRACE_ROLLOUT_GRAPH_CAPTURE=early \
    NTRACE_INCLUDE_STACK_TRACES=1 \
    NTRACE_INCLUDE_NVTX_RANGES=1 \
    NTRACE_SAVE_CPU_NVTX=0 \
    NTRACE_INCLUDE_MEMOPS=0 \
    NTRACE_GPU_METRICS_INTERVAL_US=0 \
    NTRACE_CUPTI_BACKEND=cpp \
    SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY}" \
    DRY_RUN="${DRY_RUN}" \
    "${BENCH_ROOT}/submit_bench_lyris_nemotron3_ultra_w4a16.sh"
done

echo "result_root_base=${RUN_ROOT_BASE}"
