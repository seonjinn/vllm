#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_DIR=$(cd "$(dirname "$0")" && pwd)
VLLM_SOURCE_ROOT=${VLLM_SOURCE_ROOT:-$(cd "${EXPERIMENT_DIR}/../../../../.." && pwd)}
BENCH_ROOT=${BENCH_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark-v0271-adaptive-long}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
NTRACE_RUNTIME=${NTRACE_RUNTIME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ntrace-vllm0271/runtime-4dbf6c2e-cuda13-py312-nonumpy}
BF16_MODEL=${BF16_MODEL:-/lustre/fsw/coreai_dlalgo_llm/nemo_rl_ci/nemotron_ultra/checkpoints/ultra-v3-sft-hsg-mainfeb19merge-mxfp8_fixed-hf_converted}
MXFP8_MODEL=${MXFP8_MODEL:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/nemotron3-ultra-expert-only-mxfp8-samebase}

SCOPES=${SCOPES:-"bf16 mxfp8"}
PHASES=${PHASES:-"prefill decode"}
CONCURRENCY=${CONCURRENCY:-8}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT_BASE=${RUN_ROOT_BASE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/bf16_mxfp8_ntrace_ops}
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
  "${BF16_MODEL}"
  "${MXFP8_MODEL}"
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
container_sha256=$(sha256sum "${CONTAINER_IMAGE}" | awk '{print $1}')
compilation_config='{"cudagraph_capture_sizes":[8],"cudagraph_mode":"FULL_AND_PIECEWISE","splitting_ops":["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::hpc_rope_norm_forward","vllm::unified_kv_cache_update","vllm::unified_mla_kv_cache_update"]}'

runtime_override_files=(
  vllm/config/profiler.py
  vllm/v1/worker/gpu_worker.py
)

for phase in ${PHASES}; do
  case "${phase}" in
    prefill)
      scenario=short
      isl=10000
      osl=1
      ;;
    decode)
      scenario=short
      isl=1000
      osl=256
      ;;
    *)
      echo "ERROR: unsupported phase '${phase}' (use prefill|decode)" >&2
      exit 2
      ;;
  esac

  for scope in ${SCOPES}; do
    case "${scope}" in
      bf16)
        precision=bf16
        model=${BF16_MODEL}
        linear_backend=
        ;;
      mxfp8)
        precision=mxfp8
        model=${MXFP8_MODEL}
        linear_backend=auto
        ;;
      *)
        echo "ERROR: unsupported scope '${scope}' (use bf16|mxfp8)" >&2
        exit 2
        ;;
    esac

    run_root="${RUN_ROOT_BASE}/${STAMP}_${phase}_${scope}_c${CONCURRENCY}"
    trace_root="${run_root}/ntrace"
    runtime_override_root="${run_root}/runtime_override"
    mkdir -p "${trace_root}" "${runtime_override_root}"
    for relative_path in "${runtime_override_files[@]}"; do
      mkdir -p "${runtime_override_root}/$(dirname "${relative_path}")"
      cp "${VLLM_SOURCE_ROOT}/${relative_path}" \
        "${runtime_override_root}/${relative_path}"
    done

    cat > "${run_root}/metadata.env" <<EOF
phase=${phase}
scope=${scope}
precision=${precision}
vllm_head=${vllm_head}
container=${CONTAINER_IMAGE}
container_sha256=${container_sha256}
model=${model}
ntrace_runtime=${NTRACE_RUNTIME}
ntrace_native=${ntrace_native}
tp=8
dp=1
pp=1
expert_parallel=1
batch_size=${CONCURRENCY}
concurrency=${CONCURRENCY}
num_requests=${CONCURRENCY}
isl=${isl}
osl=${osl}
cuda_graph=FULL_AND_PIECEWISE
linear_backend=${linear_backend:-vllm_default}
moe_backend=flashinfer_trtllm
kv_cache_dtype=auto
bench_seed=17
warmup_seed=117
EOF

    echo "phase=${phase} scope=${scope} run_root=${run_root}"
    env \
      PRECISIONS="${precision}" \
      STAMP="${STAMP}_${phase}_${scope}" \
      RUN_ROOT="${run_root}" \
      BENCH_PY="${VLLM_SOURCE_ROOT}/benchmarks/profiling/ntrace/benchmark_with_ntrace.py" \
      NTRACE_BENCH_TARGET="${BENCH_ROOT}/benchmark_vllm_bench_serve_static.py" \
      JOB_PREFIX="coreai_dlalgo_llm-${USER:-sna}.ops-${phase}-${scope}" \
      CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
      JOB_CACHE_DIR="${BENCH_ROOT}/.container_cache" \
      CONTAINER_MOUNTS="/lustre:/lustre,${BENCH_ROOT}/.container_cache:/root/.cache" \
      SBATCH_EXPORT_MODE=all \
      ULTRA_BF16_MODEL="${model}" \
      ULTRA_MXFP8_MODEL="${model}" \
      BF16_NNODES=2 \
      BF16_TP=8 \
      BF16_LINEAR_BACKEND="${linear_backend}" \
      BF16_MOE_BACKEND=flashinfer_trtllm \
      MXFP8_NNODES=2 \
      MXFP8_TP=8 \
      MXFP8_LINEAR_BACKEND="${linear_backend}" \
      MXFP8_MOE_BACKEND=flashinfer_trtllm \
      COMPILATION_CONFIG_JSON="${compilation_config}" \
      SCENARIOS_TO_RUN="${scenario}" \
      ISL_SHORT_VALUE="${isl}" \
      ISL_LONG_VALUE="${isl}" \
      OSL_SHORT_VALUE="${osl}" \
      OSL_LONG_VALUE="${osl}" \
      BSIZES="${CONCURRENCY}" \
      MULT=1 \
      REQUEST_SUPPLY_MODE=fixed-batch \
      MAX_MODEL_LEN=12024 \
      MAX_NUM_BATCHED_TOKENS=16384 \
      SERVER_MAX_NUM_SEQS="${CONCURRENCY}" \
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
      STRICT_WARMUP_TOKENS=1 \
      BENCH_SEED=17 \
      WARMUP_SEED=117 \
      BENCH_RETRIES=0 \
      BENCH_TIMEOUT_S=7200 \
      SERVER_HEALTH_TIMEOUT_S=7200 \
      WALLTIME="${WALLTIME}" \
      SUBMIT_DELAY_S=0 \
      NSYS_PROFILE_ENABLED=0 \
      RAY_WORKERS_USE_NSIGHT=0 \
      SERVER_NSYS_PROFILE=0 \
      FORCE_EXIT_AFTER_BENCH=1 \
      PYTHONPATH="${NTRACE_RUNTIME}" \
      VLLM_SUBPROCESS_PYTHONPATH="${NTRACE_RUNTIME}" \
      VLLM_RUNTIME_OVERRIDE_DIR="${runtime_override_root}" \
      NTRACE_ROLLOUT_OUTPUT_DIR="${trace_root}" \
      NTRACE_ROLLOUT_RANKS=0 \
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
done

echo "result_root_base=${RUN_ROOT_BASE}/${STAMP}"
