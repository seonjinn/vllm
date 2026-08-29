#!/usr/bin/env bash

set -Eeuo pipefail

readonly EXP_DIR_LOCAL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly REPO_ROOT_LOCAL=$(cd "${EXP_DIR_LOCAL}/../.." && pwd)
readonly CLUSTER=${CLUSTER:-login-lyris}
readonly ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
readonly PARTITION=${PARTITION:-gb200-backfill}
readonly QOS=${QOS:-user-restrictions}
readonly REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-/home/sna/vllm-v0271-ultra-tp8-crossover}
readonly SOURCE_COMMIT=${SOURCE_COMMIT:-e3c4137b83bd77d63245b28f99c01b79a7bf7a6c}
readonly SOURCE_ROOT=${SOURCE_ROOT:-/home/sna/vllm-v0271-mxfp8-pins/${SOURCE_COMMIT}}
readonly BENCH_COMMIT=${BENCH_COMMIT:-45b96a08f62b0687ecf732c7d57880426646bb0b}
readonly BENCH_REPO=${BENCH_REPO:-/home/sna/vllm-benchmark-layout-ncu/4f2ca9d1f0b1}
readonly BENCH_ROOT=${BENCH_ROOT:-/home/sna/vllm-benchmark-serving-crossover/${BENCH_COMMIT:0:12}}
readonly FLASHINFER_COMMIT=${FLASHINFER_COMMIT:-cec5e66dbd75a253edb5a819b2403bf410ca3223}
readonly FLASHINFER_ROOT=${FLASHINFER_ROOT:-/home/sna/flashinfer-mxfp8-pins/${FLASHINFER_COMMIT}}
readonly CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
readonly EXPECTED_CONTAINER_SHA256=${EXPECTED_CONTAINER_SHA256:-e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0}
readonly MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
readonly LOOKUP_PATH=${LOOKUP_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/mxfp8_tp8_serving_crossover_20260829_tp8census_r7_full/adaptive_oracle_20260829_tp8oracle_r1/adaptive_lookup.json}
readonly ORDERS=${ORDERS:-"cutedsl,adaptive-lookup adaptive-lookup,cutedsl"}
readonly CONCURRENCIES=${CONCURRENCIES:-"8 32"}
readonly WAVES=${WAVES:-10}
readonly STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
readonly RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/mxfp8_tp8_paired_e2e_${STAMP}}
readonly WALLTIME=${WALLTIME:-08:00:00}
readonly PRINT_PLAN=${PRINT_PLAN:-0}
readonly SBATCH_EXTRA_EXPORT_NAMES="SOURCE_ROOT FLASHINFER_ROOT FLASHINFER_COMMIT EXPECTED_CONTAINER_SHA256 EXPECTED_VLLM_VERSION MXFP8_TACTIC_LOOKUP MXFP8_TACTIC_BACKEND MXFP8_TACTIC_SCALE_LAYOUT MXFP8_TACTIC_GPU PAIR_ORDER PAIR_BENCHMARK_SCRIPT MXFP8_TACTIC_LOOKUP_PATH PYTHONDONTWRITEBYTECODE"

print_plan() {
  local order
  echo "parallelism=TP8,DP1,EP8"
  echo "workload=ISL1000,OSL10000"
  echo "concurrencies=${CONCURRENCIES}"
  echo "waves=${WAVES}"
  echo "cuda_graph=true"
  echo "allocation=2nodes,4gpus_per_node"
  for order in ${ORDERS}; do
    echo "order=${order//,/ }"
  done
}

if [[ "${PRINT_PLAN}" == 1 ]]; then
  print_plan
  exit 0
fi

if ! git -C "${REPO_ROOT_LOCAL}" diff --quiet || \
  ! git -C "${REPO_ROOT_LOCAL}" diff --cached --quiet || \
  [[ -n $(git -C "${REPO_ROOT_LOCAL}" ls-files --others --exclude-standard) ]]; then
  echo "Local vLLM worktree is dirty: ${REPO_ROOT_LOCAL}" >&2
  exit 1
fi

remote() {
  local command
  printf -v command '%q ' "$@"
  ssh "${CLUSTER}" "${command}"
}

remote git -C "${REMOTE_REPO_ROOT}" pull --ff-only
if ! remote git -C "${SOURCE_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  remote mkdir -p "$(dirname "${SOURCE_ROOT}")"
  remote git -C "${REMOTE_REPO_ROOT}" worktree add --detach \
    "${SOURCE_ROOT}" "${SOURCE_COMMIT}"
fi
if [[ $(remote git -C "${SOURCE_ROOT}" rev-parse HEAD) != "${SOURCE_COMMIT}" ]] || \
  [[ -n $(remote git -C "${SOURCE_ROOT}" status --porcelain) ]]; then
  echo "Pinned vLLM source is not clean at ${SOURCE_COMMIT}" >&2
  exit 1
fi

remote git -C "${BENCH_REPO}" fetch origin exp/mxfp8-serving-crossover-20260829
if ! remote git -C "${BENCH_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  remote mkdir -p "$(dirname "${BENCH_ROOT}")"
  remote git -C "${BENCH_REPO}" worktree add --detach \
    "${BENCH_ROOT}" "${BENCH_COMMIT}"
fi
if [[ $(remote git -C "${BENCH_ROOT}" rev-parse HEAD) != "${BENCH_COMMIT}" ]] || \
  [[ -n $(remote git -C "${BENCH_ROOT}" status --porcelain) ]]; then
  echo "Pinned benchmark harness is not clean at ${BENCH_COMMIT}" >&2
  exit 1
fi
if [[ $(remote git -C "${FLASHINFER_ROOT}" rev-parse HEAD) != \
  "${FLASHINFER_COMMIT}" ]] || \
  [[ -n $(remote git -C "${FLASHINFER_ROOT}" status --porcelain) ]]; then
  echo "Pinned FlashInfer source is not clean at ${FLASHINFER_COMMIT}" >&2
  exit 1
fi

readonly PAIRED_DRIVER="${BENCH_ROOT}/experiments/eval/mxfp8_paired_driver.py"
readonly PAIR_BENCHMARK_SCRIPT="${BENCH_ROOT}/benchmark_vllm_bench_serve_static.py"
readonly LAUNCHER="${BENCH_ROOT}/experiments/backend_sweep_v0271/submit_mxfp8_linear_workloads_tp8.sh"
remote test -s "${PAIRED_DRIVER}"
remote test -s "${PAIR_BENCHMARK_SCRIPT}"
remote test -s "${LOOKUP_PATH}"

submit_order() {
  local order=$1
  local test_only=$2
  local order_slug pair_order output
  order_slug=${order//,/-then-}
  pair_order=${order//,/ }
  output=$(remote env \
    ACCOUNT="${ACCOUNT}" \
    PARTITION="${PARTITION}" \
    QOS="${QOS}" \
    PROFILE=main \
    BACKENDS=flashinfer_cutedsl \
    REPETITIONS=0 \
    STAMP_PREFIX="${STAMP}_${order_slug}" \
    RUN_ROOT_BASE="${RESULT_ROOT}/runs/${order_slug}" \
    SBATCH_OUT_DIR="${RESULT_ROOT}/slurm" \
    JOB_PREFIX="${ACCOUNT}-sna.mx-paired-${order_slug}" \
    WALLTIME="${WALLTIME}" \
    CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    VLLM_SOURCE_ROOT="${SOURCE_ROOT}" \
    ULTRA_MXFP8_MODEL="${MODEL_PATH}" \
    BENCH_PY="${PAIRED_DRIVER}" \
    JOB_CACHE_DIR=/raid/scratch \
    CONTAINER_MOUNTS=/home/sna:/home/sna,/raid/scratch:/raid/scratch \
    HF_HOME_OVERRIDE="/raid/scratch/sna/mxfp8_tp8_paired/${STAMP}/${order_slug}/hf" \
    RAY_INSTALL_IF_MISSING=1 \
    RAY_INSTALL_PACKAGE='ray[default]' \
    SBATCH_EXTRA_EXPORT_NAMES="${SBATCH_EXTRA_EXPORT_NAMES}" \
    SBATCH_EXPORT_MODE=file \
    SBATCH_TEST_ONLY="${test_only}" \
    DRY_RUN=0 \
    SCENARIOS_TO_RUN=shortin \
    BSIZES="${CONCURRENCIES}" \
    MULT="${WAVES}" \
    MAX_MODEL_LEN=12024 \
    MAX_NUM_BATCHED_TOKENS=16384 \
    SERVER_MAX_NUM_SEQS=32 \
    GPU_MEM=0.90 \
    ASYNC_SCHEDULING=0 \
    FORCE_DISABLE_ASYNC_SCHEDULING=1 \
    SHARED_SERVER=1 \
    STRICT_RESULT_TOKENS=1 \
    STRICT_WARMUP_TOKENS=0 \
    BENCH_TIMEOUT_S=14400 \
    SERVER_HEALTH_TIMEOUT_S=7200 \
    VLLM_MXFP8_TRTLLM_LAYOUT=adaptive \
    VLLM_MXFP8_TRTLLM_SWITCH_M=256 \
    VLLM_MXFP8_TRTLLM_TACTICS= \
    PYTHONPATH="${BENCH_ROOT}" \
    VLLM_SUBPROCESS_PYTHONPATH= \
    SOURCE_ROOT="${SOURCE_ROOT}" \
    FLASHINFER_ROOT="${FLASHINFER_ROOT}" \
    FLASHINFER_COMMIT="${FLASHINFER_COMMIT}" \
    EXPECTED_CONTAINER_SHA256="${EXPECTED_CONTAINER_SHA256}" \
    EXPECTED_VLLM_VERSION=0.27.1 \
    MXFP8_TACTIC_LOOKUP= \
    MXFP8_TACTIC_BACKEND=trtllm \
    MXFP8_TACTIC_SCALE_LAYOUT=adaptive \
    MXFP8_TACTIC_GPU='NVIDIA GB200' \
    PAIR_ORDER="${pair_order}" \
    PAIR_BENCHMARK_SCRIPT="${PAIR_BENCHMARK_SCRIPT}" \
    MXFP8_TACTIC_LOOKUP_PATH="${LOOKUP_PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    "${LAUNCHER}")
  printf '%s\n' "${output}"
}

print_plan
remote mkdir -p "${RESULT_ROOT}/slurm"
for order in ${ORDERS}; do
  echo "Preflight: ${order//,/ }"
  submit_order "${order}" 1
  echo "Submit: ${order//,/ }"
  output=$(submit_order "${order}" 0)
  printf '%s\n' "${output}"
  job_id=$(printf '%s\n' "${output}" | awk '/^[0-9]+(;.*)?$/ {print $1}' | tail -n 1)
  [[ -n "${job_id}" ]] || { echo "Could not parse job ID for ${order}" >&2; exit 1; }
  printf '%s\t%s\n' "${order//,/ }" "${job_id}" | \
    remote tee -a "${RESULT_ROOT}/jobs.tsv" >/dev/null
done

echo "result_root=${RESULT_ROOT}"
