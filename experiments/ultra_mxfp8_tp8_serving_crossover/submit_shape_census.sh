#!/usr/bin/env bash

set -Eeuo pipefail

readonly EXP_DIR_LOCAL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly REPO_ROOT_LOCAL=$(cd "${EXP_DIR_LOCAL}/../.." && pwd)
readonly CLUSTER=${CLUSTER:-login-lyris}
readonly ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
readonly PARTITION=${PARTITION:-gb200}
readonly QOS=${QOS:-user-restrictions}
readonly REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-/home/sna/vllm-v0271-ultra-tp8-crossover}
readonly BENCH_COMMIT=${BENCH_COMMIT:-5504d071e6c082f7dc82347c8eabfc9cad9740ef}
readonly BENCH_ROOT=${BENCH_ROOT:-/home/sna/vllm-benchmark-serving-crossover/${BENCH_COMMIT:0:12}}
readonly FLASHINFER_ROOT=${FLASHINFER_ROOT:-/home/sna/flashinfer-mxfp8-pins/cec5e66dbd75a253edb5a819b2403bf410ca3223}
readonly FLASHINFER_COMMIT=${FLASHINFER_COMMIT:-cec5e66dbd75a253edb5a819b2403bf410ca3223}
readonly CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
readonly EXPECTED_CONTAINER_SHA256=${EXPECTED_CONTAINER_SHA256:-e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0}
readonly MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
readonly SOURCE_COMMIT=${SOURCE_COMMIT:-$(git -C "${REPO_ROOT_LOCAL}" rev-parse HEAD)}
readonly SOURCE_ROOT=${SOURCE_ROOT:-/home/sna/vllm-v0271-mxfp8-pins/${SOURCE_COMMIT}}
readonly STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
readonly RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/mxfp8_tp8_serving_crossover_${STAMP}}
readonly PHASES=${PHASES:-"low high-smoke"}
readonly PRINT_PLAN=${PRINT_PLAN:-0}
readonly SBATCH_EXTRA_EXPORT_NAMES="SOURCE_ROOT FLASHINFER_ROOT FLASHINFER_COMMIT EXPECTED_CONTAINER_SHA256 EXPECTED_VLLM_VERSION MXFP8_TACTIC_TRACE_DIR MXFP8_TACTIC_TRACE_PHASE MXFP8_TACTIC_BACKEND MXFP8_TACTIC_SCALE_LAYOUT"

phase_config() {
  case "$1" in
    low)
      PHASE_BSIZES="1 2 4 8 16 32"
      PHASE_MULT=10
      PHASE_WALLTIME=04:00:00
      ;;
    high-smoke)
      PHASE_BSIZES="128 512"
      PHASE_MULT=1
      PHASE_WALLTIME=04:00:00
      ;;
    *)
      echo "Unsupported phase: $1" >&2
      return 2
      ;;
  esac
}

print_plan() {
  local phase
  echo "model=${MODEL_PATH}"
  echo "parallelism=TP8,DP1,EP8"
  echo "workload=ISL1000,OSL10000"
  echo "layout=adaptive,switch_m=256"
  echo "ray_install_if_missing=1,node_local_tmp=true"
  echo "benchmark_commit=${BENCH_COMMIT}"
  echo "slurm_extra_exports=${SBATCH_EXTRA_EXPORT_NAMES}"
  for phase in ${PHASES}; do
    phase_config "${phase}"
    echo "phase=${phase} concurrencies=${PHASE_BSIZES} waves=${PHASE_MULT}"
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
remote_sha=$(remote git -C "${REMOTE_REPO_ROOT}" rev-parse HEAD)
if [[ "${remote_sha}" != "${SOURCE_COMMIT}" ]] || \
  [[ -n $(remote git -C "${REMOTE_REPO_ROOT}" status --porcelain) ]]; then
  echo "Remote vLLM checkout is not clean at ${SOURCE_COMMIT}" >&2
  exit 1
fi
if ! remote git -C "${SOURCE_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  remote mkdir -p "$(dirname "${SOURCE_ROOT}")"
  remote git -C "${REMOTE_REPO_ROOT}" worktree add \
    --detach "${SOURCE_ROOT}" "${SOURCE_COMMIT}"
fi
if [[ $(remote git -C "${SOURCE_ROOT}" rev-parse HEAD) != "${SOURCE_COMMIT}" ]] || \
  [[ -n $(remote git -C "${SOURCE_ROOT}" status --porcelain) ]]; then
  echo "Pinned vLLM source is not clean at ${SOURCE_COMMIT}" >&2
  exit 1
fi
if [[ $(remote git -C "${BENCH_ROOT}" rev-parse HEAD) != "${BENCH_COMMIT}" ]] || \
  [[ -n $(remote git -C "${BENCH_ROOT}" status --porcelain) ]]; then
  echo "Benchmark harness is not clean at ${BENCH_COMMIT}" >&2
  exit 1
fi
if [[ $(remote git -C "${FLASHINFER_ROOT}" rev-parse HEAD) != \
  "${FLASHINFER_COMMIT}" ]] || \
  [[ -n $(remote git -C "${FLASHINFER_ROOT}" status --porcelain) ]]; then
  echo "FlashInfer checkout is not clean at ${FLASHINFER_COMMIT}" >&2
  exit 1
fi
if [[ $(remote awk -F= '$1 == "sha256" {print $2}' \
  "${CONTAINER_IMAGE}.metadata.txt") != "${EXPECTED_CONTAINER_SHA256}" ]]; then
  echo "Container metadata does not match ${EXPECTED_CONTAINER_SHA256}" >&2
  exit 1
fi
actual_container_sha=$(remote sha256sum "${CONTAINER_IMAGE}" | awk '{print $1}')
if [[ "${actual_container_sha}" != "${EXPECTED_CONTAINER_SHA256}" ]]; then
  echo "Container bytes do not match ${EXPECTED_CONTAINER_SHA256}" >&2
  exit 1
fi
remote test -f "${MODEL_PATH}/config.json"
model_config_sha=$(remote sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')
model_index_sha=$(remote sha256sum \
  "${MODEL_PATH}/model.safetensors.index.json" | awk '{print $1}')

readonly EXP_DIR_REMOTE="${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics"
readonly LAUNCHER="${BENCH_ROOT}/experiments/backend_sweep_v0271/submit_mxfp8_linear_workloads_tp8.sh"

submit_phase() {
  local phase=$1
  local test_only=$2
  local phase_stamp phase_root run_root trace_dir max_num_seqs output
  phase_config "${phase}"
  phase_stamp="${STAMP}_${phase}"
  phase_root="${RESULT_ROOT}/${phase}"
  run_root="${phase_root}/${phase_stamp}_main_flashinfer_trtllm_rep0"
  trace_dir="${run_root}/traces"
  max_num_seqs=${PHASE_BSIZES##* }

  output=$(remote env \
    ACCOUNT="${ACCOUNT}" \
    PARTITION="${PARTITION}" \
    QOS="${QOS}" \
    PROFILE=main \
    BACKENDS=flashinfer_trtllm \
    REPETITIONS=0 \
    STAMP_PREFIX="${phase_stamp}" \
    RUN_ROOT_BASE="${phase_root}" \
    SBATCH_OUT_DIR="${RESULT_ROOT}/slurm" \
    JOB_PREFIX="${ACCOUNT}-sna.mx-tp8-census-${phase}" \
    SUBMIT_DELAY_S=0 \
    WALLTIME="${PHASE_WALLTIME}" \
    CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    VLLM_SOURCE_ROOT="${SOURCE_ROOT}" \
    ULTRA_MXFP8_MODEL="${MODEL_PATH}" \
    JOB_CACHE_DIR=/raid/scratch \
    CONTAINER_MOUNTS=/home/sna:/home/sna,/raid/scratch:/raid/scratch \
    HF_HOME_OVERRIDE="/raid/scratch/sna/mxfp8_tp8_crossover/${STAMP}/hf" \
    RAY_INSTALL_IF_MISSING=1 \
    RAY_INSTALL_PACKAGE='ray[default]' \
    SBATCH_EXTRA_EXPORT_NAMES="${SBATCH_EXTRA_EXPORT_NAMES}" \
    SBATCH_EXPORT_MODE=all \
    SBATCH_TEST_ONLY="${test_only}" \
    DRY_RUN=0 \
    SCENARIOS_TO_RUN=shortin \
    BSIZES="${PHASE_BSIZES}" \
    MULT="${PHASE_MULT}" \
    MAX_MODEL_LEN=12024 \
    MAX_NUM_BATCHED_TOKENS=16384 \
    SERVER_MAX_NUM_SEQS="${max_num_seqs}" \
    GPU_MEM=0.95 \
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
    PYTHONPATH="${EXP_DIR_REMOTE}:${SOURCE_ROOT}" \
    VLLM_SUBPROCESS_PYTHONPATH="${SOURCE_ROOT}" \
    SOURCE_ROOT="${SOURCE_ROOT}" \
    FLASHINFER_ROOT="${FLASHINFER_ROOT}" \
    FLASHINFER_COMMIT="${FLASHINFER_COMMIT}" \
    EXPECTED_CONTAINER_SHA256="${EXPECTED_CONTAINER_SHA256}" \
    EXPECTED_VLLM_VERSION=0.27.1 \
    MXFP8_TACTIC_TRACE_DIR="${trace_dir}" \
    MXFP8_TACTIC_TRACE_PHASE="${phase}" \
    MXFP8_TACTIC_BACKEND=trtllm \
    MXFP8_TACTIC_SCALE_LAYOUT=adaptive \
    "${LAUNCHER}")
  printf '%s\n' "${output}"
}

print_plan
remote mkdir -p "${RESULT_ROOT}"
remote mkdir -p "${RESULT_ROOT}/slurm"
{
  echo "source_commit=${SOURCE_COMMIT}"
  echo "benchmark_commit=${BENCH_COMMIT}"
  echo "flashinfer_commit=${FLASHINFER_COMMIT}"
  echo "container=${CONTAINER_IMAGE}"
  echo "container_sha256=${actual_container_sha}"
  echo "model=${MODEL_PATH}"
  echo "model_config_sha256=${model_config_sha}"
  echo "model_index_sha256=${model_index_sha}"
  echo "parallelism=TP8,DP1,EP8"
  echo "workload=ISL1000,OSL10000"
  echo "layout=adaptive,switch_m=256"
  echo "phases=${PHASES}"
} | remote tee "${RESULT_ROOT}/manifest.txt" >/dev/null
for phase in ${PHASES}; do
  echo "Preflight: ${phase}"
  submit_phase "${phase}" 1
  echo "Submit: ${phase}"
  submit_phase "${phase}" 0
done

echo "result_root=${RESULT_ROOT}"
