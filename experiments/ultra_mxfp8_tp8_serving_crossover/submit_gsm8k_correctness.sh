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
readonly TACTIC_RUNTIME_ROOT="${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics"
readonly BENCH_COMMIT=${BENCH_COMMIT:-85de736cd35e0873839fa6c6767b35acf9cb1b53}
readonly BENCH_REPO=${BENCH_REPO:-/home/sna/vllm-benchmark-layout-ncu/4f2ca9d1f0b1}
readonly BENCH_ROOT=${BENCH_ROOT:-/home/sna/vllm-benchmark-serving-crossover/${BENCH_COMMIT:0:12}}
readonly FLASHINFER_COMMIT=${FLASHINFER_COMMIT:-cec5e66dbd75a253edb5a819b2403bf410ca3223}
readonly FLASHINFER_ROOT=${FLASHINFER_ROOT:-/home/sna/flashinfer-mxfp8-pins/${FLASHINFER_COMMIT}}
readonly CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
readonly EXPECTED_CONTAINER_SHA256=${EXPECTED_CONTAINER_SHA256:-e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0}
readonly MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
readonly LOOKUP_PATH=${LOOKUP_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/mxfp8_tp8_serving_crossover_20260829_tp8census_r7_full/adaptive_oracle_20260829_tp8oracle_r1/adaptive_lookup.json}
readonly PHASE=${PHASE:-smoke}
readonly ARMS=${ARMS:-"cutedsl adaptive-lookup"}
readonly STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
readonly RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/mxfp8_tp8_gsm8k_${STAMP}}
readonly WALLTIME=${WALLTIME:-04:00:00}
readonly PRINT_PLAN=${PRINT_PLAN:-0}
readonly TRTLLM_SWITCH_M=${TRTLLM_SWITCH_M:-256}
readonly SBATCH_EXTRA_EXPORT_NAMES="SOURCE_ROOT FLASHINFER_ROOT FLASHINFER_COMMIT EXPECTED_CONTAINER_SHA256 EXPECTED_VLLM_VERSION MXFP8_TACTIC_LOOKUP MXFP8_TACTIC_BACKEND MXFP8_TACTIC_SCALE_LAYOUT MXFP8_TACTIC_GPU GSM8K_DATASET GSM8K_LIMIT GSM8K_TIMEOUT_SECONDS GSM8K_CONCURRENCY CORRECTNESS_ARM SOURCE_COMMIT PYTHONDONTWRITEBYTECODE"

case "${PHASE}" in
  smoke) readonly GSM8K_LIMIT=32 ;;
  full) readonly GSM8K_LIMIT=1319 ;;
  *) echo "PHASE must be smoke or full" >&2; exit 2 ;;
esac

arm_config() {
  case "$1" in
    cutedsl)
      ARM_LINEAR_BACKEND=flashinfer_cutedsl
      ARM_LAYOUT=8x4
      ARM_LOOKUP=none
      ;;
    adaptive)
      ARM_LINEAR_BACKEND=flashinfer_trtllm
      ARM_LAYOUT=adaptive
      ARM_LOOKUP=none
      ;;
    adaptive-lookup)
      ARM_LINEAR_BACKEND=flashinfer_trtllm
      ARM_LAYOUT=adaptive
      ARM_LOOKUP=exact
      ;;
    *) echo "Unsupported correctness arm: $1" >&2; return 2 ;;
  esac
}

print_plan() {
  local arm
  echo "phase=${PHASE}"
  echo "examples=${GSM8K_LIMIT}"
  echo "parallelism=TP8,DP1,EP8"
  echo "decoding=temperature0,seed0,max_tokens256"
  echo "cuda_graph=true"
  for arm in ${ARMS}; do
    arm_config "${arm}"
    echo "arm=${arm} linear_backend=${ARM_LINEAR_BACKEND} layout=${ARM_LAYOUT} lookup=${ARM_LOOKUP}"
  done
}

if [[ "${PRINT_PLAN}" == 1 ]]; then
  print_plan
  exit 0
fi

remote() {
  local command
  printf -v command '%q ' "$@"
  ssh "${CLUSTER}" "${command}"
}

if ! git -C "${REPO_ROOT_LOCAL}" diff --quiet || \
  ! git -C "${REPO_ROOT_LOCAL}" diff --cached --quiet || \
  [[ -n $(git -C "${REPO_ROOT_LOCAL}" ls-files --others --exclude-standard) ]]; then
  echo "Local vLLM worktree is dirty: ${REPO_ROOT_LOCAL}" >&2
  exit 1
fi

remote git -C "${REMOTE_REPO_ROOT}" pull --ff-only
if ! remote git -C "${SOURCE_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  remote mkdir -p "$(dirname "${SOURCE_ROOT}")"
  remote git -C "${REMOTE_REPO_ROOT}" worktree add --detach "${SOURCE_ROOT}" "${SOURCE_COMMIT}"
fi
if [[ $(remote git -C "${SOURCE_ROOT}" rev-parse HEAD) != "${SOURCE_COMMIT}" ]] || \
  [[ -n $(remote git -C "${SOURCE_ROOT}" status --porcelain) ]]; then
  echo "Pinned vLLM source is not clean at ${SOURCE_COMMIT}" >&2
  exit 1
fi

remote git -C "${BENCH_REPO}" fetch origin exp/mxfp8-serving-crossover-20260829
if ! remote git -C "${BENCH_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  remote mkdir -p "$(dirname "${BENCH_ROOT}")"
  remote git -C "${BENCH_REPO}" worktree add --detach "${BENCH_ROOT}" "${BENCH_COMMIT}"
fi
if [[ $(remote git -C "${BENCH_ROOT}" rev-parse HEAD) != "${BENCH_COMMIT}" ]] || \
  [[ -n $(remote git -C "${BENCH_ROOT}" status --porcelain) ]]; then
  echo "Pinned benchmark harness is not clean at ${BENCH_COMMIT}" >&2
  exit 1
fi

readonly GSM8K_DATASET="${BENCH_ROOT}/experiments/eval/data/gsm8k_test_openai_1319.jsonl"
readonly BENCH_PY="${BENCH_ROOT}/experiments/eval/gsm8k_v0271_driver.py"
readonly LAUNCHER="${BENCH_ROOT}/experiments/backend_sweep_v0271/submit_mxfp8_linear_workloads_tp8.sh"
remote test -s "${GSM8K_DATASET}"
remote test -s "${BENCH_PY}"
remote test -s "${LOOKUP_PATH}"
remote test -s "${TACTIC_RUNTIME_ROOT}/sitecustomize.py"

submit_arm() {
  local arm=$1
  local test_only=$2
  local output lookup
  arm_config "${arm}"
  lookup=
  if [[ "${ARM_LOOKUP}" == exact ]]; then
    lookup=${LOOKUP_PATH}
  fi
  output=$(remote env \
    ACCOUNT="${ACCOUNT}" \
    PARTITION="${PARTITION}" \
    QOS="${QOS}" \
    PROFILE=main \
    BACKENDS="${ARM_LINEAR_BACKEND}" \
    REPETITIONS=0 \
    STAMP_PREFIX="${STAMP}_${PHASE}_${arm}" \
    RUN_ROOT_BASE="${RESULT_ROOT}/${PHASE}/${arm}" \
    SBATCH_OUT_DIR="${RESULT_ROOT}/slurm" \
    JOB_PREFIX="${ACCOUNT}-sna.mx-gsm8k-${arm}" \
    WALLTIME="${WALLTIME}" \
    CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    VLLM_SOURCE_ROOT="${SOURCE_ROOT}" \
    ULTRA_MXFP8_MODEL="${MODEL_PATH}" \
    BENCH_PY="${BENCH_PY}" \
    JOB_CACHE_DIR=/raid/scratch \
    CONTAINER_MOUNTS=/home/sna:/home/sna,/raid/scratch:/raid/scratch \
    HF_HOME_OVERRIDE="/raid/scratch/sna/mxfp8_tp8_gsm8k/${STAMP}/${arm}/hf" \
    RAY_INSTALL_IF_MISSING=1 \
    RAY_INSTALL_PACKAGE='ray[default]' \
    SBATCH_EXTRA_EXPORT_NAMES="${SBATCH_EXTRA_EXPORT_NAMES}" \
    SBATCH_EXPORT_MODE=file \
    SBATCH_TEST_ONLY="${test_only}" \
    DRY_RUN=0 \
    SCENARIOS_TO_RUN=shortin \
    BSIZES=1 \
    MULT=1 \
    OSL_SHORT_VALUE=256 \
    MAX_MODEL_LEN=12024 \
    MAX_NUM_BATCHED_TOKENS=16384 \
    SERVER_MAX_NUM_SEQS=8 \
    ASYNC_SCHEDULING=0 \
    FORCE_DISABLE_ASYNC_SCHEDULING=1 \
    ENABLE_FLASHINFER_AUTOTUNE=0 \
    SHARED_SERVER=1 \
    SERVER_HEALTH_TIMEOUT_S=7200 \
    VLLM_MXFP8_TRTLLM_LAYOUT="${ARM_LAYOUT}" \
    VLLM_MXFP8_TRTLLM_SWITCH_M="${TRTLLM_SWITCH_M}" \
    VLLM_MXFP8_TRTLLM_TACTICS= \
    PYTHONPATH="${TACTIC_RUNTIME_ROOT}:${BENCH_ROOT}" \
    VLLM_SUBPROCESS_PYTHONPATH= \
    SOURCE_ROOT="${SOURCE_ROOT}" \
    SOURCE_COMMIT="${SOURCE_COMMIT}" \
    FLASHINFER_ROOT="${FLASHINFER_ROOT}" \
    FLASHINFER_COMMIT="${FLASHINFER_COMMIT}" \
    EXPECTED_CONTAINER_SHA256="${EXPECTED_CONTAINER_SHA256}" \
    EXPECTED_VLLM_VERSION=0.27.1 \
    MXFP8_TACTIC_LOOKUP="${lookup}" \
    MXFP8_TACTIC_BACKEND=trtllm \
    MXFP8_TACTIC_SCALE_LAYOUT="${ARM_LAYOUT}" \
    MXFP8_TACTIC_GPU='NVIDIA GB200' \
    GSM8K_DATASET="${GSM8K_DATASET}" \
    GSM8K_LIMIT="${GSM8K_LIMIT}" \
    GSM8K_TIMEOUT_SECONDS=90 \
    GSM8K_CONCURRENCY=8 \
    CORRECTNESS_ARM="${arm}" \
    PYTHONDONTWRITEBYTECODE=1 \
    "${LAUNCHER}")
  printf '%s\n' "${output}"
}

print_plan
remote mkdir -p "${RESULT_ROOT}/slurm"
for arm in ${ARMS}; do
  echo "Preflight: arm=${arm} phase=${PHASE}"
  submit_arm "${arm}" 1
  echo "Submit: arm=${arm} phase=${PHASE}"
  output=$(submit_arm "${arm}" 0)
  printf '%s\n' "${output}"
  job_id=$(printf '%s\n' "${output}" | awk '/^[0-9]+(;.*)?$/ {print $1}' | tail -n 1)
  [[ -n "${job_id}" ]] || { echo "Could not parse job ID for ${arm}" >&2; exit 1; }
  echo "arm=${arm} phase=${PHASE} job_id=${job_id}" | remote tee -a "${RESULT_ROOT}/jobs.txt" >/dev/null
done

echo "result_root=${RESULT_ROOT}"
