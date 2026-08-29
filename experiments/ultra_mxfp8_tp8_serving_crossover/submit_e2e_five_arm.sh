#!/usr/bin/env bash

set -Eeuo pipefail

readonly EXP_DIR_LOCAL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly REPO_ROOT_LOCAL=$(cd "${EXP_DIR_LOCAL}/../.." && pwd)
readonly CLUSTER=${CLUSTER:-login-lyris}
readonly ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
readonly PARTITION=${PARTITION:-gb200-backfill}
readonly QOS=${QOS:-user-restrictions}
readonly REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-/home/sna/vllm-v0271-ultra-tp8-crossover}
readonly BENCH_COMMIT=${BENCH_COMMIT:-5e3594d699a7b886fde5544912c9116d06858182}
readonly BENCH_ROOT=${BENCH_ROOT:-/home/sna/vllm-benchmark-serving-crossover/${BENCH_COMMIT:0:12}}
readonly FLASHINFER_COMMIT=${FLASHINFER_COMMIT:-cec5e66dbd75a253edb5a819b2403bf410ca3223}
readonly FLASHINFER_ROOT=${FLASHINFER_ROOT:-/home/sna/flashinfer-mxfp8-pins/${FLASHINFER_COMMIT}}
readonly CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
readonly EXPECTED_CONTAINER_SHA256=${EXPECTED_CONTAINER_SHA256:-e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0}
readonly MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
readonly LOOKUP_PATH=${LOOKUP_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/mxfp8_tp8_serving_crossover_20260829_tp8census_r7_full/adaptive_oracle_20260829_tp8oracle_r1/adaptive_lookup.json}
readonly SOURCE_COMMIT=${SOURCE_COMMIT:-$(git -C "${REPO_ROOT_LOCAL}" rev-parse HEAD)}
readonly SOURCE_ROOT=${SOURCE_ROOT:-/home/sna/vllm-v0271-mxfp8-pins/${SOURCE_COMMIT}}
readonly STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
readonly RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/mxfp8_tp8_serving_crossover_e2e_${STAMP}}
readonly ARMS=${ARMS:-"cutedsl trtllm-8x4 trtllm-128x4 adaptive adaptive-lookup"}
readonly REPETITIONS=${REPETITIONS:-"0"}
readonly CONCURRENCIES=${CONCURRENCIES:-"1 2 4 8 16 32"}
readonly WAVES=${WAVES:-10}
readonly WALLTIME=${WALLTIME:-04:00:00}
readonly PRINT_PLAN=${PRINT_PLAN:-0}
readonly TRTLLM_SWITCH_M=${TRTLLM_SWITCH_M:-256}
readonly SBATCH_EXTRA_EXPORT_NAMES="SOURCE_ROOT FLASHINFER_ROOT FLASHINFER_COMMIT EXPECTED_CONTAINER_SHA256 EXPECTED_VLLM_VERSION MXFP8_TACTIC_LOOKUP MXFP8_TACTIC_BACKEND MXFP8_TACTIC_SCALE_LAYOUT MXFP8_TACTIC_GPU"

arm_config() {
  case "$1" in
    cutedsl)
      ARM_LINEAR_BACKEND=flashinfer_cutedsl
      ARM_LAYOUT=8x4
      ARM_LOOKUP=none
      ;;
    trtllm-8x4)
      ARM_LINEAR_BACKEND=flashinfer_trtllm
      ARM_LAYOUT=8x4
      ARM_LOOKUP=none
      ;;
    trtllm-128x4)
      ARM_LINEAR_BACKEND=flashinfer_trtllm
      ARM_LAYOUT=128x4
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
    *)
      echo "Unsupported arm: $1" >&2
      return 2
      ;;
  esac
}

print_plan() {
  local arm
  echo "model=${MODEL_PATH}"
  echo "parallelism=TP8,DP1,EP8"
  echo "workload=ISL1000,OSL10000"
  echo "concurrencies=${CONCURRENCIES}"
  echo "waves=${WAVES}"
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
if ((WAVES <= 0)); then
  echo "WAVES must be positive" >&2
  exit 2
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
  remote git -C "${REMOTE_REPO_ROOT}" worktree add --detach \
    "${SOURCE_ROOT}" "${SOURCE_COMMIT}"
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
remote test -s "${LOOKUP_PATH}"
lookup_sha=$(remote sha256sum "${LOOKUP_PATH}" | awk '{print $1}')
lookup_entries=$(remote jq -r .entry_count "${LOOKUP_PATH}")
if [[ "${lookup_entries}" != 60 ]]; then
  echo "Expected 60 lookup entries, got ${lookup_entries}" >&2
  exit 1
fi

readonly EXP_DIR_REMOTE="${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics"
readonly LAUNCHER="${BENCH_ROOT}/experiments/backend_sweep_v0271/submit_mxfp8_linear_workloads_tp8.sh"

submit_arm() {
  local arm=$1
  local repetition=$2
  local test_only=$3
  local arm_root stamp output lookup
  arm_config "${arm}"
  arm_root="${RESULT_ROOT}/runs/${arm}/rep${repetition}"
  stamp="${STAMP}_${arm}_rep${repetition}"
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
    STAMP_PREFIX="${stamp}" \
    RUN_ROOT_BASE="${arm_root}" \
    SBATCH_OUT_DIR="${RESULT_ROOT}/slurm" \
    JOB_PREFIX="${ACCOUNT}-sna.mx-e2e-${arm}" \
    SUBMIT_DELAY_S=0 \
    WALLTIME="${WALLTIME}" \
    CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    VLLM_SOURCE_ROOT="${SOURCE_ROOT}" \
    ULTRA_MXFP8_MODEL="${MODEL_PATH}" \
    JOB_CACHE_DIR=/raid/scratch \
    CONTAINER_MOUNTS=/home/sna:/home/sna,/raid/scratch:/raid/scratch \
    HF_HOME_OVERRIDE="/raid/scratch/sna/mxfp8_tp8_e2e/${STAMP}/${arm}/rep${repetition}/hf" \
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
    ASYNC_SCHEDULING=0 \
    FORCE_DISABLE_ASYNC_SCHEDULING=1 \
    SHARED_SERVER=1 \
    STRICT_RESULT_TOKENS=1 \
    STRICT_WARMUP_TOKENS=0 \
    BENCH_TIMEOUT_S=18000 \
    SERVER_HEALTH_TIMEOUT_S=7200 \
    VLLM_MXFP8_TRTLLM_LAYOUT="${ARM_LAYOUT}" \
    VLLM_MXFP8_TRTLLM_SWITCH_M="${TRTLLM_SWITCH_M}" \
    VLLM_MXFP8_TRTLLM_TACTICS= \
    PYTHONPATH="${EXP_DIR_REMOTE}:${SOURCE_ROOT}" \
    VLLM_SUBPROCESS_PYTHONPATH="${SOURCE_ROOT}" \
    SOURCE_ROOT="${SOURCE_ROOT}" \
    FLASHINFER_ROOT="${FLASHINFER_ROOT}" \
    FLASHINFER_COMMIT="${FLASHINFER_COMMIT}" \
    EXPECTED_CONTAINER_SHA256="${EXPECTED_CONTAINER_SHA256}" \
    EXPECTED_VLLM_VERSION=0.27.1 \
    MXFP8_TACTIC_LOOKUP="${lookup}" \
    MXFP8_TACTIC_BACKEND=trtllm \
    MXFP8_TACTIC_SCALE_LAYOUT="${ARM_LAYOUT}" \
    MXFP8_TACTIC_GPU='NVIDIA GB200' \
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
  echo "parallelism=TP8,DP1,EP8"
  echo "workload=ISL1000,OSL10000"
  echo "concurrencies=${CONCURRENCIES}"
  echo "waves=${WAVES}"
  echo "lookup=${LOOKUP_PATH}"
  echo "lookup_sha256=${lookup_sha}"
  echo "lookup_entries=${lookup_entries}"
} | remote tee "${RESULT_ROOT}/manifest.txt" >/dev/null

for repetition in ${REPETITIONS}; do
  for arm in ${ARMS}; do
    echo "Preflight: arm=${arm} repetition=${repetition}"
    submit_arm "${arm}" "${repetition}" 1
    echo "Submit: arm=${arm} repetition=${repetition}"
    output=$(submit_arm "${arm}" "${repetition}" 0)
    printf '%s\n' "${output}"
    job_id=$(printf '%s\n' "${output}" | awk '/^[0-9]+(;.*)?$/ {print $1}' | tail -n 1)
    if [[ -z "${job_id}" ]]; then
      echo "Could not parse job ID for ${arm} repetition ${repetition}" >&2
      exit 1
    fi
    echo "arm=${arm} repetition=${repetition} job_id=${job_id}" | \
      remote tee -a "${RESULT_ROOT}/jobs.txt" >/dev/null
  done
done

echo "result_root=${RESULT_ROOT}"
