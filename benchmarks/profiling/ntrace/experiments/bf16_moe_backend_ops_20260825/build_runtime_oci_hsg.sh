#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_DIR=$(cd "$(dirname "$0")" && pwd)
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-normal}
WALLTIME=${WALLTIME:-00:30:00}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/home:/home,/lustre:/lustre}
NTRACE_SOURCE=${NTRACE_SOURCE:-/home/sna/ntrace-e2etrain}
NTRACE_REVISION=${NTRACE_REVISION:-$(git -C "${NTRACE_SOURCE}" rev-parse --short=8 HEAD)}
NTRACE_RUNTIME=${NTRACE_RUNTIME:-/lustre/fsw/portfolios/coreai/users/sna/ntrace-vllm0271/runtime-${NTRACE_REVISION}-cuda13-py312-nonumpy}
NTRACE_PATCH=${NTRACE_PATCH:-${EXPERIMENT_DIR}/../../patches/ntrace-cuda13-cupti-nvtx-header.patch}
BUILD_INNER_SCRIPT=${BUILD_INNER_SCRIPT:-${EXPERIMENT_DIR}/../../build_runtime_lyris.sh}
LOG_DIR=${LOG_DIR:-${EXPERIMENT_DIR}/runtime_build_logs}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}

for path in \
  "${CONTAINER_IMAGE}" \
  "${NTRACE_SOURCE}" \
  "${NTRACE_PATCH}" \
  "${BUILD_INNER_SCRIPT}"; do
  if [[ ! -e ${path} ]]; then
    echo "ERROR: required path does not exist: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${LOG_DIR}"

inner_cmd=(
  srun
  --nodes=1
  --ntasks=1
  --container-image="${CONTAINER_IMAGE}"
  --container-mounts="${CONTAINER_MOUNTS}"
  --mpi=pmix
  bash
  "${BUILD_INNER_SCRIPT}"
  --inner
  "${NTRACE_SOURCE}"
  "${NTRACE_RUNTIME}"
  "${NTRACE_PATCH}"
)
printf -v wrapped_command '%q ' "${inner_cmd[@]}"
wrapped_command=${wrapped_command% }

cmd=(
  sbatch
  --parsable
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --qos="${QOS}"
  --nodes=1
  --ntasks-per-node=1
  --gpus-per-node=4
  --mem=0
  --time="${WALLTIME}"
  --job-name="${ACCOUNT}-${USER:-sna}.ntrace-build-py312"
  --output="${LOG_DIR}/build-${NTRACE_REVISION}-cuda13-py312-nonumpy-%j.log"
  --wrap="${wrapped_command}"
)
if [[ ${SBATCH_TEST_ONLY} == 1 ]]; then
  cmd+=(--test-only)
fi

if [[ ${DRY_RUN} == 1 ]]; then
  printf 'DRY_RUN:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
