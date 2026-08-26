#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"
: "${EXPECTED_CONTAINER_SHA256:?Set EXPECTED_CONTAINER_SHA256}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
: "${FLASHINFER_ROOT:?Set FLASHINFER_ROOT}"
: "${FLASHINFER_COMMIT:?Set FLASHINFER_COMMIT}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"
: "${ORACLE_BACKEND:?Set ORACLE_BACKEND}"
: "${SCALE_LAYOUT:?Set SCALE_LAYOUT}"

export EXP_DIR="${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics"
export OBSERVED_DIR="${RESULT_ROOT}/observed"
export ORACLE_DIR="${RESULT_ROOT}/oracle"
export SHARD_COUNT=${SHARD_COUNT:-4}
export ROUNDS=${ROUNDS:-2}
export REPEAT_ITERS=${REPEAT_ITERS:-10}
export ORACLE_BACKEND SCALE_LAYOUT FLASHINFER_COMMIT
readonly mounts=/home/sna:/home/sna,/lustre:/lustre,/raid/scratch:/raid/scratch

for checkout in "${SOURCE_ROOT}:${SOURCE_COMMIT}:vLLM" \
  "${FLASHINFER_ROOT}:${FLASHINFER_COMMIT}:FlashInfer"; do
  IFS=: read -r root expected name <<<"${checkout}"
  actual=$(git -C "${root}" rev-parse HEAD)
  if [[ "${actual}" != "${expected}" ]] || \
    [[ -n $(git -C "${root}" status --porcelain) ]]; then
    echo "${name} checkout is not clean at ${expected}: ${root}" >&2
    exit 1
  fi
done
container_sha=$(awk -F= '$1 == "sha256" {print $2}' \
  "${CONTAINER_IMAGE}.metadata.txt")
if [[ "${container_sha}" != "${EXPECTED_CONTAINER_SHA256}" ]]; then
  echo "Expected container ${EXPECTED_CONTAINER_SHA256}, got ${container_sha}" >&2
  exit 1
fi

mkdir -p "${OBSERVED_DIR}" "${ORACLE_DIR}/shards"
srun --ntasks=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${mounts}" \
  --container-workdir=/tmp \
  --mpi=pmix \
  bash "${EXP_DIR}/prepare_oracle_inputs.sh"

srun \
  --ntasks="${SHARD_COUNT}" \
  --ntasks-per-node="${SHARD_COUNT}" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${mounts}" \
  --container-workdir="${FLASHINFER_ROOT}" \
  --mpi=pmix \
  bash "${EXP_DIR}/oracle_worker.sh"

srun --ntasks=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${mounts}" \
  --container-workdir=/tmp \
  --mpi=pmix \
  bash "${EXP_DIR}/finalize_lookup.sh"
