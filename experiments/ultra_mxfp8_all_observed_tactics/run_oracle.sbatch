#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${FLASHINFER_ROOT:?Set FLASHINFER_ROOT}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"

export EXP_DIR="${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics"
export OBSERVED_DIR="${RESULT_ROOT}/observed"
export ORACLE_DIR="${RESULT_ROOT}/oracle"
export SHARD_COUNT=${SHARD_COUNT:-4}
export ROUNDS=${ROUNDS:-2}
export REPEAT_ITERS=${REPEAT_ITERS:-10}
readonly mounts=/home/sna:/home/sna,/lustre:/lustre,/raid/scratch:/raid/scratch

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
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
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
