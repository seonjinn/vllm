#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"
: "${EXPECTED_CONTAINER_SHA256:?Set EXPECTED_CONTAINER_SHA256}"
: "${EXPECTED_CONTAINER_SIZE:?Set EXPECTED_CONTAINER_SIZE}"
: "${EXPECTED_CONTAINER_MTIME:?Set EXPECTED_CONTAINER_MTIME}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
: "${FLASHINFER_ROOT:?Set FLASHINFER_ROOT}"
: "${FLASHINFER_COMMIT:?Set FLASHINFER_COMMIT}"
: "${SPLIT_DIR:?Set SPLIT_DIR}"
: "${ORACLE_RESULT_ROOT:?Set ORACLE_RESULT_ROOT}"
: "${SCALE_LAYOUT:?Set SCALE_LAYOUT}"

case "${SCALE_LAYOUT}" in
  8x4 | 128x4) ;;
  *)
    echo "Unsupported SCALE_LAYOUT=${SCALE_LAYOUT}" >&2
    exit 2
    ;;
esac

export EXP_DIR="${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics"
export OBSERVED_DIR="${ORACLE_RESULT_ROOT}/${SCALE_LAYOUT}/observed"
export ORACLE_DIR="${ORACLE_RESULT_ROOT}/${SCALE_LAYOUT}/oracle"
export SHARD_COUNT=${SHARD_COUNT:-4}
export ROUNDS=${ROUNDS:-2}
export REPEAT_ITERS=${REPEAT_ITERS:-10}
export ORACLE_BACKEND=trtllm
export FLASHINFER_COMMIT SCALE_LAYOUT
readonly observed_source="${SPLIT_DIR}/observed_${SCALE_LAYOUT}.csv"
readonly mounts=/home/sna:/home/sna,/lustre:/lustre,/raid/scratch:/raid/scratch

if [[ "${SHARD_COUNT}" != 4 ]]; then
  echo "SHARD_COUNT must match the four GPUs on one Lyris GB200 node" >&2
  exit 2
fi
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
read -r container_size container_mtime < <(stat -c '%s %Y' "${CONTAINER_IMAGE}")
if [[ "${container_sha}" != "${EXPECTED_CONTAINER_SHA256}" ]] || \
  [[ "${container_size}" != "${EXPECTED_CONTAINER_SIZE}" ]] || \
  [[ "${container_mtime}" != "${EXPECTED_CONTAINER_MTIME}" ]]; then
  echo "Container changed after submission" >&2
  exit 1
fi
if [[ ! -s "${observed_source}" ]]; then
  echo "Observed layout CSV is missing: ${observed_source}" >&2
  exit 1
fi
if [[ -e "${ORACLE_RESULT_ROOT}/${SCALE_LAYOUT}/COMPLETE" ]]; then
  echo "Refusing to overwrite completed ${SCALE_LAYOUT} oracle" >&2
  exit 1
fi

mkdir -p "${OBSERVED_DIR}/shards" "${ORACLE_DIR}/shards"
cp "${observed_source}" "${OBSERVED_DIR}/observed_shapes.csv"
python3 "${EXP_DIR}/shard_shapes.py" \
  --observed "${OBSERVED_DIR}/observed_shapes.csv" \
  --output-dir "${OBSERVED_DIR}/shards" \
  --shards "${SHARD_COUNT}"

srun \
  --ntasks="${SHARD_COUNT}" \
  --ntasks-per-node="${SHARD_COUNT}" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${mounts}" \
  --container-workdir=/tmp \
  --mpi=pmix \
  bash "${EXP_DIR}/oracle_worker.sh"

srun \
  --ntasks=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${mounts}" \
  --container-workdir=/tmp \
  --mpi=pmix \
  bash "${EXP_DIR}/finalize_lookup.sh"
touch "${ORACLE_RESULT_ROOT}/${SCALE_LAYOUT}/COMPLETE"
