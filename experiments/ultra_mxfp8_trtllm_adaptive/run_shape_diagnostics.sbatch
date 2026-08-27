#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=01:00:00

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"

M=${M:-16384}
CASE_TIMEOUT=${CASE_TIMEOUT:-180}
RUN_DIR="${RESULT_ROOT}/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

PATCH_MOUNTS=(
  "${SOURCE_ROOT}/vllm/envs.py:/usr/local/lib/python3.12/dist-packages/vllm/envs.py"
  "${SOURCE_ROOT}/vllm/utils/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/flashinfer.py"
  "${SOURCE_ROOT}/experiments/ultra_mxfp8_trtllm_adaptive/diagnose_128x4_shape.py:/workspace/diagnose_128x4_shape.py"
)
CONTAINER_MOUNTS="/lustre:/lustre,$(IFS=,; echo "${PATCH_MOUNTS[*]}")"

export M CASE_TIMEOUT RUN_DIR SOURCE_COMMIT

# shellcheck disable=SC2016
srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --mpi=pmix \
  bash -lc '
set -uo pipefail

export FLASHINFER_WORKSPACE_BASE=/tmp
export PYTHONPYCACHEPREFIX=/tmp/vllm-pycache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

echo "source_commit=${SOURCE_COMMIT}" >"${RUN_DIR}/metadata.txt"
echo "m=${M}" >>"${RUN_DIR}/metadata.txt"
echo "case_timeout=${CASE_TIMEOUT}" >>"${RUN_DIR}/metadata.txt"
vllm --version >>"${RUN_DIR}/metadata.txt"
python3 -c "import flashinfer; print(flashinfer.__version__)" >>"${RUN_DIR}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  >>"${RUN_DIR}/metadata.txt"

status=0
for shape in \
  8832:8192 \
  8192:4096 \
  2560:8192 \
  8192:2560 \
  2048:8192 \
  8192:2048 \
  2304:8192; do
  IFS=: read -r n k <<<"${shape}"
  log="${RUN_DIR}/shape_m${M}_n${n}_k${k}.log"
  echo "RUN shape=(${M},${n},${k})" | tee "${log}"
  timeout --signal=TERM --kill-after=15 "${CASE_TIMEOUT}" \
    python3 /workspace/diagnose_128x4_shape.py \
      --m "${M}" --n "${n}" --k "${k}" 2>&1 | tee -a "${log}"
  rc=${PIPESTATUS[0]}
  echo "EXIT_CODE=${rc}" | tee -a "${log}"
  if [[ "${rc}" -ne 0 ]]; then
    status=1
  fi
done

touch "${RUN_DIR}/COMPLETE"
exit "${status}"
'
