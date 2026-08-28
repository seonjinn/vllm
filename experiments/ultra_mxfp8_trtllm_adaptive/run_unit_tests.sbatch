#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:10:00

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"

RUN_DIR="${RESULT_ROOT}/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

PATCH_MOUNTS=(
  "${SOURCE_ROOT}/vllm/envs.py:/usr/local/lib/python3.12/dist-packages/vllm/envs.py"
  "${SOURCE_ROOT}/vllm/model_executor/kernels/linear/__init__.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/__init__.py"
  "${SOURCE_ROOT}/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
  "${SOURCE_ROOT}/vllm/model_executor/warmup/kernel_warmup.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/kernel_warmup.py"
  "${SOURCE_ROOT}/vllm/utils/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/flashinfer.py"
  "${SOURCE_ROOT}/tests/model_executor/test_flashinfer_autotune_policy.py:/workspace/test_flashinfer_autotune_policy.py"
  "${SOURCE_ROOT}/tests/kernels/quantization/test_mxfp8_trtllm_linear.py:/workspace/test_mxfp8_trtllm_linear.py"
)
CONTAINER_MOUNTS="/lustre:/lustre,$(IFS=,; echo "${PATCH_MOUNTS[*]}")"

export PYTHONPYCACHEPREFIX=/tmp/vllm-pycache

printf 'source_commit=%s\n' "${SOURCE_COMMIT}" >"${RUN_DIR}/metadata.txt"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --mpi=pmix \
  bash -lc \
  'python3 -m pytest -q /workspace/test_flashinfer_autotune_policy.py /workspace/test_mxfp8_trtllm_linear.py' \
  2>&1 | tee "${RUN_DIR}/pytest.log"

touch "${RUN_DIR}/COMPLETE"
