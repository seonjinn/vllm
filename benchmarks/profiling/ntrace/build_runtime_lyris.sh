#!/usr/bin/env bash

set -euo pipefail

build_runtime() {
  local source_root=$1
  local runtime=$2
  local tmp_runtime="${runtime}.tmp.${SLURM_JOB_ID}"

  rm -rf "${tmp_runtime}"
  export NTRACE_BUILD_CUPTI_CPP=ON
  export NTRACE_REQUIRE_CUXXFILT=ON
  export CMAKE_ARGS='-DCUDA_INCLUDE_DIR=/usr/local/cuda-13.0/targets/sbsa-linux/include -DCUDART_LIBRARY=/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcudart.so -DCUPTI_INCLUDE_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include -DCUPTI_LIBRARY=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13'
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

  uv pip install \
    --python /usr/bin/python3 \
    --target "${tmp_runtime}" \
    "${source_root}"
  PYTHONPATH="${tmp_runtime}" /usr/bin/python3 - <<'PY'
from pathlib import Path

import ntrace
import pyarrow
from ntrace.backends import get_backend

runtime = Path(ntrace.__file__).parent.parent
native = list((runtime / "ntrace").glob("_cupti_cpp*.so"))
assert native, f"missing C++ CUPTI backend under {runtime}"
print(f"ntrace={ntrace.__file__}")
print(f"pyarrow={pyarrow.__version__}")
print(f"backend={get_backend()}")
print(f"native={native[0]}")
PY

  rm -rf "${runtime}"
  mv "${tmp_runtime}" "${runtime}"
  echo "ntrace_runtime=${runtime}"
}

if [[ ${1:-} == --inner ]]; then
  build_runtime "$2" "$3"
  exit
fi

ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-gb200}
QOS=${QOS:-user-restrictions}
WALLTIME=${WALLTIME:-00:30:00}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64.sqsh}
NTRACE_SOURCE=${NTRACE_SOURCE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ntrace-vllm0271/source}
NTRACE_REVISION=${NTRACE_REVISION:-$(git -C "${NTRACE_SOURCE}" rev-parse --short=8 HEAD)}
NTRACE_RUNTIME=${NTRACE_RUNTIME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ntrace-vllm0271/runtime-${NTRACE_REVISION}-py312}
LOG_DIR=${LOG_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ntrace-vllm0271}
DRY_RUN=${DRY_RUN:-0}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}

mkdir -p "${LOG_DIR}"

script_path=$(realpath "${BASH_SOURCE[0]}")
printf -v script_path_q '%q' "${script_path}"
printf -v ntrace_source_q '%q' "${NTRACE_SOURCE}"
printf -v ntrace_runtime_q '%q' "${NTRACE_RUNTIME}"
build_command="bash ${script_path_q} --inner ${ntrace_source_q} ${ntrace_runtime_q}"

cmd=(
  sbatch --parsable
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --qos="${QOS}"
  --nodes=1
  --time="${WALLTIME}"
  --job-name="${ACCOUNT}-sna.ntrace-build-py312"
  --output="${LOG_DIR}/build-${NTRACE_REVISION}-py312-%j.log"
  --container-image="${CONTAINER_IMAGE}"
  --container-mounts="/lustre:/lustre"
  --wrap="${build_command}"
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
