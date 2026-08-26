#!/usr/bin/env bash

set -euo pipefail

readonly shard=${SLURM_LOCALID}
readonly gpu_index=${SLURM_LOCALID}
readonly scratch="/raid/scratch/${USER}/mxfp8_observed_oracle/${SLURM_JOB_ID}/${shard}"
readonly shape_file="${OBSERVED_DIR}/shards/shard_${shard}.csv"
readonly output_dir="${ORACLE_DIR}/shards/${shard}"
readonly harness_root="${scratch}/harness"
mkdir -p "${harness_root}/benchmarks" "${output_dir}"
trap 'rm -rf "${scratch}"' EXIT
if [[ $(wc -l <"${shape_file}") -le 1 ]]; then
  echo "shard ${shard} is empty"
  exit 0
fi

cp "${FLASHINFER_ROOT}/benchmarks/bench_mxfp8_backend_tactic_oracle.py" \
  "${FLASHINFER_ROOT}/benchmarks/bench_cutedsl_mxfp8_serving_shapes.py" \
  "${harness_root}/benchmarks/"
touch "${harness_root}/benchmarks/__init__.py"

export PYTHONPATH="${harness_root}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export FLASHINFER_CUDA_ARCH_LIST=10.0a
export XDG_CACHE_HOME="${scratch}/xdg"
export FLASHINFER_JIT_DIR="${scratch}/jit"
export FLASHINFER_GEN_SRC_DIR="${scratch}/generated"
export CONTAINER_SHA256="${EXPECTED_CONTAINER_SHA256}"

python3 - <<'PY' >"${output_dir}/provenance.txt"
import flashinfer

print(f"flashinfer_version={flashinfer.__version__}")
print(f"flashinfer_file={flashinfer.__file__}")
PY
echo "flashinfer_commit=${FLASHINFER_COMMIT}" >>"${output_dir}/provenance.txt"
python3 \
  "${harness_root}/benchmarks/bench_mxfp8_backend_tactic_oracle.py" \
  --shapes "${shape_file}" \
  --selected-tactics "${shape_file}" \
  --output-dir "${output_dir}/oracle" \
  --backend "${ORACLE_BACKEND}" \
  --scale-layout "${SCALE_LAYOUT}" \
  --rounds "${ROUNDS}" \
  --dry-run-iters 3 \
  --repeat-iters "${REPEAT_ITERS}" \
  --graph-calls 10
