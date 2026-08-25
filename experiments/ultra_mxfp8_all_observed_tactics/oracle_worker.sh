#!/usr/bin/env bash

set -euo pipefail

readonly shard=${SLURM_LOCALID}
readonly gpu_index=${SLURM_LOCALID}
readonly scratch="/raid/scratch/${USER}/mxfp8_observed_oracle/${SLURM_JOB_ID}/${shard}"
readonly shape_file="${OBSERVED_DIR}/shards/shard_${shard}.csv"
readonly output_dir="${ORACLE_DIR}/shards/${shard}"
mkdir -p "${scratch}" "${output_dir}"
trap 'rm -rf "${scratch}"' EXIT
if [[ $(wc -l <"${shape_file}") -le 1 ]]; then
  echo "shard ${shard} is empty"
  exit 0
fi

export PYTHONPATH="${FLASHINFER_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export FLASHINFER_CUDA_ARCH_LIST=10.0a
export XDG_CACHE_HOME="${scratch}/xdg"
export FLASHINFER_JIT_DIR="${scratch}/jit"
export FLASHINFER_GEN_SRC_DIR="${scratch}/generated"

python3 "${EXP_DIR}/prepare_exact_cache.py" \
  --shapes "${shape_file}" \
  --output-dir "${output_dir}/cache" \
  --backend cute-dsl \
  --scale-layout 128x4
python3 \
  "${FLASHINFER_ROOT}/benchmarks/bench_mxfp8_backend_tactic_oracle.py" \
  --shapes "${shape_file}" \
  --selected-cache-dir "${output_dir}/cache" \
  --output-dir "${output_dir}/oracle" \
  --backend cute-dsl \
  --scale-layout 128x4 \
  --rounds "${ROUNDS}" \
  --dry-run-iters 3 \
  --repeat-iters "${REPEAT_ITERS}" \
  --graph-calls 10
