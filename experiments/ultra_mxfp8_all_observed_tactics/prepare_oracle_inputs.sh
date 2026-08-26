#!/usr/bin/env bash

set -euo pipefail

readonly CUDAGRAPH_COMPLETION_MARKER="Graph capturing finished"

metadata_value() {
  local metadata_path=$1
  local key=$2
  awk -F= -v key="${key}" '
    $1 == key {
      count += 1
      value = substr($0, length(key) + 2)
    }
    END {
      if (count != 1) {
        exit 1
      }
      print value
    }
  ' "${metadata_path}"
}

validate_phase_cudagraph_metadata() {
  local results_path=$1
  local phase=$2
  local expected_configured=$3
  local expected_status=$4
  local expected_evidence=$5
  local phase_dir="${results_path}/serving/${phase}"
  local complete run_dir metadata_path actual
  local -a complete_markers

  shopt -s nullglob
  complete_markers=("${phase_dir}"/*/COMPLETE)
  shopt -u nullglob
  if ((${#complete_markers[@]} == 0)); then
    echo "No completed ${phase} serving run found under ${phase_dir}" >&2
    return 1
  fi

  for complete in "${complete_markers[@]}"; do
    run_dir=${complete%/COMPLETE}
    metadata_path="${run_dir}/metadata.txt"
    for key_and_value in \
      "cudagraph_configured:${expected_configured}" \
      "cudagraph_capture_status:${expected_status}" \
      "cudagraph_capture_evidence:${expected_evidence}" \
      "cudagraph_capture_marker:${CUDAGRAPH_COMPLETION_MARKER}"; do
      local key=${key_and_value%%:*}
      local expected=${key_and_value#*:}
      if ! actual=$(metadata_value "${metadata_path}" "${key}"); then
        echo "${phase} metadata must contain exactly one ${key}: ${metadata_path}" \
          >&2
        return 1
      fi
      if [[ "${actual}" != "${expected}" ]]; then
        echo "${phase} metadata ${key} must be ${expected}, got ${actual}" >&2
        return 1
      fi
    done
    if [[ "${expected_status}" == capture_completed ]] && \
      ! grep -Fq -- "${CUDAGRAPH_COMPLETION_MARKER}" "${run_dir}/server.log"; then
      echo "${phase} server.log is missing marker: ${CUDAGRAPH_COMPLETION_MARKER}" \
        >&2
      return 1
    fi
  done
}

validate_cudagraph_metadata() {
  local results_path=$1
  validate_phase_cudagraph_metadata \
    "${results_path}" capture-graph true capture_completed \
    server_log_completion_marker
  validate_phase_cudagraph_metadata \
    "${results_path}" capture-eager false disabled_eager not_applicable
}

if [[ "${1:-}" == --validate-cudagraph-metadata ]]; then
  if (($# != 2)); then
    echo "Usage: $0 --validate-cudagraph-metadata RESULT_ROOT" >&2
    exit 2
  fi
  validate_cudagraph_metadata "$2"
  exit
fi

: "${TP:?Set TP to the serving tensor-parallel size}"
validate_cudagraph_metadata "${RESULT_ROOT}"

python3 "${EXP_DIR}/merge_shape_traces.py" \
  --trace-dir "${RESULT_ROOT}/serving" \
  --output-csv "${OBSERVED_DIR}/observed_shapes.csv" \
  --summary "${OBSERVED_DIR}/summary.json" \
  --expected-rank-count "${TP}"
python3 "${EXP_DIR}/shard_shapes.py" \
  --observed "${OBSERVED_DIR}/observed_shapes.csv" \
  --output-dir "${OBSERVED_DIR}/shards" \
  --shards "${SHARD_COUNT}"
