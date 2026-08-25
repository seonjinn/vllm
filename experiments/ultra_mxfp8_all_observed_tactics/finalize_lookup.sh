#!/usr/bin/env bash

set -euo pipefail

report_args=()
for report in "${ORACLE_DIR}"/shards/*/oracle/report.json; do
  report_args+=(--report "${report}")
done
python3 "${EXP_DIR}/build_lookup.py" \
  --observed "${OBSERVED_DIR}/observed_shapes.csv" \
  "${report_args[@]}" \
  --output "${ORACLE_DIR}/lookup.json"
touch "${ORACLE_DIR}/COMPLETE"
