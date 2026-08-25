#!/usr/bin/env bash

set -euo pipefail

uv run --no-project python "${EXP_DIR}/merge_shape_traces.py" \
  --trace-dir "${RESULT_ROOT}/serving" \
  --output-csv "${OBSERVED_DIR}/observed_shapes.csv" \
  --summary "${OBSERVED_DIR}/summary.json"
uv run --no-project python "${EXP_DIR}/shard_shapes.py" \
  --observed "${OBSERVED_DIR}/observed_shapes.csv" \
  --output-dir "${OBSERVED_DIR}/shards" \
  --shards "${SHARD_COUNT}"
