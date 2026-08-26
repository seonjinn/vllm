#!/usr/bin/env bash

set -euo pipefail

readonly CLUSTER=${CLUSTER:-sna-mfa@login-lyris}
readonly ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
readonly PARTITION=${PARTITION:-gb200}
readonly ORACLE_PARTITION=${ORACLE_PARTITION:-gb200-backfill}
readonly QOS=${QOS:-user-restrictions}
readonly SOURCE_ROOT=${SOURCE_ROOT:-/home/sna/vllm-v0271-mxfp8-all-observed-tactics}
readonly FLASHINFER_ROOT=${FLASHINFER_ROOT:-/home/sna/flashinfer-mxfp8-all-backend-exact-cg-322dd00a}
readonly CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
readonly MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
readonly STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
readonly RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/mxfp8_all_observed_tactics_${STAMP}}
readonly SOURCE_COMMIT=${SOURCE_COMMIT:-$(git rev-parse HEAD)}
readonly SERVER_TIME=${SERVER_TIME:-04:00:00}
readonly ORACLE_TIME=${ORACLE_TIME:-08:00:00}
readonly EXP_DIR="${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics"

remote_common=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --qos="${QOS}"
  --nodes=1
)
oracle_common=(
  --account="${ACCOUNT}"
  --partition="${ORACLE_PARTITION}"
  --qos="${QOS}"
  --nodes=1
)
export_common="ALL,CONTAINER_IMAGE=${CONTAINER_IMAGE},MODEL_PATH=${MODEL_PATH},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_COMMIT=${SOURCE_COMMIT},FLASHINFER_ROOT=${FLASHINFER_ROOT},RESULT_ROOT=${RESULT_ROOT}"

ssh "${CLUSTER}" git -C "${SOURCE_ROOT}" pull --ff-only
remote_sha=$(ssh "${CLUSTER}" git -C "${SOURCE_ROOT}" rev-parse HEAD)
if [[ "${remote_sha}" != "${SOURCE_COMMIT}" ]]; then
  echo "Expected ${SOURCE_COMMIT}, got ${remote_sha}" >&2
  exit 1
fi
ssh "${CLUSTER}" mkdir -p "${RESULT_ROOT}/slurm"

ssh "${CLUSTER}" sbatch --test-only "${remote_common[@]}" \
  --time="${SERVER_TIME}" \
  --export="${export_common},RUN_KIND=capture-eager" \
  "${EXP_DIR}/run_server.sbatch"
ssh "${CLUSTER}" sbatch --test-only "${oracle_common[@]}" \
  --time="${ORACLE_TIME}" \
  --export="${export_common}" \
  "${EXP_DIR}/run_oracle.sbatch"

eager_job=$(ssh "${CLUSTER}" sbatch --parsable "${remote_common[@]}" \
  --time="${SERVER_TIME}" \
  --job-name="${ACCOUNT}-mx.obs.eager" \
  --output="${RESULT_ROOT}/slurm/capture-eager-%j.out" \
  --export="${export_common},RUN_KIND=capture-eager" \
  "${EXP_DIR}/run_server.sbatch")
baseline_job=$(ssh "${CLUSTER}" sbatch --parsable "${remote_common[@]}" \
  --time="${SERVER_TIME}" \
  --job-name="${ACCOUNT}-mx.obs.baseline" \
  --output="${RESULT_ROOT}/slurm/baseline-%j.out" \
  --export="${export_common},RUN_KIND=baseline" \
  "${EXP_DIR}/run_server.sbatch")
oracle_job=$(ssh "${CLUSTER}" sbatch --parsable "${oracle_common[@]}" \
  --time="${ORACLE_TIME}" \
  --job-name="${ACCOUNT}-mx.obs.oracle" \
  --output="${RESULT_ROOT}/slurm/oracle-%j.out" \
  --dependency="afterok:${eager_job}:${baseline_job}" \
  --export="${export_common}" \
  "${EXP_DIR}/run_oracle.sbatch")

lookup_job=$(ssh "${CLUSTER}" sbatch --parsable "${remote_common[@]}" \
  --time="${SERVER_TIME}" \
  --job-name="${ACCOUNT}-mx.obs.lookup" \
  --output="${RESULT_ROOT}/slurm/lookup-%j.out" \
  --dependency="afterok:${oracle_job}" \
  --export="${export_common},RUN_KIND=lookup,LOOKUP_PATH=${RESULT_ROOT}/oracle/lookup.json" \
  "${EXP_DIR}/run_server.sbatch")

echo "capture_eager_job=${eager_job}"
echo "baseline_job=${baseline_job}"
echo "oracle_job=${oracle_job}"
echo "lookup_job=${lookup_job}"
echo "oracle_partition=${ORACLE_PARTITION}"
echo "result_root=${RESULT_ROOT}"
