#!/usr/bin/env bash

set -Eeuo pipefail

readonly LOCAL_EXP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly LOCAL_REPO_ROOT=$(cd "${LOCAL_EXP_DIR}/../.." && pwd)
readonly BACKEND=${BACKEND:-cute-dsl}
eval "$(python3 "${LOCAL_EXP_DIR}/backend_config.py" "${BACKEND}")"
readonly BACKEND_NAME LINEAR_BACKEND ORACLE_BACKEND SCALE_LAYOUT TRTLLM_LAYOUT
readonly CLUSTER=${CLUSTER:-sna-mfa@login-lyris}
readonly ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
readonly PARTITION=${PARTITION:-gb200}
readonly ORACLE_PARTITION=${ORACLE_PARTITION:-gb200-backfill}
readonly QOS=${QOS:-user-restrictions}
readonly REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-/home/sna/vllm-v0271-mxfp8-all-observed-tactics}
readonly FLASHINFER_REPO_ROOT=${FLASHINFER_REPO_ROOT:-/home/sna/flashinfer-mxfp8-serving-selected-oracle}
readonly EXPECTED_FLASHINFER_COMMIT=${EXPECTED_FLASHINFER_COMMIT:-ee72a36051a8cec8f43a087fe5365883ab2051c2}
readonly CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
readonly EXPECTED_CONTAINER_SHA256=${EXPECTED_CONTAINER_SHA256:-e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0}
readonly MODEL_PATH=${MODEL_PATH:-/lustre/fsw/coreai_dlalgo_llm/users/sna/ckpts/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase.mxfp8}
readonly STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
readonly RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/mxfp8_all_observed_tactics_${BACKEND_NAME}_${STAMP}}
readonly SOURCE_COMMIT=${SOURCE_COMMIT:-$(git -C "${LOCAL_REPO_ROOT}" rev-parse HEAD)}
readonly SOURCE_ROOT=${SOURCE_ROOT:-/home/sna/vllm-v0271-mxfp8-pins/${SOURCE_COMMIT}}
readonly FLASHINFER_ROOT=${FLASHINFER_ROOT:-/home/sna/flashinfer-mxfp8-pins/${EXPECTED_FLASHINFER_COMMIT}}
readonly SERVER_TIME=${SERVER_TIME:-04:00:00}
readonly ORACLE_TIME=${ORACLE_TIME:-08:00:00}
readonly TP=${TP:-4}
readonly EXP_DIR="${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics"
submitted_jobs=()
submission_complete=0
cleanup_submission() {
  local status=$?
  trap - EXIT
  if ((status != 0 && submission_complete == 0 && ${#submitted_jobs[@]} > 0)); then
    ssh "${CLUSTER}" scancel "${submitted_jobs[@]}" || true
  fi
  exit "${status}"
}
trap cleanup_submission EXIT

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
if ! git -C "${LOCAL_REPO_ROOT}" diff --quiet || \
  ! git -C "${LOCAL_REPO_ROOT}" diff --cached --quiet || \
  [[ -n $(git -C "${LOCAL_REPO_ROOT}" ls-files --others --exclude-standard) ]]; then
  echo "Local vLLM source tree is dirty: ${LOCAL_REPO_ROOT}" >&2
  exit 1
fi

ssh "${CLUSTER}" git -C "${REMOTE_REPO_ROOT}" pull --ff-only
if [[ -n $(ssh "${CLUSTER}" git -C "${REMOTE_REPO_ROOT}" status --porcelain) ]]; then
  echo "Remote vLLM source tree is dirty: ${REMOTE_REPO_ROOT}" >&2
  exit 1
fi
remote_sha=$(ssh "${CLUSTER}" git -C "${REMOTE_REPO_ROOT}" rev-parse HEAD)
if [[ "${remote_sha}" != "${SOURCE_COMMIT}" ]]; then
  echo "Expected ${SOURCE_COMMIT}, got ${remote_sha}" >&2
  exit 1
fi
if ! ssh "${CLUSTER}" git -C "${SOURCE_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  ssh "${CLUSTER}" mkdir -p "$(dirname "${SOURCE_ROOT}")"
  ssh "${CLUSTER}" git -C "${REMOTE_REPO_ROOT}" worktree add \
    --detach "${SOURCE_ROOT}" "${SOURCE_COMMIT}"
fi
pinned_source_sha=$(ssh "${CLUSTER}" git -C "${SOURCE_ROOT}" rev-parse HEAD)
if [[ "${pinned_source_sha}" != "${SOURCE_COMMIT}" ]] || \
  [[ -n $(ssh "${CLUSTER}" git -C "${SOURCE_ROOT}" status --porcelain) ]]; then
  echo "Pinned vLLM worktree is not clean at ${SOURCE_COMMIT}: ${SOURCE_ROOT}" >&2
  exit 1
fi

ssh "${CLUSTER}" git -C "${FLASHINFER_REPO_ROOT}" pull --ff-only
if [[ -n $(ssh "${CLUSTER}" git -C "${FLASHINFER_REPO_ROOT}" status --porcelain) ]]; then
  echo "FlashInfer source tree is dirty: ${FLASHINFER_REPO_ROOT}" >&2
  exit 1
fi
flashinfer_commit=$(ssh "${CLUSTER}" git -C "${FLASHINFER_REPO_ROOT}" rev-parse HEAD)
readonly flashinfer_commit
if [[ "${flashinfer_commit}" != "${EXPECTED_FLASHINFER_COMMIT}" ]]; then
  echo "Expected FlashInfer ${EXPECTED_FLASHINFER_COMMIT}, got ${flashinfer_commit}" >&2
  exit 1
fi
if ! ssh "${CLUSTER}" git -C "${FLASHINFER_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  ssh "${CLUSTER}" mkdir -p "$(dirname "${FLASHINFER_ROOT}")"
  ssh "${CLUSTER}" git -C "${FLASHINFER_REPO_ROOT}" worktree add \
    --detach "${FLASHINFER_ROOT}" "${flashinfer_commit}"
fi
pinned_flashinfer_sha=$(ssh "${CLUSTER}" git -C "${FLASHINFER_ROOT}" rev-parse HEAD)
if [[ "${pinned_flashinfer_sha}" != "${flashinfer_commit}" ]] || \
  [[ -n $(ssh "${CLUSTER}" git -C "${FLASHINFER_ROOT}" status --porcelain) ]]; then
  echo "Pinned FlashInfer worktree is not clean at ${flashinfer_commit}: ${FLASHINFER_ROOT}" >&2
  exit 1
fi
container_sha=$(ssh "${CLUSTER}" cat "${CONTAINER_IMAGE}.metadata.txt" | \
  awk -F= '$1 == "sha256" {print $2}')
if [[ "${container_sha}" != "${EXPECTED_CONTAINER_SHA256}" ]]; then
  echo "Expected container ${EXPECTED_CONTAINER_SHA256}, got ${container_sha}" >&2
  exit 1
fi
export_common="ALL,CONTAINER_IMAGE=${CONTAINER_IMAGE},EXPECTED_CONTAINER_SHA256=${EXPECTED_CONTAINER_SHA256},MODEL_PATH=${MODEL_PATH},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_COMMIT=${SOURCE_COMMIT},FLASHINFER_ROOT=${FLASHINFER_ROOT},FLASHINFER_COMMIT=${flashinfer_commit},RESULT_ROOT=${RESULT_ROOT},BACKEND_NAME=${BACKEND_NAME},LINEAR_BACKEND=${LINEAR_BACKEND},ORACLE_BACKEND=${ORACLE_BACKEND},SCALE_LAYOUT=${SCALE_LAYOUT},TRTLLM_LAYOUT=${TRTLLM_LAYOUT},TP=${TP}"

ssh "${CLUSTER}" mkdir -p "$(dirname "${RESULT_ROOT}")"
ssh "${CLUSTER}" mkdir "${RESULT_ROOT}"
ssh "${CLUSTER}" mkdir "${RESULT_ROOT}/slurm"

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
  --job-name="${ACCOUNT}-mx.${BACKEND_NAME}.eager" \
  --output="${RESULT_ROOT}/slurm/capture-eager-%j.out" \
  --export="${export_common},RUN_KIND=capture-eager" \
  "${EXP_DIR}/run_server.sbatch")
submitted_jobs+=("${eager_job}")
baseline_job=$(ssh "${CLUSTER}" sbatch --parsable "${remote_common[@]}" \
  --time="${SERVER_TIME}" \
  --job-name="${ACCOUNT}-mx.${BACKEND_NAME}.base" \
  --output="${RESULT_ROOT}/slurm/baseline-%j.out" \
  --export="${export_common},RUN_KIND=baseline" \
  "${EXP_DIR}/run_server.sbatch")
submitted_jobs+=("${baseline_job}")
oracle_job=$(ssh "${CLUSTER}" sbatch --parsable "${oracle_common[@]}" \
  --time="${ORACLE_TIME}" \
  --job-name="${ACCOUNT}-mx.${BACKEND_NAME}.oracle" \
  --output="${RESULT_ROOT}/slurm/oracle-%j.out" \
  --dependency="afterok:${eager_job}:${baseline_job}" \
  --export="${export_common}" \
  "${EXP_DIR}/run_oracle.sbatch")
submitted_jobs+=("${oracle_job}")

lookup_job=$(ssh "${CLUSTER}" sbatch --parsable "${remote_common[@]}" \
  --time="${SERVER_TIME}" \
  --job-name="${ACCOUNT}-mx.${BACKEND_NAME}.lookup" \
  --output="${RESULT_ROOT}/slurm/lookup-%j.out" \
  --dependency="afterok:${oracle_job}" \
  --export="${export_common},RUN_KIND=lookup,LOOKUP_PATH=${RESULT_ROOT}/oracle/lookup.json" \
  "${EXP_DIR}/run_server.sbatch")
submitted_jobs+=("${lookup_job}")

manifest_tmp="${RESULT_ROOT}/pipeline_manifest.txt.tmp.$$"
ssh "${CLUSTER}" tee "${manifest_tmp}" >/dev/null <<EOF
backend=${BACKEND_NAME}
source_root=${SOURCE_ROOT}
source_commit=${SOURCE_COMMIT}
flashinfer_root=${FLASHINFER_ROOT}
flashinfer_commit=${flashinfer_commit}
container=${CONTAINER_IMAGE}
container_sha256=${EXPECTED_CONTAINER_SHA256}
model=${MODEL_PATH}
capture_eager_job=${eager_job}
baseline_job=${baseline_job}
oracle_job=${oracle_job}
lookup_job=${lookup_job}
EOF
ssh "${CLUSTER}" mv "${manifest_tmp}" "${RESULT_ROOT}/pipeline_manifest.txt"
submission_complete=1
trap - EXIT

echo "capture_eager_job=${eager_job}"
echo "baseline_job=${baseline_job}"
echo "oracle_job=${oracle_job}"
echo "lookup_job=${lookup_job}"
echo "backend=${BACKEND_NAME}"
echo "linear_backend=${LINEAR_BACKEND}"
echo "oracle_backend=${ORACLE_BACKEND}"
echo "scale_layout=${SCALE_LAYOUT}"
echo "flashinfer_commit=${flashinfer_commit}"
echo "oracle_partition=${ORACLE_PARTITION}"
echo "result_root=${RESULT_ROOT}"
