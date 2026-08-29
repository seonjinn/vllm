#!/usr/bin/env bash

set -Eeuo pipefail

readonly EXP_DIR_LOCAL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly REPO_ROOT_LOCAL=$(cd "${EXP_DIR_LOCAL}/../.." && pwd)
readonly CLUSTER=${CLUSTER:-login-lyris}
readonly ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
readonly PARTITION=${PARTITION:-gb200-backfill}
readonly QOS=${QOS:-user-restrictions}
readonly REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-/home/sna/vllm-v0271-ultra-tp8-crossover}
readonly SOURCE_COMMIT=${SOURCE_COMMIT:-$(git -C "${REPO_ROOT_LOCAL}" rev-parse HEAD)}
readonly SOURCE_ROOT=${SOURCE_ROOT:-/home/sna/vllm-v0271-mxfp8-pins/${SOURCE_COMMIT}}
readonly FLASHINFER_ROOT=${FLASHINFER_ROOT:-/home/sna/flashinfer-mxfp8-pins/cec5e66dbd75a253edb5a819b2403bf410ca3223}
readonly FLASHINFER_COMMIT=${FLASHINFER_COMMIT:-cec5e66dbd75a253edb5a819b2403bf410ca3223}
readonly CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm_openai_v0271_aarch64_20260813_2688476.sqsh}
readonly EXPECTED_CONTAINER_SHA256=${EXPECTED_CONTAINER_SHA256:-e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0}
readonly CENSUS_ROOT=${CENSUS_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/results/mxfp8_tp8_serving_crossover_20260829_tp8census_r7_full}
readonly OBSERVED_CSV=${OBSERVED_CSV:-${CENSUS_ROOT}/observed/observed_shapes_snapshot.csv}
readonly STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
readonly RESULT_ROOT=${RESULT_ROOT:-${CENSUS_ROOT}/adaptive_oracle_${STAMP}}
readonly SWITCH_M=${SWITCH_M:-256}
readonly WALLTIME=${WALLTIME:-08:00:00}
readonly PRINT_PLAN=${PRINT_PLAN:-0}
readonly EXP_DIR_REMOTE="${SOURCE_ROOT}/experiments/ultra_mxfp8_tp8_serving_crossover"

print_plan() {
  echo "source_commit=${SOURCE_COMMIT}"
  echo "observed_csv=${OBSERVED_CSV}"
  echo "result_root=${RESULT_ROOT}"
  echo "layouts=8x4 128x4"
  echo "switch_m=${SWITCH_M}"
  echo "oracle=cuda_graph,cold_l2,rounds=2,repeat_iters=10"
  echo "hardware=1xGB200-node-per-layout,4-GPUs"
}

if [[ "${PRINT_PLAN}" == 1 ]]; then
  print_plan
  exit 0
fi
if ((SWITCH_M <= 0)); then
  echo "SWITCH_M must be positive" >&2
  exit 2
fi
if ! git -C "${REPO_ROOT_LOCAL}" diff --quiet || \
  ! git -C "${REPO_ROOT_LOCAL}" diff --cached --quiet || \
  [[ -n $(git -C "${REPO_ROOT_LOCAL}" ls-files --others --exclude-standard) ]]; then
  echo "Local vLLM worktree is dirty: ${REPO_ROOT_LOCAL}" >&2
  exit 1
fi

remote() {
  local command
  printf -v command '%q ' "$@"
  ssh "${CLUSTER}" "${command}"
}

remote git -C "${REMOTE_REPO_ROOT}" pull --ff-only
if [[ $(remote git -C "${REMOTE_REPO_ROOT}" rev-parse HEAD) != \
  "${SOURCE_COMMIT}" ]]; then
  echo "Remote source does not match ${SOURCE_COMMIT}" >&2
  exit 1
fi
if ! remote git -C "${SOURCE_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  remote mkdir -p "$(dirname "${SOURCE_ROOT}")"
  remote git -C "${REMOTE_REPO_ROOT}" worktree add \
    --detach "${SOURCE_ROOT}" "${SOURCE_COMMIT}"
fi
for checkout in "${SOURCE_ROOT}:${SOURCE_COMMIT}" \
  "${FLASHINFER_ROOT}:${FLASHINFER_COMMIT}"; do
  IFS=: read -r root expected <<<"${checkout}"
  if [[ $(remote git -C "${root}" rev-parse HEAD) != "${expected}" ]] || \
    [[ -n $(remote git -C "${root}" status --porcelain) ]]; then
    echo "Remote checkout is not clean at ${expected}: ${root}" >&2
    exit 1
  fi
done
remote test -s "${OBSERVED_CSV}"
container_sha=$(remote awk -F= '$1 == "sha256" {print $2}' \
  "${CONTAINER_IMAGE}.metadata.txt")
if [[ "${container_sha}" != "${EXPECTED_CONTAINER_SHA256}" ]]; then
  echo "Container metadata mismatch: ${container_sha}" >&2
  exit 1
fi
actual_container_sha=$(remote sha256sum "${CONTAINER_IMAGE}" | awk '{print $1}')
if [[ "${actual_container_sha}" != "${EXPECTED_CONTAINER_SHA256}" ]]; then
  echo "Container bytes mismatch: ${actual_container_sha}" >&2
  exit 1
fi
read -r container_size container_mtime < <(
  remote stat -c '%s %Y' "${CONTAINER_IMAGE}"
)

remote mkdir "${RESULT_ROOT}"
remote mkdir "${RESULT_ROOT}/split"
remote mkdir "${RESULT_ROOT}/slurm"
remote python3 "${EXP_DIR_REMOTE}/split_observed_by_layout.py" \
  --observed "${OBSERVED_CSV}" \
  --output-dir "${RESULT_ROOT}/split" \
  --switch-m "${SWITCH_M}"

common=(
  --nodes=1
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --qos="${QOS}"
  --time="${WALLTIME}"
)
export_common="ALL,CONTAINER_IMAGE=${CONTAINER_IMAGE},EXPECTED_CONTAINER_SHA256=${EXPECTED_CONTAINER_SHA256},EXPECTED_CONTAINER_SIZE=${container_size},EXPECTED_CONTAINER_MTIME=${container_mtime},SOURCE_ROOT=${SOURCE_ROOT},SOURCE_COMMIT=${SOURCE_COMMIT},FLASHINFER_ROOT=${FLASHINFER_ROOT},FLASHINFER_COMMIT=${FLASHINFER_COMMIT},SPLIT_DIR=${RESULT_ROOT}/split,ORACLE_RESULT_ROOT=${RESULT_ROOT}"
jobs=()
for layout in 8x4 128x4; do
  remote sbatch --test-only "${common[@]}" \
    --export="${export_common},SCALE_LAYOUT=${layout}" \
    "${EXP_DIR_REMOTE}/run_adaptive_oracle.sbatch"
  job=$(remote sbatch --parsable "${common[@]}" \
    --job-name="${ACCOUNT}-sna.mx-tp8-oracle-${layout}" \
    --output="${RESULT_ROOT}/slurm/oracle-${layout}-%j.out" \
    --export="${export_common},SCALE_LAYOUT=${layout}" \
    "${EXP_DIR_REMOTE}/run_adaptive_oracle.sbatch")
  jobs+=("${layout}:${job}")
done

{
  print_plan
  printf 'container=%s\n' "${CONTAINER_IMAGE}"
  printf 'container_sha256=%s\n' "${actual_container_sha}"
  printf 'jobs=%s\n' "${jobs[*]}"
} | remote tee "${RESULT_ROOT}/manifest.txt" >/dev/null

printf 'jobs=%s\n' "${jobs[*]}"
printf 'result_root=%s\n' "${RESULT_ROOT}"
