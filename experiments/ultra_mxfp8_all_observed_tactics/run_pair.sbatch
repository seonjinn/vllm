#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0

set -euo pipefail

: "${SLURM_JOB_ID:?Run inside a SLURM allocation}"
: "${PAIR_ORDER:?Set PAIR_ORDER}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"
: "${LOOKUP_PATH:?Set LOOKUP_PATH}"

readonly pipeline_result_root=${RESULT_ROOT}
readonly lookup_path=${LOOKUP_PATH}

case "${PAIR_ORDER}" in
  baseline-lookup)
    run_kinds=(baseline lookup)
    ;;
  lookup-baseline)
    run_kinds=(lookup baseline)
    ;;
  *)
    echo "PAIR_ORDER must be baseline-lookup or lookup-baseline, got ${PAIR_ORDER}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${lookup_path}" ]]; then
  echo "LOOKUP_PATH is missing or empty: ${lookup_path}" >&2
  exit 1
fi
if [[ -n ${RUN_SERVER_SCRIPT:-} ]]; then
  if [[ ${ALLOW_RUN_SERVER_SCRIPT_TEST_HOOK:-0} != 1 ]]; then
    echo "RUN_SERVER_SCRIPT is a test hook and is disabled for production" >&2
    exit 2
  fi
  readonly run_server_script=${RUN_SERVER_SCRIPT}
else
  : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
  readonly run_server_script=${SOURCE_ROOT}/experiments/ultra_mxfp8_all_observed_tactics/run_server.sbatch
fi

export PAIR_ID="${SLURM_JOB_ID}-${PAIR_ORDER}"
export PAIR_ORDER
export RESULT_ROOT="${pipeline_result_root}/pairs/${PAIR_ORDER}"
for index in "${!run_kinds[@]}"; do
  export PAIR_POSITION=$((index + 1))
  export RUN_KIND=${run_kinds[index]}
  export RUN_INSTANCE="${PAIR_ID}-${PAIR_POSITION}-${RUN_KIND}"
  if [[ "${RUN_KIND}" == lookup ]]; then
    export LOOKUP_PATH=${lookup_path}
  else
    unset LOOKUP_PATH
  fi
  bash "${run_server_script}"
done
