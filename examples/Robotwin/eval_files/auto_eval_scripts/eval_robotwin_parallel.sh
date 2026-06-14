#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EVAL_ROOT}/../../.." && pwd)"
cd "${REPO_ROOT}"

your_ckpt="${1:-${CKPT_PATH:-}}"
task_config="${2:-${TASK_CONFIG:-demo_clean}}"
worker_index="${3:-${WORKER_INDEX:-}}"
num_workers="${4:-${NUM_WORKERS:-}}"
run_tag="${5:-${RUN_TAG:-}}"

if [[ -z "${your_ckpt}" || -z "${worker_index}" || -z "${num_workers}" || -z "${run_tag}" ]]; then
  echo "Usage: $0 <ckpt_path> <task_config> <worker_index> <num_workers> <run_tag>" >&2
  exit 1
fi

STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-python}"

exec "${STAR_VLA_PYTHON}" "${EVAL_ROOT}/batched_eval_runner.py" \
  --mode worker \
  --ckpt_path "${your_ckpt}" \
  --task_config "${task_config}" \
  --run_tag "${run_tag}" \
  --worker_index "${worker_index}" \
  --num_workers "${num_workers}"
