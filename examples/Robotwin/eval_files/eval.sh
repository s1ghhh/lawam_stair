#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STARVLA_PATH=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DEFAULT_ROBOTWIN_PATH=${ROBOTWIN_PATH:-../RoboTwin}
DEFAULT_ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-python}

policy_name="model2robotwin_interface"
task_name=${1:?usage: bash eval.sh <task_name> <task_config> [ckpt_setting] [seed] [gpu_id]}
task_config=${2:?usage: bash eval.sh <task_name> <task_config> [ckpt_setting] [seed] [gpu_id]}
ckpt_setting=${3:-starvla_demo}
seed=${4:-0}
gpu_id=${5:-0} # default is 0
host=${HOST:-127.0.0.1}
port=${PORT:-6694}
policy_ckpt_path=${POLICY_CKPT_PATH:-}
unnorm_key=${UNNORM_KEY:-}
replan_steps=${ROBOTWIN_REPLAN_STEPS:-}
action_ensemble=${ROBOTWIN_ACTION_ENSEMBLE:-}
action_ensemble_alpha=${ROBOTWIN_ACTION_ENSEMBLE_ALPHA:-}
action_reorder=${ACTION_REORDER:-}

normalize_python_bool() {
  local value="${1:-1}"
  local lowered="${value,,}"
  case "${lowered}" in
    1|true|yes|on)
      printf 'True\n'
      ;;
    0|false|no|off)
      printf 'False\n'
      ;;
    *)
      echo "[ERROR] Boolean value must be one of: 1/0/true/false/yes/no/on/off, got: ${value}" >&2
      exit 1
      ;;
  esac
}

robotwin_save_video="$(normalize_python_bool "${ROBOTWIN_SAVE_VIDEO:-0}")"
robotwin_skip_get_obs_within_replan="${ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN:-1}"

ROBOTWIN_PATH=${ROBOTWIN_PATH:-$DEFAULT_ROBOTWIN_PATH}
if [[ -x "$DEFAULT_ROBOTWIN_PYTHON" ]]; then
  ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-$DEFAULT_ROBOTWIN_PYTHON}
else
  ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-python}
fi

if [[ ! -d "$ROBOTWIN_PATH" ]]; then
  echo "[ERROR] ROBOTWIN_PATH does not exist: $ROBOTWIN_PATH" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN="${robotwin_skip_get_obs_within_replan}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

EVAL_FILES_PATH=$SCRIPT_DIR
DEPLOY_POLICY_PATH=$EVAL_FILES_PATH/deploy_policy.yml

export PYTHONPATH=$ROBOTWIN_PATH:$STARVLA_PATH:$EVAL_FILES_PATH:${PYTHONPATH:-}

echo "PYTHONPATH: $PYTHONPATH"
echo "ROBOTWIN_PYTHON: $ROBOTWIN_PYTHON"
echo "ROBOTWIN_SAVE_VIDEO: $robotwin_save_video"
echo "ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN: $ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN"
echo "ROBOTWIN_REPLAN_STEPS: ${replan_steps:-<disabled>}"
echo "ROBOTWIN_ACTION_ENSEMBLE: ${action_ensemble:-<disabled>}"
echo "ROBOTWIN_ACTION_ENSEMBLE_ALPHA: ${action_ensemble_alpha:-0.0}"

overrides=(
    --task_name "${task_name}"
    --task_config "${task_config}"
    --ckpt_setting "${ckpt_setting}"
    --seed "${seed}"
    --policy_name "${policy_name}"
    --eval_video_log "${robotwin_save_video}"
)

if [[ -n "${host}" ]]; then
  overrides+=(--host "${host}")
fi
if [[ -n "${port}" ]]; then
  overrides+=(--port "${port}")
fi
if [[ -n "${policy_ckpt_path}" ]]; then
  overrides+=(--policy_ckpt_path "${policy_ckpt_path}")
fi
if [[ -n "${unnorm_key}" ]]; then
  overrides+=(--unnorm_key "${unnorm_key}")
fi
if [[ -n "${replan_steps}" ]]; then
  overrides+=(--replan_steps "${replan_steps}")
fi
if [[ -n "${action_ensemble}" ]]; then
  overrides+=(--action_ensemble "${action_ensemble}")
fi
if [[ -n "${action_ensemble_alpha}" ]]; then
  overrides+=(--action_ensemble_alpha "${action_ensemble_alpha}")
fi
if [[ -n "${action_reorder}" ]]; then
  overrides+=(--action_reorder "${action_reorder}")
fi

PYTHONWARNINGS=ignore::UserWarning \
"$ROBOTWIN_PYTHON" "$EVAL_FILES_PATH/robotwin_batch_bridge.py" --config "$DEPLOY_POLICY_PATH" \
    --overrides \
    "${overrides[@]}"
