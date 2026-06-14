#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STARVLA_PATH=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DEFAULT_ROBOTWIN_PATH=${ROBOTWIN_PATH:-../RoboTwin}

policy_name="starvla_policy"
task_name=${1:?usage: bash eval_direct.sh <task_name> <task_config> <policy_ckpt_path> [ckpt_setting] [seed] [gpu_id]}
task_config=${2:?usage: bash eval_direct.sh <task_name> <task_config> <policy_ckpt_path> [ckpt_setting] [seed] [gpu_id]}
policy_ckpt_path=${3:?usage: bash eval_direct.sh <task_name> <task_config> <policy_ckpt_path> [ckpt_setting] [seed] [gpu_id]}
ckpt_setting=${4:-$policy_ckpt_path}
seed=${5:-0}
gpu_id=${6:-0}
unnorm_key=${UNNORM_KEY:-}
replan_steps=${ROBOTWIN_REPLAN_STEPS:-}
action_ensemble=${ROBOTWIN_ACTION_ENSEMBLE:-}
action_ensemble_alpha=${ROBOTWIN_ACTION_ENSEMBLE_ALPHA:-}
action_reorder=${ACTION_REORDER:-}
guidance_scale=${GUIDANCE_SCALE:-}
num_inference_steps=${NUM_INFERENCE_STEPS:-}
image_size=${IMAGE_SIZE:-}
mixed_precision=${MIXED_PRECISION:-bf16}
device=${DEVICE:-cuda}
instruction_type=${INSTRUCTION_TYPE:-seen}

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
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-python}

if [[ ! -d "$ROBOTWIN_PATH" ]]; then
  echo "[ERROR] ROBOTWIN_PATH does not exist: $ROBOTWIN_PATH" >&2
  exit 1
fi

if [[ ! -f "$policy_ckpt_path" ]]; then
  echo "[ERROR] policy checkpoint does not exist: $policy_ckpt_path" >&2
  exit 1
fi

POLICY_SOURCE_DIR="$STARVLA_PATH/examples/Robotwin/starvla_policy"
POLICY_TARGET_DIR="$ROBOTWIN_PATH/policy/$policy_name"
if [[ ! -d "$POLICY_SOURCE_DIR" ]]; then
  echo "[ERROR] policy source directory does not exist: $POLICY_SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$ROBOTWIN_PATH/policy"
ln -sfn "$POLICY_SOURCE_DIR" "$POLICY_TARGET_DIR"

export CUDA_VISIBLE_DEVICES=${gpu_id}
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN="${robotwin_skip_get_obs_within_replan}"
CUROBO_SRC_PATH="$ROBOTWIN_PATH/envs/curobo/src"
export PYTHONPATH="$ROBOTWIN_PATH:$CUROBO_SRC_PATH:$STARVLA_PATH:${PYTHONPATH:-}"

DEPLOY_POLICY_PATH="$POLICY_TARGET_DIR/deploy_policy.yml"

echo "PYTHONPATH: $PYTHONPATH"
echo "ROBOTWIN_PYTHON: $ROBOTWIN_PYTHON"
echo "ROBOTWIN_SAVE_VIDEO: $robotwin_save_video"
echo "ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN: $ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN"
echo "ROBOTWIN_REPLAN_STEPS: ${replan_steps:-<disabled>}"
echo "ROBOTWIN_ACTION_ENSEMBLE: ${action_ensemble:-<disabled>}"
echo "ROBOTWIN_ACTION_ENSEMBLE_ALPHA: ${action_ensemble_alpha:-0.0}"
echo "INSTRUCTION_TYPE: $instruction_type"
echo "POLICY_TARGET_DIR: $POLICY_TARGET_DIR"

overrides=(
  --task_name "${task_name}"
  --task_config "${task_config}"
  --ckpt_setting "${ckpt_setting}"
  --seed "${seed}"
  --policy_name "${policy_name}"
  --policy_ckpt_path "${policy_ckpt_path}"
  --instruction_type "${instruction_type}"
  --eval_video_log "${robotwin_save_video}"
  --mixed_precision "${mixed_precision}"
  --device "${device}"
)

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
if [[ -n "${guidance_scale}" ]]; then
  overrides+=(--guidance_scale "${guidance_scale}")
fi
if [[ -n "${num_inference_steps}" ]]; then
  overrides+=(--num_inference_steps "${num_inference_steps}")
fi
if [[ -n "${image_size}" ]]; then
  overrides+=(--image_size "${image_size}")
fi

export PYTHONWARNINGS=ignore::UserWarning
(
  cd "$ROBOTWIN_PATH"
  "$ROBOTWIN_PYTHON" "script/eval_policy.py" --config "$DEPLOY_POLICY_PATH" \
    --overrides \
    "${overrides[@]}"
)
