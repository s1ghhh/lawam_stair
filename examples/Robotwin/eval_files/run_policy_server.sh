#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STARVLA_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DEFAULT_CKPT=results/Checkpoints/robotwin/20260409_231158+robotwin_eef_detach_distill_b96rp2_50k_base_3e-4_action_3e-4_world_1e-4_vlm_1e-4/final_model/pytorch_model.pt
STAR_VLA_PYTHON_CANDIDATES=(
  /root/miniconda3/envs/starvla/bin/python
)

export PYTHONPATH=$STARVLA_ROOT:${PYTHONPATH:-}
if [[ -z "${STAR_VLA_PYTHON:-}" ]]; then
  STAR_VLA_PYTHON=python
  for candidate in "${STAR_VLA_PYTHON_CANDIDATES[@]}"; do
    if [[ -x "$candidate" ]]; then
      STAR_VLA_PYTHON="$candidate"
      break
    fi
  done
fi
your_ckpt=${1:-${POLICY_CKPT_PATH:-$DEFAULT_CKPT}}
gpu_id=${2:-${GPU_ID:-0}}
port=${3:-${PORT:-6694}}
################# star Policy Server ######################

# export DEBUG=true
unset DEBUG
cd "$STARVLA_ROOT"

if [[ ! -f "$your_ckpt" ]]; then
  echo "[ERROR] checkpoint not found: $your_ckpt" >&2
  exit 1
fi

if [[ "$STAR_VLA_PYTHON" == */* ]]; then
  if [[ ! -x "$STAR_VLA_PYTHON" ]]; then
    echo "[ERROR] STAR_VLA_PYTHON is not executable: $STAR_VLA_PYTHON" >&2
    exit 1
  fi
elif ! command -v "$STAR_VLA_PYTHON" >/dev/null 2>&1; then
  echo "[ERROR] STAR_VLA_PYTHON is not on PATH: $STAR_VLA_PYTHON" >&2
  exit 1
fi

echo "STAR_VLA_PYTHON: $STAR_VLA_PYTHON"

CUDA_VISIBLE_DEVICES=$gpu_id "$STAR_VLA_PYTHON" deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
