#!/usr/bin/env bash
# =============================================================================
# cmp4 评测 | 变体 = base | GPU 7 | libero_10 (10 task) x 10 trial
# 内部串行三口径: std5_exec10 / fast_exec10 / fast_exec50 (对比协议, 各 100 ep)
#   只想跑主口径 -> SKIP_MODES="fast_exec10 fast_exec50" bash eval_base_gpu7.sh
# 用法:
#   bash eval_base_gpu7.sh                                        # 自动找正式 ckpt
#   CKPT=/abs/.../final_model/pytorch_model.pt bash eval_base_gpu7.sh      # 显式指定
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"          # -> LaWAM 仓库根

VARIANT=base
export GPU=7
export TRIALS="${TRIALS:-10}"
export SUITES="${SUITES:-libero_10}"

CKPT="${CKPT:-$(ls -dt results/Checkpoints/libero/*+cmp4_${VARIANT}/final_model/pytorch_model.pt 2>/dev/null | head -1)}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[错误] 找不到 ${VARIANT} 的正式 ckpt (本机暂无该变体 25k 产出)。" >&2
  echo "       请显式指定: CKPT=/abs/path/final_model/pytorch_model.pt bash $0" >&2
  exit 1
fi

echo "[评测] ${VARIANT} | GPU=${GPU} | ${SUITES} x ${TRIALS} trial | ${CKPT}"
exec bash run_cmp4_eval.sh "${VARIANT}" "${CKPT}"
