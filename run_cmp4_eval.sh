#!/usr/bin/env bash
# =============================================================================
# 四种去噪方式公平对比 —— 评测入口 (libero_10, 默认每任务 50 trials = 500 episodes)
#
# 每个 ckpt 跑三个口径 (server 端用 LAWAM_DECODE_MODE/LAWAM_NUM_INFERENCE_STEPS
# 控制解码方式, client 端用 EVAL_ACTION_CHUNK_LEN 控制每次执行几个动作):
#   std5_exec10 : 标准 5 步去噪, 执行前 10 个动作 (receding) —— 质量上限, 四个
#                 ckpt 协议完全一致, 用于确认训练本身没掉点
#   fast_exec10 : 各自的原生快速解码, 执行前 10 个 ——
#                   onestep    -> 1 步标准去噪
#                   has        -> FASTER 时间表 5 步 (队头 1 步就绪)
#                   eraux/base -> er50 阶梯读出 (前 10 = 组0 的 1 步等效外推)
#   fast_exec50 : 同 fast 模式, 一次执行整个 50 chunk (开环, 看后段质量)
#
# 用法:
#   bash run_cmp4_eval.sh <onestep|has|eraux|base> [ckpt路径]
#   环境变量: GPU(默认1) | TRIALS(默认50) | SUITES(默认libero_10) | NUM_WORKERS(默认8)
#             SKIP_MODES="std5_exec10 ..." 跳过已有结果
# 产出: results/eval_runs/cmp4/<变体>__<口径>/...
# =============================================================================
set -euo pipefail

VARIANT="${1:?用法: bash run_cmp4_eval.sh <onestep|has|eraux|base> [ckpt]}"
case "${VARIANT}" in onestep|has|eraux|base) ;; *) echo "[错误] 未知变体 ${VARIANT}" >&2; exit 1;; esac

# ckpt: 显式给出, 或自动取最新的 cmp4_<变体> 产出
CKPT="${2:-$(ls -dt results/Checkpoints/libero/*+cmp4_${VARIANT}/final_model/pytorch_model.pt 2>/dev/null | head -1)}"
[[ -n "${CKPT}" && -f "${CKPT}" ]] || { echo "[错误] 找不到 ${VARIANT} 的 ckpt, 请显式传入" >&2; exit 1; }

GPU="${GPU:-1}"
TRIALS="${TRIALS:-50}"
OUT="results/eval_runs/cmp4"

export LIBERO_HOME="${LIBERO_HOME:-/workspace/000000_lawam/LIBERO}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/opt/conda/envs/libero_lawam/bin/python}"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/opt/conda/envs/lawam/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# 变体 -> 快速解码方式
case "${VARIANT}" in
  onestep) FAST_MODE=std;     FAST_STEPS=1 ;;
  has)     FAST_MODE=has;     FAST_STEPS=5 ;;
  eraux|base) FAST_MODE=readout; FAST_STEPS=5 ;;
esac

run_mode() {
  local tag="$1" mode="$2" steps="$3" exec_len="$4"
  [[ " ${SKIP_MODES:-} " == *" ${tag} "* ]] && { echo "===== 跳过 ${VARIANT}/${tag}"; return; }
  echo "===== ${VARIANT}/${tag} | decode=${mode} steps=${steps} exec=${exec_len} | ${CKPT}"
  LAWAM_DECODE_MODE="${mode}" LAWAM_NUM_INFERENCE_STEPS="${steps}" \
  EVAL_ACTION_CHUNK_LEN="${exec_len}" \
  SUITES="${SUITES:-libero_10}" NUM_TRIALS_PER_TASK="${TRIALS}" \
  NUM_WORKERS="${NUM_WORKERS:-8}" GPU_IDS="${GPU}" \
  OUTPUT_ROOT="${OUT}" LIBERO_CKPT_ALIAS="${VARIANT}__${tag}" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${CKPT}"
}

run_mode "std5_exec10" std             5              10
run_mode "fast_exec10" "${FAST_MODE}"  "${FAST_STEPS}" 10
run_mode "fast_exec50" "${FAST_MODE}"  "${FAST_STEPS}" 50

echo "===================== ${VARIANT} 汇总 ====================="
for tag in std5_exec10 fast_exec10 fast_exec50; do
  E=$(ls -t ${OUT}/${VARIANT}__${tag}/*/suites/*/eval.log 2>/dev/null | head -1)
  [[ -n "${E}" ]] && echo "${tag}: $(grep -oE 'success_rate=[0-9.]+' "${E}" | tail -1)" || echo "${tag}: (无结果)"
done
