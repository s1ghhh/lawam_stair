#!/usr/bin/env bash
# =============================================================================
# cmp4 评测 —— 四变体各占一张卡, 同时后台并行 (参考 closedloop/eval_closedloop.sh)
#   onestep->GPU4  has->GPU5  eraux->GPU6  base->GPU7
#   每变体: SUITES(默认 libero_10, 10 task) x TRIALS(默认10) x 三口径
#           std5_exec10 / fast_exec10 / fast_exec50
# ckpt: 每变体用 CKPT_<变体> 传绝对路径, 否则各子脚本自动找/报错。
# 用法:
#   CKPT_onestep=/a/..pt CKPT_has=/b/..pt CKPT_eraux=/c/..pt CKPT_base=/d/..pt \
#     bash eval_all_4gpu.sh
#   (四个 ckpt 已在标准路径 -> 直接 bash eval_all_4gpu.sh)
# 覆盖: TRIALS  SUITES
# =============================================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"          # -> LaWAM 仓库根
mkdir -p logs/cmp4_eval

export TRIALS="${TRIALS:-10}"
export SUITES="${SUITES:-libero_10}"
STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/opt/conda/envs/lawam/bin/python}"
OUT="results/eval_runs/cmp4"

declare -A GPU_OF=( [onestep]=4 [has]=5 [eraux]=6 [base]=7 )
VARIANTS=(onestep has eraux base)

# ---- 并行后台启动: 四变体各绑一张卡 -----------------------------------------
declare -A PID_OF
for v in "${VARIANTS[@]}"; do
  g="${GPU_OF[$v]}"; log="logs/cmp4_eval/${v}.log"
  var="CKPT_${v}"; ckpt="${!var:-}"
  echo ">>> [$(date +%H:%M:%S)] 挂起 ${v} -> GPU ${g}  (log: ${log})"
  CKPT="${ckpt}" nohup bash "eval_${v}_gpu${g}.sh" > "${log}" 2>&1 &
  PID_OF[$v]=$!
done

echo "四个变体已并行挂到 GPU 4/5/6/7 (SUITES=${SUITES} x ${TRIALS} trial x 三口径)。"
echo "看进度: tail -f logs/cmp4_eval/onestep.log   等待全部完成..."
fail=0
for v in "${VARIANTS[@]}"; do
  wait "${PID_OF[$v]}" || { echo "  [${v}] 退出码非0, 见 logs/cmp4_eval/${v}.log"; fail=$((fail+1)); }
done

# ---- 汇总表 (变体 x 三口径 -> total_success_rate) ---------------------------
echo; echo "================ cmp4 评测汇总 (${SUITES} x ${TRIALS} trial) ================"
MODES=(std5_exec10 fast_exec10 fast_exec50)
printf '%-10s' "variant"; for m in "${MODES[@]}"; do printf '%-16s' "$m"; done; echo
for v in "${VARIANTS[@]}"; do
  printf '%-10s' "$v"
  for m in "${MODES[@]}"; do
    sj=$(ls -t "$OUT/${v}__${m}"/*/suites/*/summary.json 2>/dev/null | head -1 || true)
    if [[ -n "$sj" ]]; then
      sr=$("$STAR_VLA_PYTHON" -c "import json;print(f\"{json.load(open('$sj'))['total_success_rate']:.3f}\")" 2>/dev/null || echo NA)
    else sr=NA; fi
    printf '%-16s' "$sr"
  done
  echo
done
echo "==========================================================================="
echo "std5_exec10=质量上限(四变体同协议,确认没掉点); fast_*=各自原生快解码 (exec10 receding / exec50 开环)"
[[ $fail -gt 0 ]] && echo "[提示] 有 ${fail} 个变体异常退出, 查对应日志。"
exit 0
