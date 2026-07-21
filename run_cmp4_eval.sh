#!/usr/bin/env bash
# =============================================================================
# cmp4 评测核心调度器 —— 一个变体, 把 (suite × setting) 的 job 按 GPU 池分批挂
#
# 每个变体的 inference = 训练匹配的原生解码 (server 端 LAWAM_DECODE_MODE/STEPS):
#     onestep -> std 1 步 | has -> HAS 5 步 | eraux/base -> readout 5 步
# 两个 setting (client 端 EVAL_ACTION_CHUNK_LEN, 每次执行几个动作):
#     exec50 -> 50 全执行(开环) | exec20 -> 前 20(receding)
#
# job = 每个 suite × 每个 setting。指定 n 个 GPU, 每张卡一次一个 job,
# 一批占满 n 张卡、跑完再下一批 (分 ceil(job/n) 批)。
#   例: GPUS=4,5 + 4 suite × 2 setting = 8 job -> 分 4 批
#
# 用法:  GPUS=4,5 bash run_cmp4_eval.sh <onestep|has|eraux|base> [ckpt]
#   变量: GPUS(必填) | TRIALS(默认10) | SUITES(默认4个) | NUM_WORKERS(默认8)
# 产出: results/eval_runs/cmp4/<变体>__<suite>__<setting>/...
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

VARIANT="${1:?用法: GPUS=4,5 bash run_cmp4_eval.sh <onestep|has|eraux|base> [ckpt]}"
case "$VARIANT" in onestep|has|eraux|base) ;; *) echo "[错误] 未知变体 $VARIANT" >&2; exit 1;; esac
CKPT="${2:-${CKPT:-$(ls -dt results/Checkpoints/libero/*+cmp4_${VARIANT}/final_model/pytorch_model.pt 2>/dev/null | head -1)}}"
[[ -n "$CKPT" && -f "$CKPT" ]] || { echo "[错误] 找不到 $VARIANT 的 ckpt, 请显式传入" >&2; exit 1; }

GPUS="${GPUS:?请指定 GPU 列表, 如 GPUS=4,5}"
read -r -a GPU_ARR <<< "${GPUS//,/ }"
N=${#GPU_ARR[@]}
TRIALS="${TRIALS:-10}"
SUITES="${SUITES:-libero_10 libero_goal libero_object libero_spatial}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUT="results/eval_runs/cmp4"
mkdir -p logs/cmp4_eval

export LIBERO_HOME="${LIBERO_HOME:-/workspace/000000_lawam/LIBERO}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/opt/conda/envs/libero_lawam/bin/python}"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/opt/conda/envs/lawam/bin/python}"
# 不强制 MUJOCO_GL: 交回底层按 num_workers 自动选 (多 worker->osmesa, 单 worker->egl),
# 因为 EGL 在多进程 offscreen worker / 部分机器上会初始化失败。
# 想强制渲染后端就外部传, 如 MUJOCO_GL=osmesa bash eval_onestep.sh ...
[[ -n "${MUJOCO_GL:-}" ]] && export MUJOCO_GL || true

case "$VARIANT" in
  onestep) MODE=std;     STEPS=1 ;;
  has)     MODE=has;     STEPS=5 ;;
  eraux|base) MODE=readout; STEPS=5 ;;
esac
SETTINGS=("exec50:50" "exec20:20")

# ---- 构造 job 列表: "suite|tag|exec_len" -----------------------------------
jobs=()
for suite in $SUITES; do
  for s in "${SETTINGS[@]}"; do
    jobs+=("${suite}|${s%%:*}|${s##*:}")
  done
done
NJOBS=${#jobs[@]}
NBATCH=$(( (NJOBS + N - 1) / N ))
echo "[cmp4/$VARIANT] GPUS=$GPUS (n=$N) | $(echo $SUITES|wc -w) suite × ${#SETTINGS[@]} setting = $NJOBS job | 分 $NBATCH 批 | decode=$MODE/${STEPS}步"
echo "  ckpt=$CKPT"

run_one() {   # gpu suite tag exec_len run_index
  local gpu="$1" suite="$2" tag="$3" ex="$4" ridx="$5"
  local alias="${VARIANT}__${suite}__${tag}"
  echo "  -> GPU$gpu | $suite | $tag(exec=$ex) | log=logs/cmp4_eval/${alias}.log"
  LAWAM_DECODE_MODE="$MODE" LAWAM_NUM_INFERENCE_STEPS="$STEPS" \
  EVAL_ACTION_CHUNK_LEN="$ex" \
  SUITES="$suite" NUM_TRIALS_PER_TASK="$TRIALS" NUM_WORKERS="$NUM_WORKERS" \
  GPU_IDS="$gpu" RUN_INDEX_BASE="$ridx" \
  OUTPUT_ROOT="$OUT" LIBERO_CKPT_ALIAS="$alias" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "$CKPT" \
    > "logs/cmp4_eval/${alias}.log" 2>&1
}

# ---- 分批调度: 每批占满 n 张卡, wait 完再下一批 ----------------------------
i=0; batch=0
while (( i < NJOBS )); do
  batch=$((batch+1)); pids=()
  echo "===== 批 $batch/$NBATCH [$(date +%H:%M:%S)] ====="
  for (( k=0; k<N && i<NJOBS; k++, i++ )); do
    IFS='|' read -r suite tag ex <<< "${jobs[i]}"
    run_one "${GPU_ARR[k]}" "$suite" "$tag" "$ex" "$i" &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || echo "  [批$batch] 有 job 非0退出 (见对应 log)"; done
done

# ---- 汇总: suite × setting -> total_success_rate ---------------------------
echo; echo "======== $VARIANT 汇总 (decode=$MODE/${STEPS}步, TRIALS=$TRIALS) ========"
printf '%-18s' "suite"; for s in "${SETTINGS[@]}"; do printf '%-12s' "${s%%:*}"; done; echo
for suite in $SUITES; do
  printf '%-18s' "$suite"
  for s in "${SETTINGS[@]}"; do
    alias="${VARIANT}__${suite}__${s%%:*}"
    sj=$(ls -t "$OUT/$alias"/*/suites/*/summary.json 2>/dev/null | head -1 || true)
    if [[ -n "$sj" ]]; then
      sr=$("$STAR_VLA_PYTHON" -c "import json;print(f\"{json.load(open('$sj'))['total_success_rate']:.3f}\")" 2>/dev/null || echo NA)
    else sr=NA; fi
    printf '%-12s' "$sr"
  done
  echo
done
echo "==================================================================="
