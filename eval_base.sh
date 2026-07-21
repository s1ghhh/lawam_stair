#!/usr/bin/env bash
# =============================================================================
# cmp4 评测 | 变体 = base (解码=训练匹配的原生形态: er50 readout 5 步, 无 aux 消融)
# 把 (suite × {exec50, exec20}) 的 job 按指定 GPU 池分批挂: 每卡一次一个 job,
# 一批占满 n 卡、跑完再下一批。
# 用法:
#   GPUS=4,5 bash eval_base.sh [ckpt]        # 2 卡: 4 suite×2 setting=8 job -> 4 批
#   bash eval_base.sh                         # 默认 GPUS=4,5,6,7 (8 job -> 2 批)
#   变量: GPUS | TRIALS(默认10) | SUITES(默认4个suite) | NUM_WORKERS(默认8)
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export GPUS="${GPUS:-4,5,6,7}"
exec bash run_cmp4_eval.sh base "$@"
