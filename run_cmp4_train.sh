#!/usr/bin/env bash
# =============================================================================
# 四种去噪方式公平对比 —— 训练入口 (chunk=50, 2卡 x bs32 = global 64, 5步去噪预算)
#
# 变体 (共享 starVLA/config/training/train_libero_c50_cmp.yaml, 差异只有一个开关):
#   onestep : 训练时 tau 恒为 0 (纯噪声->动作 一步回归), 推理 1 步
#   has     : FASTER 式 Horizon-Aware Schedule, 训练按 0.5 概率混入
#             逐位置扭曲 tau_i = clip(tau/(1-u_i)), u_i=(1-j^alpha)*u0
#   eraux   : er50 早解码 + (1-tau)^2 辅助 loss (weight 0.5)
#   base    : 无任何附加 (= 官方 chunk50 SFT, er50 无辅助 loss 的 baseline)
#
# 公平性: 四者除上述开关外完全一致 —— 同 init (lawam_pretrain), 同数据, 同 25k 步,
#   同 lr/freeze/优化器, 同 chunk50, 同 num_inference_steps=5 预算。
#
# 用法:
#   cd 000_compare_2stair/LaWAM
#   GPUS=1,2 bash run_cmp4_train.sh onestep
#   GPUS=3,4 bash run_cmp4_train.sh has
#   GPUS=5,6 bash run_cmp4_train.sh eraux
#   GPUS=6,7 bash run_cmp4_train.sh base      # 卡不够时错峰跑
# 可选: MAX_STEPS=100 做冒烟; PER_DEVICE_BS 覆盖单卡bs(默认32, 48G卡冒烟建议4);
#       RUN_ID 覆盖默认 run_id; WANDB_MODE(默认 offline)
#
# 产出: results/Checkpoints/libero/<时间戳>+cmp4_<变体>/final_model/pytorch_model.pt
# =============================================================================
set -euo pipefail

VARIANT="${1:?用法: GPUS=1,2 bash run_cmp4_train.sh <onestep|has|eraux|base>}"
GPUS="${GPUS:?请指定两张卡, 如 GPUS=1,2}"
export WANDB_MODE="${WANDB_MODE:-offline}"

[[ -f "train_lawam.sh" && -d "starVLA" ]] || { echo "[错误] 请在 LaWAM repo 根目录运行" >&2; exit 1; }

case "${VARIANT}" in
  onestep) EXTRA=(--framework.action_model.flow_cfg.fixed_train_tau=0.0) ;;
  has)     EXTRA=(--framework.action_model.flow_cfg.has_train_mix_prob=0.5
                  --framework.action_model.flow_cfg.has_alpha=1.0
                  --framework.action_model.flow_cfg.has_u0=0.9) ;;
  eraux)   EXTRA=(--framework.action_model.flow_cfg.early_readout_loss_weight=0.5) ;;
  base)    EXTRA=() ;;
  *) echo "[错误] 未知变体: ${VARIANT} (onestep|has|eraux|base)" >&2; exit 1 ;;
esac

RUN_ID="${RUN_ID:-cmp4_${VARIANT}}"

# 无 ip 命令的机器上, train_lawam.sh 的网卡自动探测会在 set -e 下静默退出
if ! command -v ip >/dev/null 2>&1 && [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME=lo
  echo "[提示] 无 ip 命令, 已设 NCCL_SOCKET_IFNAME=lo (单机)"
fi

# ---- 前置检查 ----------------------------------------------------------------
for p in \
  results/Checkpoints/qwen3_weights/config.json \
  weights/dinov3-vitb16-pretrain-lvd1689m/config.json \
  latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt \
  results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt \
  dataset/libero_merged_no_noops_20hz/meta/info.json; do
  [[ -e "$p" ]] || { echo "[错误] 缺前置文件: $p (先跑 SRC=/workspace/000000_lawam/LaWAM_v2 bash link_assets.sh)" >&2; exit 1; }
done
grep -q "fixed_train_tau" starVLA/model/framework/vlas/flowmatching_expert.py || {
  echo "[错误] 代码缺少 cmp4 改动 (fixed_train_tau 等), 请先同步代码" >&2; exit 1; }

N_GPU=$(awk -F, '{print NF}' <<< "${GPUS}")
[[ "${N_GPU}" -eq 2 ]] || echo "[警告] GPUS=${GPUS} 是 ${N_GPU} 卡, 全局 bs = ${N_GPU}x32 ≠ 64"

echo "[启动] cmp4/${VARIANT} | GPUS=${GPUS} | global_bs=$((N_GPU*32)) | run_id=${RUN_ID}"
CUDA_VISIBLE_DEVICES="${GPUS}" \
bash train_lawam.sh \
  starVLA/config/training/train_libero_c50_cmp.yaml \
  --run_id="${RUN_ID}" \
  ${MAX_STEPS:+--trainer.max_train_steps="${MAX_STEPS}"} \
  ${PER_DEVICE_BS:+--datasets.vla_data.per_device_batch_size="${PER_DEVICE_BS}"} \
  --trainer.save_interval=5000 \
  "${EXTRA[@]}"

# 启动后在 logs/train.log 核对:
#   [ ] Total batch size = 64
#   [ ] config.yaml: horizon_sec: 2.5 / sec_chunk: 2.5 / num_inference_steps: 5
#   [ ] 变体开关已生效 (fixed_train_tau / has_train_mix_prob / early_readout_loss_weight)
#   [ ] relaxed finetune init checkpoint: .../lawam_pretrain/...
