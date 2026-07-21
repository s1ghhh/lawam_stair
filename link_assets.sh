#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# 把训练/评测需要的 5 个权重&数据路径 从一个已有下载好的目录 软链到当前 repo。
# 这些都是 gitignore 的大文件, 不随 git 分支走 —— 换 checkout 后用它补齐。
#
# 用法:  SRC=<已有权重的repo或目录> bash link_assets.sh
#   例:  SRC=/data/guoheng/workspace/0714/lawam_singlecol bash link_assets.sh
#        SRC=/data2/guoheng/workspace/0714/lawam_rollout  bash link_assets.sh
# 不确定 SRC 在哪? 先跑:
#   find /data /data2 /persistent_data_1 -maxdepth 6 -type d -name qwen3_weights 2>/dev/null
# =============================================================================
SRC="${SRC:?用法: SRC=<有权重的目录> bash link_assets.sh}"
DST="$(pwd)"
[[ -f "$DST/train_lawam.sh" && -d "$DST/starVLA" ]] || { echo "[错误] 请在目标 repo 根目录运行" >&2; exit 1; }
[[ "$SRC" != "$DST" ]] || { echo "[错误] SRC 不能等于当前目录" >&2; exit 1; }

# repo 相对路径 -> 需要存在的探针文件 (用于校验 SRC 里确实有)
PATHS=(
  "results/Checkpoints/qwen3_weights|config.json"
  "weights/dinov3-vitb16-pretrain-lvd1689m|config.json"
  "latent_action_model/logs/dino_large_vae/lam_release|checkpoints/pytorch_model.pt"
  "results/Checkpoints/pretrain/lawam_pretrain|final_model/pytorch_model.pt"
  "dataset/libero_merged_no_noops_20hz|meta/info.json"
)

miss=0
for entry in "${PATHS[@]}"; do
  rel="${entry%%|*}"; probe="${entry#*|}"
  src="$SRC/$rel"
  if [[ ! -e "$src/$probe" ]]; then
    echo "[SRC 缺失] $src/$probe  —— SRC 里也没有这个, 换个 SRC 或去下载" >&2
    miss=1; continue
  fi
  mkdir -p "$(dirname "$DST/$rel")"
  # 已存在(且不是坏软链)就跳过; 是软链则覆盖
  if [[ -e "$DST/$rel" && ! -L "$DST/$rel" ]]; then
    echo "[跳过] $rel 已是实体, 不动"
  else
    ln -sfn "$src" "$DST/$rel"
    echo "[链接] $rel -> $src"
  fi
done
[[ "$miss" == 0 ]] || { echo; echo "[错误] SRC 里有缺失项, 见上"; exit 1; }

echo
echo "校验(和训练脚本前置检查同款):"
ok=1
for p in \
  results/Checkpoints/qwen3_weights/config.json \
  weights/dinov3-vitb16-pretrain-lvd1689m/config.json \
  latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt \
  results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt \
  dataset/libero_merged_no_noops_20hz/meta/info.json; do
  if [[ -e "$p" ]]; then echo "  ✓ $p"; else echo "  ✗ $p"; ok=0; fi
done
[[ "$ok" == 1 ]] && echo "全部就绪, 可以跑训练了。" || { echo "仍有缺失。"; exit 1; }
