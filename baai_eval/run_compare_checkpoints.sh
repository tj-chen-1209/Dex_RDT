#!/bin/bash
# =============================================================================
# RDT-1B 多Checkpoint对比评估启动脚本
# =============================================================================
#
# 功能：对比不同训练步数的checkpoint在同一帧上的预测效果
#
# 使用方法:
#   bash run_compare_checkpoints.sh [checkpoints] [episode_idx] [step_idx] [gpu_id]
#
# 示例:
#   bash run_compare_checkpoints.sh                                    # 使用默认checkpoint和随机帧
#   bash run_compare_checkpoints.sh "checkpoint-3000,checkpoint-6000,checkpoint-14000"  # 指定checkpoints
#   bash run_compare_checkpoints.sh "" 5 100                          # 指定episode=5, step=100
#   bash run_compare_checkpoints.sh "" "" "" 1                        # 使用GPU 1
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "🎯 RDT-1B Checkpoint Comparison"
echo "=============================================="
echo "📂 Project Root: $PROJECT_ROOT"

# 解析参数
CHECKPOINTS="${1:-checkpoint-3000,checkpoint-6000,checkpoint-9000,checkpoint-12000,checkpoint-14000}"
EPISODE_IDX="${2:-}"
STEP_IDX="${3:-}"
GPU_ID="${4:-0}"

# 路径配置
CHECKPOINT_BASE="./checkpoints/rdt1b-full-action176-20251202_000048"
DATASET_PATH="./data/baai/data/lerobot_baai"
OUTPUT_DIR="./eval_results/checkpoint_compare"
CONFIG_PATH="configs/base.yaml"
VISION_ENCODER="google/siglip-so400m-patch14-384"

# 设置GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "🖥️  Using GPU: $GPU_ID"

# 检查checkpoint存在
echo ""
echo "📋 检查Checkpoints..."
IFS=',' read -ra CKPT_ARRAY <<< "$CHECKPOINTS"
for ckpt in "${CKPT_ARRAY[@]}"; do
    ckpt_path="${CHECKPOINT_BASE}/${ckpt}/pytorch_model.bin"
    if [ -f "$ckpt_path" ]; then
        echo "   ✅ $ckpt"
    else
        echo "   ❌ $ckpt (不存在)"
    fi
done

echo ""
echo "📋 配置信息:"
echo "  Checkpoints: $CHECKPOINTS"
echo "  Episode: ${EPISODE_IDX:-随机}"
echo "  Step: ${STEP_IDX:-随机}"
echo "  Output: $OUTPUT_DIR"
echo ""

# 构建命令
CMD="python baai_eval/compare_checkpoints.py"
CMD="$CMD --checkpoint_base \"$CHECKPOINT_BASE\""
CMD="$CMD --checkpoints \"$CHECKPOINTS\""
CMD="$CMD --dataset \"$DATASET_PATH\""
CMD="$CMD --config \"$CONFIG_PATH\""
CMD="$CMD --vision_encoder \"$VISION_ENCODER\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --device cuda"

if [ -n "$EPISODE_IDX" ]; then
    CMD="$CMD --episode_idx $EPISODE_IDX"
fi

if [ -n "$STEP_IDX" ]; then
    CMD="$CMD --step_idx $STEP_IDX"
fi

echo "🚀 开始对比评估..."
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "✅ 对比评估完成!"
echo "=============================================="
echo ""
echo "📊 生成的文件:"
echo "  - ep*_step*_right_arm_compare.png  : 右臂对比图"
echo "  - ep*_step*_right_hand_compare.png : 右手对比图"
echo "  - ep*_step*_left_arm_compare.png   : 左臂对比图"
echo "  - ep*_step*_left_hand_compare.png  : 左手对比图"
echo "  - ep*_step*_mse_trend.png          : MSE趋势图"
echo "  - ep*_step*_comparison.json        : 详细对比结果"
echo ""


