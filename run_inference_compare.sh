#!/bin/bash
# ==============================================================================
# RDT-1B 推理对比脚本
# ==============================================================================
# 使用checkpoint-6000和lerobot_baai数据集进行推理对比
# 随机选取一个frame，预测action chunk，与真实action对比并绘图
# ==============================================================================

# 配置路径
CHECKPOINT_PATH="./checkpoints/rdt1b-full-action176-20251202_000048/checkpoint-6000"
DATASET_PATH="./data/baai/data/lerobot_baai"
CONFIG_PATH="./configs/base.yaml"
VISION_ENCODER="google/siglip-so400m-patch14-384"
OUTPUT_DIR="./inference_results"

# 可选参数
EPISODE_IDX=""  # 留空表示随机选择，例如：EPISODE_IDX="--episode_idx 5"
STEP_IDX=""     # 留空表示随机选择，例如：STEP_IDX="--step_idx 100"
SEED=42
DEVICE="cuda"   # 使用cuda或cpu

# 显示配置
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          🎯 RDT-1B 推理对比                                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📂 Checkpoint: $CHECKPOINT_PATH"
echo "📂 Dataset: $DATASET_PATH"
echo "📂 Output: $OUTPUT_DIR"
echo "🎲 Seed: $SEED"
echo "🖥️  Device: $DEVICE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行推理脚本
python inference_compare.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --dataset "$DATASET_PATH" \
    --config "$CONFIG_PATH" \
    --vision_encoder "$VISION_ENCODER" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $EPISODE_IDX \
    $STEP_IDX

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 推理对比完成!"
echo "📁 结果保存在: $OUTPUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

