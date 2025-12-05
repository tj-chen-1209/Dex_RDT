#!/bin/bash
# =============================================================================
# RDT-1B 开环评估启动脚本 (Open-Loop Evaluation)
# =============================================================================
#
# 功能：在训练集上评估模型的预测精度，支持多时间步分析
# 
# 使用方法:
#   bash run_open_loop_eval.sh [checkpoint_name] [num_episodes] [samples_per_episode] [gpu_id]
#
# 示例:
#   bash run_open_loop_eval.sh checkpoint-14000           # 基础评估
#   bash run_open_loop_eval.sh checkpoint-14000 20 10     # 20个episodes，每个10个样本
#   bash run_open_loop_eval.sh checkpoint-14000 -1 10     # 全部episodes，每个10个采样点
#   bash run_open_loop_eval.sh checkpoint-14000 30 15 1   # 使用GPU 1
#
# 新增功能 (v2):
#   - 支持多时间步采样 (early/mid/late阶段分析)
#   - 自动生成阶段对比图 (phase_comparison.png)
#   - Step vs Error趋势分析 (step_vs_error.png)
# =============================================================================

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "🎯 RDT-1B Open-Loop Evaluation (v2)"
echo "=============================================="
echo "📂 Project Root: $PROJECT_ROOT"

# 解析参数
CHECKPOINT_NAME="${1:-checkpoint-14000}"
NUM_EPISODES="${2:-20}"           # 默认评估20个episodes
SAMPLES_PER_EPISODE="${3:-10}"    # 默认每个episode 10个采样点（覆盖初期/中期/末期）
GPU_ID="${4:-0}"                  # 默认使用GPU 0

# 设置GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "🖥️  Using GPU: $GPU_ID"

# 路径配置
CHECKPOINT_BASE="./checkpoints/rdt1b-full-action176-20251202_000048"
CHECKPOINT_PATH="${CHECKPOINT_BASE}/${CHECKPOINT_NAME}"
DATASET_PATH="./data/baai/data/lerobot_baai"
OUTPUT_DIR="./baai_eval/eval_results"
CONFIG_PATH="configs/base.yaml"
VISION_ENCODER="google/siglip-so400m-patch14-384"

# 检查checkpoint是否存在
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint不存在: $CHECKPOINT_PATH"
    echo ""
    echo "可用的checkpoints:"
    ls -1 "$CHECKPOINT_BASE" 2>/dev/null | grep "checkpoint-" || echo "  (无)"
    exit 1
fi

# 检查pytorch_model.bin是否存在
if [ ! -f "$CHECKPOINT_PATH/pytorch_model.bin" ]; then
    echo "❌ 未找到权重文件: $CHECKPOINT_PATH/pytorch_model.bin"
    echo ""
    echo "如果是DeepSpeed格式，请先转换权重:"
    echo "  python $CHECKPOINT_PATH/zero_to_fp32.py $CHECKPOINT_PATH $CHECKPOINT_PATH/pytorch_model.bin"
    exit 1
fi

# 检查数据集是否存在
if [ ! -d "$DATASET_PATH/cache" ]; then
    echo "❌ 数据集缓存不存在: $DATASET_PATH/cache"
    exit 1
fi

echo ""
echo "📋 配置信息:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Dataset: $DATASET_PATH"
echo "  Episodes: $NUM_EPISODES (-1表示全部)"
echo "  Samples/Episode: $SAMPLES_PER_EPISODE (均匀覆盖early/mid/late阶段)"
echo "  Output: $OUTPUT_DIR"
echo "  GPU: $GPU_ID"
echo ""

# 激活conda环境（如果需要）
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate rdt

# 运行评估
echo "🚀 开始评估..."
echo ""

python baai_eval/open_loop_eval.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --dataset "$DATASET_PATH" \
    --config "$CONFIG_PATH" \
    --vision_encoder "$VISION_ENCODER" \
    --num_episodes "$NUM_EPISODES" \
    --samples_per_episode "$SAMPLES_PER_EPISODE" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --seed 42 \
    --save_samples

echo ""
echo "=============================================="
echo "✅ 评估完成!"
echo "=============================================="
echo ""
echo "📊 生成的图表:"
echo "  - error_by_group.png     : 分部位误差条形图"
echo "  - error_per_joint.png    : 每个关节误差详图"
echo "  - error_distribution.png : 误差分布直方图"
echo "  - error_per_episode.png  : Episode误差对比"
echo "  - phase_comparison.png   : 阶段对比图 (early/mid/late)"
echo "  - step_vs_error.png      : Step vs Error趋势分析"
echo ""

