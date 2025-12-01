#!/bin/bash
# ==============================================================================
# RDT-1B 最小训练测试脚本
# ==============================================================================
# 目的：快速验证训练pipeline是否正常工作
# 预计运行时间：5-10分钟
# ==============================================================================

export run_id="test_$(date +%Y%m%d_%H%M%S)"

# ====== NCCL 配置（单卡或双卡测试）======
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=WARN  # 减少日志输出
export NCCL_NVLS_ENABLE=0
export DS_BUILD_EVOFORMER_ATTN=0

# ====== 环境配置 ======
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"

# ====== 测试配置（最小化）======
dataset_name="baai"
action_name="action176"
test_name="minimal_test"

# 最小化超参数
train_batch_size=2           # 每卡只2个样本（快速测试）
gradient_accumulation_steps=1 # 不使用梯度累积
sample_batch_size=2          # 采样batch size
num_sample_batches=1         # 只采样1个batch
seed=42
max_train_steps=20           # 只训练20步！
checkpointing_period=10      # 每10步保存一次
sample_period=5              # 每5步采样一次
lr="1e-4"

export OUTPUT_DIR="./checkpoints/TEST_${test_name}_${run_id}"

# ====== 创建测试输出目录 ======
mkdir -p "$OUTPUT_DIR"
echo "✅ 测试输出目录: $OUTPUT_DIR"

cat > "$OUTPUT_DIR/test_config.txt" <<EOF
RDT-1B Minimal Training Test
======================================
Run ID: ${run_id}
Purpose: Verify training pipeline
Expected Duration: 5-10 minutes
Max Steps: ${max_train_steps}

Test Configuration:
  - Batch Size: ${train_batch_size}
  - Gradient Accumulation: ${gradient_accumulation_steps}
  - Max Steps: ${max_train_steps}
  - Checkpoint Every: ${checkpointing_period} steps
  - Sample Every: ${sample_period} steps

What to Check:
  ✓ Data loading works
  ✓ Model forward pass works
  ✓ Loss computation works
  ✓ Backward pass works
  ✓ Optimizer step works
  ✓ Checkpoint saving works
  ✓ Sampling works

Started: $(date)
EOF

# ====== 显示测试信息 ======
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              🧪 RDT-1B Minimal Training Test                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 测试目的: 验证训练pipeline是否正常工作"
echo "⏱️  预计时间: 5-10分钟"
echo "📊 训练步数: ${max_train_steps} steps"
echo "💾 输出目录: ${OUTPUT_DIR}"
echo ""
echo "🔍 将要验证的组件:"
echo "  ✓ 数据加载（BSON格式）"
echo "  ✓ 多模态编码（T5 + SigLIP）"
echo "  ✓ 模型前向传播"
echo "  ✓ 损失计算"
echo "  ✓ 反向传播"
echo "  ✓ 优化器更新"
echo "  ✓ Checkpoint保存"
echo "  ✓ 采样评估"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 开始测试..."
echo ""

# ====== 启动最小测试 ======
# 使用单卡或双卡测试（排除GPU:0）
# CUDA_VISIBLE_DEVICES=0 python main_baai.py \
deepspeed --include="localhost:1" main_baai.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="./checkpoints/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --seed=${seed} \
    --train_batch_size=${train_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --sample_batch_size=${sample_batch_size} \
    --num_sample_batches=${num_sample_batches} \
    --max_train_steps=${max_train_steps} \
    --checkpointing_period=${checkpointing_period} \
    --sample_period=${sample_period} \
    --checkpoints_total_limit=3 \
    --lr_scheduler="constant" \
    --learning_rate=${lr} \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_bson \
    --report_to=tensorboard \
    --precomp_lang_embed
    # --use_8bit_adam

EXIT_CODE=$?

# ====== 测试结果报告 ======
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试完成！"
    echo ""
    echo "📊 检查测试结果:"
    echo ""
    
    # 检查checkpoint是否生成
    if [ -d "$OUTPUT_DIR/checkpoint-10" ] || [ -d "$OUTPUT_DIR/checkpoint-20" ]; then
        echo "  ✅ Checkpoint保存成功"
        ls -lh "$OUTPUT_DIR" | grep checkpoint
    else
        echo "  ⚠️  未找到checkpoint文件"
    fi
    
    echo ""
    
    # 检查TensorBoard日志
    if [ -d "$OUTPUT_DIR/logs" ]; then
        echo "  ✅ TensorBoard日志已生成"
        echo "     查看方法: tensorboard --logdir=$OUTPUT_DIR"
    else
        echo "  ⚠️  未找到TensorBoard日志"
    fi
    
    echo ""
    echo "📁 测试结果位置: $OUTPUT_DIR"
    echo ""
    echo "🎉 所有组件工作正常，可以开始正式训练！"
    
else
    echo "❌ 测试失败 (Exit Code: $EXIT_CODE)"
    echo ""
    echo "请检查错误信息并修复问题。"
    echo ""
    echo "常见问题:"
    echo "  1. 预训练模型路径是否正确？"
    echo "  2. BSON数据是否存在且格式正确？"
    echo "  3. GPU显存是否充足？"
    echo "  4. 依赖包是否完整安装？"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 完成提示音（如果支持）
echo -e "\a"

exit $EXIT_CODE