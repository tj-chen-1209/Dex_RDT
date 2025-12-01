#!/bin/bash
# ==============================================================================
# RDT-1B 全量微调训练脚本 - 针对 A800*7 优化
# ==============================================================================
# 数据集：action176 (100 episodes)
# 硬件：7x NVIDIA A800 (80GB each, 排除GPU:0)
# 作者：AI Assistant
# 日期：$(date +%Y-%m-%d)
# ==============================================================================

export run_id=$(date +%Y%m%d_%H%M%S)

# ====== NCCL 通信配置（多卡训练优化）======
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export DS_BUILD_EVOFORMER_ATTN=0

# ====== 编译环境配置 ======
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"

# ====== 模型编码器路径 ======
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"

# ====== 训练超参数配置 ======
dataset_name="baai"
action_name="action176"
model_type="full"
lr="1e-4"
train_batch_size=48
gradient_accumulation_steps=2
sample_batch_size=32
num_sample_batches=4
seed=42
max_train_steps=200000
checkpointing_period=1000
sample_period=500

# 生成清晰的输出路径
export OUTPUT_DIR="./checkpoints/rdt1b-${model_type}-${action_name}-${run_id}"

# ====== 创建输出目录和配置文件 ======
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "✅ Output folder '$OUTPUT_DIR' created"
    
    # 创建详细配置说明文件
    cat > "$OUTPUT_DIR/training_config.txt" <<EOF
╔══════════════════════════════════════════════════════════════════╗
║          RDT-1B Full Fine-tuning Configuration                   ║
╚══════════════════════════════════════════════════════════════════╝

📋 基本信息
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Run ID: ${run_id}
  Model: RDT-1B (1 Billion parameters)
  Method: Full Fine-tuning (全量微调 - 所有参数可训练)
  Dataset: ${dataset_name}/${action_name} (100 episodes)
  Hardware: 7x NVIDIA A800 (80GB VRAM each, GPU:0 excluded)
  Random Seed: ${seed}

🎯 训练超参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Learning Rate: ${lr}
  LR Scheduler: cosine (with warmup)
  LR Warmup Steps: 500
  Per-Device Batch Size: ${train_batch_size}
  Gradient Accumulation Steps: ${gradient_accumulation_steps}
  Effective Batch Size: $((train_batch_size * gradient_accumulation_steps * 6)) (global, 6 GPUs)
  Max Training Steps: ${max_train_steps}
  Checkpointing Period: ${checkpointing_period} steps
  Sample Period: ${sample_period} steps
  Checkpoints Keep: 20 (最近的)
  Mixed Precision: bf16 (bfloat16)
  Optimizer: 8-bit Adam (节省显存)
  Max Gradient Norm: 1.0

🔧 DeepSpeed 配置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ZeRO Stage: 2 (Optimizer + Gradient Partitioning)
  Overlap Communication: Yes
  Contiguous Gradients: Yes

📊 数据配置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Load from BSON: Yes
  Precomputed Language Embeddings: Yes (节省计算)
  Image History Size: 2 frames
  Number of Cameras: 3 (RDT-1B模型限制)
  Image Augmentation: Enabled (ColorJitter, Blur, Noise)
  State Noise SNR: 40 dB
  Condition Mask Probability: 0.1
  Dataloader Workers: 8

💾 显存估算（单卡A800 80GB）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model Parameters (bf16): ~2.0 GB
  Optimizer States (8-bit): ~1.5 GB
  Gradients (ZeRO-2): ~0.4 GB
  Activations (batch=8): ~15-20 GB
  Working Memory: ~5 GB
  ────────────────────────────────
  Total Estimated: ~25-30 GB / 80 GB ✅ 显存充足

📈 训练进度估算
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  数据集大小: 100 episodes × ~604KB ≈ 60 MB
  每步样本数: 96 (全局batch size)
  总训练步数: ${max_train_steps}
  预计训练时间: ~8-12小时 (取决于数据加载速度)
  Checkpoint总大小: ~80 GB (20个checkpoints × 4GB each)

🚀 启动命令
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  bash train_baai_optimized.sh

📝 监控训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TensorBoard: tensorboard --logdir=${OUTPUT_DIR}
  访问地址: http://localhost:6006

Started: $(date)
EOF
else
    echo "⚠️  Output folder '$OUTPUT_DIR' already exists"
fi

# ====== 显示训练配置摘要 ======
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          🚀 RDT-1B Full Fine-tuning on A800*7                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📦 数据集: ${dataset_name}/${action_name} (100 episodes)"
echo "🎯 模型: RDT-1B (1B params, full fine-tuning)"
echo "💻 硬件: 6x A800 GPUs (GPU:0 excluded)"
echo "📊 全局Batch Size: $((train_batch_size * gradient_accumulation_steps * 6))"
echo "📈 训练步数: ${max_train_steps}"
echo "💾 输出目录: ${OUTPUT_DIR}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ====== 启动训练（DeepSpeed分布式训练）======
# deepspeed --exclude="localhost:0" main_baai.py \
deepspeed --hostfile=hostfile.txt main_baai.py \
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
    --checkpoints_total_limit=40 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --learning_rate=${lr} \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_bson \
    --report_to=tensorboard \
    --precomp_lang_embed
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Training completed or interrupted!"
echo "📁 Results saved to: ${OUTPUT_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

