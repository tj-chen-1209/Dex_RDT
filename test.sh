#!/bin/bash
# ====================================
# RDT-1B LoRA 快速测试脚本
# 用途：验证训练流程是否正常，不进行完整训练
# ====================================

export run_id=$(date +%Y%m%d_%H%M%S)
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export DS_BUILD_EVOFORMER_ATTN=0

# 解决分布式训练网络问题
export MASTER_ADDR=127.0.0.1      # 强制使用IPv4
export MASTER_PORT=29601           # 使用独立端口避免与其他deepspeed进程冲突
export NCCL_SOCKET_FAMILY=AF_INET  # 禁用IPv6，只使用IPv4

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"

# ====== 测试配置（资源消耗小，快速验证） ======
dataset_name="baai"
model_type="lora"
lora_r=32
lora_alpha=64
lr="1e-4"
batch_size=4          # 测试用小batch size
seed=42
test_steps=20         # 只跑20步测试

# 测试输出路径
export TEST_OUTPUT_DIR="./checkpoints/test-rdt1b-lora-${run_id}"
echo "============================================"
echo "🧪 RDT-1B LoRA 快速测试模式"
echo "============================================"
echo "测试步数: ${test_steps} steps"
echo "Batch Size: ${batch_size}"
echo "输出目录: ${TEST_OUTPUT_DIR}"
echo "============================================"
# ============================================

if [ ! -d "$TEST_OUTPUT_DIR" ]; then
    mkdir -p "$TEST_OUTPUT_DIR"
    echo "✅ 测试输出文件夹已创建: '$TEST_OUTPUT_DIR'"
    
    # 创建测试配置说明
    cat > "$TEST_OUTPUT_DIR/test_config.txt" <<EOF
RDT-1B LoRA 快速测试配置
======================================
测试目的: 验证训练流程是否正常
Run ID: ${run_id}
Model: RDT-1B
Method: LoRA Fine-tuning
Dataset: ${dataset_name}

测试参数:
  - Test Steps: ${test_steps} (正式训练: 200000)
  - Batch Size: ${batch_size} (正式训练: 32)
  - GPUs: 2 (正式训练: 7-8)
  - Random Seed: ${seed}

LoRA Parameters:
  - Rank: ${lora_r}
  - Alpha: ${lora_alpha}
  - Dropout: 0.1
  - Target Modules: all

Training Hyperparameters:
  - Learning Rate: ${lr}
  - Mixed Precision: bf16
  
Command: bash test.sh
Started: $(date)

注意: 这是测试运行，不会产生可用的训练模型！
EOF
else
    echo "⚠️  测试输出文件夹已存在: '$TEST_OUTPUT_DIR'"
fi

# ====== 快速测试：只使用GPU 0 单卡运行（不使用DeepSpeed） ======
echo ""
echo "🚀 开始测试 (GPU 0 单卡模式，不使用DeepSpeed)..."
echo ""

CUDA_VISIBLE_DEVICES=0 python main_baai_lora.py \
    --pretrained_model_name_or_path="./checkpoints/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$TEST_OUTPUT_DIR \
    --seed=${seed} \
    --use_lora \
    --lora_rank=${lora_r} \
    --lora_alpha=${lora_alpha} \
    --lora_dropout=0.1 \
    --lora_target_modules="all" \
    --train_batch_size=${batch_size} \
    --sample_batch_size=${batch_size} \
    --num_sample_batches=1 \
    --max_train_steps=${test_steps} \
    --checkpointing_period=10 \
    --sample_period=10 \
    --checkpoints_total_limit=2 \
    --lr_scheduler="constant" \
    --learning_rate=${lr} \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_bson \
    --report_to=tensorboard \
    --precomp_lang_embed

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✅ 测试成功完成！"
    echo "============================================"
    echo "训练流程验证通过，可以开始正式训练。"
    echo "使用命令: bash train_baai_lora.sh"
    echo ""
    echo "测试输出位置: ${TEST_OUTPUT_DIR}"
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "❌ 测试失败！"
    echo "============================================"
    echo "请检查错误信息并修复配置。"
    echo "常见问题："
    echo "  1. 检查预训练模型路径是否正确"
    echo "  2. 检查数据集路径是否正确"
    echo "  3. 检查GPU是否可用"
    echo "  4. 检查依赖是否安装完整"
    echo "============================================"
    exit 1
fi

