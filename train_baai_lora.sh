export run_id=$(date +%Y%m%d_%H%M%S)
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export DS_BUILD_EVOFORMER_ATTN=0

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"

# ====== 路径配置优化 ======
dataset_name="baai"
model_type="lora"
lora_r=32
lora_alpha=64
lr="1e-4"
batch_size=32
seed=42  # 添加固定种子

# 生成清晰的输出路径（包含seed信息）
export LORA_OUTPUT_DIR="./checkpoints/rdt1b-${model_type}-${dataset_name}-r${lora_r}a${lora_alpha}-lr${lr}-bs${batch_size}-seed${seed}-${run_id}"
# ========================

if [ ! -d "$LORA_OUTPUT_DIR" ]; then
    mkdir -p "$LORA_OUTPUT_DIR"
    echo "LoRA output folder '$LORA_OUTPUT_DIR' created"
    
    # 创建配置说明文件
    cat > "$LORA_OUTPUT_DIR/training_config.txt" <<EOF
RDT-1B LoRA Fine-tuning Configuration
======================================
Run ID: ${run_id}
Model: RDT-1B
Method: LoRA Fine-tuning
Dataset: ${dataset_name}
Random Seed: ${seed}  # 记录种子

LoRA Parameters:
  - Rank: ${lora_r}
  - Alpha: ${lora_alpha}
  - Dropout: 0.1
  - Target Modules: all

Training Hyperparameters:
  - Learning Rate: ${lr}
  - Batch Size: ${batch_size}
  - Max Steps: 200000
  - Mixed Precision: bf16
  - Random Seed: ${seed}
  
Command: bash train_baai_lora.sh
Started: $(date)
EOF
else
    echo "LoRA output folder '$LORA_OUTPUT_DIR' already exists"
fi

deepspeed main_baai_lora.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="./checkpoints/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$LORA_OUTPUT_DIR \
    --seed=${seed} \                    # 添加这一行！
    --use_lora \
    --lora_rank=${lora_r} \
    --lora_alpha=${lora_alpha} \
    --lora_dropout=0.1 \
    --lora_target_modules="all" \
    --train_batch_size=${batch_size} \
    --sample_batch_size=${batch_size} \
    --num_sample_batches=4 \
    --max_train_steps=200000 \
    --checkpointing_period=2500 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=${lr} \
    --mixed_precision="bf16" \
    --dataloader_num_workers=16 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_bson \
    --report_to=tensorboard \
    --precomp_lang_embed