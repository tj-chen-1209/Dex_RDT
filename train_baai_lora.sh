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
# export CUTLASS_PATH="/path/to/cutlass"
dataset_name="baai"
export WANDB_PROJECT="baai_data_train"
export LORA_OUTPUT_DIR="./checkpoints/rdt-${WANDB_PROJECT}-${run_id}"

if [ ! -d "$LORA_OUTPUT_DIR" ]; then
    mkdir "$LORA_OUTPUT_DIR"
    echo "LoRA output folder '$LORA_OUTPUT_DIR' created"
else
    echo "LoRA output folder '$LORA_OUTPUT_DIR' already exists"
fi

# --use_8bit_adam \
deepspeed --exclude="localhost:0" main_baai_lora.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$LORA_OUTPUT_DIR \
    --train_batch_size=32 \
    --sample_batch_size=32 \
    --num_sample_batches=4 \
    --max_train_steps=200000 \
    --checkpointing_period=5000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="pretrain" \
    --state_noise_snr=40 \
    --load_from_bson \
    --report_to=wandb \
    --precomp_lang_embed