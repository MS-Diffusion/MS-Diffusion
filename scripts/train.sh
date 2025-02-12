#!/bin/bash
set +x
local_test=$1
datatime=$(date +"%Y_%m_%d_%H_%M_%S")

train_id="msdiffusion"
version="${datatime}_${train_id}"
echo "version:$version"

export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTHONUNBUFFERED=1

mixed_precision="fp16"
gradient_accumulation_steps=1
resolution=512
train_batch_size=8
dataloader_num_workers=0
learning_rate=1e-04
weight_decay=0.01
checkpointing_steps=10000
num_tokens=16
image_proj_type="resampler"
image_encoder_type="clip"
latent_init_mode="grounding"
resume_from_checkpoint="latest"
checkpoints_total_limit=20
num_train_epochs=1
max_grad_norm=1.0

# VAE_PATH="/path/to/vae"  # madebyollin/sdxl-vae-fp16-fix
# IMAGE_ENCODER_PATH="/path/to/image_encoder"  # laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
# MODEL_NAME="/path/to/sdxl"  # stabilityai/stable-diffusion-xl-base-1.0
VAE_PATH="/mnt/workspace/yinu/personalized_diffusion/pretrain_models/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/4df413ca49271c25289a6482ab97a433f8117d15"  # madebyollin/sdxl-vae-fp16-fix
IMAGE_ENCODER_PATH="/mnt/workspace/yinu/personalized_diffusion/pretrain_models/huggingface/hub/models--h94--IP-Adapter/snapshots/92a2d51861c754afacf8b3aaf90845254b49f219/sdxl_models/image_encoder"  # laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
MODEL_NAME="/mnt/workspace/yinu/personalized_diffusion/pretrain_models/modelscope/hub/AI-ModelScope/stable-diffusion-xl-base-1.0"  # stabilityai/stable-diffusion-xl-base-1.0

if [ ${local_test} -eq 1 ];
then
    OUTPUT_DIR="./output"
    RES_DIR="./res"

    CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file configs/accelerate/local.yaml tutorial_train_sdxl.py \
        --pretrained_model_name_or_path=$MODEL_NAME  \
        --output_dir=$OUTPUT_DIR \
        --result_dir=$RES_DIR \
        --image_encoder_path=$IMAGE_ENCODER_PATH \
        --mixed_precision=$mixed_precision \
        --gradient_accumulation_steps=$gradient_accumulation_steps \
        --resolution=$resolution \
        --num_train_epochs=$num_train_epochs \
        --train_batch_size=$train_batch_size \
        --dataloader_num_workers=$dataloader_num_workers \
        --learning_rate=$learning_rate \
        --weight_decay=$weight_decay \
        --checkpointing_steps=$checkpointing_steps \
        --num_tokens=$num_tokens \
        --pretrained_vae_model_name_or_path=$VAE_PATH \
        --image_proj_type=$image_proj_type \
        --image_encoder_type=$image_encoder_type \
        --latent_init_mode=$latent_init_mode \
        --checkpoints_total_limit=$checkpoints_total_limit \
        --train_id=$train_id \
        --max_grad_norm=$max_grad_norm \
        --resume_from_checkpoint=$resume_from_checkpoint
fi