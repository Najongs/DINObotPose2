#!/bin/bash

# DINOv3 3D Pose Training Script (Joint Angle + Camera-frame 3D)
# 2D Heatmap backbone은 freeze하고 3D Head만 학습하는 스크립트

# =============================================================================
# Global Configuration
# =============================================================================

# GPU Settings
GPU_IDS="0,1,2,3,4"
NUM_GPUS=5
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Data paths
TRAIN_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"

# 2D Pretrained Checkpoint (필수)
# 2D Heatmap 학습이 완료된 'best_heatmap.pth' 경로를 입력하세요.
CHECKPOINT="/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_heatmap/heatmap_only_20260304_002551_0.05/best_heatmap.pth"

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512

# Training hyperparameters
EPOCHS=30
BATCH_SIZE=16  # GPU당 배치 사이즈 (5개 GPU 사용 시 Total 40)
NUM_WORKERS=4
LEARNING_RATE=1e-4
ANGLE_WEIGHT=10.0
CAMERA_3D_WEIGHT=1.0

# WANDB Settings
WANDB_PROJECT="dinov3-3d-pose"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="train_3d_${TIMESTAMP}"
OUTPUT_DIR="./outputs_3d/train_3d_${TIMESTAMP}"

# =============================================================================
# Execution
# =============================================================================

echo "============================================================================="
echo "==> STARTING 3D POSE TRAINING"
echo "==> 2D Checkpoint: ${CHECKPOINT}"
echo "==> Output: ${OUTPUT_DIR}"
echo "============================================================================="

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} train_3d.py \
    --train-dir "${TRAIN_DIR}" \
    --val-dir "${VAL_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --image-size ${IMAGE_SIZE} \
    --heatmap-size ${HEATMAP_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --angle-weight ${ANGLE_WEIGHT} \
    --camera-3d-weight ${CAMERA_3D_WEIGHT} \
    --num-workers ${NUM_WORKERS} \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${RUN_NAME}"

echo "==> 3D Training Completed!"
