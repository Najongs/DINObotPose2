#!/bin/bash

# DINOv3 Pose Estimation Training Script
# This script provides various training configurations

# =============================================================================
# Configuration
# =============================================================================

# Data paths (REQUIRED - Update these paths!)
DATA_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"  # Training data directory
TRAIN_SPLIT=0.9  # Train/Val split ratio (0.9 = 90% train, 10% val)

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512
UNFREEZE_BLOCKS=2  # Number of backbone blocks to unfreeze for fine-tuning

USE_CNN_STEM=True
USE_ROBOT_CLASSIFIER=False  # Enable robot type classification (4 classes)

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=16
NUM_WORKERS=4
OPTIMIZER="adam"  # Options: adam, adamw, sgd
LEARNING_RATE=5e-4
MIN_LR=1e-8
WEIGHT_DECAY=1e-5
SCHEDULER="cosine"  # Options: step, cosine, plateau, none

# Loss weights
HEATMAP_WEIGHT=1.0
KP3D_WEIGHT=1000.0
ROBOT_CLASS_WEIGHT=1.0

# Output and logging
OUTPUT_DIR="./outputs/dinov3_base_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="dinov3-pose-estimation"
WANDB_RUN_NAME="dinov2-base-cosine"

# Other settings
SEED=42
RESUME=""  # Path to checkpoint for resuming (leave empty for new training)

# =============================================================================
# Training Modes
# =============================================================================

# Choose training mode by uncommenting one of the following:

# --- Single GPU Training ---
# TRAIN_MODE="single_gpu"

# --- Multi-GPU Training (Distributed Data Parallel) ---
TRAIN_MODE="multi_gpu"
NUM_GPUS=5  # 사용할 GPU 개수 (single GPU는 1로 설정)
GPU_IDS="0,1,2,3,4"  # 사용할 GPU ID (예: "0,1,2,3")

# =============================================================================
# Execute Training
# =============================================================================

# Build base command
BASE_CMD="python train.py \
    --data-dir ${DATA_DIR} \
    --train-split ${TRAIN_SPLIT} \
    --model-name ${MODEL_NAME} \
    --image-size ${IMAGE_SIZE} \
    --heatmap-size ${HEATMAP_SIZE} \
    --unfreeze-blocks ${UNFREEZE_BLOCKS} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --optimizer ${OPTIMIZER} \
    --learning-rate ${LEARNING_RATE} \
    --min-lr ${MIN_LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --scheduler ${SCHEDULER} \
    --heatmap-weight ${HEATMAP_WEIGHT} \
    --kp3d-weight ${KP3D_WEIGHT} \
    --robot-class-weight ${ROBOT_CLASS_WEIGHT} \
    --output-dir ${OUTPUT_DIR} \
    --wandb-project ${WANDB_PROJECT} \
    --seed ${SEED}"

# Add CNN stem flag
if [ "${USE_CNN_STEM}" = "True" ] || [ "${USE_CNN_STEM}" = "true" ]; then
    BASE_CMD="${BASE_CMD} --use-cnn-stem"
else
    BASE_CMD="${BASE_CMD} --no-cnn-stem"
fi

# Add robot classifier flag
if [ "${USE_ROBOT_CLASSIFIER}" = "True" ] || [ "${USE_ROBOT_CLASSIFIER}" = "true" ]; then
    BASE_CMD="${BASE_CMD} --use-robot-classifier"
fi

# Add WandB run name if specified
if [ -n "${WANDB_RUN_NAME}" ]; then
    BASE_CMD="${BASE_CMD} --wandb-run-name ${WANDB_RUN_NAME}"
fi

# Add resume checkpoint if specified
if [ -n "${RESUME}" ]; then
    BASE_CMD="${BASE_CMD} --resume ${RESUME}"
fi

# Execute based on training mode
if [ "${TRAIN_MODE}" = "single_gpu" ]; then
    echo "Starting single GPU training..."
    echo "Using GPU(s): ${GPU_IDS}"
    echo "Output directory: ${OUTPUT_DIR}"

    # Set CUDA_VISIBLE_DEVICES for single GPU
    export CUDA_VISIBLE_DEVICES=${GPU_IDS}
    eval ${BASE_CMD}

elif [ "${TRAIN_MODE}" = "multi_gpu" ]; then
    echo "Starting multi-GPU training with ${NUM_GPUS} GPUs..."
    echo "Using GPU(s): ${GPU_IDS}"
    echo "Output directory: ${OUTPUT_DIR}"

    # Set CUDA_VISIBLE_DEVICES for multi-GPU
    export CUDA_VISIBLE_DEVICES=${GPU_IDS}

    # Build CNN stem flag
    if [ "${USE_CNN_STEM}" = "True" ] || [ "${USE_CNN_STEM}" = "true" ]; then
        CNN_STEM_FLAG="--use-cnn-stem"
    else
        CNN_STEM_FLAG="--no-cnn-stem"
    fi

    # Build robot classifier flag
    if [ "${USE_ROBOT_CLASSIFIER}" = "True" ] || [ "${USE_ROBOT_CLASSIFIER}" = "true" ]; then
        ROBOT_CLASSIFIER_FLAG="--use-robot-classifier"
    else
        ROBOT_CLASSIFIER_FLAG=""
    fi

    # Use torchrun for distributed training
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=${NUM_GPUS} \
        train.py \
        --data-dir ${DATA_DIR} \
        --train-split ${TRAIN_SPLIT} \
        --model-name ${MODEL_NAME} \
        --image-size ${IMAGE_SIZE} \
        --heatmap-size ${HEATMAP_SIZE} \
        --unfreeze-blocks ${UNFREEZE_BLOCKS} \
        ${CNN_STEM_FLAG} \
        ${ROBOT_CLASSIFIER_FLAG} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --optimizer ${OPTIMIZER} \
        --learning-rate ${LEARNING_RATE} \
        --min-lr ${MIN_LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --scheduler ${SCHEDULER} \
        --heatmap-weight ${HEATMAP_WEIGHT} \
        --kp3d-weight ${KP3D_WEIGHT} \
        --robot-class-weight ${ROBOT_CLASS_WEIGHT} \
        --output-dir ${OUTPUT_DIR} \
        --wandb-project ${WANDB_PROJECT} \
        $([ -n "${WANDB_RUN_NAME}" ] && echo "--wandb-run-name ${WANDB_RUN_NAME}") \
        $([ -n "${RESUME}" ] && echo "--resume ${RESUME}") \
        --seed ${SEED}
else
    echo "Error: Unknown training mode: ${TRAIN_MODE}"
    exit 1
fi

echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
