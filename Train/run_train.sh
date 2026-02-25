#!/bin/bash

# DINOv3 Pose Estimation Training Script
# This script provides various training configurations

# =============================================================================
# Configuration
# =============================================================================

# Data paths (REQUIRED - Update these paths!)
DATA_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn"  # Training data directory
TRAIN_SPLIT=0.9  # Train/Val split ratio (0.9 = 90% train, 10% val)

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512
UNFREEZE_BLOCKS=2  # Number of backbone blocks to unfreeze for fine-tuning

USE_JOINT_EMBEDDING=True  # Enable joint identity embeddings in 3D head

# FDA (Fourier Domain Adaptation) for sim-to-real
FDA_REAL_DIR="/data/public/NAS/DINObotPose2/Dataset/DREAM_real"  # Real images (no labels needed)
FDA_BETA=0.01   # Low-freq replacement ratio (0.01=subtle tone shift, 0.05=strong)
FDA_PROB=0.5    # Probability of applying FDA per sample (0.0 to disable)

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
KP3D_WEIGHT=1.0

# Output and logging
OUTPUT_DIR="./outputs/dinov3_base_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="dinov3-pose-estimation"
WANDB_RUN_NAME="dinov3_base_$(date +%Y%m%d_%H%M%S)"

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
NUM_GPUS=3  # 사용할 GPU 개수 (single GPU는 1로 설정)
GPU_IDS="0,1,2"  # 사용할 GPU ID (예: "0,1,2,3")

# =============================================================================
# Execute Training
# =============================================================================

# Build joint embedding flag
if [ "${USE_JOINT_EMBEDDING}" = "True" ] || [ "${USE_JOINT_EMBEDDING}" = "true" ]; then
    JOINT_EMBEDDING_FLAG="--use-joint-embedding"
else
    JOINT_EMBEDDING_FLAG=""
fi

# Build base command
BASE_CMD="python train.py \
    --data-dir ${DATA_DIR} \
    --train-split ${TRAIN_SPLIT} \
    --model-name ${MODEL_NAME} \
    --image-size ${IMAGE_SIZE} \
    --heatmap-size ${HEATMAP_SIZE} \
    --unfreeze-blocks ${UNFREEZE_BLOCKS} \
    ${JOINT_EMBEDDING_FLAG} \
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
    --output-dir ${OUTPUT_DIR} \
    --wandb-project ${WANDB_PROJECT} \
    --seed ${SEED}"

# Add FDA flags
if [ -n "${FDA_REAL_DIR}" ] && [ "${FDA_PROB}" != "0.0" ]; then
    BASE_CMD="${BASE_CMD} --fda-real-dir ${FDA_REAL_DIR} --fda-beta ${FDA_BETA} --fda-prob ${FDA_PROB}"
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
        ${JOINT_EMBEDDING_FLAG} \
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
        --output-dir ${OUTPUT_DIR} \
        --wandb-project ${WANDB_PROJECT} \
        $([ -n "${WANDB_RUN_NAME}" ] && echo "--wandb-run-name ${WANDB_RUN_NAME}") \
        $([ -n "${RESUME}" ] && echo "--resume ${RESUME}") \
        $([ -n "${FDA_REAL_DIR}" ] && [ "${FDA_PROB}" != "0.0" ] && echo "--fda-real-dir ${FDA_REAL_DIR} --fda-beta ${FDA_BETA} --fda-prob ${FDA_PROB}") \
        --seed ${SEED}
else
    echo "Error: Unknown training mode: ${TRAIN_MODE}"
    exit 1
fi

echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
