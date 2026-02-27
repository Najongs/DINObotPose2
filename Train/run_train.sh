#!/bin/bash

# DINOv3 Pose Estimation Training Script
# This script provides various training configurations

# =============================================================================
# Configuration
# =============================================================================

# Data paths (REQUIRED - Update these paths!)
DATA_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"  # Training data directory
VAL_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"  

# DATA_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"  # Training data directory
# VAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"  # Validation data directory (separate from training)

TRAIN_SPLIT=1.0  # Train split ratio (1.0 = use all training data when VAL_DIR is specified)
VAL_SPLIT=0.5  # Validation data usage ratio (0.1 = use 10% of validation data)

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512
UNFREEZE_BLOCKS=2  # Number of backbone blocks to unfreeze for fine-tuning

USE_JOINT_EMBEDDING=True  # Enable joint identity embeddings in 3D head

# Joint angle mode loss weights
# angle loss: 라디안 단위 (범위 ~0~6), 3D loss: 미터 단위 (범위 ~0.01~0.5)
# FK_3D는 robot frame 기준이므로 실제 성능 지표(camera frame ADD)와 좌표계가 다름
# → angle loss로 자세 추정 → FK_3D로 구조적 일관성 강제 순서로 학습
ANGLE_WEIGHT=10.0    # Joint angle MSE loss weight
FK_3D_WEIGHT=100.0   # FK 3D keypoint MSE loss weight (robot frame)

# Iterative Refinement (joint_angle mode only)
USE_ITERATIVE_REFINEMENT=True  # Enable render-and-compare refinement loop
REFINEMENT_ITERATIONS=3        # Number of refinement iterations
REFINEMENT_WEIGHT=50.0         # Refinement loss weight

# Loss weights
HEATMAP_WEIGHT=1.0
KP3D_WEIGHT=100.0
HEATMAP_ONLY_TRAIN=False  # True: train only 2D heatmap branch

# FDA (Fourier Domain Adaptation) for sim-to-real
FDA_REAL_DIR="/data/public/NAS/DINObotPose2/Dataset/DREAM_real"  # Real images (no labels needed)
# FDA_REAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/DREAM_real"
FDA_BETA=0.001   # Low-freq replacement ratio (0.01=subtle tone shift, 0.05=strong)
FDA_PROB=0.5    # Probability of applying FDA per sample (0.0 to disable)

# Training hyperparameters
EPOCHS=100
BATCH_SIZE=16
NUM_WORKERS=4
OPTIMIZER="adam"  # Options: adam, adamw, sgd
LEARNING_RATE=1e-4
MIN_LR=1e-8
WEIGHT_DECAY=1e-5
SCHEDULER="cosine"  # Options: step, cosine, plateau, none
WARMUP_STEPS=200
WARMUP_START_LR=1e-8

# Loss configuration
LOSS_TYPE="smoothl1"  # Loss function type: mse, l1, smoothl1 (smoothl1 recommended for ADD AUC)

# Output and logging
OUTPUT_DIR="./outputs/dinov3_base_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="dinov3-pose-estimation"
WANDB_RUN_NAME="dinov3_base_$(date +%Y%m%d_%H%M%S)"

# Other settings
SEED=42
RESUME=""  # Path to checkpoint for resuming (leave empty for new training)
RESUME_LR=""  # Learning rate to use when resuming (leave empty for automatic calculation from scheduler)
LOAD_2D_HEAD="/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260228_013413/epoch_9.pth"  # Path to checkpoint for loading pretrained 2D heatmap head (leave empty to train from scratch)
FREEZE_2D_HEAD_EPOCHS=3  # If LOAD_2D_HEAD is set, freeze loaded 2D head for first N epochs then unfreeze

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

# Build iterative refinement flag
if [ "${USE_ITERATIVE_REFINEMENT}" = "True" ] || [ "${USE_ITERATIVE_REFINEMENT}" = "true" ]; then
    REFINEMENT_FLAG="--use-iterative-refinement --refinement-iterations ${REFINEMENT_ITERATIONS} --refinement-weight ${REFINEMENT_WEIGHT}"
else
    REFINEMENT_FLAG=""
fi

# Build heatmap-only training flag
if [ "${HEATMAP_ONLY_TRAIN}" = "True" ] || [ "${HEATMAP_ONLY_TRAIN}" = "true" ]; then
    HEATMAP_ONLY_FLAG="--heatmap-only-train"
else
    HEATMAP_ONLY_FLAG=""
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
    ${HEATMAP_ONLY_FLAG} \
    --angle-weight ${ANGLE_WEIGHT} \
    --fk-3d-weight ${FK_3D_WEIGHT} \
    ${REFINEMENT_FLAG} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --optimizer ${OPTIMIZER} \
    --learning-rate ${LEARNING_RATE} \
    --min-lr ${MIN_LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --scheduler ${SCHEDULER} \
    --warmup-steps ${WARMUP_STEPS} \
    --warmup-start-lr ${WARMUP_START_LR} \
    --freeze-2d-head-epochs ${FREEZE_2D_HEAD_EPOCHS} \
    --loss-type ${LOSS_TYPE} \
    --heatmap-weight ${HEATMAP_WEIGHT} \
    --kp3d-weight ${KP3D_WEIGHT} \
    --output-dir ${OUTPUT_DIR} \
    --wandb-project ${WANDB_PROJECT} \
    --seed ${SEED}"

# Add validation directory if specified
if [ -n "${VAL_DIR}" ]; then
    BASE_CMD="${BASE_CMD} --val-dir ${VAL_DIR} --val-split ${VAL_SPLIT}"
fi

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

# Add resume learning rate if specified
if [ -n "${RESUME_LR}" ]; then
    BASE_CMD="${BASE_CMD} --resume-lr ${RESUME_LR}"
fi

# Add pretrained 2D head checkpoint if specified
if [ -n "${LOAD_2D_HEAD}" ]; then
    BASE_CMD="${BASE_CMD} --load-2d-head ${LOAD_2D_HEAD}"
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
        $([ -n "${VAL_DIR}" ] && echo "--val-dir ${VAL_DIR} --val-split ${VAL_SPLIT}") \
        --model-name ${MODEL_NAME} \
        --image-size ${IMAGE_SIZE} \
        --heatmap-size ${HEATMAP_SIZE} \
        --unfreeze-blocks ${UNFREEZE_BLOCKS} \
        ${JOINT_EMBEDDING_FLAG} \
        ${HEATMAP_ONLY_FLAG} \
        --angle-weight ${ANGLE_WEIGHT} \
        --fk-3d-weight ${FK_3D_WEIGHT} \
        ${REFINEMENT_FLAG} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --optimizer ${OPTIMIZER} \
        --learning-rate ${LEARNING_RATE} \
        --min-lr ${MIN_LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --scheduler ${SCHEDULER} \
        --warmup-steps ${WARMUP_STEPS} \
        --warmup-start-lr ${WARMUP_START_LR} \
        --freeze-2d-head-epochs ${FREEZE_2D_HEAD_EPOCHS} \
        --loss-type ${LOSS_TYPE} \
        --heatmap-weight ${HEATMAP_WEIGHT} \
        --kp3d-weight ${KP3D_WEIGHT} \
        --output-dir ${OUTPUT_DIR} \
        --wandb-project ${WANDB_PROJECT} \
        $([ -n "${WANDB_RUN_NAME}" ] && echo "--wandb-run-name ${WANDB_RUN_NAME}") \
        $([ -n "${RESUME}" ] && echo "--resume ${RESUME}") \
        $([ -n "${RESUME_LR}" ] && echo "--resume-lr ${RESUME_LR}") \
        $([ -n "${LOAD_2D_HEAD}" ] && echo "--load-2d-head ${LOAD_2D_HEAD}") \
        $([ -n "${FDA_REAL_DIR}" ] && [ "${FDA_PROB}" != "0.0" ] && echo "--fda-real-dir ${FDA_REAL_DIR} --fda-beta ${FDA_BETA} --fda-prob ${FDA_PROB}") \
        --seed ${SEED}
else
    echo "Error: Unknown training mode: ${TRAIN_MODE}"
    exit 1
fi

echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
