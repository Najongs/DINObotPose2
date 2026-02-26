#!/bin/bash

# DINOv3 Pose Estimation Training Script
# This script provides various training configurations

# =============================================================================
# Configuration
# =============================================================================

# Data paths (REQUIRED - Update these paths!)
<<<<<<< HEAD
DATA_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"  # Training data directory
# DATA_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"  # Training data directory
TRAIN_SPLIT=0.9  # Train/Val split ratio (0.9 = 90% train, 10% val)
=======
# DATA_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"  # Training data directory
DATA_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"  # Training data directory
VAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/DREAM_syn/panda_synth_test_dr"  # Validation data directory (separate from training)
TRAIN_SPLIT=1.0  # Train split ratio (1.0 = use all training data when VAL_DIR is specified)
VAL_SPLIT=0.1  # Validation data usage ratio (0.1 = use 10% of validation data)
>>>>>>> refs/remotes/origin/main

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512
UNFREEZE_BLOCKS=2  # Number of backbone blocks to unfreeze for fine-tuning

USE_JOINT_EMBEDDING=True  # Enable joint identity embeddings in 3D head
DEPTH_ONLY_3D=False  # Predict only depth (z), recover x,y from 2D heatmap + camera K
JOINT_ANGLE_3D=True  # Predict joint angles → FK → robot-frame 3D keypoints

# Joint angle mode loss weights
ANGLE_WEIGHT=1.0     # Joint angle MSE loss weight
FK_3D_WEIGHT=10.0    # FK 3D keypoint MSE loss weight (robot frame)

# FDA (Fourier Domain Adaptation) for sim-to-real
FDA_REAL_DIR="/data/public/NAS/DINObotPose2/Dataset/DREAM_real"  # Real images (no labels needed)
# FDA_REAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/DREAM_real"
FDA_BETA=0.001   # Low-freq replacement ratio (0.01=subtle tone shift, 0.05=strong)
FDA_PROB=0.5    # Probability of applying FDA per sample (0.0 to disable)

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=16
NUM_WORKERS=4
OPTIMIZER="adam"  # Options: adam, adamw, sgd
LEARNING_RATE=1e-3
MIN_LR=1e-8
WEIGHT_DECAY=1e-5
SCHEDULER="cosine"  # Options: step, cosine, plateau, none

# Loss configuration
LOSS_TYPE="smoothl1"  # Loss function type: mse, l1, smoothl1 (smoothl1 recommended for ADD AUC)

# Loss weights
HEATMAP_WEIGHT=1.0
KP3D_WEIGHT=100.0

# Output and logging
OUTPUT_DIR="./outputs/dinov3_base_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="dinov3-pose-estimation"
WANDB_RUN_NAME="dinov3_base_$(date +%Y%m%d_%H%M%S)"

# Other settings
SEED=42
RESUME=""  # Path to checkpoint for resuming (leave empty for new training)
<<<<<<< HEAD
LOAD_2D_HEAD="/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260225_231442/best_model.pth"  # Path to checkpoint for loading pretrained 2D heatmap head (leave empty to train from scratch)
=======
LOAD_2D_HEAD="/home/najo/NAS/DIP/DINObotPose2/Train/outputs/dinov3_base_20260226_161726/best_model.pth"  # Path to checkpoint for loading pretrained 2D heatmap head (leave empty to train from scratch)
>>>>>>> refs/remotes/origin/main

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

# Build joint embedding flag
if [ "${USE_JOINT_EMBEDDING}" = "True" ] || [ "${USE_JOINT_EMBEDDING}" = "true" ]; then
    JOINT_EMBEDDING_FLAG="--use-joint-embedding"
else
    JOINT_EMBEDDING_FLAG=""
fi

# Build depth-only 3D flag
if [ "${DEPTH_ONLY_3D}" = "True" ] || [ "${DEPTH_ONLY_3D}" = "true" ]; then
    DEPTH_ONLY_FLAG="--depth-only-3d"
else
    DEPTH_ONLY_FLAG=""
fi

# Build joint angle 3D flag
if [ "${JOINT_ANGLE_3D}" = "True" ] || [ "${JOINT_ANGLE_3D}" = "true" ]; then
    JOINT_ANGLE_FLAG="--joint-angle-3d --angle-weight ${ANGLE_WEIGHT} --fk-3d-weight ${FK_3D_WEIGHT}"
else
    JOINT_ANGLE_FLAG=""
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
    ${DEPTH_ONLY_FLAG} \
    ${JOINT_ANGLE_FLAG} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --optimizer ${OPTIMIZER} \
    --learning-rate ${LEARNING_RATE} \
    --min-lr ${MIN_LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --scheduler ${SCHEDULER} \
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
        ${DEPTH_ONLY_FLAG} \
        ${JOINT_ANGLE_FLAG} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --optimizer ${OPTIMIZER} \
        --learning-rate ${LEARNING_RATE} \
        --min-lr ${MIN_LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --scheduler ${SCHEDULER} \
        --loss-type ${LOSS_TYPE} \
        --heatmap-weight ${HEATMAP_WEIGHT} \
        --kp3d-weight ${KP3D_WEIGHT} \
        --output-dir ${OUTPUT_DIR} \
        --wandb-project ${WANDB_PROJECT} \
        $([ -n "${WANDB_RUN_NAME}" ] && echo "--wandb-run-name ${WANDB_RUN_NAME}") \
        $([ -n "${RESUME}" ] && echo "--resume ${RESUME}") \
        $([ -n "${LOAD_2D_HEAD}" ] && echo "--load-2d-head ${LOAD_2D_HEAD}") \
        $([ -n "${FDA_REAL_DIR}" ] && [ "${FDA_PROB}" != "0.0" ] && echo "--fda-real-dir ${FDA_REAL_DIR} --fda-beta ${FDA_BETA} --fda-prob ${FDA_PROB}") \
        --seed ${SEED}
else
    echo "Error: Unknown training mode: ${TRAIN_MODE}"
    exit 1
fi

echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
