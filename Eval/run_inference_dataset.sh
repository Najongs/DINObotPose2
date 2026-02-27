#!/bin/bash

# Dataset Inference Script for DINOv3 Pose Estimation
# Evaluates trained model on panda-3cam_azure dataset

# Model configuration (update with your trained model path)
MODEL_PATH="/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260227_021707/best_model.pth"

# Dataset (choose one of the following datasets)
DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"
# /home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure
# /home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_kinect360
# /home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_realsense
# /home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-orb
# Output
OUTPUT_DIR="./eval_outputs"

# Inference parameters
BATCH_SIZE=64
NUM_WORKERS=4

# Metrics thresholds (matching DREAM defaults)
KP_AUC_THRESHOLD=20.0    # pixels
ADD_AUC_THRESHOLD=0.1    # meters

# Run inference (model-name, image-size, heatmap-size are read from config.yaml automatically)
python inference_dataset.py \
    --model-path "$MODEL_PATH" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --kp-auc-threshold $KP_AUC_THRESHOLD \
    --add-auc-threshold $ADD_AUC_THRESHOLD
