#!/bin/bash

# Real Image Inference Script for DINOv3 Pose Estimation
# Input: JSON annotation file (contains image path + GT keypoints)
# Output: GT vs Prediction visualizations + error metrics

# =============================================================================
# Configuration
# =============================================================================

# Model checkpoint
# MODEL_PATH="/home/najo/NAS/DIP/DINObotPose2/Train/outputs/dinov3_base_20260226_161726/best_model.pth"
MODEL_PATH="/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260227_021707/best_model.pth"

# Input annotation JSON (contains image_path + GT keypoints + camera K)
JSON_PATH="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure/000006.json"
# JSON_PATH="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure/000000.json"

# Output directory
OUTPUT_DIR="./real_inference_output"

# =============================================================================
# Run Inference
# =============================================================================

echo "=========================================="
echo "  Real Image Inference (GT vs Prediction)"
echo "=========================================="
echo "  Model: ${MODEL_PATH}"
echo "  JSON:  ${JSON_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python inference_with_real.py \
    -j "${JSON_PATH}" \
    -p "${MODEL_PATH}" \
    -o "${OUTPUT_DIR}"

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "  01_gt_vs_pred_keypoints.png  - Green=GT, Red=Prediction"
echo "  02_belief_map_mosaic.png     - Per-joint belief maps"
echo "  03_belief_maps_overlay_mosaic.png - Belief maps on image"
echo "  04_combined_on_original.png  - Combined heatmap on original"
echo "  metrics.json                 - Per-keypoint error metrics"
