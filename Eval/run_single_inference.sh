#!/bin/bash

# Single Image Inference Script for DINOv3 Pose Estimation

# Model configuration
MODEL_PATH="/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260225_125129/best_model.pth"

# Input image (change this to your image path)
IMAGE_PATH="/data/public/NAS/DINObotPose2/Dataset/DREAM_real/panda-orb/panda-orb/003000.rgb.jpg"

# Output directory
OUTPUT_DIR="./inference_output"

# Run inference (model-name, image-size, heatmap-size are read from config.yaml automatically)
python inference_single_image.py \
    --model-path "$MODEL_PATH" \
    --image-path "$IMAGE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --save-heatmaps \
    --save-combined

echo ""
echo "Inference completed! Check results in: $OUTPUT_DIR"
