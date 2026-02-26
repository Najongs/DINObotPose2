#!/bin/bash
# Qualitative visualization: mesh overlay + keypoint skeleton
# Usage: bash Vis/run_visualize.sh

MODEL_PATH="/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260226_112034/best_model.pth"
OUTPUT_DIR="/data/public/NAS/DINObotPose2/Vis/qualitative_output"
DATA_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_photo"

# Select sample images (evenly spaced, odd-numbered files only)
JSON_FILES=(
    "${DATA_DIR}/000100.json"
    "${DATA_DIR}/000638.json"
    "${DATA_DIR}/001000.json"
)

conda run -n dino python visualize_qualitative.py \
    -j ${JSON_FILES[@]} \
    -p "${MODEL_PATH}" \
    -o "${OUTPUT_DIR}"
