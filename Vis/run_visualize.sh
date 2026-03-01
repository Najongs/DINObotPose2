#!/bin/bash
# Qualitative visualization: mesh overlay + keypoint skeleton
# Usage: bash Vis/run_visualize.sh [optional_model_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${ROOT_DIR}/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_photo"
OUTPUT_DIR="${ROOT_DIR}/Vis/qualitative_output"

MODEL_PATH="/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260301_011937/best_model.pth"
if [[ $# -ge 1 ]]; then
    MODEL_PATH="$1"
fi
# Pick representative samples (edit as needed)
JSON_FILES=(
    "${DATA_DIR}/000001.json"
    "${DATA_DIR}/000638.json"
    "${DATA_DIR}/001000.json"
)

for jf in "${JSON_FILES[@]}"; do
    if [[ ! -f "${jf}" ]]; then
        echo "[ERROR] JSON not found: ${jf}"
        exit 1
    fi
done

echo "[INFO] Model: ${MODEL_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "[ERROR] Model not found: ${MODEL_PATH}"
    exit 1
fi
USE_MESH="${USE_MESH:-0}"
KP_MIN_CONFIDENCE="${KP_MIN_CONFIDENCE:-0.20}"
if [[ "${USE_MESH}" == "1" ]]; then
    echo "[INFO] Running visualization with mesh overlay (IK enabled)"
    MESH_FLAG=()
else
    echo "[INFO] Running visualization (skeleton + 3D metrics, mesh disabled by default)"
    MESH_FLAG=(--no-mesh)
fi

FORCE_IK="${FORCE_IK:-0}"
if [[ "${FORCE_IK}" == "1" ]]; then
    echo "[INFO] Forcing IK for predicted mesh"
    IK_FLAG=(--force-ik)
else
    IK_FLAG=()
fi

conda run -n dino python "${SCRIPT_DIR}/visualize_qualitative.py" \
    -j "${JSON_FILES[@]}" \
    -p "${MODEL_PATH}" \
    -o "${OUTPUT_DIR}" \
    --kp-min-confidence "${KP_MIN_CONFIDENCE}" \
    "${MESH_FLAG[@]}" \
    "${IK_FLAG[@]}"
