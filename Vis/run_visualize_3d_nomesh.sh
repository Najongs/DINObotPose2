#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${1:-/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260302_152637/epoch_12.pth}"
JSON_PATH="${2:-/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure/000300.json}"
OUTPUT_DIR="${3:-${ROOT_DIR}/Vis/nomesh_3d_output}"
PRED_3D_SOURCE="${PRED_3D_SOURCE:-fk}"

echo "[INFO] model: ${MODEL_PATH}"
echo "[INFO] json: ${JSON_PATH}"
echo "[INFO] output: ${OUTPUT_DIR}"
echo "[INFO] pred-3d-source: ${PRED_3D_SOURCE}"

python "${SCRIPT_DIR}/visualize_3d_keypoints_nomesh.py" \
  -j "${JSON_PATH}" \
  -p "${MODEL_PATH}" \
  -o "${OUTPUT_DIR}" \
  --pred-3d-source "${PRED_3D_SOURCE}"

