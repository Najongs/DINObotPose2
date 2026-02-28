#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model and dataset
MODEL_PATH="/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260228_161218/best_model.pth"
DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"

# Output
OUTPUT_DIR="${SCRIPT_DIR}/eval_outputs_outlier"

# Inference
BATCH_SIZE=64
NUM_WORKERS=4

# Metrics thresholds
KP_AUC_THRESHOLD=20.0
ADD_AUC_THRESHOLD=0.1

# Outlier report
OUTLIER_TOPK=200

# Execution mode
INFER_MODE="multi_gpu"
NUM_GPUS=3
GPU_IDS="0,1,2"

if [ "${INFER_MODE}" = "single_gpu" ]; then
    echo "Running single-GPU outlier analysis..."
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    python "${SCRIPT_DIR}/inference_dataset.py" \
        --model-path "$MODEL_PATH" \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --kp-auc-threshold $KP_AUC_THRESHOLD \
        --add-auc-threshold $ADD_AUC_THRESHOLD \
        --save-per-frame-errors \
        --outlier-topk $OUTLIER_TOPK
elif [ "${INFER_MODE}" = "multi_gpu" ]; then
    echo "Running distributed outlier analysis with ${NUM_GPUS} GPUs..."
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=${NUM_GPUS} \
        "${SCRIPT_DIR}/inference_dataset.py" \
        --distributed \
        --model-path "$MODEL_PATH" \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --kp-auc-threshold $KP_AUC_THRESHOLD \
        --add-auc-threshold $ADD_AUC_THRESHOLD \
        --save-per-frame-errors \
        --outlier-topk $OUTLIER_TOPK
else
    echo "Error: Unknown INFER_MODE=${INFER_MODE}"
    exit 1
fi

echo "Outlier analysis completed."
echo "Check files in: ${OUTPUT_DIR}"
echo "  - eval_results.json"
echo "  - per_frame_3d_errors.json"
echo "  - outlier_topk_3d_errors.json"
echo "  - per_keypoint_3d_error_summary.json"
