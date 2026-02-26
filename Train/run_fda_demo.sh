#!/bin/bash

# FDA Demo 실행 스크립트
# 사용법: bash run_fda_demo.sh

# =============================================================================
# 이미지 경로 설정
# =============================================================================

# Synthetic 이미지 (FDA 적용 대상)
# SYN_PATH="/data/public/NAS/DINObotPose2/Dataset/DREAM_syn/panda_synth_train_dr/000007.rgb.jpg"
SYN_PATH="/data/public/NAS/DINObotPose2/Dataset/DREAM_syn/panda_synth_test_photo/000001.rgb.jpg"
# Real 이미지 (스타일 소스)
REAL_PATH="/data/public/NAS/DINObotPose2/Dataset/DREAM_real/panda-3cam_realsense/panda-3cam_realsense/000300.rgb.jpg"

# =============================================================================
# FDA 파라미터
# =============================================================================

# Beta 값들 (쉼표 구분, 예: "0.005,0.01,0.03,0.05,0.09,0.15")
BETAS="0.001,0.005,0.01"

# Soft mask 사용 여부 (true/false)
USE_SOFT=false

# =============================================================================
# 출력 설정
# =============================================================================

OUTPUT_DEMO="/data/public/NAS/DINObotPose2/Train/fda_demo_result.png"
OUTPUT_SPECTRUM="/data/public/NAS/DINObotPose2/Train/fda_spectrum_result.png"
OUTPUT_DIFF="/data/public/NAS/DINObotPose2/Train/fda_diff_map.png"

# =============================================================================
# 실행
# =============================================================================

# Soft flag
if [ "${USE_SOFT}" = "true" ]; then
    SOFT_FLAG="--soft"
else
    SOFT_FLAG=""
fi

# 1) Hard vs Soft 비교 (fda_demo.py)
echo "=== FDA Demo (Hard vs Soft comparison) ==="
conda run -n dino python fda_demo.py \
    --syn-path "${SYN_PATH}" \
    --real-path "${REAL_PATH}" \
    --betas ${BETAS} \
    --output "${OUTPUT_DEMO}"

# 2) Spectrum map (fda_spectrum.py)
echo ""
echo "=== FDA Spectrum Map ==="
conda run -n dino python fda_spectrum.py \
    --syn-path "${SYN_PATH}" \
    --real-path "${REAL_PATH}" \
    --betas ${BETAS} \
    ${SOFT_FLAG} \
    --output "${OUTPUT_SPECTRUM}"

# 3) Diff map: Syn + Delta = FDA (fda_diff_map.py)
echo ""
echo "=== FDA Diff Map ==="
conda run -n dino python fda_diff_map.py \
    --syn-path "${SYN_PATH}" \
    --real-path "${REAL_PATH}" \
    --betas ${BETAS} \
    ${SOFT_FLAG} \
    --output "${OUTPUT_DIFF}"
