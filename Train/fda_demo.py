"""
FDA (Fourier Domain Adaptation) Demo
- Synthetic 이미지의 저주파 스펙트럼을 Real 이미지의 것으로 교체
- Label은 그대로 유지하면서 이미지만 real-looking으로 변환
- Reference: "FDA: Fourier Domain Adaptation for Semantic Segmentation" (Yang et al., CVPR 2020)
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def fda_transfer(src_img: np.ndarray, trg_img: np.ndarray, beta: float = 0.01) -> np.ndarray:
    """
    FDA: 소스 이미지의 저주파(low-frequency) 성분을 타겟 이미지의 것으로 교체.

    저주파 = 전체적인 색감, 밝기, 톤 (domain-specific)
    고주파 = 엣지, 텍스처, 구조 (task-relevant)

    Args:
        src_img: Synthetic 이미지 (H, W, 3), float32 [0, 255]
        trg_img: Real 이미지 (H, W, 3), float32 [0, 255]
        beta: 교체할 저주파 영역 비율 (0.01 ~ 0.15)
              작을수록 색감만 살짝, 클수록 더 많이 real처럼
    Returns:
        FDA 적용된 이미지 (H, W, 3), uint8 [0, 255]
    """
    src = src_img.astype(np.float32)
    trg = trg_img.astype(np.float32)

    # 타겟을 소스 크기에 맞춤
    if src.shape != trg.shape:
        trg = np.array(Image.fromarray(trg.astype(np.uint8)).resize(
            (src.shape[1], src.shape[0]), Image.BILINEAR
        )).astype(np.float32)

    result = np.zeros_like(src)

    for ch in range(3):
        # FFT
        fft_src = np.fft.fft2(src[:, :, ch])
        fft_trg = np.fft.fft2(trg[:, :, ch])

        # 중심으로 이동 (저주파가 중앙에 오도록)
        fft_src_shifted = np.fft.fftshift(fft_src)
        fft_trg_shifted = np.fft.fftshift(fft_trg)

        # 진폭(amplitude)과 위상(phase) 분리
        amp_src = np.abs(fft_src_shifted)
        amp_trg = np.abs(fft_trg_shifted)
        phase_src = np.angle(fft_src_shifted)

        # 저주파 마스크 생성 (중앙의 beta 비율만큼)
        h, w = src.shape[:2]
        cy, cx = h // 2, w // 2
        bh, bw = int(h * beta), int(w * beta)
        bh = max(bh, 1)
        bw = max(bw, 1)

        # 소스의 저주파 진폭 → 타겟의 저주파 진폭으로 교체
        amp_src[cy - bh:cy + bh, cx - bw:cx + bw] = amp_trg[cy - bh:cy + bh, cx - bw:cx + bw]

        # 수정된 진폭 + 원래 위상으로 복원
        fft_result = amp_src * np.exp(1j * phase_src)
        fft_result = np.fft.ifftshift(fft_result)
        result[:, :, ch] = np.real(np.fft.ifft2(fft_result))

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    syn_path = "/data/public/NAS/DINObotPose2/Dataset/DREAM_syn/panda_synth_train_dr/000007.rgb.jpg"
    real_path = "/data/public/NAS/DINObotPose2/Dataset/DREAM_real/panda-3cam_realsense/panda-3cam_realsense/000000.rgb.jpg"

    syn_img = np.array(Image.open(syn_path).convert("RGB"))
    real_img = np.array(Image.open(real_path).convert("RGB"))

    # 다양한 beta 값으로 FDA 적용
    betas = [0.01, 0.03, 0.05, 0.09, 0.15]
    fda_results = []
    for b in betas:
        fda_results.append(fda_transfer(syn_img, real_img, beta=b))

    # 시각화
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 상단: 원본들 + 대표 FDA 결과
    axes[0, 0].imshow(syn_img)
    axes[0, 0].set_title("Synthetic (Original)", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(real_img)
    axes[0, 1].set_title("Real (Style Source)", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(fda_results[0])
    axes[0, 2].set_title(f"FDA β={betas[0]}\n(subtle)", fontsize=13)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(fda_results[2])
    axes[0, 3].set_title(f"FDA β={betas[2]}\n(moderate)", fontsize=13)
    axes[0, 3].axis('off')

    # 하단: 전체 beta sweep
    for i, (b, result) in enumerate(zip(betas, fda_results)):
        if i < 4:
            axes[1, i].imshow(result)
            axes[1, i].set_title(f"β = {b}", fontsize=13)
            axes[1, i].axis('off')
    # 마지막 beta가 5개인데 칸이 4개이므로 4번째 칸에 마지막 것
    if len(betas) > 4:
        axes[1, 3].imshow(fda_results[-1])
        axes[1, 3].set_title(f"β = {betas[-1]}\n(aggressive)", fontsize=13)
        axes[1, 3].axis('off')

    plt.suptitle("FDA (Fourier Domain Adaptation): Synthetic → Real Style Transfer\n"
                 "Low-freq (color/tone) from Real + High-freq (structure/edges) from Synthetic",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = "/data/public/NAS/DINObotPose2/Train/fda_demo_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
