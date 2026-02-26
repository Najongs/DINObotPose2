"""
FDA (Fourier Domain Adaptation) Demo
- Synthetic 이미지의 저주파 스펙트럼을 Real 이미지의 것으로 교체
- Label은 그대로 유지하면서 이미지만 real-looking으로 변환
- Reference: "FDA: Fourier Domain Adaptation for Semantic Segmentation" (Yang et al., CVPR 2020)
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def fda_transfer(src_img: np.ndarray, trg_img: np.ndarray, beta: float = 0.01,
                 soft: bool = False) -> np.ndarray:
    """
    FDA: 소스 이미지의 저주파(low-frequency) 성분을 타겟 이미지의 것으로 교체.

    저주파 = 전체적인 색감, 밝기, 톤 (domain-specific)
    고주파 = 엣지, 텍스처, 구조 (task-relevant)

    Args:
        src_img: Synthetic 이미지 (H, W, 3), float32 [0, 255]
        trg_img: Real 이미지 (H, W, 3), float32 [0, 255]
        beta: 교체할 저주파 영역 비율 (0.01 ~ 0.15)
        soft: True이면 Gaussian soft mask, False이면 원본 hard rectangular mask
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

    h, w = src.shape[:2]
    cy, cx = h // 2, w // 2

    # 마스크 생성 (한 번만)
    if soft:
        # Gaussian soft mask: 중앙이 1, 외곽으로 부드럽게 0
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        sigma = max(h * beta, 1)
        mask = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    else:
        # Hard rectangular mask (원본 FDA)
        mask = np.zeros((h, w), dtype=np.float32)
        bh, bw = max(int(h * beta), 1), max(int(w * beta), 1)
        mask[cy - bh:cy + bh, cx - bw:cx + bw] = 1.0

    result = np.zeros_like(src)

    for ch in range(3):
        fft_src = np.fft.fft2(src[:, :, ch])
        fft_trg = np.fft.fft2(trg[:, :, ch])

        fft_src_shifted = np.fft.fftshift(fft_src)
        fft_trg_shifted = np.fft.fftshift(fft_trg)

        amp_src = np.abs(fft_src_shifted)
        amp_trg = np.abs(fft_trg_shifted)
        phase_src = np.angle(fft_src_shifted)

        # 마스크로 블렌딩: mask=1인 곳은 trg, mask=0인 곳은 src
        amp_blended = (1 - mask) * amp_src + mask * amp_trg

        fft_result = amp_blended * np.exp(1j * phase_src)
        fft_result = np.fft.ifftshift(fft_result)
        result[:, :, ch] = np.real(np.fft.ifft2(fft_result))

    return np.clip(result, 0, 255).astype(np.uint8)


def compute_diff(img1: np.ndarray, img2: np.ndarray, amplify: float = 3.0) -> np.ndarray:
    """두 이미지 차이를 시각화 (증폭)"""
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32)) * amplify
    return np.clip(diff, 0, 255).astype(np.uint8)


def get_fft_spectrum(img: np.ndarray) -> tuple:
    """
    이미지의 FFT amplitude/phase spectrum 계산.
    Returns: (amplitude_vis, phase_vis) 각각 (H, W) log-scale 시각화용
    """
    gray = np.mean(img.astype(np.float32), axis=2)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)

    amplitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)

    # Log scale for visibility
    amp_log = np.log1p(amplitude)
    amp_vis = (amp_log - amp_log.min()) / (amp_log.max() - amp_log.min() + 1e-8)

    phase_vis = (phase - phase.min()) / (phase.max() - phase.min() + 1e-8)

    return amp_vis, phase_vis


def get_fft_per_channel(img: np.ndarray) -> np.ndarray:
    """각 채널별 FFT amplitude를 RGB로 합성하여 반환"""
    h, w = img.shape[:2]
    amp_rgb = np.zeros((h, w, 3), dtype=np.float32)

    for ch in range(3):
        fft = np.fft.fft2(img[:, :, ch].astype(np.float32))
        fft_shifted = np.fft.fftshift(fft)
        amp = np.log1p(np.abs(fft_shifted))
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
        amp_rgb[:, :, ch] = amp

    return (amp_rgb * 255).astype(np.uint8)


def get_fft_radial_profile(img: np.ndarray) -> tuple:
    """방사형 주파수 에너지 프로파일 계산"""
    gray = np.mean(img.astype(np.float32), axis=2)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    amplitude = np.abs(fft_shifted)

    h, w = amplitude.shape
    cy, cx = h // 2, w // 2

    # 각 픽셀의 중심으로부터 거리 계산
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)

    max_r = min(cx, cy)
    radial_profile = np.zeros(max_r)
    for i in range(max_r):
        mask = r == i
        if mask.sum() > 0:
            radial_profile[i] = np.mean(amplitude[mask])

    return np.log1p(radial_profile)


def main():
    parser = argparse.ArgumentParser(description='FDA Fourier Domain Analysis Demo')
    parser.add_argument('--syn-path', type=str,
                        default="/data/public/NAS/DINObotPose2/Dataset/DREAM_syn/panda_synth_train_dr/000007.rgb.jpg",
                        help='Synthetic image path')
    parser.add_argument('--real-path', type=str,
                        default="/data/public/NAS/DINObotPose2/Dataset/DREAM_real/panda-3cam_realsense/panda-3cam_realsense/000300.rgb.jpg",
                        help='Real image path (style source)')
    parser.add_argument('--betas', type=str, default="0.01,0.05,0.15",
                        help='Comma-separated beta values (e.g. "0.01,0.05,0.15")')
    parser.add_argument('--output', type=str,
                        default="/data/public/NAS/DINObotPose2/Train/fda_demo_result.png",
                        help='Output image path')
    args = parser.parse_args()

    syn_img = np.array(Image.open(args.syn_path).convert("RGB"))
    real_img = np.array(Image.open(args.real_path).convert("RGB"))
    print(f"Synthetic: {args.syn_path} ({syn_img.shape})")
    print(f"Real:      {args.real_path} ({real_img.shape})")

    # Real을 Syn 크기에 맞춤
    if syn_img.shape != real_img.shape:
        real_img = np.array(Image.fromarray(real_img).resize(
            (syn_img.shape[1], syn_img.shape[0]), Image.BILINEAR
        ))

    betas = [float(b.strip()) for b in args.betas.split(',')]
    n_betas = len(betas)

    # Hard vs Soft FDA 결과 생성
    hard_results = [fda_transfer(syn_img, real_img, beta=b, soft=False) for b in betas]
    soft_results = [fda_transfer(syn_img, real_img, beta=b, soft=True) for b in betas]

    # ================================================================
    # 시각화: Hard vs Soft 비교
    # Row 0: Syn | Real | Hard mask 시각화 | Soft mask 시각화
    # Row 1~N: 각 beta별 [Hard image | Hard diff | Soft image | Soft diff]
    # Last Row: Radial profile
    # ================================================================

    n_rows = 1 + n_betas + 1  # header + betas + radial
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))

    # --- Row 0: 원본 + 마스크 시각화 ---
    axes[0, 0].imshow(syn_img)
    axes[0, 0].set_title("Synthetic (Original)", fontsize=12, fontweight='bold', color='blue')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(real_img)
    axes[0, 1].set_title("Real (Style Source)", fontsize=12, fontweight='bold', color='red')
    axes[0, 1].axis('off')

    # Hard mask 예시 (중간 beta)
    h, w = syn_img.shape[:2]
    cy, cx = h // 2, w // 2
    mid_beta = betas[len(betas) // 2]

    hard_mask = np.zeros((h, w), dtype=np.float32)
    bh, bw = max(int(h * mid_beta), 1), max(int(w * mid_beta), 1)
    hard_mask[cy - bh:cy + bh, cx - bw:cx + bw] = 1.0
    axes[0, 2].imshow(hard_mask, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f"Hard Mask (β={mid_beta})\nSharp rectangular", fontsize=11)
    axes[0, 2].axis('off')

    # Soft mask 예시
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    sigma = max(h * mid_beta, 1)
    soft_mask = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    axes[0, 3].imshow(soft_mask, cmap='hot', vmin=0, vmax=1)
    axes[0, 3].set_title(f"Soft Mask (β={mid_beta})\nGaussian transition", fontsize=11)
    axes[0, 3].axis('off')

    # --- Row 1~N: 각 beta별 비교 ---
    for i, b in enumerate(betas):
        row = i + 1

        axes[row, 0].imshow(hard_results[i])
        axes[row, 0].set_title(f"Hard FDA β={b}", fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')

        # Hard diff (×5 증폭)
        hard_diff = np.abs(hard_results[i].astype(float) - syn_img.astype(float)) * 5
        axes[row, 1].imshow(np.clip(hard_diff, 0, 255).astype(np.uint8))
        axes[row, 1].set_title(f"Hard |FDA-Syn| ×5", fontsize=11)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(soft_results[i])
        axes[row, 2].set_title(f"Soft FDA β={b}", fontsize=12, fontweight='bold')
        axes[row, 2].axis('off')

        # Soft diff (×5 증폭)
        soft_diff = np.abs(soft_results[i].astype(float) - syn_img.astype(float)) * 5
        axes[row, 3].imshow(np.clip(soft_diff, 0, 255).astype(np.uint8))
        axes[row, 3].set_title(f"Soft |FDA-Syn| ×5", fontsize=11)
        axes[row, 3].axis('off')

    # --- Last Row: Radial profile ---
    syn_radial = get_fft_radial_profile(syn_img)
    real_radial = get_fft_radial_profile(real_img)

    ax_left = axes[n_rows - 1, 0]
    ax_left.set_position(axes[n_rows - 1, 0].get_position())  # keep default
    # Merge left 2 cols for Hard radial
    axes[n_rows - 1, 1].axis('off')
    ax_left.plot(syn_radial, 'b--', lw=1.5, label='Syn', alpha=0.5)
    ax_left.plot(real_radial, 'r--', lw=1.5, label='Real', alpha=0.5)
    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
    for i, b in enumerate(betas):
        r = get_fft_radial_profile(hard_results[i])
        ax_left.plot(r, '-', lw=2, label=f'Hard β={b}', color=colors[i % len(colors)])
    ax_left.set_xlabel("Frequency"); ax_left.set_ylabel("Log Amp")
    ax_left.set_title("Hard FDA Radial Profile", fontsize=12, fontweight='bold')
    ax_left.legend(fontsize=9); ax_left.grid(True, alpha=0.3)

    ax_right = axes[n_rows - 1, 2]
    axes[n_rows - 1, 3].axis('off')
    ax_right.plot(syn_radial, 'b--', lw=1.5, label='Syn', alpha=0.5)
    ax_right.plot(real_radial, 'r--', lw=1.5, label='Real', alpha=0.5)
    for i, b in enumerate(betas):
        r = get_fft_radial_profile(soft_results[i])
        ax_right.plot(r, '-', lw=2, label=f'Soft β={b}', color=colors[i % len(colors)])
    ax_right.set_xlabel("Frequency"); ax_right.set_ylabel("Log Amp")
    ax_right.set_title("Soft FDA Radial Profile", fontsize=12, fontweight='bold')
    ax_right.legend(fontsize=9); ax_right.grid(True, alpha=0.3)

    plt.suptitle("FDA Comparison: Hard (Rectangular) vs Soft (Gaussian) Mask\n"
                 "Left: Hard FDA | Right: Soft FDA | Bottom: Radial frequency profiles",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
