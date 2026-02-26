"""
FDA Spectrum Visualization
- Synthetic / Real / FDA 적용 결과의 FFT Amplitude Spectrum 비교
- Radial frequency profile 포함
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def fda_transfer(src_img, trg_img, beta=0.01, soft=False):
    """FDA transfer (hard or soft mask)"""
    src = src_img.astype(np.float32)
    trg = trg_img.astype(np.float32)

    if src.shape != trg.shape:
        trg = np.array(Image.fromarray(trg.astype(np.uint8)).resize(
            (src.shape[1], src.shape[0]), Image.BILINEAR
        )).astype(np.float32)

    h, w = src.shape[:2]
    cy, cx = h // 2, w // 2

    if soft:
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        sigma = max(h * beta, 1)
        mask = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    else:
        mask = np.zeros((h, w), dtype=np.float32)
        bh, bw = max(int(h * beta), 1), max(int(w * beta), 1)
        mask[cy - bh:cy + bh, cx - bw:cx + bw] = 1.0

    result = np.zeros_like(src)
    for ch in range(3):
        fft_src = np.fft.fftshift(np.fft.fft2(src[:, :, ch]))
        fft_trg = np.fft.fftshift(np.fft.fft2(trg[:, :, ch]))
        amp_src, amp_trg = np.abs(fft_src), np.abs(fft_trg)
        phase_src = np.angle(fft_src)
        amp_blended = (1 - mask) * amp_src + mask * amp_trg
        fft_result = np.fft.ifftshift(amp_blended * np.exp(1j * phase_src))
        result[:, :, ch] = np.real(np.fft.ifft2(fft_result))

    return np.clip(result, 0, 255).astype(np.uint8)


def get_amplitude(img):
    """Grayscale FFT amplitude (log scale, normalized 0~1)"""
    gray = np.mean(img.astype(np.float32), axis=2)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    amp = np.log1p(np.abs(fft))
    return (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)


def get_amplitude_rgb(img):
    """Per-channel FFT amplitude as RGB"""
    h, w = img.shape[:2]
    amp_rgb = np.zeros((h, w, 3), dtype=np.float32)
    for ch in range(3):
        fft = np.fft.fftshift(np.fft.fft2(img[:, :, ch].astype(np.float32)))
        amp = np.log1p(np.abs(fft))
        amp_rgb[:, :, ch] = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
    return (amp_rgb * 255).astype(np.uint8)


def get_phase(img):
    """Grayscale FFT phase (normalized 0~1)"""
    gray = np.mean(img.astype(np.float32), axis=2)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    phase = np.angle(fft)
    return (phase - phase.min()) / (phase.max() - phase.min() + 1e-8)


def get_radial_profile(img):
    """Radial frequency energy profile"""
    gray = np.mean(img.astype(np.float32), axis=2)
    amp = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = amp.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    profile = np.zeros(max_r)
    for i in range(max_r):
        m = r == i
        if m.sum() > 0:
            profile[i] = np.mean(amp[m])
    return np.log1p(profile)


def main():
    parser = argparse.ArgumentParser(description='FDA Spectrum Map Visualization')
    parser.add_argument('--syn-path', type=str,
                        default="/data/public/NAS/DINObotPose2/Dataset/DREAM_syn/panda_synth_train_dr/000007.rgb.jpg")
    parser.add_argument('--real-path', type=str,
                        default="/data/public/NAS/DINObotPose2/Dataset/DREAM_real/panda-3cam_realsense/panda-3cam_realsense/000300.rgb.jpg")
    parser.add_argument('--betas', type=str, default="0.01,0.05,0.15")
    parser.add_argument('--soft', action='store_true', help='Use soft Gaussian mask')
    parser.add_argument('--output', type=str,
                        default="/data/public/NAS/DINObotPose2/Train/fda_spectrum_result.png")
    args = parser.parse_args()

    syn_img = np.array(Image.open(args.syn_path).convert("RGB"))
    real_img = np.array(Image.open(args.real_path).convert("RGB"))
    print(f"Synthetic: {args.syn_path} ({syn_img.shape})")
    print(f"Real:      {args.real_path} ({real_img.shape})")

    if syn_img.shape != real_img.shape:
        real_img = np.array(Image.fromarray(real_img).resize(
            (syn_img.shape[1], syn_img.shape[0]), Image.BILINEAR
        ))

    betas = [float(b.strip()) for b in args.betas.split(',')]
    mode_label = "Soft" if args.soft else "Hard"

    fda_results = [fda_transfer(syn_img, real_img, beta=b, soft=args.soft) for b in betas]

    # 전체 이미지 리스트: Syn, Real, FDA_0, FDA_1, ...
    all_images = [syn_img, real_img] + fda_results
    all_labels = ["Synthetic", "Real"] + [f"FDA β={b}" for b in betas]
    n = len(all_images)

    # ================================================================
    # 4 rows x n cols
    # Row 0: 원본 이미지
    # Row 1: Amplitude spectrum (grayscale, log)
    # Row 2: Amplitude spectrum (per-channel RGB)
    # Row 3: Phase
    # + 마지막에 Radial profile (별도)
    # ================================================================

    fig, axes = plt.subplots(4, n, figsize=(4.5 * n, 18))

    for j, (img, label) in enumerate(zip(all_images, all_labels)):
        # Row 0: Image
        axes[0, j].imshow(img)
        color = 'blue' if j == 0 else ('red' if j == 1 else 'black')
        axes[0, j].set_title(label, fontsize=13, fontweight='bold', color=color)
        axes[0, j].axis('off')

        # Row 1: Amplitude (grayscale)
        amp = get_amplitude(img)
        axes[1, j].imshow(amp, cmap='hot')
        axes[1, j].set_title("Amplitude (log)", fontsize=11)
        axes[1, j].axis('off')

        # Row 2: Amplitude (RGB)
        amp_rgb = get_amplitude_rgb(img)
        axes[2, j].imshow(amp_rgb)
        axes[2, j].set_title("Amplitude (RGB)", fontsize=11)
        axes[2, j].axis('off')

        # Row 3: Phase
        phase = get_phase(img)
        axes[3, j].imshow(phase, cmap='twilight')
        axes[3, j].set_title("Phase", fontsize=11)
        axes[3, j].axis('off')

    # Row labels
    row_labels = ["Image", "Amplitude\n(log, gray)", "Amplitude\n(per-ch RGB)", "Phase"]
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=13, fontweight='bold', rotation=0,
                               labelpad=80, va='center')

    plt.suptitle(f"Fourier Spectrum: Synthetic vs Real vs FDA ({mode_label} mask)\n"
                 f"Betas: {betas}",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spectrum map: {args.output}")

    # ================================================================
    # 별도 Radial profile 그래프
    # ================================================================
    radial_output = args.output.replace('.png', '_radial.png')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    syn_r = get_radial_profile(syn_img)
    real_r = get_radial_profile(real_img)
    ax.plot(syn_r, 'b-', lw=2, label='Synthetic', alpha=0.8)
    ax.plot(real_r, 'r-', lw=2, label='Real', alpha=0.8)
    ax.fill_between(range(len(syn_r)), syn_r, real_r, alpha=0.1, color='purple')

    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
    for i, (b, fda_img) in enumerate(zip(betas, fda_results)):
        r = get_radial_profile(fda_img)
        ax.plot(r, '-', lw=2, label=f'FDA β={b} ({mode_label})', color=colors[i % len(colors)])

    ax.set_xlabel("Frequency (low → high)", fontsize=13)
    ax.set_ylabel("Log Amplitude", fontsize=13)
    ax.set_title(f"Radial Frequency Profile ({mode_label} FDA)", fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(radial_output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved radial profile: {radial_output}")


if __name__ == "__main__":
    main()
