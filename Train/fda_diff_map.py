"""
FDA Difference Map Visualization
원본 Amplitude + Delta(차이) = FDA 결과 Amplitude 를 시각적으로 보여줌

구성:
  Row per beta: [Syn Amp] [+] [Delta Amp (Real-Syn, masked)] [=] [FDA Amp] | [Syn Image] → [FDA Image]
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def fda_transfer(src_img, trg_img, beta=0.01, soft=False):
    src = src_img.astype(np.float32)
    trg = trg_img.astype(np.float32)
    if src.shape != trg.shape:
        trg = np.array(Image.fromarray(trg.astype(np.uint8)).resize(
            (src.shape[1], src.shape[0]), Image.BILINEAR)).astype(np.float32)

    h, w = src.shape[:2]
    cy, cx = h // 2, w // 2
    if soft:
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        sigma = max(h * beta, 1)
        mask = np.exp(-(dist**2) / (2 * sigma**2))
    else:
        mask = np.zeros((h, w), dtype=np.float32)
        bh, bw = max(int(h * beta), 1), max(int(w * beta), 1)
        mask[cy-bh:cy+bh, cx-bw:cx+bw] = 1.0

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


def get_amp_log(img):
    """Grayscale log amplitude"""
    gray = np.mean(img.astype(np.float32), axis=2)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    return np.log1p(np.abs(fft))


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def main():
    parser = argparse.ArgumentParser(description='FDA Difference Map')
    parser.add_argument('--syn-path', type=str,
                        default="/data/public/NAS/DINObotPose2/Dataset/DREAM_syn/panda_synth_train_dr/000007.rgb.jpg")
    parser.add_argument('--real-path', type=str,
                        default="/data/public/NAS/DINObotPose2/Dataset/DREAM_real/panda-3cam_realsense/panda-3cam_realsense/000300.rgb.jpg")
    parser.add_argument('--betas', type=str, default="0.001,0.005,0.01")
    parser.add_argument('--soft', action='store_true')
    parser.add_argument('--output', type=str,
                        default="/data/public/NAS/DINObotPose2/Train/fda_diff_map.png")
    args = parser.parse_args()

    syn_img = np.array(Image.open(args.syn_path).convert("RGB"))
    real_img = np.array(Image.open(args.real_path).convert("RGB"))
    if syn_img.shape != real_img.shape:
        real_img = np.array(Image.fromarray(real_img).resize(
            (syn_img.shape[1], syn_img.shape[0]), Image.BILINEAR))

    betas = [float(b.strip()) for b in args.betas.split(',')]
    mode = "Soft" if args.soft else "Hard"

    syn_amp = get_amp_log(syn_img)
    real_amp = get_amp_log(real_img)

    # 공통 normalization 범위 (syn_amp 기준)
    amp_min, amp_max = syn_amp.min(), syn_amp.max()

    h, w = syn_img.shape[:2]
    cy, cx = h // 2, w // 2

    # Layout: (1 + n_betas) rows x 7 cols
    # Row 0: Syn image | Syn Amp | ... | Real image | Real Amp
    # Row i: Syn Amp | + | Delta (masked) | = | FDA Amp | Syn img | FDA img
    n_betas = len(betas)

    fig = plt.figure(figsize=(28, 5 * (1 + n_betas)))
    gs = gridspec.GridSpec(1 + n_betas, 7, figure=fig, hspace=0.3, wspace=0.15,
                           width_ratios=[3, 0.5, 3, 0.5, 3, 3, 3])

    # === Row 0: 원본 정보 ===
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(syn_img); ax.set_title("Synthetic Image", fontsize=13, fontweight='bold', color='blue'); ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(normalize(syn_amp), cmap='hot'); ax.set_title("Syn Amplitude (log)", fontsize=12); ax.axis('off')

    # operator text
    ax = fig.add_subplot(gs[0, 3]); ax.axis('off')

    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(normalize(real_amp), cmap='hot'); ax.set_title("Real Amplitude (log)", fontsize=12); ax.axis('off')

    ax = fig.add_subplot(gs[0, 5])
    ax.imshow(real_img); ax.set_title("Real Image", fontsize=13, fontweight='bold', color='red'); ax.axis('off')

    # Amplitude 차이 전체
    ax = fig.add_subplot(gs[0, 6])
    full_delta = real_amp - syn_amp
    vmax = max(abs(full_delta.min()), abs(full_delta.max()))
    ax.imshow(full_delta, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title("Full Delta\n(Real - Syn)", fontsize=12, fontweight='bold'); ax.axis('off')

    # hide unused
    ax = fig.add_subplot(gs[0, 1]); ax.axis('off')

    # === Row 1~N: 각 beta별 ===
    for i, b in enumerate(betas):
        row = i + 1
        fda_img = fda_transfer(syn_img, real_img, beta=b, soft=args.soft)
        fda_amp = get_amp_log(fda_img)

        # Delta = FDA_amp - Syn_amp (실제로 교체된 부분만 non-zero)
        delta = fda_amp - syn_amp

        # Mask 시각화 (어떤 영역이 교체되었는지)
        if args.soft:
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            sigma = max(h * b, 1)
            mask_vis = np.exp(-(dist**2) / (2 * sigma**2))
        else:
            mask_vis = np.zeros((h, w))
            bh, bw = max(int(h * b), 1), max(int(w * b), 1)
            mask_vis[cy-bh:cy+bh, cx-bw:cx+bw] = 1.0

        # Col 0: Syn Amplitude
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(normalize(syn_amp), cmap='hot')
        ax.set_title("Syn Amplitude", fontsize=11)
        ax.axis('off')

        # Col 1: + 기호
        ax = fig.add_subplot(gs[row, 1])
        ax.text(0.5, 0.5, "+", fontsize=36, fontweight='bold', ha='center', va='center',
                transform=ax.transAxes)
        ax.axis('off')

        # Col 2: Delta (차이값, 마스크 영역만)
        ax = fig.add_subplot(gs[row, 2])
        delta_vmax = max(abs(delta.min()), abs(delta.max()), 0.01)
        im = ax.imshow(delta, cmap='RdBu_r', vmin=-delta_vmax, vmax=delta_vmax)
        # 마스크 경계 표시
        ax.contour(mask_vis, levels=[0.5], colors='lime', linewidths=1.5)
        ax.set_title(f"Δ Amplitude (β={b})\nRed=increased, Blue=decreased", fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Col 3: = 기호
        ax = fig.add_subplot(gs[row, 3])
        ax.text(0.5, 0.5, "=", fontsize=36, fontweight='bold', ha='center', va='center',
                transform=ax.transAxes)
        ax.axis('off')

        # Col 4: FDA Amplitude
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(normalize(fda_amp), cmap='hot')
        ax.set_title(f"FDA Amplitude (β={b})", fontsize=11)
        ax.axis('off')

        # Col 5: Syn Image
        ax = fig.add_subplot(gs[row, 5])
        ax.imshow(syn_img)
        ax.set_title("Syn Image", fontsize=11)
        ax.axis('off')

        # Col 6: FDA Image (→ 화살표 느낌)
        ax = fig.add_subplot(gs[row, 6])
        ax.imshow(fda_img)
        ax.set_title(f"FDA Image (β={b})", fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f"FDA Amplitude Decomposition: Syn + Δ(masked from Real) = FDA  [{mode} mask]\n"
                 f"Green contour = mask boundary (replaced region in frequency domain)",
                 fontsize=16, fontweight='bold', y=1.01)

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
