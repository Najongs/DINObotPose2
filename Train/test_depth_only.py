"""
Test: 2D heatmap prediction (u,v) + GT depth (z) → recovered 3D vs GT 3D

Purpose: Verify that if we use the model's 2D heatmap predictions to get (u,v),
then recover x,y using camera intrinsics K and GT z, the resulting 3D coordinates
are close to GT. This validates the "depth-only prediction" approach.

x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = z_gt
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DREAM')))

from model import DINOv3PoseEstimator
from dataset import PoseEstimationDataset


def soft_argmax_2d(heatmaps):
    """
    Differentiable soft-argmax to extract (u, v) from heatmaps.
    heatmaps: (B, N, H, W)
    Returns: (B, N, 2) [x, y] in heatmap pixel coordinates
    """
    B, N, H, W = heatmaps.shape

    # Create coordinate grids
    device = heatmaps.device
    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    y_coords = torch.arange(H, device=device, dtype=torch.float32)

    # Softmax over spatial dimensions
    heatmaps_flat = heatmaps.view(B, N, -1)
    weights = torch.softmax(heatmaps_flat * 10.0, dim=-1)  # temperature scaling
    weights = weights.view(B, N, H, W)

    # Weighted sum of coordinates
    x = (weights.sum(dim=2) * x_coords).sum(dim=-1)  # (B, N)
    y = (weights.sum(dim=3) * y_coords).sum(dim=-1)  # (B, N)

    return torch.stack([x, y], dim=-1)  # (B, N, 2)


def hard_argmax_2d(heatmaps):
    """
    Hard argmax to extract (u, v) from heatmaps.
    heatmaps: (B, N, H, W)
    Returns: (B, N, 2) [x, y] in heatmap pixel coordinates
    """
    B, N, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)
    y = max_indices // W
    x = max_indices % W
    return torch.stack([x, y], dim=-1).float()


def recover_3d_from_2d_and_depth(uv_heatmap, z_gt, K, heatmap_size, original_size):
    """
    Recover 3D coordinates from 2D heatmap predictions + GT depth + camera K.

    Args:
        uv_heatmap: (B, N, 2) predicted keypoints in heatmap pixel space
        z_gt: (B, N) GT depth values
        K: (3, 3) camera intrinsic matrix (in original image pixel space)
        heatmap_size: (H, W) of heatmap
        original_size: (W, H) of original image

    Returns:
        xyz_recovered: (B, N, 3) recovered 3D coordinates
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Scale heatmap coords to original image coords
    scale_x = original_size[0] / heatmap_size[1]  # W / heatmap_W
    scale_y = original_size[1] / heatmap_size[0]  # H / heatmap_H

    u_orig = uv_heatmap[:, :, 0] * scale_x  # (B, N)
    v_orig = uv_heatmap[:, :, 1] * scale_y  # (B, N)

    # Recover x, y using pinhole camera model
    x_3d = (u_orig - cx) * z_gt / fx
    y_3d = (v_orig - cy) * z_gt / fy

    return torch.stack([x_3d, y_3d, z_gt], dim=-1)  # (B, N, 3)


def main():
    parser = argparse.ArgumentParser(description='Test depth-only 3D recovery')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of samples to evaluate')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)

    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]

    # Load dataset (no augmentation)
    dataset = PoseEstimationDataset(
        data_dir=args.data_dir,
        keypoint_names=keypoint_names,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=False
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2
    )

    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)

    # Detect model config from checkpoint
    config = checkpoint.get('config', {})
    model_name = config.get('model_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m')
    use_joint_embedding = config.get('use_joint_embedding', False)

    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0,
        use_joint_embedding=use_joint_embedding
    ).to(device)

    state_dict = checkpoint['model_state_dict']
    # Remove DDP 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")

    # We need camera K per sample - load from JSON
    # Collect errors
    errors_direct_3d = []      # Model's direct 3D prediction vs GT
    errors_recovered_hard = []  # Hard argmax 2D + GT z vs GT 3D
    errors_recovered_soft = []  # Soft argmax 2D + GT z vs GT 3D
    errors_gt_2d_recovery = []  # GT 2D + GT z vs GT 3D (upper bound)
    per_link_errors = {name: {'direct': [], 'hard': [], 'soft': [], 'gt2d': []} for name in keypoint_names}

    num_evaluated = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=min(args.num_samples, len(loader)))):
            if num_evaluated >= args.num_samples:
                break

            images = batch['image'].to(device)
            gt_kp3d = batch['keypoints_3d'].to(device)  # (B, 7, 3)
            gt_kp2d = batch['keypoints'].to(device)      # (B, 7, 2) in heatmap coords
            valid_mask = batch['valid_mask'].to(device)   # (B, 7)

            # Load camera K from annotation JSON
            sample_info = dataset.samples[i]
            with open(sample_info['annotation_path'], 'r') as f:
                ann_data = json.load(f)

            K_list = ann_data.get('meta', {}).get('K', None)
            if K_list is None:
                continue
            K = torch.tensor(K_list, dtype=torch.float32, device=device)

            # Get original image size
            from PIL import Image as PILImage
            orig_img = PILImage.open(sample_info['image_path'])
            original_size = orig_img.size  # (W, H)

            # Forward pass
            pred_dict = model(images)
            pred_heatmaps = pred_dict['heatmaps_2d']  # (B, 7, H, W)
            pred_kp3d = pred_dict['keypoints_3d']      # (B, 7, 3)

            # Handle synthetic data cm->m conversion for GT
            gt_z = gt_kp3d[:, :, 2]  # (B, 7)

            # Extract 2D keypoints from heatmaps
            pred_uv_hard = hard_argmax_2d(pred_heatmaps)  # (B, 7, 2)
            pred_uv_soft = soft_argmax_2d(pred_heatmaps)  # (B, 7, 2)

            # Recover 3D using predicted 2D + GT z
            recovered_hard = recover_3d_from_2d_and_depth(
                pred_uv_hard, gt_z, K,
                (args.heatmap_size, args.heatmap_size), original_size
            )
            recovered_soft = recover_3d_from_2d_and_depth(
                pred_uv_soft, gt_z, K,
                (args.heatmap_size, args.heatmap_size), original_size
            )

            # Also test with GT 2D keypoints + GT z (upper bound)
            recovered_gt2d = recover_3d_from_2d_and_depth(
                gt_kp2d, gt_z, K,
                (args.heatmap_size, args.heatmap_size), original_size
            )

            # Compute errors (only valid keypoints)
            for b in range(images.shape[0]):
                for j in range(len(keypoint_names)):
                    if not valid_mask[b, j]:
                        continue

                    gt = gt_kp3d[b, j].cpu().numpy()

                    # Direct 3D prediction error
                    err_direct = np.linalg.norm(pred_kp3d[b, j].cpu().numpy() - gt)
                    errors_direct_3d.append(err_direct)
                    per_link_errors[keypoint_names[j]]['direct'].append(err_direct)

                    # Hard argmax recovery error
                    err_hard = np.linalg.norm(recovered_hard[b, j].cpu().numpy() - gt)
                    errors_recovered_hard.append(err_hard)
                    per_link_errors[keypoint_names[j]]['hard'].append(err_hard)

                    # Soft argmax recovery error
                    err_soft = np.linalg.norm(recovered_soft[b, j].cpu().numpy() - gt)
                    errors_recovered_soft.append(err_soft)
                    per_link_errors[keypoint_names[j]]['soft'].append(err_soft)

                    # GT 2D recovery error (upper bound)
                    err_gt2d = np.linalg.norm(recovered_gt2d[b, j].cpu().numpy() - gt)
                    errors_gt_2d_recovery.append(err_gt2d)
                    per_link_errors[keypoint_names[j]]['gt2d'].append(err_gt2d)

            num_evaluated += 1

    # ============================================================
    # Print results
    # ============================================================
    print("\n" + "=" * 90)
    print("DEPTH-ONLY 3D RECOVERY TEST RESULTS")
    print(f"Evaluated {num_evaluated} samples from: {args.data_dir}")
    print("=" * 90)

    def print_stats(name, errors):
        errors = np.array(errors)
        print(f"  {name:<35} mean={errors.mean()*1000:>8.2f}mm  median={np.median(errors)*1000:>8.2f}mm  std={errors.std()*1000:>8.2f}mm")

    print("\n--- OVERALL (L2 distance, meters → shown in mm) ---")
    print_stats("Direct 3D prediction", errors_direct_3d)
    print_stats("Hard argmax 2D + GT z", errors_recovered_hard)
    print_stats("Soft argmax 2D + GT z", errors_recovered_soft)
    print_stats("GT 2D + GT z (upper bound)", errors_gt_2d_recovery)

    print("\n--- PER-LINK ERRORS (mm) ---")
    header = f"{'Link':<18} {'Direct 3D':>12} {'Hard+GTz':>12} {'Soft+GTz':>12} {'GT2D+GTz':>12}"
    print(header)
    print("-" * len(header))
    for name in keypoint_names:
        d = per_link_errors[name]
        if len(d['direct']) == 0:
            continue
        print(f"{name:<18} "
              f"{np.mean(d['direct'])*1000:>10.2f}mm "
              f"{np.mean(d['hard'])*1000:>10.2f}mm "
              f"{np.mean(d['soft'])*1000:>10.2f}mm "
              f"{np.mean(d['gt2d'])*1000:>10.2f}mm")

    # ADD AUC calculation
    print("\n--- ADD AUC (threshold=100mm) ---")
    for label, errors in [("Direct 3D", errors_direct_3d),
                           ("Hard+GTz", errors_recovered_hard),
                           ("Soft+GTz", errors_recovered_soft),
                           ("GT2D+GTz", errors_gt_2d_recovery)]:
        errors_m = np.array(errors)
        threshold = 0.1  # 100mm in meters
        delta = 0.00001
        thresholds = np.arange(0.0, threshold, delta)
        counts = [np.mean(errors_m <= t) for t in thresholds]
        auc = np.trapz(counts, dx=delta) / threshold
        print(f"  {label:<35} ADD AUC = {auc*100:.2f}%")


if __name__ == '__main__':
    main()
