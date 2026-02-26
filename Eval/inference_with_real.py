"""
Real Image Inference Script for DINOv3 Pose Estimation
- JSON 어노테이션 파일을 입력받아 이미지 경로/GT를 자동 로드
- GT vs Prediction 비교 시각화 + 정량적 메트릭 출력
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from PIL import Image as PILImage

import numpy as np
import torch
import torchvision.transforms as TVTransforms
import yaml

# Import DREAM utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DREAM')))
import dream

# Import model from Train directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Train')))
from model import DINOv3PoseEstimator, MODE_DIRECT, MODE_DEPTH_ONLY


def get_keypoints_from_heatmaps(heatmaps_tensor):
    """Extract keypoint coordinates from heatmaps using argmax."""
    B, N, H, W = heatmaps_tensor.shape
    heatmaps_flat = heatmaps_tensor.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)
    y = max_indices // W
    x = max_indices % W
    keypoints = torch.stack([x, y], dim=-1).float()
    return keypoints[0].cpu().numpy()


def load_annotation(json_path, keypoint_names):
    """
    JSON 어노테이션에서 GT 정보를 로드.
    Returns:
        image_path: 이미지 절대경로
        gt_2d: (N, 2) projected keypoint 좌표
        gt_3d: (N, 3) 3D keypoint 좌표
        camera_K: (3, 3) camera intrinsic matrix (있으면)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract image path
    image_path = None
    camera_K = None
    if 'meta' in data:
        if 'image_path' in data['meta']:
            raw_path = data['meta']['image_path']
            # Fix relative path: ../dataset/... -> resolve from JSON dir
            if raw_path.startswith('../dataset/'):
                raw_path = raw_path.replace('../dataset/', '../../../', 1)
            if not os.path.isabs(raw_path):
                image_path = os.path.normpath(os.path.join(os.path.dirname(json_path), raw_path))
            else:
                image_path = raw_path
        if 'K' in data['meta']:
            camera_K = np.array(data['meta']['K'], dtype=np.float64)

    # Extract keypoints
    gt_2d = np.zeros((len(keypoint_names), 2), dtype=np.float32)
    gt_3d = np.zeros((len(keypoint_names), 3), dtype=np.float32)
    found = [False] * len(keypoint_names)

    if 'objects' in data:
        for obj in data['objects']:
            if 'keypoints' in obj:
                for kp in obj['keypoints']:
                    if kp['name'] in keypoint_names:
                        idx = keypoint_names.index(kp['name'])
                        gt_2d[idx] = kp['projected_location']
                        if 'location' in kp:
                            gt_3d[idx] = kp['location']
                        found[idx] = True

    return image_path, gt_2d, gt_3d, camera_K, found


def compute_metrics(pred_2d, gt_2d, pred_3d, gt_3d, keypoint_names, found, orig_image_dim):
    """Compute per-keypoint and overall error metrics."""
    metrics = {}

    # 2D pixel error (L2 distance)
    errors_2d = []
    for i, name in enumerate(keypoint_names):
        if found[i]:
            err = np.linalg.norm(pred_2d[i] - gt_2d[i])
            errors_2d.append(err)
            metrics[f'{name}_2d_px'] = err
        else:
            metrics[f'{name}_2d_px'] = float('nan')

    if errors_2d:
        metrics['mean_2d_px'] = np.mean(errors_2d)
        metrics['max_2d_px'] = np.max(errors_2d)
        metrics['median_2d_px'] = np.median(errors_2d)

    # 3D Euclidean error (meters)
    errors_3d = []
    for i, name in enumerate(keypoint_names):
        if found[i] and not np.allclose(gt_3d[i], 0):
            err = np.linalg.norm(pred_3d[i] - gt_3d[i])
            errors_3d.append(err)
            metrics[f'{name}_3d_m'] = err
        else:
            metrics[f'{name}_3d_m'] = float('nan')

    if errors_3d:
        metrics['mean_3d_m'] = np.mean(errors_3d)
        metrics['max_3d_m'] = np.max(errors_3d)
        metrics['median_3d_m'] = np.median(errors_3d)

    # Normalized 2D error (% of image diagonal)
    diag = np.sqrt(orig_image_dim[0]**2 + orig_image_dim[1]**2)
    if errors_2d and diag > 0:
        metrics['mean_2d_norm'] = np.mean(errors_2d) / diag * 100  # percentage

    return metrics


def network_inference(args):

    assert os.path.exists(args.json_path), \
        f'JSON path "{args.json_path}" does not exist.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"# Using device: {device}")

    # Default keypoint names
    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]

    # Load training config
    checkpoint_dir = Path(args.model_path).parent
    config_path = checkpoint_dir / 'config.yaml'

    use_joint_embedding = False
    depth_only_3d = False
    model_name = args.model_name
    image_size = 512
    heatmap_size = 512

    if config_path.exists():
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        use_joint_embedding = train_config.get('use_joint_embedding', False)
        depth_only_3d = train_config.get('depth_only_3d', False)
        model_name = train_config.get('model_name', model_name)
        image_size = int(train_config.get('image_size', image_size))
        heatmap_size = int(train_config.get('heatmap_size', heatmap_size))
        if 'keypoint_names' in train_config:
            keypoint_names = train_config['keypoint_names']
        print(f"# Config: use_joint_embedding={use_joint_embedding}, depth_only_3d={depth_only_3d}")

    # Load annotation JSON
    print(f"\n# Loading annotation: {args.json_path}")
    image_path, gt_2d, gt_3d, camera_K, found = load_annotation(args.json_path, keypoint_names)

    if image_path is None or not os.path.exists(image_path):
        print(f"# ERROR: Image not found at resolved path: {image_path}")
        print(f"# Check 'meta.image_path' in JSON and relative path resolution")
        return

    print(f"# Image path: {image_path}")
    print(f"# GT keypoints found: {sum(found)}/{len(found)}")
    if camera_K is not None:
        print(f"# Camera K:\n{camera_K}")

    # Create model
    mode_3d = MODE_DEPTH_ONLY if depth_only_3d else MODE_DIRECT
    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=(heatmap_size, heatmap_size),
        unfreeze_blocks=0,
        use_joint_embedding=use_joint_embedding,
        mode_3d=mode_3d
    ).to(device)

    # Load checkpoint
    print(f"# Loading weights: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if 'backbone.model.embeddings.mask_token' in state_dict:
        mask_shape = state_dict['backbone.model.embeddings.mask_token'].shape
        if len(mask_shape) == 3 and mask_shape[1] == 1:
            del state_dict['backbone.model.embeddings.mask_token']

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    if 'epoch' in checkpoint:
        print(f"# Checkpoint epoch: {checkpoint['epoch']}")

    # Load and preprocess image
    image_pil = PILImage.open(image_path).convert("RGB")
    orig_dim = image_pil.size  # (W, H)

    transform = TVTransforms.Compose([
        TVTransforms.Resize((image_size, image_size)),
        TVTransforms.ToTensor(),
        TVTransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Run inference
    print("\n# Running inference...")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Prepare camera_K tensor for depth_only mode
    camera_K_tensor = None
    if camera_K is not None:
        camera_K_tensor = torch.tensor(camera_K, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor, camera_K=camera_K_tensor, original_size=orig_dim)
        pred_heatmaps = outputs["heatmaps_2d"]
        pred_kpts_3d = outputs["keypoints_3d"]

    # Extract 2D keypoints from heatmaps (in heatmap coordinate space)
    pred_2d_heatmap = get_keypoints_from_heatmaps(pred_heatmaps)

    # Scale predicted 2D to original image coordinates
    pred_2d_orig = pred_2d_heatmap.copy()
    pred_2d_orig[:, 0] *= orig_dim[0] / heatmap_size
    pred_2d_orig[:, 1] *= orig_dim[1] / heatmap_size

    # 3D predictions
    pred_3d = pred_kpts_3d[0].cpu().numpy()

    # === Compute Metrics ===
    metrics = compute_metrics(pred_2d_orig, gt_2d, pred_3d, gt_3d, keypoint_names, found, orig_dim)

    # Print results
    print("\n" + "=" * 80)
    print("  RESULTS: GT (green) vs Prediction (red)")
    print("=" * 80)

    print(f"\n{'Keypoint':<20} {'GT 2D (px)':<22} {'Pred 2D (px)':<22} {'2D Err (px)':<12}")
    print("-" * 76)
    for i, name in enumerate(keypoint_names):
        gt_str = f"({gt_2d[i][0]:7.1f}, {gt_2d[i][1]:7.1f})" if found[i] else "  N/A"
        pred_str = f"({pred_2d_orig[i][0]:7.1f}, {pred_2d_orig[i][1]:7.1f})"
        err_str = f"{metrics.get(f'{name}_2d_px', float('nan')):8.2f}" if found[i] else "  N/A"
        print(f"  {name:<18} {gt_str:<22} {pred_str:<22} {err_str}")

    print(f"\n  Mean 2D error:   {metrics.get('mean_2d_px', float('nan')):.2f} px")
    print(f"  Median 2D error: {metrics.get('median_2d_px', float('nan')):.2f} px")
    print(f"  Max 2D error:    {metrics.get('max_2d_px', float('nan')):.2f} px")
    print(f"  Normalized error: {metrics.get('mean_2d_norm', float('nan')):.2f}% of image diagonal")

    print(f"\n{'Keypoint':<20} {'GT 3D (m)':<30} {'Pred 3D (m)':<30} {'3D Err (m)':<12}")
    print("-" * 92)
    for i, name in enumerate(keypoint_names):
        if found[i] and not np.allclose(gt_3d[i], 0):
            gt_str = f"({gt_3d[i][0]:8.4f}, {gt_3d[i][1]:8.4f}, {gt_3d[i][2]:8.4f})"
            pred_str = f"({pred_3d[i][0]:8.4f}, {pred_3d[i][1]:8.4f}, {pred_3d[i][2]:8.4f})"
            err_str = f"{metrics.get(f'{name}_3d_m', float('nan')):.4f}"
        else:
            gt_str = "  N/A"
            pred_str = f"({pred_3d[i][0]:8.4f}, {pred_3d[i][1]:8.4f}, {pred_3d[i][2]:8.4f})"
            err_str = "  N/A"
        print(f"  {name:<18} {gt_str:<30} {pred_str:<30} {err_str}")

    if 'mean_3d_m' in metrics:
        print(f"\n  Mean 3D error:   {metrics['mean_3d_m']:.4f} m ({metrics['mean_3d_m']*100:.2f} cm)")
        print(f"  Median 3D error: {metrics['median_3d_m']:.4f} m ({metrics['median_3d_m']*100:.2f} cm)")
        print(f"  Max 3D error:    {metrics['max_3d_m']:.4f} m ({metrics['max_3d_m']*100:.2f} cm)")

    print("=" * 80)

    # === Visualizations ===
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n# Saving visualizations to: {args.output_dir}")

        input_dim = (image_size, image_size)
        image_resized = image_pil.resize(input_dim, resample=PILImage.BILINEAR)

        # Scale GT to network input coords
        gt_2d_input = gt_2d.copy()
        gt_2d_input[:, 0] *= input_dim[0] / orig_dim[0]
        gt_2d_input[:, 1] *= input_dim[1] / orig_dim[1]

        pred_2d_input = pred_2d_heatmap.copy()
        pred_2d_input[:, 0] *= input_dim[0] / heatmap_size
        pred_2d_input[:, 1] *= input_dim[1] / heatmap_size

        # Filter valid GTs
        gt_2d_input_list = [gt_2d_input[i].tolist() if found[i] else None for i in range(len(keypoint_names))]
        gt_2d_input_valid = [pt for pt in gt_2d_input_list if pt is not None]
        gt_names_valid = [name for i, name in enumerate(keypoint_names) if found[i]]

        # 1. GT (green) + Pred (red) on image
        overlay = image_resized.copy()
        if gt_2d_input_valid:
            overlay = dream.image_proc.overlay_points_on_image(
                overlay, gt_2d_input_valid, gt_names_valid,
                annotation_color_dot="green", annotation_color_text="white",
            )
        overlay = dream.image_proc.overlay_points_on_image(
            overlay, pred_2d_input, keypoint_names,
            annotation_color_dot="red", annotation_color_text="white",
        )
        out_path = os.path.join(args.output_dir, "01_gt_vs_pred_keypoints.png")
        overlay.save(out_path)
        print(f"  Saved: {out_path}")

        # 2. Belief map mosaic
        gt_2d_heatmap_list = None
        if any(found):
            gt_2d_heatmap = gt_2d.copy()
            gt_2d_heatmap[:, 0] *= heatmap_size / orig_dim[0]
            gt_2d_heatmap[:, 1] *= heatmap_size / orig_dim[1]
            gt_2d_heatmap_list = [gt_2d_heatmap[i].tolist() if found[i] else None
                                  for i in range(len(keypoint_names))]

        belief_map_images = dream.image_proc.images_from_belief_maps(
            pred_heatmaps[0], normalization_method=6
        )
        belief_map_images_kp = []
        for kp_idx in range(len(keypoint_names)):
            points = [pred_2d_heatmap[kp_idx]]
            colors = ["red"]
            if gt_2d_heatmap_list and gt_2d_heatmap_list[kp_idx] is not None:
                points.insert(0, gt_2d_heatmap_list[kp_idx])
                colors.insert(0, "green")
            bm_kp = dream.image_proc.overlay_points_on_image(
                belief_map_images[kp_idx], points,
                annotation_color_dot=colors, annotation_color_text=colors,
                point_diameter=4,
            )
            belief_map_images_kp.append(bm_kp)

        n_cols = int(math.ceil(len(keypoint_names) / 2.0))
        mosaic = dream.image_proc.mosaic_images(
            belief_map_images_kp, rows=2, cols=n_cols,
            inner_padding_px=10, fill_color_rgb=(0, 0, 0),
        )
        out_path = os.path.join(args.output_dir, "02_belief_map_mosaic.png")
        mosaic.save(out_path)
        print(f"  Saved: {out_path}")

        # 3. Per-joint belief maps overlaid on image
        blended_array = []
        for n in range(len(keypoint_names)):
            bm = belief_map_images[n].resize(input_dim, resample=PILImage.BILINEAR)
            blended = PILImage.blend(image_resized, bm, alpha=0.5)
            blended = dream.image_proc.overlay_points_on_image(
                blended, [pred_2d_input[n]], [keypoint_names[n]],
                annotation_color_dot="red", annotation_color_text="white",
            )
            if found[n]:
                blended = dream.image_proc.overlay_points_on_image(
                    blended, [gt_2d_input[n].tolist()], [keypoint_names[n]],
                    annotation_color_dot="green", annotation_color_text="white",
                )
            blended_array.append(blended)

        mosaic2 = dream.image_proc.mosaic_images(
            blended_array, rows=2, cols=n_cols, fill_color_rgb=(0, 0, 0)
        )
        out_path = os.path.join(args.output_dir, "03_belief_maps_overlay_mosaic.png")
        mosaic2.save(out_path)
        print(f"  Saved: {out_path}")

        # 4. Combined belief map on original image
        belief_combined = pred_heatmaps[0].sum(dim=0)
        belief_combined_img = dream.image_proc.image_from_belief_map(
            belief_combined, normalization_method=6
        )
        belief_orig = belief_combined_img.resize(orig_dim, resample=PILImage.BILINEAR)
        orig_overlay = PILImage.blend(image_pil, belief_orig, alpha=0.5)
        if any(found):
            gt_valid_orig = [gt_2d[i].tolist() for i in range(len(keypoint_names)) if found[i]]
            orig_overlay = dream.image_proc.overlay_points_on_image(
                orig_overlay, gt_valid_orig, gt_names_valid,
                annotation_color_dot="green", annotation_color_text="white",
            )
        orig_overlay = dream.image_proc.overlay_points_on_image(
            orig_overlay, pred_2d_orig, keypoint_names,
            annotation_color_dot="red", annotation_color_text="white",
        )
        out_path = os.path.join(args.output_dir, "04_combined_on_original.png")
        orig_overlay.save(out_path)
        print(f"  Saved: {out_path}")

        # 5. Save metrics to JSON
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        # Convert numpy types for JSON serialization
        metrics_json = {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()}
        metrics_json['json_path'] = args.json_path
        metrics_json['image_path'] = image_path
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"  Saved: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-j", "--json-path", required=True,
                        help="Path to annotation JSON (contains image path + GT keypoints)")
    parser.add_argument("-p", "--model-path", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Directory to save visualizations and metrics")
    parser.add_argument("--model-name", default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="DINOv3 model name (overridden by config.yaml)")
    args = parser.parse_args()

    network_inference(args)
