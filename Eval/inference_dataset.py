"""
Dataset Inference Script for DINOv3 Pose Estimation
Evaluates a trained model on a dataset with metrics:
- L2 error (px) for in-frame keypoints with AUC
- ADD (m) from model's 3D head: pred_3d vs GT_3d direct comparison
- ADD (m) from PnP baseline (DREAM-style): 2D pred + GT_3d + K -> PnP -> ADD
"""

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Import DREAM utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DREAM')))
import dream
from dream import analysis as dream_analysis

# Import model from TRAIN directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Train')))
from model import DINOv3PoseEstimator, MODE_DIRECT, MODE_DEPTH_ONLY


class InferenceDataset(Dataset):
    """Dataset for inference on converted DREAM-format data (json contains meta.K, meta.image_path)"""

    def __init__(self, data_dir: str, keypoint_names: List[str], image_size: Tuple[int, int]=(512,512)):
        self.data_dir = Path(data_dir)
        self.keypoint_names = keypoint_names
        self.image_size = image_size

        self.json_files = sorted(list(self.data_dir.glob("*.json")))
        if len(self.json_files) == 0:
            raise ValueError(f"No JSON files found in {data_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        print(f"Found {len(self.json_files)} json frames in {data_dir}")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, "r") as f:
            data = json.load(f)

        # Resolve image path from meta.image_path
        img_path_str = data.get("meta", {}).get("image_path", None)
        if img_path_str is None:
            raise KeyError(f"'meta.image_path' missing in {json_path}")

        # Fix incorrect relative path: ../dataset/... should be ../../../...
        if img_path_str.startswith('../dataset/'):
            img_path_str = img_path_str.replace('../dataset/', '../../../', 1)

        img_path = (json_path.parent / img_path_str).resolve()
        if not img_path.exists():
            img_path = (self.data_dir / img_path_str).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for {json_path}: {img_path_str}")

        image = Image.open(img_path).convert("RGB")

        # Extract keypoints (same logic as before)
        keypoints_2d = []
        keypoints_3d = []

        if "objects" in data and len(data["objects"]) > 0:
            obj = data["objects"][0]
            if "keypoints" in obj:
                kp_dict = {kp["name"]: kp for kp in obj["keypoints"]}
                for kp_name in self.keypoint_names:
                    if kp_name in kp_dict:
                        kp = kp_dict[kp_name]
                        keypoints_2d.append(kp["projected_location"])
                        keypoints_3d.append(kp["location"])
                    else:
                        keypoints_2d.append([-999.0, -999.0])
                        keypoints_3d.append([0.0, 0.0, 0.0])

        keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
        keypoints_3d = np.array(keypoints_3d, dtype=np.float32)

        image_tensor = self.transform(image)

        # Angles (same)
        angles = np.zeros(9, dtype=np.float32)
        if "sim_state" in data and "joints" in data["sim_state"]:
            joints = data["sim_state"]["joints"]
            for i, joint in enumerate(joints[:9]):
                if "position" in joint:
                    angles[i] = joint["position"]

        return {
            "image": image_tensor,
            "keypoints": keypoints_2d,
            "keypoints_3d": keypoints_3d,
            "angles": angles,
            "image_path": str(img_path),
            "name": json_path.stem,
        }


def get_keypoints_from_heatmaps(heatmaps: torch.Tensor) -> np.ndarray:
    """
    Extract keypoint coordinates from heatmaps using argmax.

    Args:
        heatmaps: (B, N, H, W) tensor

    Returns:
        keypoints: (B, N, 2) numpy array [x, y]
    """
    B, N, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)

    y = max_indices // W
    x = max_indices % W

    keypoints = torch.stack([x, y], dim=-1).float()
    return keypoints.cpu().numpy()


def compute_keypoint_metrics(
    kp_detected: np.ndarray,
    kp_gt: np.ndarray,
    image_resolution: Tuple[int, int],
    auc_threshold: float = 20.0
) -> Dict:
    """
    Compute keypoint metrics similar to DREAM.

    Args:
        kp_detected: (N, 2) detected keypoints
        kp_gt: (N, 2) ground truth keypoints
        image_resolution: (width, height)
        auc_threshold: AUC threshold in pixels

    Returns:
        metrics dictionary
    """
    num_gt_inframe = 0
    num_found_gt_inframe = 0
    kp_errors = []

    for kp_det, kp_g in zip(kp_detected, kp_gt):
        # Check if GT is in frame
        if (0.0 <= kp_g[0] <= image_resolution[0] and
            0.0 <= kp_g[1] <= image_resolution[1]):
            num_gt_inframe += 1

            # Check if detected
            if kp_det[0] > -999.0 and kp_det[1] > -999.0:
                num_found_gt_inframe += 1
                kp_errors.append(kp_det - kp_g)

    kp_errors = np.array(kp_errors)

    if len(kp_errors) > 0:
        kp_l2_errors = np.linalg.norm(kp_errors, axis=1)
        kp_l2_error_mean = np.mean(kp_l2_errors)
        kp_l2_error_median = np.median(kp_l2_errors)
        kp_l2_error_std = np.std(kp_l2_errors)

        # Compute AUC
        delta_pixel = 0.01
        pck_values = np.arange(0, auc_threshold, delta_pixel)
        y_values = []

        for value in pck_values:
            valids = len(np.where(kp_l2_errors < value)[0])
            y_values.append(valids)

        kp_auc = (
            np.trapz(y_values, dx=delta_pixel) /
            float(auc_threshold) /
            float(num_gt_inframe)
        )
    else:
        kp_l2_error_mean = None
        kp_l2_error_median = None
        kp_l2_error_std = None
        kp_auc = None

    return {
        'num_gt_inframe': num_gt_inframe,
        'num_found_gt_inframe': num_found_gt_inframe,
        'l2_error_mean_px': kp_l2_error_mean,
        'l2_error_median_px': kp_l2_error_median,
        'l2_error_std_px': kp_l2_error_std,
        'l2_error_auc': kp_auc,
        'l2_error_auc_thresh_px': auc_threshold,
    }


def compute_pnp_metrics(
    pnp_add: List[float],
    num_inframe_projs_gt: List[int],
    num_min_inframe_projs_gt_for_pnp: int = 4,
    add_auc_threshold: float = 0.1,
    pnp_magic_number: float = -999.0
) -> Dict:
    """
    Compute PnP metrics similar to DREAM.

    Args:
        pnp_add: List of ADD values
        num_inframe_projs_gt: Number of in-frame GT keypoints per sample
        num_min_inframe_projs_gt_for_pnp: Minimum keypoints for PnP
        add_auc_threshold: AUC threshold in meters
        pnp_magic_number: Magic number for failed PnP

    Returns:
        metrics dictionary
    """
    pnp_add = np.array(pnp_add)
    num_inframe_projs_gt = np.array(num_inframe_projs_gt)

    idx_pnp_found = np.where(pnp_add > pnp_magic_number)[0]
    add_pnp_found = pnp_add[idx_pnp_found]
    num_pnp_found = len(idx_pnp_found)

    mean_add = np.mean(add_pnp_found)
    median_add = np.median(add_pnp_found)
    std_add = np.std(add_pnp_found)

    num_pnp_possible = len(
        np.where(num_inframe_projs_gt >= num_min_inframe_projs_gt_for_pnp)[0]
    )
    num_pnp_not_found = num_pnp_possible - num_pnp_found

    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, add_auc_threshold, delta_threshold)

    counts = []
    for value in add_threshold_values:
        under_threshold = len(np.where(add_pnp_found <= value)[0]) / float(
            num_pnp_possible
        )
        counts.append(under_threshold)

    auc = np.trapz(counts, dx=delta_threshold) / float(add_auc_threshold)

    return {
        'num_pnp_found': num_pnp_found,
        'num_pnp_not_found': num_pnp_not_found,
        'num_pnp_possible': num_pnp_possible,
        'add_mean': mean_add,
        'add_median': median_add,
        'add_std': std_add,
        'add_auc': auc,
        'add_auc_thresh': add_auc_threshold,
    }

def compute_direct_add_metrics(
    pred_3d_all: np.ndarray,
    gt_3d_all: np.ndarray,
    add_auc_threshold: float = 0.1,
) -> Dict:
    """
    Compute ADD metrics by directly comparing model's predicted 3D keypoints
    with GT 3D keypoints (like HoRoPose / RoboPEPP evaluation).

    Args:
        pred_3d_all: (N_frames, N_kp, 3) predicted 3D keypoints from model
        gt_3d_all: (N_frames, N_kp, 3) ground truth 3D keypoints
        add_auc_threshold: AUC threshold in meters

    Returns:
        metrics dictionary
    """
    # Per-frame mean 3D distance (ADD = average of per-keypoint L2 distances)
    # Only use valid GT keypoints (not all-zero)
    frame_adds = []
    per_kp_errors = []

    for pred_3d, gt_3d in zip(pred_3d_all, gt_3d_all):
        # Valid mask: GT keypoints that are not zero
        valid = np.any(gt_3d != 0, axis=-1)  # (N_kp,)
        if valid.sum() == 0:
            continue

        pred_valid = pred_3d[valid]
        gt_valid = gt_3d[valid]

        # Per-keypoint L2 distance
        kp_dists = np.linalg.norm(pred_valid - gt_valid, axis=-1)  # (N_valid,)
        per_kp_errors.extend(kp_dists.tolist())

        # ADD = mean per-keypoint distance for this frame
        frame_add = np.mean(kp_dists)
        frame_adds.append(frame_add)

    frame_adds = np.array(frame_adds)
    per_kp_errors = np.array(per_kp_errors)
    n_frames = len(frame_adds)

    if n_frames == 0:
        return {
            'num_frames': 0,
            'add_mean': None,
            'add_median': None,
            'add_std': None,
            'add_auc': None,
            'add_auc_thresh': add_auc_threshold,
            'per_kp_mean': None,
            'per_kp_median': None,
        }

    # AUC computation (same method as RoboPEPP)
    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, add_auc_threshold, delta_threshold)
    counts = []
    for value in add_threshold_values:
        under_threshold = np.mean(frame_adds <= value)
        counts.append(under_threshold)
    auc = np.trapz(counts, dx=delta_threshold) / float(add_auc_threshold)

    return {
        'num_frames': n_frames,
        'add_mean': float(np.mean(frame_adds)),
        'add_median': float(np.median(frame_adds)),
        'add_std': float(np.std(frame_adds)),
        'add_auc': float(auc),
        'add_auc_thresh': float(add_auc_threshold),
        'per_kp_mean': float(np.mean(per_kp_errors)),
        'per_kp_median': float(np.median(per_kp_errors)),
    }


def load_camera_from_first_frame(dataset_dir: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load camera intrinsics K and raw image resolution (W,H) from the first frame json.
    Expected json format:
    {
      "meta": {
        "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
        "image_path": "path/to/000000.rgb.jpg"  # optional but recommended
      }
    }
    """
    json_files = sorted(dataset_dir.glob("*.json"))
    if len(json_files) == 0:
        raise FileNotFoundError(f"No json files found in {dataset_dir}")

    first_json = json_files[0]
    with open(first_json, "r") as f:
        data = json.load(f)

    if "meta" not in data or "K" not in data["meta"]:
        raise KeyError(f"'meta.K' not found in {first_json}")

    K = np.array(data["meta"]["K"], dtype=np.float32)

    # Determine raw resolution by opening the referenced image
    img_path_str = data["meta"].get("image_path", None)
    if img_path_str is None:
        raise KeyError(f"'meta.image_path' not found in {first_json} (needed to get resolution)")

    # Fix incorrect relative path: ../dataset/... should be ../../../...
    if img_path_str.startswith('../dataset/'):
        img_path_str = img_path_str.replace('../dataset/', '../../../', 1)

    # Resolve image path relative to the json file location
    img_path = (first_json.parent / img_path_str).resolve()
    if not img_path.exists():
        # fallback: try resolving relative to dataset_dir
        img_path = (dataset_dir / img_path_str).resolve()

    if not img_path.exists():
        raise FileNotFoundError(f"Image for resolution not found. Tried: {img_path}")

    with Image.open(img_path) as im:
        w, h = im.size

    return K, (w, h)

@torch.no_grad()
def run_inference(args):
    """Run inference on dataset and compute metrics"""

    dataset_dir = Path(args.dataset_dir)
    camera_K, raw_resolution = load_camera_from_first_frame(dataset_dir)

    print(f"Camera intrinsics:\n{camera_K}")
    print(f"Raw resolution: {raw_resolution}")

    # Load training config from checkpoint directory
    checkpoint_dir = Path(args.model_path).parent
    config_path = checkpoint_dir / 'config.yaml'

    # Defaults
    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]
    train_config = {}

    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        if 'keypoint_names' in train_config:
            keypoint_names = train_config['keypoint_names']
        print(f"Loaded training config from {config_path}")
        print(f"  model_name: {train_config.get('model_name', 'N/A')}")
        print(f"  keypoint_names ({len(keypoint_names)}): {keypoint_names}")
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")

    # Resolve config values (CLI args override config.yaml)
    model_name = args.model_name or train_config.get('model_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m')
    image_size = args.image_size or int(train_config.get('image_size', 512))
    heatmap_size = args.heatmap_size or int(train_config.get('heatmap_size', 512))
    use_joint_embedding = train_config.get('use_joint_embedding', False)
    depth_only_3d = train_config.get('depth_only_3d', False)
    mode_3d = MODE_DEPTH_ONLY if depth_only_3d else MODE_DIRECT

    print(f"  model_name: {model_name}")
    print(f"  image_size: {image_size}, heatmap_size: {heatmap_size}")
    print(f"  use_joint_embedding: {use_joint_embedding}")
    print(f"  depth_only_3d: {depth_only_3d} (mode: {mode_3d})")

    # Create dataset
    dataset = InferenceDataset(
        data_dir=args.dataset_dir,
        keypoint_names=keypoint_names,
        image_size=(image_size, image_size)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Load model
    print(f"\nLoading model from {args.model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=(heatmap_size, heatmap_size),
        unfreeze_blocks=0,  # Not needed for inference
        use_joint_embedding=use_joint_embedding,
        mode_3d=mode_3d
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint

    # Handle DDP wrapper (module. prefix)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Remove mask_token if shape mismatch (transformers version difference)
    if 'backbone.model.embeddings.mask_token' in state_dict:
        mask_token_shape = state_dict['backbone.model.embeddings.mask_token'].shape
        if len(mask_token_shape) == 3 and mask_token_shape[1] == 1:
            # Old format: (1, 1, 768) - remove to avoid mismatch
            print(f"Removing mask_token from state_dict due to shape mismatch (transformers version difference)")
            del state_dict['backbone.model.embeddings.mask_token']

    model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Collect predictions
    all_kp_projs_detected = []
    all_kp_projs_gt = []
    all_kp_pos_gt = []
    all_n_inframe_projs_gt = []
    all_pred_3d = []  # Model's direct 3D predictions

    # Prepare camera_K tensor for depth_only mode
    camera_K_tensor = torch.tensor(camera_K, dtype=torch.float32).to(device) if camera_K is not None else None

    print(f"\nRunning inference on {len(dataset)} images...")

    for batch in tqdm(dataloader):
        images = batch['image'].to(device)
        gt_keypoints = batch['keypoints'].numpy()
        gt_keypoints_3d = batch['keypoints_3d'].numpy()

        # Prepare batched camera_K for depth_only mode
        batch_camera_K = None
        if camera_K_tensor is not None:
            batch_camera_K = camera_K_tensor.unsqueeze(0).expand(images.shape[0], -1, -1)

        # Forward pass
        outputs = model(images, camera_K=batch_camera_K, original_size=raw_resolution)
        pred_heatmaps = outputs["heatmaps_2d"]
        pred_kpts_3d = outputs["keypoints_3d"].cpu().numpy()  # (B, N_kp, 3)

        # Extract keypoints from heatmaps
        pred_keypoints = get_keypoints_from_heatmaps(pred_heatmaps)

        # Scale to raw resolution
        scale_x = raw_resolution[0] / heatmap_size
        scale_y = raw_resolution[1] / heatmap_size

        for i in range(len(pred_keypoints)):
            # Scale predictions to raw resolution
            pred_kp_scaled = pred_keypoints[i].copy()
            pred_kp_scaled[:, 0] *= scale_x
            pred_kp_scaled[:, 1] *= scale_y

            all_kp_projs_detected.append(pred_kp_scaled)
            all_kp_projs_gt.append(gt_keypoints[i])
            all_kp_pos_gt.append(gt_keypoints_3d[i])
            all_pred_3d.append(pred_kpts_3d[i])

            # Count in-frame GT keypoints
            n_inframe = 0
            for kp in gt_keypoints[i]:
                if 0 <= kp[0] <= raw_resolution[0] and 0 <= kp[1] <= raw_resolution[1]:
                    n_inframe += 1
            all_n_inframe_projs_gt.append(n_inframe)

    all_kp_projs_detected = np.array(all_kp_projs_detected)
    all_kp_projs_gt = np.array(all_kp_projs_gt)
    all_kp_pos_gt = np.array(all_kp_pos_gt)
    all_pred_3d = np.array(all_pred_3d)

    # Compute keypoint metrics (2D)
    print("\nComputing keypoint metrics...")
    n_samples = len(all_kp_projs_detected)
    kp_metrics = compute_keypoint_metrics(
        all_kp_projs_detected.reshape(n_samples * len(keypoint_names), 2),
        all_kp_projs_gt.reshape(n_samples * len(keypoint_names), 2),
        raw_resolution,
        auc_threshold=args.kp_auc_threshold
    )

    # ===== Direct ADD: model's 3D head output vs GT 3D =====
    print("Computing Direct ADD metrics (model 3D head vs GT 3D)...")
    direct_add_metrics = compute_direct_add_metrics(
        all_pred_3d,
        all_kp_pos_gt,
        add_auc_threshold=args.add_auc_threshold
    )

    # ===== PnP ADD (DREAM baseline): 2D pred + GT 3D + K -> PnP =====
    print("Computing PnP ADD metrics (DREAM baseline)...")
    pnp_adds = []

    for kp_det, kp_gt, kp_3d, n_inframe in tqdm(
        zip(all_kp_projs_detected, all_kp_projs_gt, all_kp_pos_gt, all_n_inframe_projs_gt),
        total=len(all_kp_projs_detected),
        desc="PnP solving"
    ):
        # Filter valid detections
        idx_good = np.where((kp_det[:, 0] > -999.0) & (kp_det[:, 1] > -999.0))[0]

        if len(idx_good) >= 4:  # Need at least 4 points for PnP
            kp_det_pnp = kp_det[idx_good]
            kp_3d_pnp = kp_3d[idx_good]

            # Solve PnP
            pnp_retval, translation, quaternion = dream.geometric_vision.solve_pnp(
                kp_3d_pnp, kp_det_pnp, camera_K
            )

            if pnp_retval:
                # Compute ADD
                add = dream.geometric_vision.add_from_pose(
                    translation, quaternion, kp_3d_pnp, camera_K
                )
                pnp_adds.append(add)
            else:
                pnp_adds.append(-999.0)
        else:
            pnp_adds.append(-999.0)

    pnp_metrics = compute_pnp_metrics(
        pnp_adds,
        all_n_inframe_projs_gt,
        add_auc_threshold=args.add_auc_threshold
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nDataset: {args.dataset_dir}")
    print(f"Model: {args.model_path}")
    print(f"Number of frames: {n_samples}")

    # 2D Keypoint metrics
    print(f"\n# L2 error (px) for in-frame keypoints (n = {kp_metrics['num_gt_inframe']}):")
    if kp_metrics['l2_error_auc'] is not None:
        print(f"#    AUC: {kp_metrics['l2_error_auc']:.5f}")
        print(f"#       AUC threshold: {kp_metrics['l2_error_auc_thresh_px']:.5f}")
        print(f"#    Mean: {kp_metrics['l2_error_mean_px']:.5f}")
        print(f"#    Median: {kp_metrics['l2_error_median_px']:.5f}")
        print(f"#    Std Dev: {kp_metrics['l2_error_std_px']:.5f}")
    else:
        print("#    No valid keypoints found")

    # Direct ADD (model's 3D head)
    print(f"\n# Direct ADD (m) - Model 3D head vs GT 3D (n = {direct_add_metrics['num_frames']}):")
    if direct_add_metrics['add_auc'] is not None:
        print(f"#    AUC: {direct_add_metrics['add_auc']:.5f}")
        print(f"#       AUC threshold: {direct_add_metrics['add_auc_thresh']:.5f}")
        print(f"#    Mean: {direct_add_metrics['add_mean']:.5f}")
        print(f"#    Median: {direct_add_metrics['add_median']:.5f}")
        print(f"#    Std Dev: {direct_add_metrics['add_std']:.5f}")
        print(f"#    Per-keypoint Mean: {direct_add_metrics['per_kp_mean']:.5f}")
        print(f"#    Per-keypoint Median: {direct_add_metrics['per_kp_median']:.5f}")
    else:
        print("#    No valid frames for direct ADD")

    # PnP ADD (DREAM baseline)
    print(f"\n# PnP ADD (m) - DREAM baseline: 2D pred + GT_3D + K (n = {pnp_metrics['num_pnp_found']}):")
    if pnp_metrics['num_pnp_found'] > 0:
        print(f"#    AUC: {pnp_metrics['add_auc']:.5f}")
        print(f"#       AUC threshold: {pnp_metrics['add_auc_thresh']:.5f}")
        print(f"#    Mean: {pnp_metrics['add_mean']:.5f}")
        print(f"#    Median: {pnp_metrics['add_median']:.5f}")
        print(f"#    Std Dev: {pnp_metrics['add_std']:.5f}")
        print(f"#    PnP Success Rate: {pnp_metrics['num_pnp_found']}/{pnp_metrics['num_pnp_possible']} ({pnp_metrics['num_pnp_found']/max(1, pnp_metrics['num_pnp_possible'])*100:.1f}%)")
    else:
        print("#    No successful PnP solutions")

    print("=" * 80)

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'dataset': str(args.dataset_dir),
            'model': str(args.model_path),
            'num_frames': n_samples,
            'keypoint_metrics': {k: float(v) if v is not None else None for k, v in kp_metrics.items()},
            'direct_add_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in direct_add_metrics.items()},
            'pnp_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in pnp_metrics.items()}
        }

        results_path = output_dir / 'eval_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Inference on Dataset with DREAM-style Metrics')

    # Model
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model-name', type=str, default=None,
                        help='DINOv3 model name (auto-read from config.yaml if not specified)')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Input image size (auto-read from config.yaml if not specified)')
    parser.add_argument('--heatmap-size', type=int, default=None,
                        help='Output heatmap size (auto-read from config.yaml if not specified)')

    # Dataset
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to NDDS dataset directory')

    # Inference
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Metrics
    parser.add_argument('--kp-auc-threshold', type=float, default=20.0,
                        help='AUC threshold for keypoint L2 error (pixels)')
    parser.add_argument('--add-auc-threshold', type=float, default=0.1,
                        help='AUC threshold for ADD metric (meters)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    run_inference(args)


if __name__ == '__main__':
    main()
