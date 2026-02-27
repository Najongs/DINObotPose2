"""
DINOv3 Pose Estimation Training Script
DREAM 학습 방식을 참고한 학습 코드
"""

import argparse
import os
import time
import pickle
import random
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import yaml
import wandb
import cv2

from model import DINOv3PoseEstimator, panda_forward_kinematics

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DREAM')))
import dream
from dream import analysis as dream_analysis

def solve_pnp_epnp(object_points, image_points, camera_matrix):
    """
    Solve PnP using EPnP algorithm (faster and more stable than iterative).

    Args:
        object_points: (N, 3) 3D points in object/robot frame
        image_points: (N, 2) 2D points in image coordinates
        camera_matrix: (3, 3) camera intrinsic matrix

    Returns:
        success (bool): Whether PnP succeeded
        R (3, 3): Rotation matrix
        t (3,): Translation vector
    """
    try:
        # EPnP requires at least 4 points
        if len(object_points) < 4:
            return False, None, None

        # Ensure correct data types
        object_points = object_points.astype(np.float64)
        image_points = image_points.astype(np.float64)
        camera_matrix = camera_matrix.astype(np.float64)

        # Solve PnP with EPnP algorithm
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            None,  # No distortion
            flags=cv2.SOLVEPNP_EPNP  # Use EPnP algorithm
        )

        if not success:
            return False, None, None

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        return True, R, t

    except Exception as e:
        # PnP can fail for degenerate cases
        return False, None, None


def get_keypoints_from_heatmaps(heatmaps):
    """
    Extract keypoint coordinates from heatmaps using argmax.
    heatmaps: (B, N, H, W)
    Returns: (B, N, 2) [x, y] coordinates
    """
    B, N, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)

    y = max_indices // W
    x = max_indices % W

    return torch.stack([x, y], dim=-1).float()


class UnifiedPoseLoss(nn.Module):
    """
    Multi-task loss for Pose Estimation:
    1. 2D Heatmap Loss (configurable: MSE/L1/SmoothL1)
    2. 3D Keypoint Lifting Loss (configurable: MSE/L1/SmoothL1)
    3. Camera-frame 3D loss (for joint_angle mode with PnP transform)
    4. Adaptive loss weighting (uncertainty-based)
    5. PnP failure penalty

    Loss type recommendation:
    - MSE: Traditional, sensitive to outliers
    - L1: Robust to outliers, aligns with ADD metric
    - SmoothL1 (Huber): Best of both - L2 for small errors, L1 for large errors
    """
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        kp3d_weight: float = 10.0,  # 3D 좌표는 값의 범위가 작으므로 가중치를 높게 설정
        heatmap_size: int = 512,
        angle_weight: float = 0.0,  # Joint angle MSE loss weight
        fk_3d_weight: float = 0.0,  # FK 3D keypoint MSE loss weight (robot frame)
        camera_3d_weight: float = 0.0,  # Camera-frame 3D loss weight (with PnP transform)
        loss_type: str = 'smoothl1',  # Loss function type: 'mse', 'l1', 'smoothl1'
        use_adaptive_weighting: bool = True,  # Use uncertainty-based adaptive weighting
        pnp_failure_penalty_weight: float = 0.1,  # PnP failure penalty weight
        refinement_weight: float = 0.0,  # Iterative refinement loss weight
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.kp3d_weight = kp3d_weight
        self.heatmap_size = heatmap_size
        self.angle_weight = angle_weight
        self.fk_3d_weight = fk_3d_weight
        self.camera_3d_weight = camera_3d_weight
        self.loss_type = loss_type
        self.use_adaptive_weighting = use_adaptive_weighting
        self.pnp_failure_penalty_weight = pnp_failure_penalty_weight
        self.refinement_weight = refinement_weight

        # Heatmap loss: 항상 MSE 사용 (값 범위 0~1, Gaussian GT와 MSE가 자연스러움)
        self.heatmap_loss_fn = nn.MSELoss(reduction='none')

        # 3D / angle loss: loss_type 인수에 따라 선택
        # SmoothL1(beta=0.01): 3D 오차 수 cm 기준, 1cm 이상이면 L1(robust) 전환
        # beta=1.0(구버전)은 오차 1m 기준이라 수 cm 오차에서 항상 L2만 작동했음
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'smoothl1':
            self.loss_fn = nn.SmoothL1Loss(reduction='none', beta=0.01)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Adaptive loss weighting parameters (learnable uncertainty)
        if use_adaptive_weighting:
            self.log_sigma_heatmap = nn.Parameter(torch.zeros(1))
            self.log_sigma_3d = nn.Parameter(torch.zeros(1))

        self.eps = 1e-6  # Numerical stability

    def forward(self, pred_dict, gt_dict):
        """
        Args:
            pred_dict: {
                'heatmaps_2d': (B, MAX_JOINTS, H, W),
                'keypoints_3d': (B, MAX_JOINTS, 3),
            }
            gt_dict: {
                'heatmaps_2d': (B, MAX_JOINTS, H, W),
                'keypoints_3d': (B, MAX_JOINTS, 3),
                'valid_mask': (B, MAX_JOINTS) - bool mask,
            }
        """
        # Get valid mask
        valid_mask = gt_dict.get('valid_mask', None)  # (B, MAX_JOINTS)

        # 1. 2D Heatmap Loss (항상 MSE: Gaussian GT와 자연스러운 조합)
        if valid_mask is not None:
            mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).float()  # (B, MAX_JOINTS, 1, 1)
            heatmap_diff = self.heatmap_loss_fn(pred_dict['heatmaps_2d'], gt_dict['heatmaps_2d'])
            heatmap_diff_masked = heatmap_diff * mask_expanded
            loss_heatmap = heatmap_diff_masked.sum() / (mask_expanded.sum() + self.eps)
        else:
            heatmap_diff = self.heatmap_loss_fn(pred_dict['heatmaps_2d'], gt_dict['heatmaps_2d'])
            loss_heatmap = heatmap_diff.mean()

        # 2. 3D Keypoint Loss - with masking
        kp3d_pred = pred_dict['keypoints_3d']
        kp3d_gt = gt_dict['keypoints_3d']

        if valid_mask is not None:
            mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, MAX_JOINTS, 1)
            kp3d_diff = self.loss_fn(kp3d_pred, kp3d_gt)  # (B, MAX_JOINTS, 3)
            kp3d_diff_masked = kp3d_diff * mask_expanded
            loss_kp3d = kp3d_diff_masked.sum() / (mask_expanded.sum() + self.eps)
        else:
            kp3d_diff = self.loss_fn(kp3d_pred, kp3d_gt)
            loss_kp3d = kp3d_diff.mean()

        # Total Loss with Adaptive Weighting or Fixed Weights
        # kp3d_weight=0이면 (joint_angle 모드 등) kp3d loss를 total에서 완전히 제외
        use_kp3d = self.kp3d_weight > 0

        if self.use_adaptive_weighting:
            # Uncertainty-based weighting: loss / (2 * exp(log_sigma)) + log_sigma
            # joint_angle 모드에서는 kp3d_weight=0이므로 kp3d 항 제외
            # (robot frame vs camera frame 비교가 되는 잘못된 gradient 방지)
            if use_kp3d:
                total_loss = (
                    loss_heatmap / (2 * torch.exp(self.log_sigma_heatmap)) + self.log_sigma_heatmap +
                    loss_kp3d / (2 * torch.exp(self.log_sigma_3d)) + self.log_sigma_3d
                )
            else:
                total_loss = (
                    loss_heatmap / (2 * torch.exp(self.log_sigma_heatmap)) + self.log_sigma_heatmap
                )
            loss_dict = {
                'total': total_loss.item(),
                'heatmap': loss_heatmap.item(),
                'kp3d': loss_kp3d.item(),
                'sigma_hm': torch.exp(self.log_sigma_heatmap).item(),
                'sigma_3d': torch.exp(self.log_sigma_3d).item(),
            }
        else:
            # Fixed weighting
            total_loss = (
                self.heatmap_weight * loss_heatmap +
                self.kp3d_weight * loss_kp3d  # kp3d_weight=0이면 자동으로 0
            )
            loss_dict = {
                'total': total_loss.item(),
                'heatmap': loss_heatmap.item(),
                'kp3d': loss_kp3d.item(),
            }

        # Joint angle loss (joint_angle mode only)
        if self.angle_weight > 0 and 'joint_angles' in pred_dict and 'angles' in gt_dict:
            pred_angles = pred_dict['joint_angles']  # (B, 7)
            gt_angles = gt_dict['angles']  # (B, 7)
            loss_angle = self.loss_fn(pred_angles, gt_angles).mean()
            total_loss = total_loss + self.angle_weight * loss_angle
            loss_dict['angle'] = loss_angle.item()

        # FK 3D loss: compare FK(pred_angles) vs FK(gt_angles) in robot frame
        if self.fk_3d_weight > 0 and 'joint_angles' in pred_dict and 'angles' in gt_dict:
            pred_kp_robot = pred_dict['keypoints_3d_robot']  # (B, 7, 3) already FK'd
            gt_angles = gt_dict['angles']  # (B, 7)
            gt_kp_robot = panda_forward_kinematics(gt_angles)  # (B, 7, 3)
            loss_fk_3d = self.loss_fn(pred_kp_robot, gt_kp_robot).mean()
            total_loss = total_loss + self.fk_3d_weight * loss_fk_3d
            loss_dict['fk_3d'] = loss_fk_3d.item()

        # Camera-frame 3D loss: Transform robot-frame to camera-frame using PnP
        if self.camera_3d_weight > 0 and 'joint_angles' in pred_dict and 'keypoints_3d_robot' in pred_dict:
            if 'keypoints' in gt_dict and 'camera_K' in gt_dict and 'original_size' in gt_dict:
                try:
                    pred_kp_robot = pred_dict['keypoints_3d_robot']  # (B, 7, 3)
                    gt_kp_camera = gt_dict['keypoints_3d']  # (B, 7, 3) camera frame
                    gt_kp_2d_heatmap = gt_dict['keypoints']  # (B, 7, 2) in heatmap coords
                    camera_K = gt_dict['camera_K']  # (B, 3, 3)
                    original_size = gt_dict['original_size']  # (B, 2) [W, H]
                    valid_mask = gt_dict.get('valid_mask', None)  # (B, 7)

                    # gt_kp_2d is in heatmap coords → scale to original image coords for PnP
                    # camera_K is defined on original image (e.g. 640x480)
                    pred_heatmaps = pred_dict['heatmaps_2d']  # (B, 7, H, W)
                    H_hm, W_hm = pred_heatmaps.shape[2], pred_heatmaps.shape[3]
                    scale_x = original_size[:, 0:1] / W_hm  # (B, 1)
                    scale_y = original_size[:, 1:2] / H_hm  # (B, 1)
                    # gt 2D: heatmap → original image coords
                    gt_kp_2d_x = gt_kp_2d_heatmap[:, :, 0] * scale_x  # (B, 7)
                    gt_kp_2d_y = gt_kp_2d_heatmap[:, :, 1] * scale_y  # (B, 7)
                    gt_kp_2d = torch.stack([gt_kp_2d_x, gt_kp_2d_y], dim=-1)  # (B, 7, 2)

                    # Transform robot-frame to camera-frame using GT-based PnP (for stability)
                    B = pred_kp_robot.shape[0]
                    pred_kp_camera_list = []
                    gt_kp_camera_list = []
                    valid_indices = []  # Track successful PnP samples

                    for b in range(B):
                        # Use GT robot-frame (from FK) for PnP to get stable transform
                        gt_angles_b = gt_dict['angles'][b]  # (7,)
                        gt_kp_robot_b = panda_forward_kinematics(gt_angles_b.unsqueeze(0))[0]  # (7, 3)
                        gt_kp_2d_b = gt_kp_2d[b]  # (7, 2) in original image coords
                        camera_K_b = camera_K[b]  # (3, 3)

                        # Convert to numpy for PnP
                        gt_kp_robot_np = gt_kp_robot_b.detach().cpu().numpy()
                        gt_kp_2d_np = gt_kp_2d_b.detach().cpu().numpy()
                        camera_K_np = camera_K_b.detach().cpu().numpy()

                        # Solve PnP using EPnP algorithm to get robot-to-camera transform
                        success, R, t = solve_pnp_epnp(
                            gt_kp_robot_np, gt_kp_2d_np, camera_K_np
                        )

                        if success:
                            # Convert to tensors
                            R_tensor = torch.from_numpy(R).float().to(pred_kp_robot.device)
                            t_tensor = torch.from_numpy(t).float().to(pred_kp_robot.device)

                            # Transform predicted robot-frame to camera-frame
                            pred_kp_robot_b = pred_kp_robot[b]  # (7, 3)
                            pred_kp_camera_b = (R_tensor @ pred_kp_robot_b.T).T + t_tensor  # (7, 3)

                            pred_kp_camera_list.append(pred_kp_camera_b)
                            gt_kp_camera_list.append(gt_kp_camera[b])
                            valid_indices.append(b)
                        # else: Skip failed PnP samples entirely (don't add to loss)

                    # Compute loss only for successful PnP samples
                    if len(pred_kp_camera_list) > 0:
                        pred_kp_camera = torch.stack(pred_kp_camera_list, dim=0)  # (N_valid, 7, 3)
                        gt_kp_camera_valid = torch.stack(gt_kp_camera_list, dim=0)  # (N_valid, 7, 3)

                        # Apply valid_mask if available (only for valid PnP samples)
                        if valid_mask is not None:
                            valid_mask_subset = valid_mask[valid_indices]  # (N_valid, 7)
                            mask_expanded = valid_mask_subset.unsqueeze(-1).float()  # (N_valid, 7, 1)
                            camera_3d_diff = self.loss_fn(pred_kp_camera, gt_kp_camera_valid)
                            camera_3d_diff_masked = camera_3d_diff * mask_expanded
                            loss_camera_3d = camera_3d_diff_masked.sum() / (mask_expanded.sum() + self.eps)
                        else:
                            loss_camera_3d = self.loss_fn(pred_kp_camera, gt_kp_camera_valid).mean()

                        total_loss = total_loss + self.camera_3d_weight * loss_camera_3d
                        loss_dict['camera_3d'] = loss_camera_3d.item()

                        # Track PnP success rate and add failure penalty
                        pnp_success_rate = len(valid_indices) / B
                        loss_dict['pnp_success_rate'] = pnp_success_rate

                        # Add penalty if PnP success rate is low (< 90%)
                        if pnp_success_rate < 0.9:
                            pnp_penalty = (1 - pnp_success_rate) * self.pnp_failure_penalty_weight
                            total_loss = total_loss + pnp_penalty
                            loss_dict['pnp_penalty'] = pnp_penalty
                    else:
                        # All PnP failed - high penalty
                        loss_dict['pnp_success_rate'] = 0.0
                        pnp_penalty = self.pnp_failure_penalty_weight
                        total_loss = total_loss + pnp_penalty
                        loss_dict['pnp_penalty'] = pnp_penalty

                except Exception as e:
                    # PnP can fail, gracefully skip camera-frame loss
                    import traceback
                    print(f"[Warning] Camera-frame 3D loss failed: {e}\n{traceback.format_exc()}")
                    pass

        # Iterative refinement loss: penalize all intermediate steps with decaying weight
        if self.refinement_weight > 0 and 'all_refined_angles' in pred_dict and 'angles' in gt_dict:
            all_angles = pred_dict['all_refined_angles']  # list of (B, 7), length N+1
            all_kp3d = pred_dict['all_refined_kp3d_robot']  # list of (B, 7, 3), length N+1
            gt_angles = gt_dict['angles']  # (B, 7)
            gt_kp_robot = panda_forward_kinematics(gt_angles)  # (B, 7, 3)
            valid_mask_refine = pred_dict.get('refinement_valid_mask', None)  # (B,)

            N = len(all_angles) - 1  # number of refinement iterations
            decay = 0.5  # decay factor for earlier iterations
            refine_loss = torch.tensor(0.0, device=total_loss.device)
            refine_count = 0

            # Skip index 0 (initial prediction, already covered by angle/fk_3d loss)
            for step_i in range(1, len(all_angles)):
                w = decay ** (N - step_i)  # later steps get higher weight

                # Angle loss for this step
                step_angle_loss = self.loss_fn(all_angles[step_i], gt_angles).mean()

                # FK 3D loss for this step
                step_fk_loss = self.loss_fn(all_kp3d[step_i], gt_kp_robot).mean()

                step_loss = step_angle_loss + step_fk_loss

                # Mask out invalid PnP samples if mask available
                if valid_mask_refine is not None:
                    # Only count loss if at least one valid sample
                    n_valid = valid_mask_refine.float().sum()
                    if n_valid > 0:
                        refine_loss = refine_loss + w * step_loss
                        refine_count += 1
                else:
                    refine_loss = refine_loss + w * step_loss
                    refine_count += 1

            if refine_count > 0:
                refine_loss = refine_loss / refine_count
                total_loss = total_loss + self.refinement_weight * refine_loss
                loss_dict['refinement'] = refine_loss.item()

        # Update total in loss_dict
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class Trainer:
    """Training manager for DINOv3 pose estimation"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        device: torch.device,
        output_dir: str,
        config: Dict,
        camera_K: Optional[np.ndarray] = None,
        raw_res: Tuple[int, int] = (640, 480),
        resume_from: Optional[str] = None,
        resume_lr: Optional[float] = None,
        local_rank: int = -1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config
        self.camera_K = camera_K
        self.raw_res = raw_res
        self.resume_lr = resume_lr
        self.local_rank = local_rank
        self.is_distributed = local_rank != -1
        self.is_main_process = (not self.is_distributed) or (local_rank == 0)

        # Create output directory (only on main process)
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb (only on main process)
        if self.is_main_process:
            try:
                # Create a unique run name if not specified
                run_name = config.get('wandb_run_name', None)
                if run_name is None:
                    run_name = f"{config.get('model_name', 'dinov3').split('/')[-1]}_{self.output_dir.name}"

                wandb.init(
                    project=config.get('wandb_project', 'dinov3-pose-estimation'),
                    name=run_name,
                    config=config,
                    dir=str(self.output_dir),
                    resume='allow' if resume_from else None,
                    reinit=False  # Prevent reinitialization
                )
                print(f"WandB initialized successfully: {wandb.run.name}")
                print(f"WandB URL: {wandb.run.get_url()}")
            except Exception as e:
                print(f"Warning: Failed to initialize WandB: {e}")
                print("Training will continue without WandB logging")

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.train_log = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'timestamps': []
        }

        # Resume from checkpoint
        if resume_from:
            self._load_checkpoint(resume_from)

        # Save config (only on main process)
        if self.is_main_process:
            with open(self.output_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f)

    def _calculate_target_lr_at_epoch(self, epoch: int) -> float:
        """Calculate target LR at given epoch based on scheduler type"""
        import torch.optim as optim
        from torch.optim.lr_scheduler import SequentialLR, LinearLR

        # Get base LR from optimizer
        base_lr = self.optimizer.param_groups[0]['initial_lr'] if 'initial_lr' in self.optimizer.param_groups[0] else self.optimizer.param_groups[0]['lr']

        # Calculate based on scheduler type
        if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingLR):
            # Cosine annealing formula: eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
            T_max = self.scheduler.T_max
            eta_min = self.scheduler.eta_min
            import math
            return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2

        elif isinstance(self.scheduler, SequentialLR):
            # SequentialLR with warmup + cosine
            # Check if we're still in warmup
            milestones = self.scheduler._milestones
            if epoch < milestones[0]:
                # In warmup phase (LinearLR)
                warmup_scheduler = self.scheduler._schedulers[0]
                start_factor = warmup_scheduler.start_factor
                total_iters = warmup_scheduler.total_iters
                return base_lr * (start_factor + (1 - start_factor) * epoch / total_iters)
            else:
                # In cosine phase
                cosine_scheduler = self.scheduler._schedulers[1]
                T_max = cosine_scheduler.T_max
                eta_min = cosine_scheduler.eta_min
                epoch_in_cosine = epoch - milestones[0]
                import math
                return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * epoch_in_cosine / T_max)) / 2

        elif isinstance(self.scheduler, optim.lr_scheduler.StepLR):
            # StepLR: gamma ** (epoch // step_size)
            step_size = self.scheduler.step_size
            gamma = self.scheduler.gamma
            return base_lr * (gamma ** (epoch // step_size))

        else:
            # Unknown scheduler, return current LR
            return self.optimizer.param_groups[0]['lr']

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        if self.is_main_process:
            print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state (handle DDP wrapper)
        state_dict = checkpoint['model_state_dict']
        if self.is_distributed and not any(k.startswith('module.') for k in state_dict.keys()):
            # Add 'module.' prefix if loading non-DDP checkpoint into DDP model
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not self.is_distributed and any(k.startswith('module.') for k in state_dict.keys()):
            # Remove 'module.' prefix if loading DDP checkpoint into non-DDP model
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Filter out keys with shape mismatch (e.g., after architecture changes like PixelShuffle)
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {}
        shape_mismatch_keys = []

        for key, value in state_dict.items():
            if key in model_state_dict:
                if value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    shape_mismatch_keys.append(key)
                    if self.is_main_process:
                        print(f"⚠️  Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state_dict[key].shape}")
            else:
                # Key doesn't exist in current model (will be handled as unexpected_keys)
                filtered_state_dict[key] = value

        # Load model state with strict=False to allow partial loading (missing keys will be randomly initialized)
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)

        if self.is_main_process:
            if missing_keys:
                print(f"⚠️  Missing keys in checkpoint (will be randomly initialized):")
                for key in missing_keys:
                    print(f"   - {key}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys in checkpoint (ignored):")
                for key in unexpected_keys:
                    print(f"   - {key}")
            if not missing_keys and not unexpected_keys:
                print("✓ All model parameters loaded successfully")

        # Try to load optimizer state (may fail if model structure changed)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.is_main_process:
                print("✓ Optimizer state restored")
        except (ValueError, KeyError) as e:
            if self.is_main_process:
                print(f"⚠️  Failed to load optimizer state (model structure may have changed): {e}")
                print("   Optimizer will start with fresh state")

        # Load epoch and training state
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_log = checkpoint.get('train_log', self.train_log)

        # Handle learning rate on resume
        if self.resume_lr is not None:
            # Manual LR specified - override scheduler state
            if self.is_main_process:
                print(f"⚠️  Using manually specified resume LR: {self.resume_lr:.2e}")

            # Set LR immediately
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.resume_lr

            # Try to load scheduler state but LR will be overridden
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if self.is_main_process:
                        print(f"   → Scheduler state loaded (but LR overridden)")
                except (ValueError, KeyError) as e:
                    if self.is_main_process:
                        print(f"   → Failed to load scheduler state: {e}")
                    # Set scheduler's last_epoch to start_epoch - 1
                    if self.scheduler:
                        self.scheduler.last_epoch = self.start_epoch - 1
            else:
                # No scheduler state, set last_epoch
                if self.scheduler:
                    self.scheduler.last_epoch = self.start_epoch - 1

            # Override LR again after scheduler load
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.resume_lr

        else:
            # No manual LR - use scheduler state or calculate
            scheduler_loaded = False
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    scheduler_loaded = True
                    if self.is_main_process:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"✓ Scheduler state restored. Current LR: {current_lr:.2e}")
                except (ValueError, KeyError) as e:
                    if self.is_main_process:
                        print(f"⚠️  Failed to load scheduler state: {e}")

            # If scheduler state failed to load, calculate LR from scheduler
            if self.scheduler and not scheduler_loaded:
                if self.is_main_process:
                    print(f"⚠️  Adjusting scheduler to resume from epoch {self.start_epoch}")

                # Calculate target LR at start_epoch using scheduler formula
                target_lr = self._calculate_target_lr_at_epoch(self.start_epoch)
                if self.is_main_process:
                    print(f"   → Calculated LR from scheduler: {target_lr:.2e}")

                # Set LR to target value immediately
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = target_lr

                # Set scheduler's last_epoch to start_epoch - 1 (so next step() gives correct LR)
                self.scheduler.last_epoch = self.start_epoch - 1

        if self.is_main_process:
            print(f"\n{'='*80}")
            print(f"Resume Summary:")
            print(f"  Starting from epoch: {self.start_epoch}")
            print(f"  Best validation loss: {self.best_val_loss:.4f}")
            print(f"  Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*80}\n")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint (only on main process)"""
        if not self.is_main_process:
            return

        # Get model state dict (unwrap DDP if needed)
        model_state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_log': self.train_log,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save current epoch checkpoint
        checkpoint_path = self.output_dir / f'epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

        # Maintain last 3 checkpoints
        all_ckpts = sorted(list(self.output_dir.glob('epoch_*.pth')), key=lambda x: int(x.stem.split('_')[1]))
        if len(all_ckpts) > 3:
            for old_ckpt in all_ckpts[:-3]:
                try:
                    old_ckpt.unlink()
                    print(f"Removed old checkpoint: {old_ckpt}")
                except Exception as e:
                    print(f"Error removing old checkpoint {old_ckpt}: {e}")

    def _prepare_batch(self, batch):
        """batch를 device로 이동하고 gt_dict, forward kwargs 구성"""
        images = batch['image'].to(self.device)
        gt_heatmaps = batch['heatmaps'].to(self.device)
        gt_keypoints_3d = batch['keypoints_3d'].to(self.device)
        gt_valid_mask = batch['valid_mask'].to(self.device)
        gt_keypoints_2d = batch['keypoints'].to(self.device) if 'keypoints' in batch else None
        camera_K = batch['camera_K'].to(self.device) if 'camera_K' in batch else None
        original_size = batch['original_size'].to(self.device) if 'original_size' in batch else None
        gt_angles = batch['angles'].to(self.device) if 'angles' in batch else None

        # GT 2D → original image coords for refinement
        gt_2d_image = None
        if gt_keypoints_2d is not None and original_size is not None:
            hm_size = self.config.get('heatmap_size', 512)
            scale_x = original_size[:, 0:1] / hm_size
            scale_y = original_size[:, 1:2] / hm_size
            gt_2d_image = torch.stack([
                gt_keypoints_2d[:, :, 0] * scale_x,
                gt_keypoints_2d[:, :, 1] * scale_y,
            ], dim=-1)

        gt_dict = {
            'heatmaps_2d': gt_heatmaps,
            'keypoints_3d': gt_keypoints_3d,
            'valid_mask': gt_valid_mask,
        }
        if gt_angles is not None:
            gt_dict['angles'] = gt_angles
        if gt_keypoints_2d is not None:
            gt_dict['keypoints'] = gt_keypoints_2d
        if camera_K is not None:
            gt_dict['camera_K'] = camera_K
        if original_size is not None:
            gt_dict['original_size'] = original_size

        orig_size_list = original_size[0].tolist() if original_size is not None else None

        return images, gt_dict, camera_K, orig_size_list, gt_angles, gt_2d_image

    # Mapping from loss_dict keys to short display names
    _LOSS_EXTRA_KEYS = [('angle', 'ang'), ('fk_3d', 'fk'), ('camera_3d', 'cam3d'), ('refinement', 'ref')]

    def _format_postfix(self, loss_dict):
        """loss_dict → tqdm postfix dict"""
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']

        postfix = {
            'loss': f"{loss_dict['total']:.6f}",
            'hm': f"{loss_dict['heatmap']:.6f}",
            'kp3d': f"{loss_dict['kp3d']:.6f}",
            'lr': f"{current_lr:.2e}",  # Show LR in scientific notation
        }
        for key, short in self._LOSS_EXTRA_KEYS:
            if key in loss_dict:
                postfix[short] = f"{loss_dict[key]:.6f}"
        return postfix

    def _format_loss_extra(self, losses):
        """epoch summary용 extra loss 문자열"""
        parts = []
        for key, short in self._LOSS_EXTRA_KEYS:
            if key in losses:
                parts.append(f"{short}: {losses[key]:.6f}")
        return ', '.join(parts)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        # Set sampler epoch for proper shuffling in distributed training
        if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        epoch_losses = {
            'total': [],
            'heatmap': [],
            'kp3d': [],
        }

        # Only show progress bar on main process
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        else:
            pbar = self.train_loader

        batch_idx = 0
        for batch in pbar:
            batch_idx += 1
            images, gt_dict, camera_K, orig_size_list, gt_angles, gt_2d_image = self._prepare_batch(batch)

            # Forward pass
            pred_dict = self.model(
                images, camera_K=camera_K, original_size=orig_size_list,
                gt_angles=gt_angles, gt_2d_image=gt_2d_image
            )

            # Compute loss
            loss, loss_dict = self.criterion(pred_dict, gt_dict)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Record losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)

            # Update progress bar (only on main process)
            if self.is_main_process:
                pbar.set_postfix(self._format_postfix(loss_dict))

                # Log batch losses to wandb every N batches
                if wandb.run is not None and batch_idx % 10 == 0:
                    try:
                        global_step = epoch * len(self.train_loader) + batch_idx
                        batch_log = {
                            'batch/train_loss': loss_dict['total'],
                            'batch/train_heatmap_loss': loss_dict['heatmap'],
                            'batch/train_kp3d_loss': loss_dict['kp3d'],
                        }
                        for key, wandb_key in [('angle', 'batch/train_angle_loss'),
                                                ('fk_3d', 'batch/train_fk3d_loss'),
                                                ('camera_3d', 'batch/train_camera3d_loss'),
                                                ('pnp_success_rate', 'batch/pnp_success_rate'),
                                                ('refinement', 'batch/train_refinement_loss')]:
                            if key in loss_dict:
                                batch_log[wandb_key] = loss_dict[key]
                        wandb.log(batch_log, step=global_step)
                    except Exception:
                        pass  # Silently ignore batch logging errors

        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}

        # Synchronize losses across processes in distributed training
        if self.is_distributed:
            for key in avg_losses:
                avg_tensor = torch.tensor(avg_losses[key], device=self.device)
                dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
                avg_losses[key] = (avg_tensor / dist.get_world_size()).item()

        return avg_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        epoch_losses = {
            'total': [],
            'heatmap': [],
            'kp3d': [],
        }

        # Metrics collection
        all_kp_projs_est = []
        all_kp_projs_gt = []
        all_kp_pos_gt = []

        # Only show progress bar on main process
        if self.is_main_process:
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        else:
            pbar = self.val_loader

        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            gt_heatmaps = batch['heatmaps'].to(self.device)
            gt_keypoints_3d = batch['keypoints_3d'].to(self.device)
            gt_valid_mask = batch['valid_mask'].to(self.device)
            gt_keypoints_2d = batch['keypoints'].to(self.device) if 'keypoints' in batch else None

            # Camera K and original size for depth_only and joint_angle modes
            camera_K = batch['camera_K'].to(self.device) if 'camera_K' in batch else None
            original_size = batch['original_size'].to(self.device) if 'original_size' in batch else None

            # Joint angles for joint_angle mode
            gt_angles = batch['angles'].to(self.device) if 'angles' in batch else None

            # Prepare GT 2D in original image coords for refinement module
            gt_2d_image = None
            if gt_keypoints_2d is not None and original_size is not None:
                H_hm = self.config.get('heatmap_size', 512)
                W_hm = H_hm
                scale_x = original_size[:, 0:1] / W_hm  # (B, 1)
                scale_y = original_size[:, 1:2] / H_hm  # (B, 1)
                gt_2d_image = torch.stack([
                    gt_keypoints_2d[:, :, 0] * scale_x,
                    gt_keypoints_2d[:, :, 1] * scale_y,
                ], dim=-1)  # (B, 7, 2) in original image coords

            # Forward pass
            orig_size_list = original_size[0].tolist() if original_size is not None else None
            pred_dict = self.model(
                images, camera_K=camera_K, original_size=orig_size_list,
                gt_angles=gt_angles, gt_2d_image=gt_2d_image
            )

            # Prepare ground truth dictionary
            gt_dict = {
                'heatmaps_2d': gt_heatmaps,
                'keypoints_3d': gt_keypoints_3d,
                'valid_mask': gt_valid_mask
            }
            if gt_angles is not None:
                gt_dict['angles'] = gt_angles
            if gt_keypoints_2d is not None:
                gt_dict['keypoints'] = gt_keypoints_2d
            if camera_K is not None:
                gt_dict['camera_K'] = camera_K
            if original_size is not None:
                gt_dict['original_size'] = original_size

            # Compute loss
            _, loss_dict = self.criterion(pred_dict, gt_dict)

            # Record losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)

            # Collect data for metrics
            if self.camera_K is not None:
                # Extract coordinates from predicted heatmaps
                pred_kps = get_keypoints_from_heatmaps(pred_dict['heatmaps_2d']).cpu().numpy()
                gt_kps = batch['keypoints'].numpy()
                gt_kps_3d = batch['keypoints_3d'].numpy()
                
                # Rescale to raw resolution
                h_size = self.config.get('heatmap_size', 512)
                scale_x = self.raw_res[0] / h_size
                scale_y = self.raw_res[1] / h_size
                
                for i in range(len(pred_kps)):
                    p_kps = pred_kps[i].copy()
                    g_kps = gt_kps[i].copy()
                    p_kps[:, 0] *= scale_x
                    p_kps[:, 1] *= scale_y
                    g_kps[:, 0] *= scale_x
                    g_kps[:, 1] *= scale_y
                    
                    # Valid keypoints only for evaluation
                    valid_kp_mask = (g_kps[:, 0] >= 0) & (g_kps[:, 1] >= 0)
                    if valid_kp_mask.any():
                        all_kp_projs_est.append(p_kps)
                        all_kp_projs_gt.append(g_kps)
                        all_kp_pos_gt.append(gt_kps_3d[i])

            # Update progress bar (only on main process)
            if self.is_main_process:
                current_lr = self.optimizer.param_groups[0]['lr']
                postfix = {
                    'loss': f"{loss_dict['total']:.6f}",
                    'hm': f"{loss_dict['heatmap']:.6f}",
                    'kp3d': f"{loss_dict['kp3d']:.6f}",
                    'lr': f"{current_lr:.2e}"
                }
                if 'angle' in loss_dict:
                    postfix['ang'] = f"{loss_dict['angle']:.6f}"
                if 'fk_3d' in loss_dict:
                    postfix['fk'] = f"{loss_dict['fk_3d']:.6f}"
                if 'camera_3d' in loss_dict:
                    postfix['cam3d'] = f"{loss_dict['camera_3d']:.6f}"
                if 'refinement' in loss_dict:
                    postfix['ref'] = f"{loss_dict['refinement']:.6f}"
                pbar.set_postfix(postfix)

        # Average losses
        avg_losses = {key: np.mean(values) if values else 0.0 for key, values in epoch_losses.items()}

        # Synchronize basic losses across processes in distributed training FIRST
        # This prevents NCCL timeout when rank 0 computes expensive metrics
        if self.is_distributed:
            for key in list(avg_losses.keys()):
                avg_tensor = torch.tensor(avg_losses[key], device=self.device)
                dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
                avg_losses[key] = (avg_tensor / dist.get_world_size()).item()

        # Compute PCK and ADD if enough data collected (only on main process, AFTER sync)
        if self.is_main_process and self.camera_K is not None and len(all_kp_projs_est) > 0:
            all_kp_projs_est_np = np.array(all_kp_projs_est)
            all_kp_projs_gt_np = np.array(all_kp_projs_gt)
            all_kp_pos_gt_np = np.array(all_kp_pos_gt)

            # 2D PCK / AUC
            kp_metrics = dream_analysis.keypoint_metrics(
                all_kp_projs_est_np.reshape(-1, 2),
                all_kp_projs_gt_np.reshape(-1, 2),
                self.raw_res
            )
            avg_losses['val_pck_auc'] = kp_metrics.get('l2_error_auc', 0.0)
            avg_losses['val_kp_mean_err_px'] = kp_metrics.get('l2_error_mean_px', 0.0)

            # 3D ADD (PnP)
            pnp_adds = []
            n_inframe_list = []

            # Limit PnP calculation to keep validation fast
            subset_size = min(500, len(all_kp_projs_est))
            subset_indices = np.random.choice(len(all_kp_projs_est), subset_size, replace=False)

            for idx in subset_indices:
                p_est = all_kp_projs_est[idx]
                p_gt = all_kp_projs_gt[idx]
                pos_gt = all_kp_pos_gt[idx]

                # Filter only valid keypoints for PnP
                valid_mask = (p_gt[:, 0] >= 0) & (p_gt[:, 1] >= 0)
                if valid_mask.sum() < 4: # PnP needs at least 4 points
                    continue
                
                p_est_valid = p_est[valid_mask]
                p_gt_valid = p_gt[valid_mask]
                pos_gt_valid = pos_gt[valid_mask]

                n_inframe = 0
                for pt in p_gt_valid:
                    if 0 <= pt[0] < self.raw_res[0] and 0 <= pt[1] < self.raw_res[1]:
                        n_inframe += 1
                n_inframe_list.append(n_inframe)

                pnp_retval, translation, quaternion = dream.geometric_vision.solve_pnp(
                    pos_gt_valid, p_est_valid, self.camera_K
                )
                if pnp_retval:
                    add = dream.geometric_vision.add_from_pose(
                        translation, quaternion, pos_gt_valid, self.camera_K
                    )
                    # Convert to mm
                    if add < 10.0: # Filter out crazy outliers (larger than 10m)
                        pnp_adds.append(add * 1000.0)
                    else:
                        pnp_adds.append(-999.0)
                else:
                    pnp_adds.append(-999.0)

            if len(pnp_adds) > 0:
                pnp_metrics = dream_analysis.pnp_metrics(pnp_adds, n_inframe_list, add_auc_threshold=100.0) # 100mm threshold
                avg_losses['val_add_auc'] = pnp_metrics.get('add_auc', 0.0)
                avg_losses['val_add_mean_mm'] = pnp_metrics.get('add_mean', 0.0)
                avg_losses['val_pnp_success_rate'] = pnp_metrics.get('num_pnp_found', 0) / max(1, pnp_metrics.get('num_pnp_possible', 1))
            else:
                avg_losses['val_add_auc'] = 0.0
                avg_losses['val_add_mean_mm'] = 0.0
                avg_losses['val_pnp_success_rate'] = 0.0

        return avg_losses

    def train(self, num_epochs: int):
        """Main training loop"""
        if self.is_main_process:
            print("\n" + "=" * 80)
            print("Starting Training")
            print("=" * 80 + "\n")

        start_time = time.time()

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            # Print current LR at start of epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.is_main_process:
                print(f"\n{'='*80}")
                print(f"Epoch {epoch}/{num_epochs-1} | Learning Rate: {current_lr:.2e}")
                print(f"{'='*80}")

            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_losses = self.validate(epoch)

            # Synchronize all processes after validation
            if self.is_distributed:
                dist.barrier()

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Log to wandb (only on main process)
            if self.is_main_process and wandb.run is not None:
                try:
                    log_dict = {
                        'epoch': epoch,
                        'train/total_loss': train_losses['total'],
                        'train/heatmap_loss': train_losses['heatmap'],
                        'train/kp3d_loss': train_losses['kp3d'],
                        'learning_rate': current_lr,
                        'best_val_loss': self.best_val_loss
                    }
                    if 'angle' in train_losses:
                        log_dict['train/angle_loss'] = train_losses['angle']
                    if 'fk_3d' in train_losses:
                        log_dict['train/fk3d_loss'] = train_losses['fk_3d']
                    if 'camera_3d' in train_losses:
                        log_dict['train/camera3d_loss'] = train_losses['camera_3d']
                    if 'refinement' in train_losses:
                        log_dict['train/refinement_loss'] = train_losses['refinement']
                    if 'pnp_success_rate' in train_losses:
                        log_dict['train/pnp_success_rate'] = train_losses['pnp_success_rate']
                    # Add all validation metrics
                    for k, v in val_losses.items():
                        log_dict[f'val/{k}'] = v

                    wandb.log(log_dict)
                except Exception as e:
                    print(f"Warning: Failed to log to WandB: {e}")

            # Print epoch summary (only on main process)
            if self.is_main_process:
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch} Summary:")

                train_extra = ""
                val_extra = ""
                if 'angle' in train_losses:
                    train_extra += f", ang: {train_losses['angle']:.6f}"
                if 'fk_3d' in train_losses:
                    train_extra += f", fk: {train_losses['fk_3d']:.6f}"
                if 'camera_3d' in train_losses:
                    train_extra += f", cam3d: {train_losses['camera_3d']:.6f}"
                if 'refinement' in train_losses:
                    train_extra += f", ref: {train_losses['refinement']:.6f}"
                if 'angle' in val_losses:
                    val_extra += f", ang: {val_losses['angle']:.6f}"
                if 'fk_3d' in val_losses:
                    val_extra += f", fk: {val_losses['fk_3d']:.6f}"
                if 'camera_3d' in val_losses:
                    val_extra += f", cam3d: {val_losses['camera_3d']:.6f}"
                if 'refinement' in val_losses:
                    val_extra += f", ref: {val_losses['refinement']:.6f}"
                print(f"  Train Loss: {train_losses['total']:.6f} (hm: {train_losses['heatmap']:.6f}, kp3d: {train_losses['kp3d']:.6f}{train_extra})")
                print(f"  Val Loss:   {val_losses['total']:.6f} (hm: {val_losses['heatmap']:.6f}, kp3d: {val_losses['kp3d']:.6f}{val_extra})")
                if 'val_add_auc' in val_losses:
                    print(f"  Val ADD AUC: {val_losses['val_add_auc']:.4f}, ADD Mean: {val_losses['val_add_mean_mm']:.2f}mm, PnP Succ: {val_losses['val_pnp_success_rate']:.2%}")
                if 'val_pck_auc' in val_losses:
                    print(f"  Val PCK AUC: {val_losses['val_pck_auc']:.4f}, KP Mean Err: {val_losses['val_kp_mean_err_px']:.2f}px")
                print(f"  Learning Rate: {current_lr:.2e}")
                print(f"  Time: {epoch_time:.2f}s")

            # Save checkpoint (only on main process)
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                if self.is_main_process:
                    print(f"  New best model! (previous: {self.best_val_loss:.4f})")
                self.best_val_loss = val_losses['total']

            self._save_checkpoint(epoch, is_best=is_best)

            # Update training log (only on main process)
            if self.is_main_process:
                self.train_log['epochs'].append(epoch)
                self.train_log['train_losses'].append(train_losses)
                self.train_log['val_losses'].append(val_losses)
                self.train_log['learning_rates'].append(current_lr)
                self.train_log['timestamps'].append(time.time() - start_time)

                # Save training log
                with open(self.output_dir / 'training_log.pkl', 'wb') as f:
                    pickle.dump(self.train_log, f)

                print()

            # Synchronize all processes after checkpoint saving
            # This ensures rank 0 finishes saving before other ranks start next epoch
            if self.is_distributed:
                dist.barrier()

        if self.is_main_process:
            total_time = time.time() - start_time
            print("=" * 80)
            print(f"Training completed in {total_time / 3600:.2f} hours")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print("=" * 80)

            # Finish wandb run
            if wandb.run is not None:
                try:
                    # Log final summary
                    wandb.summary['best_val_loss'] = self.best_val_loss
                    wandb.summary['total_training_time_hours'] = total_time / 3600
                    wandb.summary['total_epochs'] = num_epochs
                    wandb.finish()
                    print("WandB run finished successfully")
                except Exception as e:
                    print(f"Warning: Error finishing WandB run: {e}")


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = -1
        world_size = -1
        local_rank = -1

    if local_rank != -1:
        torch.cuda.set_device(local_rank)

        # Set NCCL timeout to 30 minutes (default is 10 minutes)
        # This helps prevent timeout during expensive validation metrics computation
        os.environ.setdefault('NCCL_TIMEOUT', '1800')

        # Additional NCCL settings for stability
        os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')  # Better error handling
        os.environ.setdefault('NCCL_DEBUG', 'WARN')  # Less verbose logging

        # Initialize process group with timeout
        timeout_minutes = 30
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=timeout_minutes)
        )

        if rank == 0:
            print(f"Distributed training initialized with NCCL timeout: {timeout_minutes} minutes")

    return local_rank, rank, world_size


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_2d_head_weights(model, checkpoint_path, is_distributed=False, is_main_process=True):
    """
    Load pretrained 2D heatmap head (keypoint_head) weights from a checkpoint.

    Args:
        model: The model to load weights into (can be DDP-wrapped)
        checkpoint_path: Path to the checkpoint file
        is_distributed: Whether using distributed training
        is_main_process: Whether this is the main process
    """
    if is_main_process:
        print(f"\nLoading pretrained 2D heatmap head from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Extract only keypoint_head weights
    keypoint_head_state = {}
    prefix = 'keypoint_head.'

    # Handle both DDP (module.keypoint_head.) and non-DDP (keypoint_head.) prefixes
    for key, value in state_dict.items():
        if 'keypoint_head' in key:
            # Remove 'module.' prefix if present (from DDP checkpoint)
            clean_key = key.replace('module.', '')
            # Extract the keypoint_head part
            if clean_key.startswith(prefix):
                # Keep the keypoint_head prefix
                keypoint_head_state[clean_key] = value

    if len(keypoint_head_state) == 0:
        if is_main_process:
            print("Warning: No keypoint_head weights found in checkpoint!")
        return

    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if is_distributed else model

    # Load the weights into the keypoint_head
    try:
        missing_keys, unexpected_keys = actual_model.load_state_dict(
            keypoint_head_state, strict=False
        )

        if is_main_process:
            print(f"Successfully loaded {len(keypoint_head_state)} keypoint_head parameters")
            if missing_keys:
                # Filter out non-keypoint_head missing keys (expected)
                kp_missing = [k for k in missing_keys if 'keypoint_head' in k]
                if kp_missing:
                    print(f"Warning: Missing keypoint_head keys: {kp_missing}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys: {unexpected_keys}")
            print("2D heatmap head weights loaded successfully!")
    except Exception as e:
        if is_main_process:
            print(f"Error loading 2D heatmap head weights: {e}")
        raise


def worker_init_fn(worker_id):
    """
    DataLoader worker initialization function for reproducibility
    Each worker gets a different but deterministic seed
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(args):
    # Setup distributed training
    local_rank, rank, world_size = setup_distributed()
    is_distributed = local_rank != -1
    is_main_process = (not is_distributed) or (rank == 0)

    # Set random seed for reproducibility
    if args.seed is not None:
        seed = args.seed + rank if is_distributed else args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

        # Enable deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        if is_main_process:
            print(f"Random seed set to {args.seed} (rank offset: {rank if is_distributed else 0})")

    # Device
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_main_process:
        print(f"Using device: {device}")
        if is_distributed:
            print(f"Distributed training: world_size={world_size}, rank={rank}, local_rank={local_rank}")

    # Keypoint names (Panda robot example)
    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]

    # Create dataloaders
    if is_main_process:
        print(f"\nLoading datasets from: {args.data_dir}")

    if args.val_dir:
        # Separate train/val directories
        # Build datasets directly to support DistributedSampler
        from dataset import PoseEstimationDataset
        import torch.utils.data as tud

        train_dataset = PoseEstimationDataset(
            data_dir=args.data_dir,
            keypoint_names=keypoint_names,
            image_size=(args.image_size, args.image_size),
            heatmap_size=(args.heatmap_size, args.heatmap_size),
            augment=True,
            multi_robot=args.multi_robot,
            robot_types=args.robot_types,
            fda_real_dir=args.fda_real_dir,
            fda_beta=args.fda_beta,
            fda_prob=args.fda_prob
        )

        val_dataset_full = PoseEstimationDataset(
            data_dir=args.val_dir,
            keypoint_names=keypoint_names,
            image_size=(args.image_size, args.image_size),
            heatmap_size=(args.heatmap_size, args.heatmap_size),
            augment=False,
            multi_robot=args.multi_robot,
            robot_types=args.robot_types,
            fda_real_dir=None,
            fda_beta=args.fda_beta,
            fda_prob=0.0
        )

        if args.val_split < 1.0:
            val_size = int(len(val_dataset_full) * args.val_split)
            unused_size = len(val_dataset_full) - val_size
            generator = torch.Generator().manual_seed(args.seed if args.seed is not None else 42)
            val_dataset, _ = tud.random_split(val_dataset_full, [val_size, unused_size], generator=generator)
        else:
            val_dataset = val_dataset_full

        if is_distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = tud.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        val_loader = tud.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        # Create separate train and val datasets with different augmentation settings
        # This is more robust than modifying augment flag after random_split
        base_dataset = PoseEstimationDataset(
            data_dir=args.data_dir,
            keypoint_names=keypoint_names,
            image_size=(args.image_size, args.image_size),
            heatmap_size=(args.heatmap_size, args.heatmap_size),
            augment=False,  # Temporarily disable to get indices
            multi_robot=args.multi_robot,
            robot_types=args.robot_types,
            fda_real_dir=args.fda_real_dir,
            fda_beta=args.fda_beta,
            fda_prob=args.fda_prob
        )

        # Split indices with reproducible split
        train_size = int(args.train_split * len(base_dataset))
        val_size = len(base_dataset) - train_size
        generator = torch.Generator().manual_seed(args.seed if args.seed is not None else 42)
        train_indices, val_indices = random_split(
            range(len(base_dataset)), [train_size, val_size], generator=generator
        )

        # Create train dataset with augmentation
        train_dataset_full = PoseEstimationDataset(
            data_dir=args.data_dir,
            keypoint_names=keypoint_names,
            image_size=(args.image_size, args.image_size),
            heatmap_size=(args.heatmap_size, args.heatmap_size),
            augment=False,  # Disable augmentation for training
            multi_robot=args.multi_robot,
            robot_types=args.robot_types,
            fda_real_dir=args.fda_real_dir,
            fda_beta=args.fda_beta,
            fda_prob=args.fda_prob
        )
        train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices.indices)

        # Create val dataset without augmentation
        val_dataset_full = PoseEstimationDataset(
            data_dir=args.data_dir,
            keypoint_names=keypoint_names,
            image_size=(args.image_size, args.image_size),
            heatmap_size=(args.heatmap_size, args.heatmap_size),
            augment=False,  # Disable augmentation for validation
            multi_robot=args.multi_robot,
            robot_types=args.robot_types,
            fda_real_dir=None,  # No FDA for validation
            fda_beta=args.fda_beta,
            fda_prob=0.0  # No FDA for validation
        )
        val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices.indices)

        # Create samplers for distributed training
        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
            # worker_init_fn=worker_init_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
            # worker_init_fn=worker_init_fn
        )

    if is_main_process:
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    if is_main_process:
        print("\nCreating model...")

    # Load camera intrinsics if available
    camera_K = None
    raw_res = (640, 480)
    cam_settings_path = None

    # Try to find _camera_settings.json in data directory
    temp_path = Path(args.data_dir) / "_camera_settings.json"
    if temp_path.exists():
        cam_settings_path = temp_path
    else:
        # If not found in root, search recursively
        for p in Path(args.data_dir).rglob("_camera_settings.json"):
            cam_settings_path = p
            break
    
    if cam_settings_path is not None and cam_settings_path.exists():
        try:
            camera_K = dream.utilities.load_camera_intrinsics(str(cam_settings_path))
            raw_res = dream.utilities.load_image_resolution(str(cam_settings_path))
            if is_main_process:
                print(f"Loaded camera intrinsics from {cam_settings_path}")
                print(f"Raw resolution: {raw_res}")
        except Exception as e:
            if is_main_process:
                print(f"Warning: Failed to load camera settings: {e}")

    if is_main_process:
        print(f"3D prediction mode: joint_angle")

    use_iter_refine = getattr(args, 'use_iterative_refinement', False)
    refine_iters = getattr(args, 'refinement_iterations', 3)

    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=args.unfreeze_blocks,
        use_joint_embedding=args.use_joint_embedding,
        use_iterative_refinement=use_iter_refine,
        refinement_iterations=refine_iters,
    ).to(device)

    # Wrap model with DistributedDataParallel for multi-GPU training
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=use_iter_refine,  # refinement module may have unused params
            broadcast_buffers=True,
            gradient_as_bucket_view=True  # gradient stride 경고 완화
        )

    if is_main_process:
        print(f"Model: {args.model_name}")
        print(f"Joint Embedding: {'Enabled' if args.use_joint_embedding else 'Disabled'}")
        print(f"Fine-tuning: Partial (Last {args.unfreeze_blocks} blocks)")

        model_to_count = model.module if is_distributed else model
        print(f"Number of parameters: {sum(p.numel() for p in model_to_count.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model_to_count.parameters() if p.requires_grad):,}")

    # Load pretrained 2D heatmap head weights if specified
    if args.load_2d_head:
        load_2d_head_weights(
            model=model,
            checkpoint_path=args.load_2d_head,
            is_distributed=is_distributed,
            is_main_process=is_main_process
        )

    # Loss function
    angle_w = args.angle_weight
    fk_3d_w = args.fk_3d_weight
    camera_3d_w = args.kp3d_weight  # Camera-frame 3D loss (PnP-based transform)
    refinement_w = getattr(args, 'refinement_weight', 0.0) if use_iter_refine else 0.0
    criterion = UnifiedPoseLoss(
        heatmap_weight=args.heatmap_weight,
        kp3d_weight=0.0,  # Not used in joint_angle mode (use angle/FK/camera_3d losses instead)
        heatmap_size=args.heatmap_size,
        angle_weight=angle_w,
        fk_3d_weight=fk_3d_w,
        camera_3d_weight=camera_3d_w,  # Camera-frame 3D loss (PnP-based transform)
        loss_type=args.loss_type,  # Loss function type: 'mse', 'l1', 'smoothl1'
        refinement_weight=refinement_w,  # Iterative refinement loss weight
    ).to(device)

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Learning rate scheduler with warmup (skip warmup if resuming)
    scheduler = None
    warmup_epochs = 5  # Number of warmup epochs
    use_warmup = args.scheduler == 'cosine' and not args.resume  # Only warmup for cosine and new training

    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
    elif args.scheduler == 'cosine':
        if use_warmup:
            # Warmup + Cosine schedule (only for new training, not resume)
            from torch.optim.lr_scheduler import SequentialLR, LinearLR
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - warmup_epochs,
                eta_min=args.min_lr
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            if is_main_process:
                print(f"Using Warmup({warmup_epochs}ep) + CosineAnnealingLR: initial_lr={args.learning_rate}, min_lr={args.min_lr}, T_max={args.epochs}")
        else:
            # Just cosine (for resume)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.min_lr
            )
            if is_main_process:
                print(f"Using CosineAnnealingLR: initial_lr={args.learning_rate}, min_lr={args.min_lr}, T_max={args.epochs}")
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

    # Training config
    config = {
        'model_name': args.model_name,
        'image_size': args.image_size,
        'heatmap_size': args.heatmap_size,
        'unfreeze_blocks': args.unfreeze_blocks,
        'use_joint_embedding': args.use_joint_embedding,
        'joint_angle_3d': True,
        'angle_weight': angle_w,
        'fk_3d_weight': fk_3d_w,
        'use_iterative_refinement': use_iter_refine,
        'refinement_iterations': refine_iters,
        'refinement_weight': refinement_w,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'min_lr': args.min_lr,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'heatmap_weight': args.heatmap_weight,
        'kp3d_weight': args.kp3d_weight,
        'keypoint_names': keypoint_names,
        'wandb_project': args.wandb_project,
        'wandb_run_name': args.wandb_run_name
    }

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        config=config,
        camera_K=camera_K,
        raw_res=raw_res,
        resume_from=args.resume,
        resume_lr=args.resume_lr,
        local_rank=local_rank
    )

    # Start training
    try:
        trainer.train(args.epochs)
    finally:
        # Cleanup distributed training
        cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DINOv3 Pose Estimation Model')

    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='Path to validation data directory (optional)')
    parser.add_argument('--train-split', type=float, default=0.9,
                        help='Train/val split ratio if val-dir not provided (DREAM uses 0.8, we use 0.9)')
    parser.add_argument('--val-split', type=float, default=1.0,
                        help='Fraction of validation data to use (default=1.0 for all data)')
    parser.add_argument('--multi-robot', action='store_true',
                        help='Load data from multiple robot subdirectories for unified model')
    parser.add_argument('--robot-types', type=str, nargs='+', default=None,
                        help='Filter specific robot types (e.g., panda kuka baxter)')

    # FDA (Fourier Domain Adaptation) for sim-to-real
    parser.add_argument('--fda-real-dir', type=str, default=None,
                        help='Real image directory for FDA style transfer (no labels needed)')
    parser.add_argument('--fda-beta', type=float, default=0.01,
                        help='FDA low-frequency replacement ratio (0.01=subtle, 0.05=strong)')
    parser.add_argument('--fda-prob', type=float, default=0.5,
                        help='Probability of applying FDA per sample')

    # Model
    parser.add_argument('--model-name', type=str,
                        default='facebook/dinov3-vitb16-pretrain-lvd1689m',
                        help='DINOv3 model name from HuggingFace')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--heatmap-size', type=int, default=512,
                        help='Output heatmap size')
    parser.add_argument('--unfreeze-blocks', type=int, default=2,
                        help='Number of backbone blocks to unfreeze')
    parser.add_argument('--use-joint-embedding', action='store_true', default=False,
                        help='Enable joint identity embeddings in 3D head for kinematic constraint learning')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-8,
                        help='Minimum learning rate for CosineAnnealingLR scheduler')

    # Loss weights
    parser.add_argument('--loss-type', type=str, default='smoothl1',
                        choices=['mse', 'l1', 'smoothl1'],
                        help='Loss function type (smoothl1 recommended for ADD AUC)')
    parser.add_argument('--heatmap-weight', type=float, default=1.0,
                        help='Weight for heatmap loss')
    parser.add_argument('--kp3d-weight', type=float, default=10.0,
                        help='Weight for 3D keypoint loss')
    parser.add_argument('--angle-weight', type=float, default=1.0,
                        help='Weight for joint angle MSE loss (joint_angle mode)')
    parser.add_argument('--fk-3d-weight', type=float, default=10.0,
                        help='Weight for FK 3D keypoint MSE loss (joint_angle mode)')

    # Iterative Refinement
    parser.add_argument('--use-iterative-refinement', action='store_true', default=False,
                        help='Enable iterative refinement for joint angle mode')
    parser.add_argument('--refinement-iterations', type=int, default=3,
                        help='Number of refinement iterations')
    parser.add_argument('--refinement-weight', type=float, default=50.0,
                        help='Weight for iterative refinement loss')

    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--resume-lr', type=float, default=None,
                        help='Learning rate to use when resuming (if not specified, calculated from scheduler)')

    # Load pretrained 2D heatmap head
    parser.add_argument('--load-2d-head', type=str, default=None,
                        help='Path to checkpoint to load pretrained 2D heatmap head weights from')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Wandb
    parser.add_argument('--wandb-project', type=str, default='dinov3-pose-estimation',
                        help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name (optional)')

    args = parser.parse_args()

    main(args)
