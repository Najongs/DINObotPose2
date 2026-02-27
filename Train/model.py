import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel
from torchvision.ops import roi_align
import numpy as np
import cv2

FEATURE_DIM = 512
NUM_JOINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.

# 3D prediction mode
MODE_JOINT_ANGLE = 'joint_angle'  # Predict joint angles → FK → robot-frame 3D keypoints


def soft_argmax_2d(heatmaps, temperature=10.0):
    """
    Differentiable soft-argmax to extract (u, v) from heatmaps.
    Args:
        heatmaps: (B, N, H, W)
        temperature: scaling factor for softmax sharpness (float or Tensor)
    Returns:
        (B, N, 2) [x, y] coordinates in heatmap pixel space
    """
    B, N, H, W = heatmaps.shape
    device = heatmaps.device

    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    y_coords = torch.arange(H, device=device, dtype=torch.float32)

    heatmaps_flat = heatmaps.reshape(B, N, -1)

    # Support both fixed temperature and learnable parameter
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.clamp(min=0.1, max=50.0)  # Prevent extreme values

    weights = F.softmax(heatmaps_flat * temperature, dim=-1)
    weights = weights.reshape(B, N, H, W)

    x = (weights.sum(dim=2) * x_coords).sum(dim=-1)  # (B, N)
    y = (weights.sum(dim=3) * y_coords).sum(dim=-1)  # (B, N)

    return torch.stack([x, y], dim=-1)  # (B, N, 2)

class DINOv3Backbone(nn.Module):
    def __init__(self, model_name, unfreeze_blocks=2):
        super().__init__()
        self.model_name = model_name
        if "siglip" in model_name:
            self.model = SiglipVisionModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name)

        # Freeze backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze last N blocks for fine-tuning
        if unfreeze_blocks > 0:
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
                # ViT / DINOv2 / SigLIP style
                layers = self.model.encoder.layers
                for i in range(len(layers) - unfreeze_blocks, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
            elif hasattr(self.model, "blocks"):
                # Alternative ViT style
                layers = self.model.blocks
                for i in range(len(layers) - unfreeze_blocks, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True

    def forward(self, image_tensor_batch):
        # Removed torch.no_grad() to allow gradient flow for downstream heads
        if "siglip" in self.model_name:
            outputs = self.model(
                pixel_values=image_tensor_batch,
                interpolate_pos_encoding=True)
            tokens = outputs.last_hidden_state
            patch_tokens = tokens[:, 1:, :]
        else: # DINOv3 계열
            outputs = self.model(pixel_values=image_tensor_batch)
            tokens = outputs.last_hidden_state
            num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
            patch_tokens = tokens[:, 1 + num_reg :, :]
        return patch_tokens

class AdaptiveNorm2d(nn.Module):
    """Adaptive normalization mixing GroupNorm and LayerNorm for sim-to-real robustness"""
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.ln = nn.LayerNorm(num_channels)
        # Learnable mixing coefficient
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B, C, H, W)
        gn_out = self.gn(x)
        # LayerNorm over channels
        ln_out = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Mix with learnable alpha (clamped to [0, 1])
        alpha = torch.sigmoid(self.alpha)
        return alpha * gn_out + (1 - alpha) * ln_out


class TokenFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # bias=True를 추가하여 PyTorch CUDA kernel의 gradient stride 문제 완화
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(out_channels, num_groups=32),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(out_channels, num_groups=32)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        projected = self.projection(x)
        refined = self.refine_blocks(projected)
        residual = self.residual_conv(x)
        return torch.nn.functional.gelu(refined + residual)

class ViTKeypointHead(nn.Module):
    def __init__(self, input_dim=768, num_joints=NUM_JOINTS, heatmap_size=(512, 512)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.token_fuser = TokenFuser(input_dim, 256)

        # ViT-only decoder with adaptive normalization
        self.decoder_block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(128, num_groups=32),
            nn.GELU()
        )
        self.decoder_block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(64, num_groups=16),
            nn.GELU()
        )
        self.decoder_block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(32, num_groups=8),
            nn.GELU()
        )

        self.heatmap_predictor = nn.Conv2d(32, num_joints, kernel_size=3, padding=1)

        # Learned upsampling with transposed convolution (better than bilinear)
        # From 32x32 (after 3 decoder blocks) to 512x512 requires 4x upsampling
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(num_joints, num_joints, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(num_joints, num_joints, kernel_size=4, stride=2, padding=1, bias=False)
        )

    def forward(self, dino_features):
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        if h * w != n:
            n_new = h * w
            dino_features = dino_features[:, :n_new, :]
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)

        x = self.token_fuser(x)
        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)

        heatmaps = self.heatmap_predictor(x)

        # Use learned upsampling instead of bilinear interpolation
        heatmaps = self.final_upsample(heatmaps)

        # Final resize to exact target size if needed
        if heatmaps.shape[2:] != self.heatmap_size:
            heatmaps = F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)

        return heatmaps

# =============================================================================
# Forward Kinematics (Panda robot, URDF-based DH parameters)
# =============================================================================

def _rotation_matrix_z(theta):
    """Rotation matrix around Z axis. theta: (B,) or (B,1)"""
    c = torch.cos(theta)
    s = torch.sin(theta)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    # (B, 3, 3)
    return torch.stack([
        torch.stack([c, -s, zero], dim=-1),
        torch.stack([s,  c, zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ], dim=-2)


def _rotation_matrix_x(angle):
    """Fixed rotation matrix around X axis. angle: scalar (float)"""
    c = math.cos(angle)
    s = math.sin(angle)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _rotation_matrix_z_fixed(angle):
    """Fixed rotation matrix around Z axis. angle: scalar (float)"""
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _make_transform(xyz, rpy):
    """
    Create a 4x4 homogeneous transform from xyz translation and rpy rotation.
    Returns a list-of-lists (will be converted to tensor later).
    rpy = (roll, pitch, yaw) = rotations around (x, y, z) in that order.
    """
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    # For Panda URDF, only roll (rx) rotations appear (pitch=yaw=0 mostly)
    rx, ry, rz = rpy
    # Build rotation: Rz @ Ry @ Rx
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    R = [
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx            ],
    ]
    T = [
        [R[0][0], R[0][1], R[0][2], xyz[0]],
        [R[1][0], R[1][1], R[1][2], xyz[1]],
        [R[2][0], R[2][1], R[2][2], xyz[2]],
        [0,       0,       0,       1      ],
    ]
    return T


# Panda URDF joint parameters: (origin_xyz, origin_rpy, axis)
# Joint axis is always z for Panda revolute joints
_PANDA_JOINTS = [
    # J1: panda_joint1
    {'xyz': (0, 0, 0.333), 'rpy': (0, 0, 0)},
    # J2: panda_joint2
    {'xyz': (0, 0, 0), 'rpy': (-math.pi/2, 0, 0)},
    # J3: panda_joint3
    {'xyz': (0, -0.316, 0), 'rpy': (math.pi/2, 0, 0)},
    # J4: panda_joint4
    {'xyz': (0.0825, 0, 0), 'rpy': (math.pi/2, 0, 0)},
    # J5: panda_joint5
    {'xyz': (-0.0825, 0.384, 0), 'rpy': (-math.pi/2, 0, 0)},
    # J6: panda_joint6
    {'xyz': (0, 0, 0), 'rpy': (math.pi/2, 0, 0)},
    # J7: panda_joint7
    {'xyz': (0.088, 0, 0), 'rpy': (math.pi/2, 0, 0)},
]

# Fixed transforms after joint7
_PANDA_FIXED_J8 = {'xyz': (0, 0, 0.107), 'rpy': (0, 0, 0)}
_PANDA_FIXED_HAND = {'xyz': (0, 0, 0), 'rpy': (0, 0, -math.pi/4)}

# Panda joint limits (radians) from URDF
_PANDA_JOINT_LIMITS = [
    (-2.8973, 2.8973),   # J1
    (-1.7628, 1.7628),   # J2
    (-2.8973, 2.8973),   # J3
    (-3.0718, -0.0698),  # J4
    (-2.8973, 2.8973),   # J5
    (-0.0175, 3.7525),   # J6
    (-2.8973, 2.8973),   # J7
]

# Keypoint-to-joint mapping:
# link0 = base (before any joint)
# link2 = after J1, J2
# link3 = after J1, J2, J3
# link4 = after J1, J2, J3, J4
# link6 = after J1, ..., J6
# link7 = after J1, ..., J7
# hand  = after J1, ..., J7, J8_fixed, hand_fixed
_KEYPOINT_JOINT_INDICES = [0, 2, 3, 4, 6, 7, 8]  # 8 = hand (after all joints + fixed)


def panda_forward_kinematics(joint_angles):
    """
    Compute forward kinematics for Panda robot.

    Args:
        joint_angles: (B, 7) joint angles in radians

    Returns:
        keypoint_positions: (B, 7, 3) keypoint positions in robot base frame (meters)
    """
    B = joint_angles.shape[0]
    device = joint_angles.device
    dtype = joint_angles.dtype

    # Precompute fixed origin transforms as tensors
    fixed_transforms = []
    for j_info in _PANDA_JOINTS:
        T = _make_transform(j_info['xyz'], j_info['rpy'])
        fixed_transforms.append(torch.tensor(T, device=device, dtype=dtype))

    T_j8 = torch.tensor(_make_transform(_PANDA_FIXED_J8['xyz'], _PANDA_FIXED_J8['rpy']),
                         device=device, dtype=dtype)
    T_hand = torch.tensor(_make_transform(_PANDA_FIXED_HAND['xyz'], _PANDA_FIXED_HAND['rpy']),
                          device=device, dtype=dtype)

    # Identity for base
    eye4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    # Accumulate transforms: T_cumul[i] = T_0 @ ... @ T_i
    # where T_i = fixed_origin_i @ Rz(theta_i)
    cumul = eye4.clone()  # (B, 4, 4) - base frame

    # Store cumulative transforms after each joint
    # Index 0..6 = after J1..J7, index 7 = after J8_fixed, index 8 = after hand_fixed
    all_transforms = [cumul.clone()]  # [0] = base (before J1)

    for i in range(7):
        # Fixed origin transform (broadcast to batch)
        T_fixed = fixed_transforms[i].unsqueeze(0).expand(B, -1, -1)  # (B, 4, 4)

        # Joint rotation around z
        theta = joint_angles[:, i]  # (B,)
        R_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        R_joint[:, :3, :3] = _rotation_matrix_z(theta)

        # Cumulative: cumul = cumul @ T_fixed @ R_joint
        cumul = cumul @ T_fixed @ R_joint

        all_transforms.append(cumul.clone())  # [i+1] = after J(i+1)

    # After J7, apply fixed J8 and hand transforms
    T_j8_batch = T_j8.unsqueeze(0).expand(B, -1, -1)
    T_hand_batch = T_hand.unsqueeze(0).expand(B, -1, -1)

    cumul_j8 = cumul @ T_j8_batch
    all_transforms.append(cumul_j8.clone())  # [8] = after J8 fixed

    cumul_hand = cumul_j8 @ T_hand_batch
    all_transforms.append(cumul_hand.clone())  # [9] = after hand fixed

    # Extract keypoint positions
    # link0=base=[0], link2=after J2=[2], link3=after J3=[3],
    # link4=after J4=[4], link6=after J6=[6], link7=after J7=[7], hand=[9]
    kp_indices = [0, 2, 3, 4, 6, 7, 9]
    keypoints = []
    for idx in kp_indices:
        pos = all_transforms[idx][:, :3, 3]  # (B, 3)
        keypoints.append(pos)

    return torch.stack(keypoints, dim=1)  # (B, 7, 3)


class JointAngleHead(nn.Module):
    """
    Predicts 7 joint angles for Panda robot from visual features + heatmaps.
    Uses FK to produce 3D keypoints in robot base frame.
    """

    def __init__(self, input_dim=FEATURE_DIM, num_joints=NUM_JOINTS, num_angles=7):
        super().__init__()
        self.num_joints = num_joints
        self.num_angles = num_angles
        self.hidden_dim = 256

        # Joint limits as buffers
        limits = torch.tensor(_PANDA_JOINT_LIMITS, dtype=torch.float32)  # (7, 2)
        self.register_buffer('joint_lower', limits[:, 0])  # (7,)
        self.register_buffer('joint_upper', limits[:, 1])  # (7,)
        self.register_buffer('joint_mid', (limits[:, 0] + limits[:, 1]) / 2)  # (7,)
        self.register_buffer('joint_range', (limits[:, 1] - limits[:, 0]) / 2)  # (7,)

        # Learnable temperature for soft-argmax
        self.temperature = nn.Parameter(torch.tensor(10.0))

        # 1. Per-joint feature refinement
        # Input: visual feature (input_dim) + normalized 2D coords (2)
        self.joint_feature_net = nn.Sequential(
            nn.Linear(input_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # 2. Self-attention for kinematic constraint learning (reduced from 4 to 2 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.joint_relation_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Global state → 7 angles at once
        # After self-attention, mean-pool all joint tokens into a single robot state vector,
        # then predict all 7 joint angles together.
        # This is kinematically correct: joint angles are a global property of the robot
        # configuration, not independently decodable from individual keypoint tokens.
        self.angle_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.num_angles)  # Predict all 7 angles at once
        )

    def forward(self, dino_features, predicted_heatmaps):
        """
        Args:
            dino_features: (B, N, D) backbone patch tokens
            predicted_heatmaps: (B, NUM_JOINTS, H, W) 2D belief maps
        Returns:
            joint_angles: (B, 7) predicted joint angles (radians, within limits)
            keypoints_3d_robot: (B, 7, 3) FK-computed 3D keypoints in robot base frame
        """
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        feat_map = dino_features.permute(0, 2, 1).reshape(b, d, h, w)

        # Extract 2D keypoint locations from heatmaps with learnable temperature
        uv_heatmap = soft_argmax_2d(predicted_heatmaps, self.temperature)  # (B, NJ, 2)

        # Normalize 2D coords to [-1, 1] (in-place 회피)
        hm_h, hm_w = predicted_heatmaps.shape[2], predicted_heatmaps.shape[3]
        uv_norm_x = (uv_heatmap[:, :, 0] / hm_w) * 2.0 - 1.0
        uv_norm_y = (uv_heatmap[:, :, 1] / hm_h) * 2.0 - 1.0
        uv_normalized = torch.stack([uv_norm_x, uv_norm_y], dim=-1)  # (B, NJ, 2)

        # Scale keypoint coords from heatmap space to feature map space (ROIAlign용)
        scale_x = w / hm_w
        scale_y = h / hm_h
        uv_feat = torch.stack([uv_heatmap[:, :, 0] * scale_x,
                               uv_heatmap[:, :, 1] * scale_y], dim=-1)  # (B, NJ, 2)

        # Create RoI boxes — 벡터화 (roi_size=2: keypoint 겹침 방지)
        roi_size = 4
        batch_ids = torch.arange(b, device=feat_map.device, dtype=torch.float32).unsqueeze(1).expand(b, self.num_joints)
        cx = uv_feat[:, :, 0]
        cy = uv_feat[:, :, 1]
        rois = torch.stack([batch_ids, cx - roi_size/2, cy - roi_size/2,
                            cx + roi_size/2, cy + roi_size/2], dim=-1).reshape(-1, 5).detach()

        # ROIAlign + pooling
        roi_features = roi_align(feat_map, rois, output_size=(3, 3), spatial_scale=1.0)
        joint_features = roi_features.mean(dim=[2, 3])
        joint_features = joint_features.view(b, self.num_joints, d)

        # Concatenate 2D coordinates
        joint_features = torch.cat([joint_features, uv_normalized], dim=-1)  # (B, NJ, D+2)

        # Feature refinement + self-attention
        refined = self.joint_feature_net(joint_features)  # (B, NJ, 256)
        related = self.joint_relation_net(refined)  # (B, NJ, 256)

        # Global pooling: aggregate all joint tokens into single robot state
        global_state = related.mean(dim=1)  # (B, 256)

        # Predict all 7 joint angles from global robot state
        raw_angles = self.angle_predictor(global_state)  # (B, 7)

        # Apply joint limits via tanh scaling: mid + tanh(raw) * range
        joint_angles = self.joint_mid.unsqueeze(0) + torch.tanh(raw_angles) * self.joint_range.unsqueeze(0)

        # Forward kinematics
        keypoints_3d_robot = panda_forward_kinematics(joint_angles)  # (B, 7, 3)

        return joint_angles, keypoints_3d_robot


def compute_extrinsics_from_pnp(kp3d_robot, kp2d_image, camera_K):
    """
    Batch PnP solver: compute robot-to-camera extrinsics [R, t].

    Args:
        kp3d_robot: (B, 7, 3) 3D keypoints in robot frame (detached numpy or tensor)
        kp2d_image: (B, 7, 2) 2D keypoints in original image coords (detached numpy or tensor)
        camera_K: (B, 3, 3) camera intrinsic matrices

    Returns:
        R_batch: (B, 3, 3) rotation matrices (detached tensors on same device as input)
        t_batch: (B, 3) translation vectors (detached tensors)
        valid_mask: (B,) bool tensor indicating which samples had successful PnP
    """
    if isinstance(kp3d_robot, torch.Tensor):
        device = kp3d_robot.device
        kp3d_np = kp3d_robot.detach().cpu().numpy()
        kp2d_np = kp2d_image.detach().cpu().numpy()
        K_np = camera_K.detach().cpu().numpy()
    else:
        device = torch.device('cpu')
        kp3d_np = kp3d_robot
        kp2d_np = kp2d_image
        K_np = camera_K

    B = kp3d_np.shape[0]
    R_batch = torch.zeros(B, 3, 3, device=device)
    t_batch = torch.zeros(B, 3, device=device)
    valid_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for b in range(B):
        try:
            obj_pts = kp3d_np[b].astype(np.float64)
            img_pts = kp2d_np[b].astype(np.float64)
            K = K_np[b].astype(np.float64)

            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_EPNP
            )
            if success:
                R, _ = cv2.Rodrigues(rvec)
                R_batch[b] = torch.from_numpy(R).float().to(device)
                t_batch[b] = torch.from_numpy(tvec.flatten()).float().to(device)
                valid_mask[b] = True
        except Exception:
            pass

    return R_batch, t_batch, valid_mask


class IterativeRefinementModule(nn.Module):
    """
    Iterative refinement for joint angle prediction via render-and-compare.

    Pipeline per iteration:
        FK(θᵢ) → kp3d_robot → project(R,t,K) → pred_2d
        Δuv = target_2d - pred_2d
        CorrectionNet(Δuv, θᵢ, |Δuv|) → Δθ
        θᵢ₊₁ = clamp(θᵢ + step_scale * Δθ)

    R, t are obtained via PnP (detached, no gradient).
    """

    def __init__(self, num_iterations=3, num_angles=7):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_angles = num_angles

        # Joint limits as buffers (same as JointAngleHead)
        limits = torch.tensor(_PANDA_JOINT_LIMITS, dtype=torch.float32)
        self.register_buffer('joint_lower', limits[:, 0])
        self.register_buffer('joint_upper', limits[:, 1])

        # Shared correction network (weight-tied across iterations)
        # Input: Δuv (7*2=14) + θ_current (7) + |Δuv| (7) = 28
        self.correction_net = nn.Sequential(
            nn.Linear(28, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_angles),
        )

        # Learnable step scale per iteration (sigmoid-gated, init ~0.1)
        # logit for sigmoid(x)=0.1 is ln(0.1/0.9) ≈ -2.2
        self.step_scale_logits = nn.Parameter(
            torch.full((num_iterations,), -2.2)
        )

    def _project_to_2d(self, kp3d_robot, R, t, K):
        """
        Differentiable projection: robot frame → camera frame → 2D image coords.

        Args:
            kp3d_robot: (B, 7, 3) robot-frame keypoints (differentiable)
            R: (B, 3, 3) rotation matrices (detached)
            t: (B, 3) translation vectors (detached)
            K: (B, 3, 3) camera intrinsics (detached)

        Returns:
            uv: (B, 7, 2) projected 2D coordinates in original image space
        """
        # Transform to camera frame: kp_cam = R @ kp_robot^T + t
        # (B, 3, 3) @ (B, 3, 7) → (B, 3, 7)
        kp_cam = torch.bmm(R, kp3d_robot.transpose(1, 2)) + t.unsqueeze(-1)
        kp_cam = kp_cam.transpose(1, 2)  # (B, 7, 3)

        # Project: uv = K @ kp_cam / z
        z = kp_cam[:, :, 2:3].clamp(min=1e-6)  # (B, 7, 1)
        kp_norm = kp_cam / z  # (B, 7, 3) - [x/z, y/z, 1]

        # Apply intrinsics: (B, 3, 3) @ (B, 3, 7) → (B, 3, 7)
        uv_h = torch.bmm(K, kp_norm.transpose(1, 2)).transpose(1, 2)  # (B, 7, 3)
        uv = uv_h[:, :, :2]  # (B, 7, 2)

        return uv

    def forward(self, initial_angles, target_2d, camera_K, original_size,
                R_ext, t_ext, valid_mask):
        """
        Run iterative refinement loop.

        Args:
            initial_angles: (B, 7) initial joint angle predictions from JointAngleHead
            target_2d: (B, 7, 2) target 2D keypoints in original image coords
                       (from heatmap hard-argmax during inference, or GT during training)
            camera_K: (B, 3, 3) camera intrinsic matrices
            original_size: (W, H) tuple of original image size
            R_ext: (B, 3, 3) rotation matrices from PnP (detached)
            t_ext: (B, 3) translation vectors from PnP (detached)
            valid_mask: (B,) bool mask for valid PnP samples

        Returns:
            dict with:
                'refined_angles': (B, 7) final refined angles
                'refined_kp3d_robot': (B, 7, 3) FK keypoints from refined angles
                'all_angles': list of (B, 7) angles at each iteration
                'all_kp3d_robot': list of (B, 7, 3) FK keypoints at each iteration
        """
        B = initial_angles.shape[0]
        theta = initial_angles  # (B, 7) - gradients flow through this

        all_angles = [theta]
        all_kp3d_robot = [panda_forward_kinematics(theta)]

        for i in range(self.num_iterations):
            # FK to get current 3D keypoints in robot frame
            kp3d_robot = panda_forward_kinematics(theta)  # (B, 7, 3)

            # Project to 2D (differentiable through FK and projection, R/t detached)
            pred_2d = self._project_to_2d(kp3d_robot, R_ext, t_ext, camera_K)  # (B, 7, 2)

            # Compute reprojection error
            delta_uv = target_2d - pred_2d  # (B, 7, 2)

            # For invalid PnP samples, zero out delta to avoid garbage gradients
            delta_uv = delta_uv * valid_mask.float().unsqueeze(-1).unsqueeze(-1)

            # Prepare correction net input
            delta_uv_flat = delta_uv.reshape(B, -1)  # (B, 14)
            delta_uv_mag = delta_uv.norm(dim=-1)  # (B, 7)
            correction_input = torch.cat([delta_uv_flat, theta, delta_uv_mag], dim=-1)  # (B, 28)

            # Predict angle correction
            delta_theta = self.correction_net(correction_input)  # (B, 7)

            # Apply step scale (sigmoid-gated)
            step_scale = torch.sigmoid(self.step_scale_logits[i])

            # Update angles with clamping to joint limits
            theta = torch.clamp(
                theta + step_scale * delta_theta,
                min=self.joint_lower.unsqueeze(0),
                max=self.joint_upper.unsqueeze(0)
            )

            all_angles.append(theta)
            all_kp3d_robot.append(panda_forward_kinematics(theta))

        return {
            'refined_angles': theta,
            'refined_kp3d_robot': all_kp3d_robot[-1],
            'all_angles': all_angles,
            'all_kp3d_robot': all_kp3d_robot,
        }


class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, unfreeze_blocks=2,
                 use_joint_embedding=False,
                 use_iterative_refinement=False, refinement_iterations=3):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.heatmap_size = heatmap_size  # (H, W) tuple
        self.use_iterative_refinement = use_iterative_refinement
        self.backbone = DINOv3Backbone(dino_model_name, unfreeze_blocks=unfreeze_blocks)

        if "siglip" in self.dino_model_name:
            config = self.backbone.model.config
            feature_dim = config.hidden_size
        else: # DINOv3 계열
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in self.dino_model_name else config.hidden_size

        # 1. 2D Heatmap Predictor
        self.keypoint_head = ViTKeypointHead(
            input_dim=feature_dim,
            heatmap_size=heatmap_size
        )

        # 2. Joint Angle Head → FK → robot-frame 3D keypoints
        self.joint_angle_head = JointAngleHead(
            input_dim=feature_dim,
            num_joints=NUM_JOINTS,
            num_angles=7
        )

        # 3. Iterative Refinement Module (optional)
        if use_iterative_refinement:
            self.refinement_module = IterativeRefinementModule(
                num_iterations=refinement_iterations,
                num_angles=7
            )
        else:
            self.refinement_module = None

    def forward(self, image_tensor_batch, camera_K=None, original_size=None,
                gt_angles=None, gt_2d_image=None, use_refinement=None):
        """
        Args:
            image_tensor_batch: (B, 3, H, W) input images
            camera_K: (B, 3, 3) camera intrinsics (for iterative refinement PnP)
            original_size: (W, H) original image resolution (for iterative refinement)
            gt_angles: (B, 7) GT joint angles (for training refinement with stable PnP)
            gt_2d_image: (B, 7, 2) GT 2D keypoints in original image coords (for training refinement)
            use_refinement: bool override (None = use self.use_iterative_refinement)
        """
        # 1. Extract visual representations
        dino_features = self.backbone(image_tensor_batch)

        # 2. Estimate 2D pixel locations (Heatmaps)
        predicted_heatmaps = self.keypoint_head(dino_features)

        # 3. Predict joint angles → FK → robot-frame 3D keypoints
        joint_angles, kpts_3d_robot = self.joint_angle_head(
            dino_features, predicted_heatmaps
        )
        result = {
            'heatmaps_2d': predicted_heatmaps,
            'keypoints_3d': kpts_3d_robot,  # robot base frame
            'joint_angles': joint_angles,
            'keypoints_3d_robot': kpts_3d_robot,
        }

        # 4. Iterative Refinement (if enabled)
        do_refine = use_refinement if use_refinement is not None else self.use_iterative_refinement
        if do_refine and self.refinement_module is not None and camera_K is not None:
            B = image_tensor_batch.shape[0]
            hm_h, hm_w = predicted_heatmaps.shape[2], predicted_heatmaps.shape[3]

            # Get 2D target for refinement
            if gt_2d_image is not None:
                # Training: use GT 2D keypoints (more stable)
                target_2d = gt_2d_image  # already in original image coords
            else:
                # Inference: use heatmap hard-argmax → scale to original image coords
                heatmaps_flat = predicted_heatmaps.detach().reshape(B, NUM_JOINTS, -1)
                max_indices = torch.argmax(heatmaps_flat, dim=-1)
                x = (max_indices % hm_w).float()
                y = (max_indices // hm_w).float()
                target_2d_hm = torch.stack([x, y], dim=-1)  # (B, 7, 2) in heatmap coords

                # Scale to original image coords
                if original_size is not None:
                    if isinstance(original_size, (list, tuple)):
                        orig_w, orig_h = original_size[0], original_size[1]
                    else:
                        orig_w = original_size[:, 0:1].unsqueeze(-1)  # (B, 1, 1)
                        orig_h = original_size[:, 1:2].unsqueeze(-1)
                    scale_x = orig_w / hm_w
                    scale_y = orig_h / hm_h
                    target_2d = torch.stack([
                        target_2d_hm[:, :, 0] * scale_x if isinstance(scale_x, float) else target_2d_hm[:, :, 0] * scale_x.squeeze(-1),
                        target_2d_hm[:, :, 1] * scale_y if isinstance(scale_y, float) else target_2d_hm[:, :, 1] * scale_y.squeeze(-1),
                    ], dim=-1)
                else:
                    target_2d = target_2d_hm

            # Compute extrinsics via PnP
            if gt_angles is not None:
                # Training: use GT angles for stable PnP
                with torch.no_grad():
                    gt_kp3d_robot = panda_forward_kinematics(gt_angles)
                R_ext, t_ext, valid_mask = compute_extrinsics_from_pnp(
                    gt_kp3d_robot, target_2d, camera_K
                )
            else:
                # Inference: use predicted angles
                R_ext, t_ext, valid_mask = compute_extrinsics_from_pnp(
                    kpts_3d_robot.detach(), target_2d.detach(), camera_K
                )

            # Detach R, t to prevent gradient flow through PnP
            R_ext = R_ext.detach()
            t_ext = t_ext.detach()

            # Run refinement
            refine_out = self.refinement_module(
                initial_angles=joint_angles,
                target_2d=target_2d,
                camera_K=camera_K,
                original_size=original_size,
                R_ext=R_ext,
                t_ext=t_ext,
                valid_mask=valid_mask,
            )

            result['refined_angles'] = refine_out['refined_angles']
            result['refined_kp3d_robot'] = refine_out['refined_kp3d_robot']
            result['all_refined_angles'] = refine_out['all_angles']
            result['all_refined_kp3d_robot'] = refine_out['all_kp3d_robot']
            result['refinement_valid_mask'] = valid_mask

            # Update main outputs to use refined results
            result['joint_angles'] = refine_out['refined_angles']
            result['keypoints_3d'] = refine_out['refined_kp3d_robot']
            result['keypoints_3d_robot'] = refine_out['refined_kp3d_robot']

        return result