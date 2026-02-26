import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel
from torchvision.ops import roi_align

FEATURE_DIM = 512
NUM_JOINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.

# 3D prediction modes
MODE_DIRECT = 'direct'        # Predict (x, y, z) directly from features
MODE_DEPTH_ONLY = 'depth_only'  # Predict z only, recover x,y from 2D heatmap + camera K
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

class Keypoint3DHead(nn.Module):
    """
    Predicts 3D coordinates for each joint.

    Supports two modes:
    - 'direct': Predict (x, y, z) directly from features (original behavior)
    - 'depth_only': Predict z only, recover x,y from 2D heatmap + camera K

    Improvements over naive regression:
    - Explicit 2D coordinate injection: soft-argmax (u,v) concatenated to joint features
    - Mean pose + residual: predicts offset from dataset mean, reducing output range
    """
    # Mean 3D pose computed from DREAM synthetic training set (meters, camera frame)
    # Order: link0, link2, link3, link4, link6, link7, hand
    MEAN_POSE = [
        [ 0.0008,  0.2679,  1.1670],
        [ 0.0008,  0.0224,  0.9769],
        [-0.0070, -0.1568,  0.8290],
        [-0.0082, -0.1421,  0.8336],
        [-0.0058, -0.0479,  0.8884],
        [-0.0059, -0.0340,  0.8937],
        [-0.0022,  0.0054,  0.9236],
    ]

    def __init__(self, input_dim=FEATURE_DIM, num_joints=NUM_JOINTS,
                 use_joint_embedding=False, mode=MODE_DIRECT):
        super().__init__()
        self.num_joints = num_joints
        self.use_joint_embedding = use_joint_embedding
        self.hidden_dim = 256
        self.mode = mode

        # Register mean pose as buffer (moves to correct device automatically)
        self.register_buffer('mean_pose',
                             torch.tensor(self.MEAN_POSE, dtype=torch.float32))  # (NJ, 3)

        # Learnable temperature for soft-argmax (initialized to 10.0)
        self.temperature = nn.Parameter(torch.tensor(10.0))

        # 1. Per-joint feature refinement
        # Input: visual feature (input_dim) + normalized 2D coords (2)
        self.joint_feature_net = nn.Sequential(
            nn.Linear(input_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # 2. Joint Identity Embedding (optional)
        if use_joint_embedding:
            self.joint_embedding = nn.Embedding(num_joints, self.hidden_dim)

        # 3. Self-attention for kinematic constraint learning (reduced from 4 to 2 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.joint_relation_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Output head depends on mode
        if mode == MODE_DEPTH_ONLY:
            # Predict only depth residual (Δz) per joint
            self.depth_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, 128),
                nn.GELU(),
                nn.LayerNorm(128),
                nn.Linear(128, 1)
            )
        else:
            # Predict residual (Δx, Δy, Δz) per joint
            self.coord_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, 128),
                nn.GELU(),
                nn.LayerNorm(128),
                nn.Linear(128, 3)
            )

    def forward(self, dino_features, predicted_heatmaps, camera_K=None,
                heatmap_size=None, original_size=None):
        """
        Args:
            dino_features: (B, N, D) backbone patch tokens
            predicted_heatmaps: (B, NUM_JOINTS, H, W) 2D belief maps
            camera_K: (B, 3, 3) camera intrinsic matrix (required for depth_only)
            heatmap_size: (H, W) tuple of heatmap resolution
            original_size: (W, H) tuple of original image resolution
        Returns:
            pred_kpts_3d: (B, NUM_JOINTS, 3) predicted 3D coordinates
        """
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        feat_map = dino_features.permute(0, 2, 1).reshape(b, d, h, w)

        # Extract 2D keypoint locations from heatmaps with learnable temperature
        uv_heatmap = soft_argmax_2d(predicted_heatmaps, self.temperature)  # (B, NJ, 2) in heatmap pixels

        # Normalize 2D coords to [-1, 1] for feature input (in-place 회피)
        hm_h, hm_w = predicted_heatmaps.shape[2], predicted_heatmaps.shape[3]
        uv_norm_x = (uv_heatmap[:, :, 0] / hm_w) * 2.0 - 1.0
        uv_norm_y = (uv_heatmap[:, :, 1] / hm_h) * 2.0 - 1.0
        uv_normalized = torch.stack([uv_norm_x, uv_norm_y], dim=-1)  # (B, NJ, 2)

        # Scale keypoint coords from heatmap space to feature map space (ROIAlign용)
        scale_x = w / hm_w
        scale_y = h / hm_h
        uv_feat = torch.stack([uv_heatmap[:, :, 0] * scale_x,
                               uv_heatmap[:, :, 1] * scale_y], dim=-1)  # (B, NJ, 2)

        # Create RoI boxes: [batch_idx, x1, y1, x2, y2] — 벡터화로 Python loop 제거
        # ViT patch size=14 → 512px input → feature map ~36x36
        # roi_size=2: keypoint당 좁은 영역만 추출, 인접 keypoint 겹침 방지
        roi_size = 2
        batch_ids = torch.arange(b, device=feat_map.device, dtype=torch.float32)  # (B,)
        batch_ids = batch_ids.unsqueeze(1).expand(b, self.num_joints)  # (B, NJ)
        cx = uv_feat[:, :, 0]  # (B, NJ)
        cy = uv_feat[:, :, 1]  # (B, NJ)
        x1 = cx - roi_size / 2
        y1 = cy - roi_size / 2
        x2 = cx + roi_size / 2
        y2 = cy + roi_size / 2
        # Stack: (B*NJ, 5) — detach box coords (ROIAlign은 box 위치에 대해 미분불가)
        rois = torch.stack([batch_ids, x1, y1, x2, y2], dim=-1).reshape(-1, 5).detach()

        # ROIAlign: extract 3x3 features for each keypoint
        roi_features = roi_align(feat_map, rois, output_size=(3, 3), spatial_scale=1.0)  # (B*NJ, D, 3, 3)

        # Global average pooling over spatial dimensions
        joint_features = roi_features.mean(dim=[2, 3])  # (B*NJ, D)
        joint_features = joint_features.view(b, self.num_joints, d)  # (B, NJ, D)

        # Concatenate explicit 2D coordinates to visual features
        joint_features = torch.cat([joint_features, uv_normalized], dim=-1)  # (B, NJ, D+2)

        # 1. Feature refinement
        refined_features = self.joint_feature_net(joint_features)  # (B, NJ, 256)

        # 2. Optionally add Joint Identity Embedding
        if self.use_joint_embedding:
            joint_ids = torch.arange(self.num_joints, device=dino_features.device).expand(b, self.num_joints)
            joint_embeds = self.joint_embedding(joint_ids)  # (B, NJ, 256)
            refined_features = refined_features + joint_embeds

        # 3. Learn inter-joint kinematic relations
        related_features = self.joint_relation_net(refined_features)  # (B, NJ, 256)

        # 4. Predict 3D coordinates (mean pose + residual)
        mean = self.mean_pose.unsqueeze(0).expand(b, -1, -1)  # (B, NJ, 3)

        if self.mode == MODE_DEPTH_ONLY:
            # Predict depth residual only
            pred_dz = self.depth_predictor(related_features).squeeze(-1)  # (B, NJ)
            pred_z = mean[:, :, 2] + pred_dz  # (B, NJ)

            # Scale 2D coords to original image pixels
            assert heatmap_size is not None and original_size is not None, \
                "heatmap_size and original_size required for depth_only mode"

            scale_x = original_size[0] / heatmap_size[1]
            scale_y = original_size[1] / heatmap_size[0]

            u_orig = uv_heatmap[:, :, 0] * scale_x
            v_orig = uv_heatmap[:, :, 1] * scale_y

            # Unproject using camera intrinsics
            assert camera_K is not None, "camera_K required for depth_only mode"
            fx = camera_K[:, 0, 0]
            fy = camera_K[:, 1, 1]
            cx = camera_K[:, 0, 2]
            cy = camera_K[:, 1, 2]

            pred_x = (u_orig - cx.unsqueeze(1)) * pred_z / fx.unsqueeze(1)
            pred_y = (v_orig - cy.unsqueeze(1)) * pred_z / fy.unsqueeze(1)

            pred_kpts_3d = torch.stack([pred_x, pred_y, pred_z], dim=-1)
        else:
            # Predict residual from mean pose
            residual = self.coord_predictor(related_features)  # (B, NJ, 3)
            pred_kpts_3d = mean + residual

        return pred_kpts_3d

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
        roi_size = 2
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


class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, unfreeze_blocks=2,
                 use_joint_embedding=False, mode_3d=MODE_DIRECT):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.heatmap_size = heatmap_size  # (H, W) tuple
        self.mode_3d = mode_3d
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

        # 2. 3D Keypoint Predictor (or Joint Angle Head)
        if mode_3d == MODE_JOINT_ANGLE:
            self.joint_angle_head = JointAngleHead(
                input_dim=feature_dim,
                num_joints=NUM_JOINTS,
                num_angles=7
            )
            self.keypoint_3d_head = None
        else:
            self.keypoint_3d_head = Keypoint3DHead(
                input_dim=feature_dim,
                num_joints=NUM_JOINTS,
                use_joint_embedding=use_joint_embedding,
                mode=mode_3d
            )
            self.joint_angle_head = None

    def forward(self, image_tensor_batch, camera_K=None, original_size=None):
        """
        Args:
            image_tensor_batch: (B, 3, H, W) input images
            camera_K: (B, 3, 3) camera intrinsics (required for depth_only mode)
            original_size: (W, H) original image resolution (required for depth_only mode)
        """
        # 1. Extract visual representations
        dino_features = self.backbone(image_tensor_batch)

        # 2. Estimate 2D pixel locations (Heatmaps)
        predicted_heatmaps = self.keypoint_head(dino_features)

        # 3. Lift 2D representations to 3D spatial coordinates
        if self.mode_3d == MODE_JOINT_ANGLE:
            joint_angles, kpts_3d_robot = self.joint_angle_head(
                dino_features, predicted_heatmaps
            )
            return {
                'heatmaps_2d': predicted_heatmaps,
                'keypoints_3d': kpts_3d_robot,  # robot base frame
                'joint_angles': joint_angles,
                'keypoints_3d_robot': kpts_3d_robot,
            }
        else:
            predicted_kpts_3d = self.keypoint_3d_head(
                dino_features, predicted_heatmaps,
                camera_K=camera_K,
                heatmap_size=self.heatmap_size,
                original_size=original_size
            )
            return {
                'heatmaps_2d': predicted_heatmaps,
                'keypoints_3d': predicted_kpts_3d
            }