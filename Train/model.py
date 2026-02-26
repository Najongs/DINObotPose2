import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel

FEATURE_DIM = 512
NUM_JOINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.

# 3D prediction modes
MODE_DIRECT = 'direct'        # Predict (x, y, z) directly from features
MODE_DEPTH_ONLY = 'depth_only'  # Predict z only, recover x,y from 2D heatmap + camera K


def soft_argmax_2d(heatmaps, temperature=10.0):
    """
    Differentiable soft-argmax to extract (u, v) from heatmaps.
    Args:
        heatmaps: (B, N, H, W)
        temperature: scaling factor for softmax sharpness
    Returns:
        (B, N, 2) [x, y] coordinates in heatmap pixel space
    """
    B, N, H, W = heatmaps.shape
    device = heatmaps.device

    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    y_coords = torch.arange(H, device=device, dtype=torch.float32)

    heatmaps_flat = heatmaps.reshape(B, N, -1)
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

class TokenFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # bias=True를 추가하여 PyTorch CUDA kernel의 gradient stride 문제 완화
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels)
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

        # ViT-only decoder (GroupNorm for sim-to-real robustness)
        self.decoder_block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.GELU()
        )
        self.decoder_block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU()
        )
        self.decoder_block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )

        self.heatmap_predictor = nn.Conv2d(32, num_joints, kernel_size=3, padding=1)

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

        return F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)

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

        # 3. Self-attention for kinematic constraint learning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.joint_relation_net = nn.TransformerEncoder(encoder_layer, num_layers=4)

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

        # Extract 2D keypoint locations from heatmaps
        uv_heatmap = soft_argmax_2d(predicted_heatmaps)  # (B, NJ, 2) in heatmap pixels

        # Normalize 2D coords to [-1, 1] for feature input
        hm_h, hm_w = predicted_heatmaps.shape[2], predicted_heatmaps.shape[3]
        uv_normalized = uv_heatmap.clone()
        uv_normalized[:, :, 0] = (uv_normalized[:, :, 0] / hm_w) * 2.0 - 1.0
        uv_normalized[:, :, 1] = (uv_normalized[:, :, 1] / hm_h) * 2.0 - 1.0  # (B, NJ, 2)

        # Spatial Softmax Pooling for visual features
        weights = F.interpolate(predicted_heatmaps, size=(h, w), mode='bilinear', align_corners=False)
        weights = torch.clamp(weights, min=0)

        weights_flat = weights.reshape(b, self.num_joints, -1)
        weights_norm = F.softmax(weights_flat / 0.1, dim=-1)
        weights_norm = weights_norm.reshape(b, self.num_joints, h, w)

        joint_features = torch.einsum('bdhw,bjhw->bjd', feat_map, weights_norm)  # (B, NJ, D)

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

        # 2. 3D Keypoint Predictor
        self.keypoint_3d_head = Keypoint3DHead(
            input_dim=feature_dim,
            num_joints=NUM_JOINTS,
            use_joint_embedding=use_joint_embedding,
            mode=mode_3d
        )

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