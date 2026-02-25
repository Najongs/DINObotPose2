import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel

FEATURE_DIM = 512
NUM_JOINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.

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
    Predicts 3D coordinates (x, y, z) for each joint by pooling backbone features
    at predicted 2D keypoint locations and reasoning about joint relationships.
    Optionally adds Joint Identity Embeddings to learn kinematic constraints.
    """
    def __init__(self, input_dim=FEATURE_DIM, num_joints=NUM_JOINTS, use_joint_embedding=False):
        super().__init__()
        self.num_joints = num_joints
        self.use_joint_embedding = use_joint_embedding
        self.hidden_dim = 256

        # 1. Per-joint feature refinement
        self.joint_feature_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # 2. Joint Identity Embedding (optional)
        if use_joint_embedding:
            self.joint_embedding = nn.Embedding(num_joints, self.hidden_dim)

        # 3. Self-attention for kinematic constraint learning``
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.joint_relation_net = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 4. Predict 3D coordinates (x, y, z)
        self.coord_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 3)
        )

    def forward(self, dino_features, predicted_heatmaps):
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        feat_map = dino_features.permute(0, 2, 1).reshape(b, d, h, w)

        # Spatial Softmax Pooling
        weights = F.interpolate(predicted_heatmaps, size=(h, w), mode='bilinear', align_corners=False)
        weights = torch.clamp(weights, min=0)

        weights_flat = weights.reshape(b, self.num_joints, -1)
        weights_norm = F.softmax(weights_flat / 0.1, dim=-1)
        weights_norm = weights_norm.reshape(b, self.num_joints, h, w)

        joint_features = torch.einsum('bdhw,bjhw->bjd', feat_map, weights_norm)  # (B, NJ, D)

        # 1. Feature refinement
        refined_features = self.joint_feature_net(joint_features)  # (B, NJ, 256)

        # 2. Optionally add Joint Identity Embedding
        if self.use_joint_embedding:
            joint_ids = torch.arange(self.num_joints, device=dino_features.device).expand(b, self.num_joints)
            joint_embeds = self.joint_embedding(joint_ids)  # (B, NJ, 256)
            refined_features = refined_features + joint_embeds

        # 3. Learn inter-joint kinematic relations
        related_features = self.joint_relation_net(refined_features)  # (B, NJ, 256)

        # 4. 3D coordinate regression
        pred_kpts_3d = self.coord_predictor(related_features)  # (B, NJ, 3)

        return pred_kpts_3d

class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, unfreeze_blocks=2, use_joint_embedding=False):
        super().__init__()
        self.dino_model_name = dino_model_name
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
            use_joint_embedding=use_joint_embedding
        )

    def forward(self, image_tensor_batch):
        # 1. Extract visual representations
        dino_features = self.backbone(image_tensor_batch)

        # 2. Estimate 2D pixel locations (Heatmaps)
        predicted_heatmaps = self.keypoint_head(dino_features)

        # 3. Lift 2D representations to 3D spatial coordinates
        predicted_kpts_3d = self.keypoint_3d_head(dino_features, predicted_heatmaps)

        return {
            'heatmaps_2d': predicted_heatmaps,
            'keypoints_3d': predicted_kpts_3d
        }