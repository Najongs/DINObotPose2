import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel

FEATURE_DIM = 512
NUM_JOINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.
NUM_ROBOT_CLASSES = 4  # Number of robot types
ROBOT_TYPE_NAMES = ['franka_panda', 'meca500', 'fr5', 'franka_research3']

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

class LightCNNStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # 1/2
            nn.GroupNorm(4, 16),
            nn.GELU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 1/4
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 1/8
            nn.GroupNorm(16, 64),
            nn.GELU()
        )
        
    def forward(self, x):
        feat_2 = self.conv_block1(x)  # 1/2
        feat_4 = self.conv_block2(feat_2)  # 1/4
        feat_8 = self.conv_block3(feat_4) # 1/8
        return feat_2, feat_4, feat_8

class FusedUpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.GELU()
        )

    def forward(self, x, skip_feature):
        x = self.upsample(x)
        if x.shape[-2:] != skip_feature.shape[-2:]:
            skip_feature = F.interpolate(
                skip_feature, 
                size=x.shape[-2:], # target H, W
                mode='bilinear', 
                align_corners=False
            )

        fused = torch.cat([x, skip_feature], dim=1)
        return self.refine_conv(fused)

class UNetViTKeypointHead(nn.Module):
    def __init__(self, input_dim=768, num_joints=NUM_JOINTS, heatmap_size=(512, 512), use_cnn_stem=True):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.use_cnn_stem = use_cnn_stem
        self.token_fuser = TokenFuser(input_dim, 256)

        if use_cnn_stem:
            # UNet-style decoder with CNN skip connections
            self.decoder_block1 = FusedUpsampleBlock(in_channels=256, skip_channels=64, out_channels=128)
            self.decoder_block2 = FusedUpsampleBlock(in_channels=128, skip_channels=32, out_channels=64)
            self.decoder_block3 = FusedUpsampleBlock(in_channels=64, skip_channels=16, out_channels=32)
        else:
            # ViT-only decoder without skip connections (GroupNorm for sim-to-real robustness)
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

    def forward(self, dino_features, cnn_features=None):
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        if h * w != n:
            n_new = h * w
            dino_features = dino_features[:, :n_new, :]
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)

        x = self.token_fuser(x)

        if self.use_cnn_stem and cnn_features is not None:
            # UNet-style with skip connections
            cnn_feat_2, cnn_feat_4, cnn_feat_8 = cnn_features
            x = self.decoder_block1(x, cnn_feat_8)
            x = self.decoder_block2(x, cnn_feat_4)
            x = self.decoder_block3(x, cnn_feat_2)
        else:
            # ViT-only without skip connections
            x = self.decoder_block1(x)
            x = self.decoder_block2(x)
            x = self.decoder_block3(x)

        heatmaps = self.heatmap_predictor(x)

        return F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)

class RobotClassifierHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=NUM_ROBOT_CLASSES):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, dino_features):
        # dino_features: (B, N, D) -> (B, D, N)
        x = dino_features.permute(0, 2, 1)
        x = self.pooler(x).squeeze(-1) # (B, D)
        return self.classifier(x)

class EnhancedKeypoint3DHead(nn.Module):
    """
    Predicts 3D coordinates (x, y, z) by combining DINOv3 features with
    explicit Joint Embeddings to learn kinematic constraints.
    """
    def __init__(self, input_dim=FEATURE_DIM, num_joints=NUM_JOINTS):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = 256

        # 1. Per-joint feature refinement
        self.joint_feature_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # 2. Joint Identity Embedding
        # Tells the transformer "this feature is 'Base'", "this is 'Wrist'" etc.
        self.joint_embedding = nn.Embedding(num_joints, self.hidden_dim)

        # 3. Deeper Self-attention for kinematic constraint learning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=1024,
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

        # 2. Add Joint Identity Embedding
        joint_ids = torch.arange(self.num_joints, device=dino_features.device).expand(b, self.num_joints)
        joint_embeds = self.joint_embedding(joint_ids)  # (B, NJ, 256)
        fused_features = refined_features + joint_embeds

        # 3. Learn inter-joint kinematic relations
        related_features = self.joint_relation_net(fused_features)  # (B, NJ, 256)

        # 4. 3D coordinate regression
        pred_kpts_3d = self.coord_predictor(related_features)  # (B, NJ, 3)

        return pred_kpts_3d

class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, unfreeze_blocks=2, use_cnn_stem=True, use_robot_classifier=False):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.use_cnn_stem = use_cnn_stem
        self.use_robot_classifier = use_robot_classifier
        self.backbone = DINOv3Backbone(dino_model_name, unfreeze_blocks=unfreeze_blocks)

        if "siglip" in self.dino_model_name:
            config = self.backbone.model.config
            feature_dim = config.hidden_size
        else: # DINOv3 계열
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in self.dino_model_name else config.hidden_size

        # CNN stem (optional)
        if use_cnn_stem:
            self.cnn_stem = LightCNNStem()
        else:
            self.cnn_stem = None

        # 0. Robot Type Classifier (optional)
        if use_robot_classifier:
            self.robot_classifier = RobotClassifierHead(input_dim=feature_dim, num_classes=NUM_ROBOT_CLASSES)
        else:
            self.robot_classifier = None

        # 1. 2D Heatmap Predictor
        self.keypoint_head = UNetViTKeypointHead(
            input_dim=feature_dim,
            heatmap_size=heatmap_size,
            use_cnn_stem=use_cnn_stem
        )

        # 2. 3D Keypoint Predictor with Joint Embeddings
        self.keypoint_3d_head = EnhancedKeypoint3DHead(input_dim=feature_dim, num_joints=NUM_JOINTS)

    def forward(self, image_tensor_batch):
        # 1. Extract visual representations
        dino_features = self.backbone(image_tensor_batch)

        # 2. Extract CNN features if using CNN stem
        if self.use_cnn_stem and self.cnn_stem is not None:
            cnn_stem_features = self.cnn_stem(image_tensor_batch)
        else:
            cnn_stem_features = None

        # 3. Robot type classification (optional)
        if self.use_robot_classifier and self.robot_classifier is not None:
            robot_logits = self.robot_classifier(dino_features)
        else:
            robot_logits = None

        # 4. Estimate 2D pixel locations (Heatmaps)
        predicted_heatmaps = self.keypoint_head(dino_features, cnn_stem_features)

        # 5. Lift 2D representations to 3D spatial coordinates
        predicted_kpts_3d = self.keypoint_3d_head(dino_features, predicted_heatmaps)

        # Return dictionary format for training compatibility
        result = {
            'heatmaps_2d': predicted_heatmaps,
            'keypoints_3d': predicted_kpts_3d
        }

        if robot_logits is not None:
            result['robot_type'] = robot_logits

        return result