# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DINObotPose2 is a robot pose estimation system using DINOv2/v3 backbone with 2D heatmap prediction and 3D keypoint estimation. Trained on DREAM synthetic data, targeting real-world inference on Franka Panda robots. The system predicts 7 keypoints: panda_link0, link2, link3, link4, link6, link7, hand.

## Environment

- Python environment: `conda run -n dino` (or `conda activate dino`)
- GPU training uses PyTorch DDP via `torchrun`
- Key dependencies: torch, torchvision, transformers, albumentations, wandb, opencv (for PnP), pyrr

## Common Commands

```bash
# Training (edit Train/run_train.sh config section first)
cd Train && bash run_train.sh

# Multi-GPU training (default in run_train.sh)
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes=1 --nproc_per_node=3 train.py [args...]

# Evaluate on dataset (Direct ADD + PnP ADD metrics)
cd Eval && bash run_inference_dataset.sh

# Single real image inference with GT comparison
cd Eval && bash run_real_inference.sh

# Single image inference (no GT needed)
cd Eval && bash run_single_inference.sh

# Qualitative visualization (mesh overlay + keypoint skeleton)
cd Vis && bash run_visualize.sh
```

## Training Configuration

All training hyperparameters are set in the config section of `Train/run_train.sh` (not via CLI). Key settings:
- `MODEL_NAME`: DINOv3 backbone variant (default: `facebook/dinov3-vitb16-pretrain-lvd1689m`)
- `JOINT_ANGLE_3D` / `DEPTH_ONLY_3D`: Toggle 3D mode (set one True, other False)
- Loss weights: `HEATMAP_WEIGHT`, `KP3D_WEIGHT`, `ANGLE_WEIGHT`, `FK_3D_WEIGHT`
- Two-stage training: set `LOAD_2D_HEAD` to a 2D-only checkpoint path, then train 3D head
- Checkpoints save to `Train/outputs/dinov3_base_YYYYMMDD_HHMMSS/` with `config.yaml`
- WandB logging enabled by default (`--wandb-project`)

## Architecture

The model (`DINOv3PoseEstimator` in `Train/model.py`) has three stages:

1. **Backbone** (`DINOv3Backbone`): Frozen DINOv2/v3 ViT with last N blocks unfrozen. Outputs patch tokens `(B, N, D)`.
2. **2D Head** (`ViTKeypointHead`): TokenFuser → decoder blocks with AdaptiveNorm2d → `(B, 7, 512, 512)` heatmaps.
3. **3D Head** (one of three modes set via `mode_3d`):
   - `direct`: `Keypoint3DHead` predicts (x,y,z) residuals from mean pose using ROIAlign + transformer.
   - `depth_only`: Predicts z only, recovers x,y from 2D heatmap + camera intrinsics K.
   - `joint_angle`: `JointAngleHead` predicts 7 joint angles → forward kinematics → 3D keypoints in robot frame.

Loss (`UnifiedPoseLoss` in `Train/train.py`): weighted sum of 2D heatmap loss + 3D keypoint loss (+ angle loss + FK loss for joint_angle mode).

## Data Format

Uses DREAM-converted format. Each sample is a `NNNNNN.json` + `NNNNNN.rgb.png` pair:
- JSON contains: `meta.K` (3x3 camera intrinsics), `objects[0].keypoints` (7 keypoints with `projected_location` [u,v] and `location` [x,y,z])
- **Unit warning**: DREAM_syn `location` is in cm (divide by 100 for meters). DREAM_real `location` is already in meters.

## Data Paths

- SYN train: `Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr` (~104K samples)
- SYN test: `Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr` (~6K)
- REAL: `Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure` (~6.4K)
- FDA real images: `Dataset/DREAM_real`

## Key Design Decisions

- `NUM_JOINTS = 7` is hardcoded and intentional — do not change.
- 3D head uses mean pose + residual prediction to reduce output range.
- Soft-argmax with learnable temperature bridges 2D heatmaps → 3D head.
- FDA (Fourier Domain Adaptation) applies real-image low-frequency spectrum to synthetic images for sim-to-real transfer.
- AdaptiveNorm2d mixes GroupNorm + LayerNorm with learnable alpha for domain robustness.
- ROIAlign extracts per-keypoint features from backbone feature map (roi_size=2 to avoid overlap).
- The `--load-2d-head` flag enables two-stage training: first train 2D head, then freeze and train 3D head.
- Forward kinematics in `panda_forward_kinematics()` (model.py) is fully differentiable in PyTorch using embedded DH parameters.
- `_KEYPOINT_JOINT_INDICES = [0, 2, 3, 4, 6, 7, 8]` maps the 7 keypoints to Panda link chain indices (8 = hand after fixed transforms).
- Multi-GPU: DDP with `module.*` prefix handling for checkpoint loading. Only rank 0 saves checkpoints and logs.

## Evaluation Metrics

- **Keypoint L2 AUC**: 2D pixel error (threshold configurable, default 20px)
- **Direct ADD AUC**: 3D ADD metric without PnP — uses model's predicted 3D keypoints directly
- **PnP ADD AUC**: Baseline using OpenCV EPnP to solve pose from 2D keypoints + camera K (threshold default 0.1m)

## Project Structure

- `Train/` — Model definition, dataset, training loop, launch script
- `Eval/` — Inference scripts (single image, dataset evaluation, real image with GT)
- `Vis/` — Qualitative visualization
- `Dataset/` — Data conversion scripts and converted datasets
- `DREAM/` — Original DREAM codebase (used for analysis utilities and metrics)
- `robopose/` — Reference RoboPose codebase
