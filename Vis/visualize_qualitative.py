"""
Qualitative Comparison Visualization for DINObotPose2
- Row 1: Original RGB image
- Row 2: Robot mesh overlay (GT green, Pred blue)
- Row 3: Predicted keypoint skeleton
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import torch
import torchvision.transforms as TVTransforms
import yaml
from PIL import Image as PILImage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Train')))
from model import DINOv3PoseEstimator, MODE_DIRECT, MODE_DEPTH_ONLY


# ─── Constants ───────────────────────────────────────────────────────────────

KEYPOINT_NAMES = [
    'panda_link0', 'panda_link2', 'panda_link3',
    'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
]

SKELETON_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
]

KEYPOINT_COLORS_BGR = [
    (0, 0, 255),     # link0: red
    (0, 128, 255),   # link2: orange
    (0, 255, 255),   # link3: yellow
    (0, 255, 0),     # link4: green
    (255, 255, 0),   # link6: cyan
    (255, 0, 0),     # link7: blue
    (255, 0, 255),   # hand: magenta
]

LINK_COLORS_BGR = [
    (0, 64, 255), (0, 192, 255), (0, 255, 192),
    (0, 255, 0), (255, 255, 0), (255, 0, 128),
]

URDF_PATH = os.path.join(os.path.dirname(__file__),
                          'panda-description/patched_urdf/panda.urdf')

DREAM_JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]

# All link names that exist in both sim_state.links and objects.keypoints
ALL_LINK_NAMES = [
    'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3',
    'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7',
]


# ─── Camera Transform ───────────────────────────────────────────────────────

def compute_T_cw(links_world_pos, kp_cam_pos_m):
    """Compute world→camera rigid transform via Procrustes alignment.

    Args:
        links_world_pos: dict {link_name: [x,y,z] in world meters}
        kp_cam_pos_m: dict {link_name: [x,y,z] in camera meters}

    Returns:
        R_cw: (3,3) rotation matrix
        t_cw: (3,) translation vector
        Such that p_cam = R_cw @ p_world + t_cw
    """
    names = sorted(set(links_world_pos.keys()) & set(kp_cam_pos_m.keys()))
    if len(names) < 3:
        return None, None

    pts_w = np.array([links_world_pos[n] for n in names])
    pts_c = np.array([kp_cam_pos_m[n] for n in names])

    cw = pts_w.mean(0)
    cc = pts_c.mean(0)
    H = (pts_w - cw).T @ (pts_c - cc)
    U, S, Vt = np.linalg.svd(H)
    d_det = np.linalg.det(Vt.T @ U.T)
    R_cw = Vt.T @ np.diag([1, 1, d_det]) @ U.T
    t_cw = cc - R_cw @ cw

    return R_cw, t_cw


# ─── Panda Renderer ─────────────────────────────────────────────────────────

class PandaRenderer:
    """Render Panda robot using pybullet's built-in renderer."""

    def __init__(self, urdf_path=URDF_PATH):
        self.physics_client = p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(
            urdf_path, basePosition=[0, 0, 0],
            useFixedBase=True, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL
        )
        self.num_joints = p.getNumJoints(self.robot_id)

    def set_pose(self, base_position, base_orientation_xyzw, joint_angles):
        p.resetBasePositionAndOrientation(
            self.robot_id, base_position, base_orientation_xyzw
        )
        for i in range(min(7, len(joint_angles))):
            p.resetJointState(self.robot_id, i, joint_angles[i])

    def render_with_T_cw(self, camera_K, width, height, R_cw, t_cw):
        """Render using explicit world→camera transform (OpenCV convention).

        R_cw, t_cw define: p_cam = R_cw @ p_world + t_cw
        where camera frame is OpenCV: X-right, Y-down, Z-forward.
        """
        # Convert OpenCV camera→OpenGL camera: flip Y and Z
        flip = np.diag([1.0, -1.0, -1.0])
        R_gl = flip @ R_cw
        t_gl = flip @ t_cw

        # Build 4x4 view matrix (column-major for pybullet)
        view_4x4 = np.eye(4)
        view_4x4[:3, :3] = R_gl
        view_4x4[:3, 3] = t_gl
        view_list = view_4x4.T.flatten().tolist()

        # Projection matrix from intrinsics
        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]
        near, far = 0.01, 10.0

        proj = np.zeros((4, 4))
        proj[0, 0] = 2 * fx / width
        proj[1, 1] = 2 * fy / height
        proj[0, 2] = (width - 2 * cx) / width
        proj[1, 2] = -(height - 2 * cy) / height
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1.0
        proj_list = proj.T.flatten().tolist()

        _, _, rgba, depth, seg = p.getCameraImage(
            width, height,
            viewMatrix=view_list,
            projectionMatrix=proj_list,
            renderer=p.ER_TINY_RENDERER,
        )

        rgba = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
        seg = np.array(seg, dtype=np.int32).reshape(height, width)
        # seg == -1 means background, >= 0 means robot body
        robot_mask = seg >= 0
        return rgba, robot_mask

    def disconnect(self):
        p.disconnect(self.physics_client)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_dream_annotation(json_path):
    """Load DREAM annotation with all needed data."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Image path
    image_path = None
    camera_K = None
    if 'meta' in data:
        raw_path = data['meta'].get('image_path', '')
        if raw_path.startswith('../dataset/'):
            raw_path = raw_path.replace('../dataset/', '../../../', 1)
        if not os.path.isabs(raw_path):
            image_path = os.path.normpath(os.path.join(os.path.dirname(json_path), raw_path))
        else:
            image_path = raw_path
        if 'K' in data['meta']:
            camera_K = np.array(data['meta']['K'], dtype=np.float64)

    # GT 2D/3D keypoints (in camera frame, cm for synthetic)
    gt_2d = {}
    gt_3d_cam = {}  # camera frame
    is_synthetic = 'syn' in json_path.lower()
    if 'objects' in data:
        for obj in data['objects']:
            for kp in obj.get('keypoints', []):
                gt_2d[kp['name']] = np.array(kp['projected_location'])
                loc = np.array(kp['location'])
                if is_synthetic:
                    loc = loc / 100.0  # cm → m
                gt_3d_cam[kp['name']] = loc

    # Joint angles
    joint_angles = [0.0] * 7
    if 'sim_state' in data and 'joints' in data['sim_state']:
        for jt in data['sim_state']['joints']:
            short_name = jt['name'].split('/')[-1]
            if short_name in DREAM_JOINT_NAMES:
                idx = DREAM_JOINT_NAMES.index(short_name)
                joint_angles[idx] = jt['position']

    # Entity poses
    robot_base_pose = None
    if 'sim_state' in data and 'entities' in data['sim_state']:
        for ent in data['sim_state']['entities']:
            if 'panda' in ent['name'].lower():
                robot_base_pose = ent['pose']

    # Link world positions (for computing T_cw)
    links_world_pos = {}
    if 'sim_state' in data and 'links' in data['sim_state']:
        for link in data['sim_state']['links']:
            name = link['name'].split('/')[-1]
            links_world_pos[name] = np.array(link['pose']['position'])

    # Compute T_cw from matched link/keypoint positions
    R_cw, t_cw = None, None
    kp_cam_m = {n: gt_3d_cam[n] for n in ALL_LINK_NAMES if n in gt_3d_cam}
    lw = {n: links_world_pos[n] for n in ALL_LINK_NAMES if n in links_world_pos}
    if kp_cam_m and lw:
        R_cw, t_cw = compute_T_cw(lw, kp_cam_m)

    return {
        'image_path': image_path,
        'camera_K': camera_K,
        'gt_2d': gt_2d,
        'gt_3d_cam': gt_3d_cam,
        'joint_angles': joint_angles,
        'robot_base_pose': robot_base_pose,
        'R_cw': R_cw,
        't_cw': t_cw,
        'is_synthetic': is_synthetic,
    }


# ─── Model Inference ─────────────────────────────────────────────────────────

def load_model(model_path, device):
    checkpoint_dir = Path(model_path).parent
    config_path = checkpoint_dir / 'config.yaml'

    model_name = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    use_joint_embedding = False
    depth_only_3d = False
    image_size = 512
    heatmap_size = 512

    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        model_name = cfg.get('model_name', model_name)
        use_joint_embedding = cfg.get('use_joint_embedding', False)
        depth_only_3d = cfg.get('depth_only_3d', False)
        image_size = int(cfg.get('image_size', image_size))
        heatmap_size = int(cfg.get('heatmap_size', heatmap_size))

    mode_3d = MODE_DEPTH_ONLY if depth_only_3d else MODE_DIRECT
    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=(heatmap_size, heatmap_size),
        unfreeze_blocks=0,
        use_joint_embedding=use_joint_embedding,
        mode_3d=mode_3d
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if 'backbone.model.embeddings.mask_token' in state_dict:
        del state_dict['backbone.model.embeddings.mask_token']
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, image_size, heatmap_size


def run_inference(model, image_pil, camera_K, device, image_size, heatmap_size):
    orig_dim = image_pil.size
    transform = TVTransforms.Compose([
        TVTransforms.Resize((image_size, image_size)),
        TVTransforms.ToTensor(),
        TVTransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    camera_K_tensor = None
    if camera_K is not None:
        camera_K_tensor = torch.tensor(camera_K, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor, camera_K=camera_K_tensor, original_size=orig_dim)

    heatmaps = outputs['heatmaps_2d']
    B, N, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)
    y = max_indices // W
    x = max_indices % W
    pred_2d_heatmap = torch.stack([x, y], dim=-1).float()[0].cpu().numpy()

    pred_2d = pred_2d_heatmap.copy()
    pred_2d[:, 0] *= orig_dim[0] / heatmap_size
    pred_2d[:, 1] *= orig_dim[1] / heatmap_size

    pred_3d = outputs['keypoints_3d'][0].cpu().numpy()

    return pred_2d, pred_3d


# ─── Overlay Functions ───────────────────────────────────────────────────────

def make_mesh_overlay(original_bgr, rendered_rgba, robot_mask, color_bgr=(0, 200, 0), alpha=0.5):
    """Blend rendered robot mesh onto original image with tinted color.

    Uses segmentation mask from pybullet for precise robot/background separation.
    """
    overlay = original_bgr.copy()
    if not robot_mask.any():
        return overlay

    robot_rgb = rendered_rgba[:, :, :3]
    robot_bgr = robot_rgb[:, :, ::-1]  # RGB→BGR
    tinted = robot_bgr[robot_mask].astype(np.float32) * 0.3 + np.array(color_bgr, dtype=np.float32) * 0.7

    overlay[robot_mask] = (
        original_bgr[robot_mask].astype(np.float32) * (1 - alpha) +
        tinted * alpha
    ).clip(0, 255).astype(np.uint8)

    return overlay


def draw_skeleton(image_bgr, keypoints_2d, gt_keypoints_2d=None):
    result = image_bgr.copy()

    if gt_keypoints_2d is not None:
        for (i, j), color in zip(SKELETON_LINKS, LINK_COLORS_BGR):
            pt1 = tuple(gt_keypoints_2d[i].astype(int))
            pt2 = tuple(gt_keypoints_2d[j].astype(int))
            cv2.line(result, pt1, pt2, (0, 180, 0), 2, cv2.LINE_AA)
        for pt in gt_keypoints_2d:
            center = tuple(pt.astype(int))
            cv2.circle(result, center, 5, (0, 220, 0), -1, cv2.LINE_AA)
            cv2.circle(result, center, 5, (255, 255, 255), 1, cv2.LINE_AA)

    for (i, j), color in zip(SKELETON_LINKS, LINK_COLORS_BGR):
        pt1 = tuple(keypoints_2d[i].astype(int))
        pt2 = tuple(keypoints_2d[j].astype(int))
        cv2.line(result, pt1, pt2, color, 3, cv2.LINE_AA)

    for pt, color in zip(keypoints_2d, KEYPOINT_COLORS_BGR):
        center = tuple(pt.astype(int))
        cv2.circle(result, center, 7, color, -1, cv2.LINE_AA)
        cv2.circle(result, center, 7, (255, 255, 255), 2, cv2.LINE_AA)

    return result


def draw_label(image_bgr, text, position='top-left', color=(255, 255, 255),
               bg_color=(0, 0, 0), font_scale=0.7, thickness=2):
    result = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    org = (10, th + 10)
    cv2.rectangle(result,
                  (org[0] - 5, org[1] - th - 5),
                  (org[0] + tw + 5, org[1] + baseline + 5),
                  bg_color, -1)
    cv2.putText(result, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    return result


# ─── IK Solver ───────────────────────────────────────────────────────────────

def solve_ik_from_keypoints(pred_3d_cam, R_cw, t_cw, robot_base_pos, robot_base_orn):
    """Solve IK from predicted 3D keypoints (in camera frame) → joint angles.

    Transforms pred 3D from camera frame to world frame, then uses pybullet IK.
    """
    # p_world = R_cw^T @ (p_cam - t_cw)
    R_wc = R_cw.T
    t_wc = -R_cw.T @ t_cw

    # Transform hand position to world frame
    hand_cam = pred_3d_cam[6]  # panda_hand
    hand_world = R_wc @ hand_cam + t_wc

    cid = p.connect(p.DIRECT)
    robot_id = p.loadURDF(URDF_PATH, basePosition=robot_base_pos,
                           baseOrientation=robot_base_orn, useFixedBase=True)

    joint_angles = p.calculateInverseKinematics(
        robot_id, 8, hand_world.tolist(),  # joint 8 = panda_hand_joint
        maxNumIterations=200,
        residualThreshold=1e-4,
    )

    p.disconnect(cid)
    return list(joint_angles[:7])


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def create_grid(images_per_row, num_rows, pad=5, bg_color=(40, 40, 40)):
    assert len(images_per_row) == num_rows
    h, w = images_per_row[0][0].shape[:2]
    n_cols = len(images_per_row[0])

    grid_h = num_rows * h + (num_rows - 1) * pad
    grid_w = n_cols * w + (n_cols - 1) * pad
    grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

    for r, row_imgs in enumerate(images_per_row):
        for c, img in enumerate(row_imgs):
            y0 = r * (h + pad)
            x0 = c * (w + pad)
            grid[y0:y0+h, x0:x0+w] = img

    return grid


def process_single_sample(json_path, model, device, image_size, heatmap_size, renderer):
    ann = load_dream_annotation(json_path)

    if ann['image_path'] is None or not os.path.exists(ann['image_path']):
        print(f"  Image not found: {ann['image_path']}")
        return None, None

    image_pil = PILImage.open(ann['image_path']).convert('RGB')
    orig_w, orig_h = image_pil.size
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # GT keypoints (2D in pixels, 3D in camera frame meters)
    gt_2d = np.array([ann['gt_2d'].get(name, [0, 0]) for name in KEYPOINT_NAMES])
    gt_3d = np.array([ann['gt_3d_cam'].get(name, [0, 0, 0]) for name in KEYPOINT_NAMES])

    # Run model inference
    pred_2d, pred_3d = run_inference(
        model, image_pil, ann['camera_K'], device, image_size, heatmap_size
    )

    # ── Row 1: Original image ──
    row1 = draw_label(image_bgr, 'Input Image')

    # ── Row 2: Mesh overlays ──
    gt_overlay = image_bgr.copy()
    pred_overlay = image_bgr.copy()

    R_cw = ann['R_cw']
    t_cw = ann['t_cw']

    if renderer is not None and R_cw is not None and ann['robot_base_pose'] is not None:
        base_pos = ann['robot_base_pose']['position']
        base_orn = ann['robot_base_pose']['orientation']

        # GT mesh overlay
        renderer.set_pose(base_pos, base_orn, ann['joint_angles'])
        gt_rgba, gt_mask = renderer.render_with_T_cw(ann['camera_K'], orig_w, orig_h, R_cw, t_cw)
        gt_overlay = make_mesh_overlay(image_bgr, gt_rgba, gt_mask, color_bgr=(0, 200, 0), alpha=0.6)
        gt_overlay = draw_label(gt_overlay, 'GT Mesh Overlay', color=(0, 255, 0))

        # Pred mesh overlay via IK
        try:
            pred_joints = solve_ik_from_keypoints(pred_3d, R_cw, t_cw, base_pos, base_orn)
            renderer.set_pose(base_pos, base_orn, pred_joints)
            pred_rgba, pred_mask = renderer.render_with_T_cw(ann['camera_K'], orig_w, orig_h, R_cw, t_cw)
            pred_overlay = make_mesh_overlay(image_bgr, pred_rgba, pred_mask, color_bgr=(200, 100, 0), alpha=0.6)
            pred_overlay = draw_label(pred_overlay, 'Pred Mesh (IK)', color=(0, 200, 255))
        except Exception as e:
            print(f"  IK failed: {e}")
            pred_overlay = draw_label(pred_overlay, 'Pred Mesh (IK failed)', color=(0, 0, 255))
    else:
        gt_overlay = draw_label(gt_overlay, 'No mesh data')
        pred_overlay = draw_label(pred_overlay, 'No mesh data')

    # ── Row 3: Skeletons ──
    gt_skel = draw_skeleton(image_bgr, gt_2d)
    gt_skel = draw_label(gt_skel, 'GT Skeleton', color=(0, 255, 0))

    pred_skel = draw_skeleton(image_bgr, pred_2d)
    pred_skel = draw_label(pred_skel, 'Pred Skeleton', color=(0, 200, 255))

    combined_skel = draw_skeleton(image_bgr, pred_2d, gt_keypoints_2d=gt_2d)
    combined_skel = draw_label(combined_skel, 'GT(green) + Pred(color)')

    # Build 3x3 grid
    grid = create_grid(
        [
            [row1, row1.copy(), row1.copy()],
            [gt_overlay, pred_overlay, image_bgr],
            [gt_skel, pred_skel, combined_skel],
        ],
        num_rows=3
    )

    add = np.mean(np.linalg.norm(pred_3d - gt_3d, axis=-1)) * 1000  # mm
    info = {
        'json_path': json_path,
        'add_mm': float(add),
        'pred_2d': pred_2d.tolist(),
        'gt_2d': gt_2d.tolist(),
    }

    return grid, info


def main():
    parser = argparse.ArgumentParser(
        description='Qualitative visualization: mesh overlay + skeleton',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-j', '--json-paths', nargs='+', required=True,
                        help='One or more DREAM JSON annotation paths')
    parser.add_argument('-p', '--model-path', required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('-o', '--output-dir', default='Vis/qualitative_output',
                        help='Output directory')
    parser.add_argument('--no-mesh', action='store_true',
                        help='Skip mesh rendering (faster, skeleton only)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading model: {args.model_path}")
    model, image_size, heatmap_size = load_model(args.model_path, device)

    renderer = None
    if not args.no_mesh:
        print(f"Initializing pybullet renderer...")
        renderer = PandaRenderer(URDF_PATH)

    results = []
    for idx, json_path in enumerate(args.json_paths):
        print(f"\n[{idx+1}/{len(args.json_paths)}] Processing: {json_path}")
        grid, info = process_single_sample(
            json_path, model, device, image_size, heatmap_size, renderer
        )
        if grid is not None:
            basename = Path(json_path).stem
            out_path = os.path.join(args.output_dir, f'{basename}_qualitative.jpg')
            cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  Saved: {out_path}")
            print(f"  ADD: {info['add_mm']:.1f} mm")
            results.append(info)

    if results:
        summary_path = os.path.join(args.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved: {summary_path}")
        mean_add = np.mean([r['add_mm'] for r in results])
        print(f"Mean ADD: {mean_add:.1f} mm")

    if renderer is not None:
        renderer.disconnect()

    print("\nDone.")


if __name__ == '__main__':
    main()
