"""
3D keypoint visualization without mesh/joint-angle dependency.

Inputs:
- Annotation JSON (image path + GT 2D/3D + camera K)
- Model checkpoint

Outputs:
- 3D scatter plot in camera frame (GT vs Pred)
- XY/XZ/YZ projection plots
- Reprojection overlay on original image
- JSON dump with per-keypoint 3D values/errors
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import torch
import torchvision.transforms as TVTransforms
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Train")))
from model import DINOv3PoseEstimator


KEYPOINT_NAMES_DEFAULT = [
    "panda_link0", "panda_link2", "panda_link3",
    "panda_link4", "panda_link6", "panda_link7", "panda_hand",
]

SKELETON_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]


def get_keypoints_from_heatmaps(heatmaps_tensor):
    """Argmax 2D from heatmaps."""
    b, n, h, w = heatmaps_tensor.shape
    heatmaps_flat = heatmaps_tensor.view(b, n, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)
    y = max_indices // w
    x = max_indices % w
    keypoints = torch.stack([x, y], dim=-1).float()
    peak_logits = heatmaps_flat.amax(dim=-1)
    confidences = torch.sigmoid(peak_logits)
    return keypoints[0].cpu().numpy(), confidences[0].cpu().numpy(), peak_logits[0].cpu().numpy()


def load_annotation(json_path, keypoint_names):
    with open(json_path, "r") as f:
        data = json.load(f)

    image_path = None
    camera_k = None
    if "meta" in data:
        raw_path = data["meta"].get("image_path")
        if raw_path:
            if raw_path.startswith("../dataset/"):
                raw_path = raw_path.replace("../dataset/", "../../../", 1)
            if not os.path.isabs(raw_path):
                image_path = os.path.normpath(os.path.join(os.path.dirname(json_path), raw_path))
            else:
                image_path = raw_path
        if "K" in data["meta"]:
            camera_k = np.array(data["meta"]["K"], dtype=np.float64)

    gt_2d = np.zeros((len(keypoint_names), 2), dtype=np.float32)
    gt_3d = np.zeros((len(keypoint_names), 3), dtype=np.float32)
    found = np.zeros((len(keypoint_names),), dtype=bool)

    for obj in data.get("objects", []):
        for kp in obj.get("keypoints", []):
            name = kp.get("name")
            if name in keypoint_names:
                i = keypoint_names.index(name)
                gt_2d[i] = kp.get("projected_location", [-999.0, -999.0])
                if "location" in kp:
                    gt_3d[i] = kp["location"]
                found[i] = True

    if "syn" in json_path.lower():
        gt_3d = gt_3d / 100.0

    return image_path, gt_2d, gt_3d, camera_k, found


def solve_pnp_and_transform(robot_kpts, pred_2d, camera_k):
    """Estimate robot->camera via EPnP and transform all 3D keypoints."""
    valid = (
        np.isfinite(robot_kpts).all(axis=1)
        & np.isfinite(pred_2d).all(axis=1)
        & (pred_2d[:, 0] > -900.0)
        & (pred_2d[:, 1] > -900.0)
    )
    idx = np.where(valid)[0]
    if idx.shape[0] < 4:
        return None, None, None

    ok, rvec, tvec = cv2.solvePnP(
        robot_kpts[idx].astype(np.float64),
        pred_2d[idx].astype(np.float64),
        camera_k.astype(np.float64),
        None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    pred_cam = (rmat @ robot_kpts.T).T + t.reshape(1, 3)
    return pred_cam, rmat, t


def project_camera_points(points_cam, camera_k):
    """Project camera-frame 3D points to pixels."""
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = camera_k[0, 0] * (points_cam[:, 0] / z) + camera_k[0, 2]
    v = camera_k[1, 1] * (points_cam[:, 1] / z) + camera_k[1, 2]
    return np.stack([u, v], axis=1)


def draw_skeleton_3d(ax, points, color, alpha=0.9):
    for i, j in SKELETON_LINKS:
        p1, p2 = points[i], points[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=alpha, linewidth=2)


def draw_skeleton_2d(ax, points, color, alpha=0.9):
    for i, j in SKELETON_LINKS:
        p1, p2 = points[i], points[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=alpha, linewidth=2)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-j", "--json-path", required=True, help="Path to annotation JSON")
    parser.add_argument("-p", "--model-path", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory")
    parser.add_argument("--pred-3d-source", type=str, default="fk", choices=["fk", "fused"],
                        help="Robot-frame 3D source before PnP")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "nomesh_3d_output")
    os.makedirs(output_dir, exist_ok=True)

    keypoint_names = list(KEYPOINT_NAMES_DEFAULT)

    checkpoint_dir = Path(args.model_path).parent
    config_path = checkpoint_dir / "config.yaml"
    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    image_size = 512
    heatmap_size = 512
    use_joint_embedding = False
    fix_joint7_zero = False

    if config_path.exists():
        with open(config_path, "r") as f:
            train_config = yaml.safe_load(f)
        model_name = train_config.get("model_name", model_name)
        image_size = int(train_config.get("image_size", image_size))
        heatmap_size = int(train_config.get("heatmap_size", heatmap_size))
        use_joint_embedding = bool(train_config.get("use_joint_embedding", False))
        fix_joint7_zero = bool(train_config.get("fix_joint7_zero", False))
        if "keypoint_names" in train_config:
            keypoint_names = train_config["keypoint_names"]

    image_path, gt_2d, gt_3d, camera_k, found = load_annotation(args.json_path, keypoint_names)
    if image_path is None or not os.path.exists(image_path):
        raise FileNotFoundError(f"Failed to resolve image path from JSON: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=(heatmap_size, heatmap_size),
        unfreeze_blocks=0,
        use_joint_embedding=use_joint_embedding,
        use_iterative_refinement=False,
        refinement_iterations=0,
        fix_joint7_zero=fix_joint7_zero,
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    image_pil = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size
    transform = TVTransforms.Compose([
        TVTransforms.Resize((image_size, image_size)),
        TVTransforms.ToTensor(),
        TVTransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    camera_k_tensor = None if camera_k is None else torch.tensor(camera_k, dtype=torch.float32).unsqueeze(0).to(device)
    orig_size_tensor = torch.tensor([[orig_w, orig_h]], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(image_tensor, camera_K=camera_k_tensor, original_size=orig_size_tensor, use_refinement=False)
        pred_heatmaps = out["heatmaps_2d"]
        if args.pred_3d_source == "fk":
            pred_3d_robot = out["keypoints_3d_fk"][0].cpu().numpy() if "keypoints_3d_fk" in out else out["keypoints_3d"][0].cpu().numpy()
        else:
            pred_3d_robot = out["keypoints_3d"][0].cpu().numpy()

    pred_2d_hm, pred_conf, pred_peak = get_keypoints_from_heatmaps(pred_heatmaps)
    pred_2d_px = pred_2d_hm.copy()
    pred_2d_px[:, 0] *= orig_w / heatmap_size
    pred_2d_px[:, 1] *= orig_h / heatmap_size

    if camera_k is None:
        raise RuntimeError("camera K is required for camera-frame 3D visualization.")

    pred_3d_cam, rmat, tvec = solve_pnp_and_transform(pred_3d_robot, pred_2d_px, camera_k)
    if pred_3d_cam is None:
        raise RuntimeError("PnP failed: cannot transform predicted 3D to camera frame.")

    valid_3d = found & (~np.all(np.isclose(gt_3d, 0.0), axis=1))
    per_err = np.linalg.norm(pred_3d_cam - gt_3d, axis=1)

    # 1) 3D scatter in camera frame
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], c="g", s=60, label="GT")
    ax.scatter(pred_3d_cam[:, 0], pred_3d_cam[:, 1], pred_3d_cam[:, 2], c="r", s=60, label="Pred")
    draw_skeleton_3d(ax, gt_3d, "green", alpha=0.8)
    draw_skeleton_3d(ax, pred_3d_cam, "red", alpha=0.8)
    for i, name in enumerate(keypoint_names):
        ax.text(gt_3d[i, 0], gt_3d[i, 1], gt_3d[i, 2], f"GT:{name}", color="green", fontsize=7)
        ax.text(pred_3d_cam[i, 0], pred_3d_cam[i, 1], pred_3d_cam[i, 2], f"P:{name}", color="red", fontsize=7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Keypoints in Camera Frame (No Mesh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_3d_scatter_camera_frame.png"), dpi=180)
    plt.close(fig)

    # 2) XY/XZ/YZ projections
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    planes = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)]
    for ax2, (xl, yl, a, b) in zip(axes, planes):
        ax2.scatter(gt_3d[:, a], gt_3d[:, b], c="g", s=50, label="GT")
        ax2.scatter(pred_3d_cam[:, a], pred_3d_cam[:, b], c="r", s=50, label="Pred")
        draw_skeleton_2d(ax2, gt_3d[:, [a, b]], "green", alpha=0.8)
        draw_skeleton_2d(ax2, pred_3d_cam[:, [a, b]], "red", alpha=0.8)
        for i, name in enumerate(keypoint_names):
            ax2.text(gt_3d[i, a], gt_3d[i, b], f"{name}", color="green", fontsize=7)
            ax2.text(pred_3d_cam[i, a], pred_3d_cam[i, b], f"{name}", color="red", fontsize=7)
        ax2.set_xlabel(f"{xl} (m)")
        ax2.set_ylabel(f"{yl} (m)")
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"{xl}-{yl} plane")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_3d_plane_projections.png"), dpi=180)
    plt.close(fig)

    # 3) Reprojection from predicted camera-frame 3D
    reproj = project_camera_points(pred_3d_cam, camera_k)
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    for i, name in enumerate(keypoint_names):
        gt_ok = found[i]
        if gt_ok:
            gx, gy = int(round(float(gt_2d[i, 0]))), int(round(float(gt_2d[i, 1])))
            cv2.circle(img_bgr, (gx, gy), 5, (0, 255, 0), -1)
        rx, ry = int(round(float(reproj[i, 0]))), int(round(float(reproj[i, 1])))
        cv2.circle(img_bgr, (rx, ry), 4, (0, 0, 255), -1)
        cv2.putText(img_bgr, name, (rx + 4, ry - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if gt_ok:
            cv2.line(img_bgr, (gx, gy), (rx, ry), (255, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, "03_reprojection_from_pred3d.png"), img_bgr)

    summary = {
        "json_path": args.json_path,
        "image_path": image_path,
        "model_path": args.model_path,
        "pred_3d_source": args.pred_3d_source,
        "mean_3d_error_m_valid_gt": float(np.mean(per_err[valid_3d])) if np.any(valid_3d) else None,
        "median_3d_error_m_valid_gt": float(np.median(per_err[valid_3d])) if np.any(valid_3d) else None,
        "pnp_rmat": rmat.tolist() if rmat is not None else None,
        "pnp_tvec": tvec.tolist() if tvec is not None else None,
        "per_keypoint": [],
    }
    for i, name in enumerate(keypoint_names):
        summary["per_keypoint"].append({
            "name": name,
            "found_gt": bool(found[i]),
            "peak_logit": float(pred_peak[i]),
            "peak_sigmoid": float(pred_conf[i]),
            "gt_2d_px": [float(gt_2d[i, 0]), float(gt_2d[i, 1])] if found[i] else None,
            "pred_2d_px": [float(pred_2d_px[i, 0]), float(pred_2d_px[i, 1])],
            "gt_3d_m": [float(x) for x in gt_3d[i].tolist()] if found[i] else None,
            "pred_3d_cam_m": [float(x) for x in pred_3d_cam[i].tolist()],
            "err_3d_m": float(per_err[i]) if found[i] else None,
        })
    with open(os.path.join(output_dir, "summary_3d_nomesh.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved to: {output_dir}")
    print("  01_3d_scatter_camera_frame.png")
    print("  02_3d_plane_projections.png")
    print("  03_reprojection_from_pred3d.png")
    print("  summary_3d_nomesh.json")


if __name__ == "__main__":
    main()
