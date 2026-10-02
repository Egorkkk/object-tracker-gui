import os

# Важно поставить до импорта pyrender/OpenGL.
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils import net_util, renderer_builder, structs


def load_mesh(path):
    mesh = trimesh.load(path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    return mesh


def project_vertices(vertices, T, K):
    pts = (T[:3, :3] @ vertices.T).T + T[:3, 3]

    good = pts[:, 2] > 1.0
    pts = pts[good]

    if len(pts) == 0:
        return np.empty((0, 2))

    uv = np.empty((len(pts), 2), np.float64)

    uv[:, 0] = K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2]
    uv[:, 1] = K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]

    return uv


def draw_pose(image, vertices, T, K, color, label):
    out = image.copy()

    uv = project_vertices(vertices, T, K)

    h, w = out.shape[:2]

    reasonable = (
        (uv[:, 0] > -w)
        & (uv[:, 0] < 2 * w)
        & (uv[:, 1] > -h)
        & (uv[:, 1] < 2 * h)
    )

    uv = uv[reasonable]

    if len(uv) >= 3:
        pts = np.round(uv).astype(np.int32)
        hull = cv2.convexHull(pts)

        cv2.polylines(
            out,
            [hull],
            True,
            color,
            3,
            cv2.LINE_AA,
        )

        step = max(1, len(pts) // 2000)

        for x, y in pts[::step]:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(out, (x, y), 1, color, -1)

    cv2.putText(
        out,
        label,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--image", default="custom/frame000.png")
    ap.add_argument("--mesh", default="custom/phone_mm.ply")
    ap.add_argument(
        "--pose",
        default="custom/alignment/initial_pose.npy"
    )

    ap.add_argument("--fx", type=float, default=1867.0)
    ap.add_argument("--fy", type=float, default=1867.0)
    ap.add_argument("--cx", type=float, default=960.0)
    ap.add_argument("--cy", type=float, default=540.0)

    ap.add_argument(
        "--out",
        default="custom/refine_one"
    )

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Image
    # ----------------------------------------------------------

    image_bgr = cv2.imread(args.image)

    if image_bgr is None:
        raise RuntimeError(f"Cannot read {args.image}")

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    h, w = image_rgb.shape[:2]

    print(f"Image: {w}x{h}")

    # ----------------------------------------------------------
    # Intrinsics
    # ----------------------------------------------------------

    K = np.array([
        [args.fx, 0.0, args.cx],
        [0.0, args.fy, args.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    print("K:")
    print(K)

    camera = structs.PinholePlaneCameraModel(
        width=w,
        height=h,
        f=(args.fx, args.fy),
        c=(args.cx, args.cy),
        T_world_from_eye=np.eye(4, dtype=np.float32),
    )

    # ----------------------------------------------------------
    # Mesh — coordinates MUST be mm
    # ----------------------------------------------------------

    mesh = load_mesh(args.mesh)

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float32
    )

    print("Mesh extents mm:", mesh.extents)
    print("Vertices:", len(vertices))

    # GoTrack itself only needs a subset for crop computation.
    if len(vertices) > 1000:
        rng = np.random.default_rng(12345)

        ids = rng.choice(
            len(vertices),
            1000,
            replace=False
        )

        crop_vertices = vertices[ids]
    else:
        crop_vertices = vertices

    # ----------------------------------------------------------
    # Initial pose, model -> camera, millimeters
    # ----------------------------------------------------------

    init_pose = np.load(args.pose).astype(np.float32)

    if init_pose.shape != (4, 4):
        raise RuntimeError(
            f"Pose must be 4x4, got {init_pose.shape}"
        )

    print()
    print("Initial T_cam_from_model:")
    print(init_pose)

    # ----------------------------------------------------------
    # Build GoTrack
    # ----------------------------------------------------------

    print()
    print("Creating GoTrack...")

    cfg = OmegaConf.load(
        "configs/model/gotrack.yaml"
    )

    model = instantiate(cfg)

    # Не запускаем внутреннюю debug-визуализацию BOP.
    model.opts = model.opts._replace(debug=False)

    print("Loading GoTrack checkpoint...")

    net_util.load_checkpoint(
        model=model,
        checkpoint_path="gotrack_checkpoint.pt",
        checkpoint_key="model_state_dict",
        prefix="models.1.",
    )

    model = model.cuda().eval()

    # ----------------------------------------------------------
    # Custom renderer + geometry
    # ----------------------------------------------------------

    model.renderer = renderer_builder.build(
        renderer_type=renderer_builder.RendererType.PYRENDER_RASTERIZER,
        model_path=str(Path(args.mesh).resolve()),
    )

    # Object ID = 1
    model.object_vertices = {
        1: crop_vertices
    }

    # forward_pipeline() требует result_dir даже если нам нужен
    # только возвращаемый pose.
    model.result_dir = out_dir
    model.result_file_name = "phone"

    # ----------------------------------------------------------
    # Construct GoTrack inputs directly
    # ----------------------------------------------------------

    image_tensor = (
        torch.from_numpy(image_rgb)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    images = structs.Collection()

    images.bitmaps = image_tensor.unsqueeze(0)
    images.cameras = [camera]
    images.scene_ids = torch.tensor(
        [0],
        dtype=torch.int32
    )
    images.im_ids = torch.tensor(
        [0],
        dtype=torch.int32
    )
    images.times = torch.tensor(
        [0.0],
        dtype=torch.float32
    )

    objects = structs.Collection()

    objects.labels = torch.tensor(
        [1],
        dtype=torch.int32
    )

    objects.frame_ids = torch.tensor(
        [0],
        dtype=torch.int32
    )

    objects.inst_ids = torch.tensor(
        [0],
        dtype=torch.int32
    )

    pose_tensor = torch.from_numpy(
        init_pose
    ).unsqueeze(0)

    # Camera == world для первого кадра.
    objects.poses_cam_from_model = pose_tensor.clone()
    objects.poses_world_from_model = pose_tensor.clone()

    inputs = {
        "images": images,
        "objects": objects,
    }

    # ----------------------------------------------------------
    # Run refinement
    # ----------------------------------------------------------

    print()
    print("Running GoTrack refinement...")
    print(
        f"Iterations: {model.opts.num_iterations_test}"
    )

    with torch.inference_mode():
        outputs = model.forward_pipeline(
            inputs,
            batch_idx=0
        )

    refined_pose = (
        outputs["objects"]
        .poses_cam_from_model[0]
        .detach()
        .cpu()
        .numpy()
    )

    score = float(
        outputs["objects"]
        .pose_scores[0]
        .detach()
        .cpu()
    )

    print()
    print("========================================")
    print("REFINEMENT COMPLETE")
    print("========================================")

    print()
    print("Initial:")
    print(init_pose)

    print()
    print("Refined:")
    print(refined_pose)

    print()
    print("Score:", score)

    print()
    print(
        "Translation initial:",
        init_pose[:3, 3]
    )

    print(
        "Translation refined:",
        refined_pose[:3, 3]
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    np.save(
        out_dir / "refined_pose.npy",
        refined_pose
    )

    np.savetxt(
        out_dir / "refined_pose.txt",
        refined_pose,
        fmt="%.9f"
    )

    # ----------------------------------------------------------
    # Visual comparison
    # ----------------------------------------------------------

    initial_vis = draw_pose(
        image_bgr,
        vertices,
        init_pose,
        K,
        (0, 255, 255),
        "INITIAL",
    )

    refined_vis = draw_pose(
        image_bgr,
        vertices,
        refined_pose,
        K,
        (0, 255, 0),
        "REFINED",
    )

    # Both on one image.
    compare = image_bgr.copy()

    uv_init = project_vertices(
        vertices,
        init_pose,
        K
    )

    uv_ref = project_vertices(
        vertices,
        refined_pose,
        K
    )

    def hull(uv):
        hh, ww = compare.shape[:2]

        ok = (
            (uv[:, 0] > -ww)
            & (uv[:, 0] < 2 * ww)
            & (uv[:, 1] > -hh)
            & (uv[:, 1] < 2 * hh)
        )

        uv = uv[ok]

        if len(uv) < 3:
            return None

        return cv2.convexHull(
            np.round(uv).astype(np.int32)
        )

    h0 = hull(uv_init)
    h1 = hull(uv_ref)

    if h0 is not None:
        cv2.polylines(
            compare,
            [h0],
            True,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

    if h1 is not None:
        cv2.polylines(
            compare,
            [h1],
            True,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    cv2.putText(
        compare,
        "YELLOW = initial    GREEN = GoTrack",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(out_dir / "initial.png"),
        initial_vis
    )

    cv2.imwrite(
        str(out_dir / "refined.png"),
        refined_vis
    )

    cv2.imwrite(
        str(out_dir / "compare.png"),
        compare
    )

    print()
    print("Saved:")
    print(out_dir / "refined_pose.npy")
    print(out_dir / "refined_pose.txt")
    print(out_dir / "initial.png")
    print(out_dir / "refined.png")
    print(out_dir / "compare.png")


if __name__ == "__main__":
    main()