import os
import sys
import csv
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np
import torch
import trimesh

from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils import net_util, renderer_builder, structs


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

FRAMES_GLOB = "custom/frames/*.png"
MESH_PATH = "custom/phone_mm.ply"
INITIAL_POSE_PATH = "custom/alignment/initial_pose.npy"
OUT_DIR = Path("custom/track60")

MAX_FRAMES = 60

FX = 1867.0
FY = 1867.0
CX = 960.0
CY = 540.0


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------

def load_mesh(path):
    mesh = trimesh.load(path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(mesh.geometry.values())
        )

    return mesh


def project_vertices(vertices, T, K):
    pts = (T[:3, :3] @ vertices.T).T + T[:3, 3]

    valid = pts[:, 2] > 1.0
    pts = pts[valid]

    if len(pts) == 0:
        return np.empty((0, 2))

    uv = np.empty((len(pts), 2), dtype=np.float64)

    uv[:, 0] = (
        K[0, 0] * pts[:, 0] / pts[:, 2]
        + K[0, 2]
    )

    uv[:, 1] = (
        K[1, 1] * pts[:, 1] / pts[:, 2]
        + K[1, 2]
    )

    return uv


def make_hull(uv, width, height):
    if len(uv) < 3:
        return None

    valid = (
        (uv[:, 0] > -width)
        & (uv[:, 0] < 2 * width)
        & (uv[:, 1] > -height)
        & (uv[:, 1] < 2 * height)
    )

    uv = uv[valid]

    if len(uv) < 3:
        return None

    return cv2.convexHull(
        np.round(uv).astype(np.int32)
    )


def draw_pose(image, vertices, T, K, color, thickness=2):
    h, w = image.shape[:2]

    uv = project_vertices(
        vertices,
        T,
        K
    )

    hull = make_hull(uv, w, h)

    if hull is not None:
        cv2.polylines(
            image,
            [hull],
            True,
            color,
            thickness,
            cv2.LINE_AA
        )


def rotation_delta_deg(T0, T1):
    R0 = T0[:3, :3]
    R1 = T1[:3, :3]

    R = R1 @ R0.T

    cos_angle = (
        np.trace(R) - 1.0
    ) / 2.0

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )

    return float(
        np.degrees(np.arccos(cos_angle))
    )


# ------------------------------------------------------------
# GoTrack input
# ------------------------------------------------------------

def make_inputs(image_rgb, camera, pose):
    tensor = (
        torch.from_numpy(image_rgb)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    images = structs.Collection()

    images.bitmaps = tensor.unsqueeze(0)
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
        pose.astype(np.float32)
    ).unsqueeze(0)

    # Camera == world for this experiment.
    objects.poses_cam_from_model = (
        pose_tensor.clone()
    )

    objects.poses_world_from_model = (
        pose_tensor.clone()
    )

    return {
        "images": images,
        "objects": objects,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pose_dir = OUT_DIR / "poses"
    overlay_dir = OUT_DIR / "overlays"
    internal_dir = OUT_DIR / "gotrack_internal"

    pose_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    internal_dir.mkdir(exist_ok=True)

    frame_paths = sorted(
        glob.glob(FRAMES_GLOB)
    )

    frame_paths = frame_paths[:MAX_FRAMES]

    if not frame_paths:
        raise RuntimeError(
            f"No frames found: {FRAMES_GLOB}"
        )

    print(
        f"Found {len(frame_paths)} frames"
    )

    # --------------------------------------------------------
    # First frame / camera
    # --------------------------------------------------------

    first = cv2.imread(frame_paths[0])

    if first is None:
        raise RuntimeError(
            f"Cannot read {frame_paths[0]}"
        )

    height, width = first.shape[:2]

    print(
        f"Resolution: {width}x{height}"
    )

    K = np.array([
        [FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    camera = structs.PinholePlaneCameraModel(
        width=width,
        height=height,
        f=(FX, FY),
        c=(CX, CY),
        T_world_from_eye=np.eye(
            4,
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    mesh = load_mesh(MESH_PATH)

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float32
    )

    print(
        "Mesh extents:",
        mesh.extents
    )

    print(
        "Mesh vertices:",
        len(vertices)
    )

    # GoTrack crop calculation does not need 175k vertices.
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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Creating GoTrack...")

    cfg = OmegaConf.load(
        "configs/model/gotrack.yaml"
    )

    model = instantiate(cfg)

    model.opts = model.opts._replace(
        debug=False
    )

    print(
        "Loading checkpoint..."
    )

    net_util.load_checkpoint(
        model=model,
        checkpoint_path="gotrack_checkpoint.pt",
        checkpoint_key="model_state_dict",
        prefix="models.1."
    )

    model = model.cuda().eval()

    model.renderer = renderer_builder.build(
        renderer_type=(
            renderer_builder.RendererType
            .PYRENDER_RASTERIZER
        ),
        model_path=str(
            Path(MESH_PATH).resolve()
        )
    )

    model.object_vertices = {
        1: crop_vertices
    }

    model.result_dir = internal_dir
    model.result_file_name = "phone"

    # Don't generate GoTrack's own large visualization.
    model.get_vis_tiles = lambda outputs, save_path: None

    # --------------------------------------------------------
    # Initial pose
    # --------------------------------------------------------

    current_pose = np.load(
        INITIAL_POSE_PATH
    ).astype(np.float32)

    print()
    print("Starting pose:")
    print(current_pose)

    # --------------------------------------------------------
    # Preview video
    # --------------------------------------------------------

    preview_path = (
        OUT_DIR / "tracking_preview.mp4"
    )

    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        24.0,
        (width, height)
    )

    csv_path = OUT_DIR / "tracking.csv"

    csv_file = open(
        csv_path,
        "w",
        newline=""
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "filename",
        "score",
        "translation_delta_mm",
        "rotation_delta_deg",
        "tx",
        "ty",
        "tz"
    ])

    previous_refined = None

    # --------------------------------------------------------
    # Tracking loop
    # --------------------------------------------------------

    for frame_idx, path in enumerate(frame_paths):

        print()
        print("=" * 60)

        print(
            f"Frame {frame_idx:04d} / "
            f"{len(frame_paths)-1:04d}"
        )

        print(
            Path(path).name
        )

        image_bgr = cv2.imread(path)

        if image_bgr is None:
            print(
                "WARNING: cannot read frame"
            )
            continue

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        propagated_pose = (
            current_pose.copy()
        )

        inputs = make_inputs(
            image_rgb,
            camera,
            propagated_pose
        )

        with torch.inference_mode():

            outputs = model.forward_pipeline(
                inputs,
                batch_idx=frame_idx
            )

        refined_pose = (
            outputs["objects"]
            .poses_cam_from_model[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        score = float(
            outputs["objects"]
            .pose_scores[0]
            .detach()
            .cpu()
        )

        # ----------------------------------------------
        # Motion diagnostics
        # ----------------------------------------------

        if previous_refined is None:

            translation_delta = 0.0
            rotation_delta = 0.0

        else:

            translation_delta = float(
                np.linalg.norm(
                    refined_pose[:3, 3]
                    - previous_refined[:3, 3]
                )
            )

            rotation_delta = (
                rotation_delta_deg(
                    previous_refined,
                    refined_pose
                )
            )

        print(
            f"score = {score:.4f}"
        )

        print(
            f"dT = {translation_delta:.3f} mm"
        )

        print(
            f"dR = {rotation_delta:.3f} deg"
        )

        print(
            "T =",
            refined_pose[:3, 3]
        )

        # ----------------------------------------------
        # Save pose
        # ----------------------------------------------

        np.save(
            pose_dir /
            f"{frame_idx:04d}.npy",
            refined_pose
        )

        np.savetxt(
            pose_dir /
            f"{frame_idx:04d}.txt",
            refined_pose,
            fmt="%.9f"
        )

        # ----------------------------------------------
        # Overlay
        # ----------------------------------------------

        overlay = image_bgr.copy()

        # propagated initial = yellow
        draw_pose(
            overlay,
            vertices,
            propagated_pose,
            K,
            (0, 255, 255),
            2
        )

        # refined = green
        draw_pose(
            overlay,
            vertices,
            refined_pose,
            K,
            (0, 255, 0),
            3
        )

        cv2.putText(
            overlay,
            f"frame {frame_idx:04d}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            overlay,
            f"score {score:.3f}",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            overlay,
            (
                f"dT {translation_delta:.1f} mm   "
                f"dR {rotation_delta:.1f} deg"
            ),
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            overlay,
            "yellow: propagated   green: refined",
            (30, height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imwrite(
            str(
                overlay_dir /
                f"{frame_idx:04d}.png"
            ),
            overlay
        )

        writer.write(overlay)

        # ----------------------------------------------
        # CSV
        # ----------------------------------------------

        csv_writer.writerow([
            frame_idx,
            Path(path).name,
            score,
            translation_delta,
            rotation_delta,
            refined_pose[0, 3],
            refined_pose[1, 3],
            refined_pose[2, 3]
        ])

        csv_file.flush()

        # ----------------------------------------------
        # Pose propagation
        # ----------------------------------------------

        previous_refined = (
            refined_pose.copy()
        )

        current_pose = (
            refined_pose.copy()
        )

    # --------------------------------------------------------

    writer.release()
    csv_file.close()

    print()
    print("=" * 60)
    print("DONE")
    print()

    print(
        "Preview:",
        preview_path
    )

    print(
        "CSV:",
        csv_path
    )

    print(
        "Poses:",
        pose_dir
    )

    print(
        "Overlays:",
        overlay_dir
    )


if __name__ == "__main__":
    main()