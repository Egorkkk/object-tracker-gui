"""Pose dispersion and raw motion diagnostics, without filtering output poses."""
import numpy as np
from scipy.spatial.transform import Rotation


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(rms=None, median=None, mad=None, p95=None, std=None)
    median = float(np.median(values))
    return dict(rms=float(np.sqrt(np.mean(values**2))), median=median,
                mad=float(np.median(np.abs(values-median))),
                p95=float(np.percentile(values, 95)), std=float(np.std(values)))


def project(points, matrix, camera):
    xyz = points @ matrix[:3, :3].T + matrix[:3, 3]
    uv = xyz[:, :2] / xyz[:, 2:3] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
    uv[xyz[:, 2] <= 1] = np.nan
    return uv


def dispersion(matrices, points, camera):
    matrices = np.asarray(matrices, dtype=float)
    rotations = Rotation.from_matrix(matrices[:, :3, :3])
    reference = np.eye(4)
    reference[:3, :3] = rotations.mean().as_matrix()
    reference[:3, 3] = np.median(matrices[:, :3, 3], axis=0)
    translation = np.linalg.norm(matrices[:, :3, 3]-reference[:3, 3], axis=1)
    rotation = np.rad2deg((Rotation.from_matrix(reference[:3, :3]).inv()*rotations).magnitude())
    uv = np.asarray([project(points, m, camera) for m in matrices])
    reference_uv = project(points, reference, camera)
    # Fixed point identities and common positive-depth support across the group.
    valid = np.isfinite(uv).all(axis=(0, 2)) & np.isfinite(reference_uv).all(axis=1)
    screen = (np.sqrt(np.mean(np.sum((uv[:, valid]-reference_uv[valid])**2, axis=2), axis=1))
              if valid.any() else np.full(len(matrices), np.nan))
    return dict(translation=stats(translation), rotation=stats(rotation), screen=stats(screen),
                translation_axis_std_mm=np.std(matrices[:, :3, 3], axis=0).tolist(),
                translation_std_mm=float(np.linalg.norm(np.std(matrices[:, :3, 3], axis=0))),
                screen_points=int(valid.sum()), reference=reference.tolist()), dict(translation=translation, rotation=rotation, screen=screen)
