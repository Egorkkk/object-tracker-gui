"""Pinhole projection in the prototype's OpenCV camera convention."""

import numpy as np

from .types import CameraIntrinsics, Pose


def project_points(vertices_mm: np.ndarray, pose: Pose, camera: CameraIntrinsics):
    """Return visible UVs and the original vertex mask, using z > 1 mm.

    Distortion storage is supported, but this baseline projector is pinhole-only.
    """
    if camera.distortion is not None and any(camera.distortion):
        raise ValueError("Pinhole projection does not support nonzero distortion")
    matrix = pose.T_cam_from_object
    points = (matrix[:3, :3] @ vertices_mm.T).T + matrix[:3, 3]
    valid = points[:, 2] > 1.0
    points = points[valid]
    if len(points) == 0:
        return np.empty((0, 2)), valid
    uv = np.empty((len(points), 2), dtype=np.float64)
    uv[:, 0] = camera.fx * points[:, 0] / points[:, 2] + camera.cx
    uv[:, 1] = camera.fy * points[:, 1] / points[:, 2] + camera.cy
    return uv, valid
