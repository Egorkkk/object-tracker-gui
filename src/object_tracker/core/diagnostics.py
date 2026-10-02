"""Raw pose differences; no smoothing or correction of tracking output."""

import numpy as np

from .types import Pose


def rotation_delta_deg(previous: Pose, current: Pose) -> float:
    R0 = previous.T_cam_from_object[:3, :3]
    R1 = current.T_cam_from_object[:3, :3]
    R = R1 @ R0.T
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def pose_delta(previous: Pose, current: Pose) -> tuple[float, float]:
    translation = float(np.linalg.norm(
        current.T_cam_from_object[:3, 3] - previous.T_cam_from_object[:3, 3]
    ))
    return translation, rotation_delta_deg(previous, current)


def tracking_delta(previous: Pose | None, current: Pose) -> tuple[float, float]:
    """Sequence diagnostics use zero deltas on the first tracked frame."""
    return (0.0, 0.0) if previous is None else pose_delta(previous, current)
