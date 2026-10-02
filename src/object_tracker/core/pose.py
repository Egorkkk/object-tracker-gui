"""Pose editing math extracted from prototypes/align_phone.py."""

import numpy as np

from .types import Pose


def euler_to_matrix(rx, ry, rz):
    """Euler degrees -> rotation; preserve the prototype order Rz @ Ry @ Rx."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)],
    ], dtype=np.float64)
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float64)
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1],
    ], dtype=np.float64)
    return Rz @ Ry @ Rx


def make_pose(rx, ry, rz, tx, ty, tz) -> Pose:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = euler_to_matrix(rx, ry, rz)
    matrix[:3, 3] = [tx, ty, tz]
    return Pose(matrix)
