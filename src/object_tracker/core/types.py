"""Canonical coordinates: object -> OpenCV camera, translation in millimeters."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True, eq=False)
class Pose:
    T_cam_from_object: np.ndarray

    def __post_init__(self):
        # Preserve float32 results from the tracker; do not re-orthogonalize R.
        matrix = np.array(self.T_cam_from_object, copy=True)
        if matrix.dtype not in (np.dtype("float32"), np.dtype("float64")):
            matrix = matrix.astype(np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Pose must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
            raise ValueError("Pose must have homogeneous last row [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0):
            raise ValueError("Pose rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4, rtol=0):
            raise ValueError("Pose rotation must have determinant +1")
        matrix.setflags(write=False)
        object.__setattr__(self, "T_cam_from_object", matrix)


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Optional[tuple[float, ...]] = None

    def __post_init__(self):
        if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0
               for n in (self.width, self.height)):
            raise ValueError("Camera dimensions must be positive integers")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("Camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")
        if self.distortion is not None:
            coefficients = tuple(self.distortion)
            if not np.isfinite(coefficients).all():
                raise ValueError("Distortion coefficients must be finite")
            object.__setattr__(self, "distortion", coefficients)

    def matrix(self, dtype=np.float32) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
                        dtype=dtype)


@dataclass(frozen=True, eq=False)
class MeshData:
    source_path: Path
    vertices_mm: np.ndarray
    faces: np.ndarray
    source_transform: np.ndarray

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.vertices_mm.min(axis=0), self.vertices_mm.max(axis=0)])

    @property
    def extents(self) -> np.ndarray:
        bounds = self.bounds
        return bounds[1] - bounds[0]


@dataclass(frozen=True, eq=False)
class FrameMasks:
    """Separate semantic layers; neither is enabled in the GoTrack baseline."""

    object_mask: Optional[np.ndarray] = None
    occlusion_mask: Optional[np.ndarray] = None


class FrameStatus(str, Enum):
    UNTRACKED = "UNTRACKED"
    TRACKED = "TRACKED"
    MANUAL = "MANUAL"
    WARNING = "WARNING"
    FAILED = "FAILED"


@dataclass
class FrameResult:
    frame_index: int
    pose: Optional[Pose] = None
    score: Optional[float] = None
    translation_delta_mm: Optional[float] = None
    rotation_delta_deg: Optional[float] = None
    status: FrameStatus = FrameStatus.UNTRACKED
    runtime: Optional[float] = None  # Wall time in seconds.
    backend_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackingRange:
    """Inclusive sequence indices; start may be greater than end."""

    start_frame: int
    end_frame: int

    def __post_init__(self):
        if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0
               for n in (self.start_frame, self.end_frame)):
            raise ValueError("Frame indices must be non-negative integers")

    def indices(self) -> range:
        direction = 1 if self.end_frame >= self.start_frame else -1
        return range(self.start_frame, self.end_frame + direction, direction)


@dataclass
class TrackingResult:
    range: TrackingRange
    backend_id: str
    backend_version: str
    camera: CameraIntrinsics
    mesh_reference: str
    frame_results: dict[int, FrameResult] = field(default_factory=dict)
