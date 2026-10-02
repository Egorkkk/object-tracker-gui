from dataclasses import dataclass
from typing import Protocol

from object_tracker.core.types import CameraIntrinsics, FrameResult, MeshData, Pose


@dataclass(frozen=True)
class BackendCapabilities:
    requires_mesh: bool = True
    requires_initial_pose: bool = True
    requires_depth: bool = False
    supports_refine: bool = True
    supports_tracking: bool = True
    supports_object_mask: bool = False
    supports_occlusion_mask: bool = False
    supports_relocalization: bool = False


class TrackingBackend(Protocol):
    backend_id: str
    capabilities: BackendCapabilities

    def initialize(self): ...
    def shutdown(self): ...
    def refine_frame(self, image, mesh: MeshData, camera: CameraIntrinsics,
                     initial_pose: Pose, frame_index: int = 0,
                     object_mask=None, occlusion_mask=None) -> FrameResult: ...
