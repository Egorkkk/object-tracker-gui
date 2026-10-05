from pathlib import Path
import time

import numpy as np

from object_tracker.backends.base import BackendCapabilities
from object_tracker.core.diagnostics import pose_delta, tracking_delta
from object_tracker.core.types import FrameResult, FrameStatus, Pose
from . import adapter
from .diagnostics import check_runtime, checkpoint_identity
from .environment import activate_runtime, backend_version, prepare_runtime
from .experimental import correspondence_selection
from .quality import capture_quality


class GoTrackBackend:
    """Single-thread-owned resident model; inputs are RGB uint8 and mm poses."""

    backend_id = "gotrack"
    capabilities = BackendCapabilities()

    def __init__(self, source, checkpoint, output, runtime_cache):
        self.source = Path(source).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.output = Path(output).resolve()
        self.runtime_cache = Path(runtime_cache).resolve()
        self.model = None
        self.metadata = {}
        self.mesh_path = None
        self.correspondence_selector = 'random'
        self.version = backend_version(self.source)

    def initialize(self):
        if self.model is not None:
            return
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"GoTrack checkpoint not found: {self.checkpoint}")
        runtime = prepare_runtime(self.source, self.runtime_cache)
        activate_runtime(runtime)
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from utils import net_util
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check WSL GPU and LD_LIBRARY_PATH")
        self.output.mkdir(parents=True, exist_ok=True)
        environment = check_runtime(self.output)
        cfg = OmegaConf.load(runtime / "configs/model/gotrack.yaml")
        self.metadata = dict(environment=environment, checkpoint=checkpoint_identity(self.checkpoint),
                             config=OmegaConf.to_container(cfg, resolve=True),
                             runtime_source_hash=(runtime / ".ready").read_text())
        model = instantiate(cfg)
        model.opts = model.opts._replace(debug=False)
        net_util.load_checkpoint(model=model, checkpoint_path=str(self.checkpoint),
                                 checkpoint_key="model_state_dict", prefix="models.1.")
        self.model = model.cuda().eval()
        self.model.result_file_name = "object"

    def _release_renderer(self):
        renderer = getattr(self.model, "renderer", None)
        if renderer is not None and renderer.renderer is not None:
            renderer.renderer.delete()
        if self.model is not None:
            self.model.renderer = None
        self.mesh_path = None

    def shutdown(self):
        if self.model is not None:
            self._release_renderer()
            self.model = None
            import torch
            torch.cuda.empty_cache()

    def refine_frame(self, image, mesh, camera, initial_pose, frame_index=0,
                     object_mask=None, occlusion_mask=None):
        if image.dtype != np.uint8 or image.shape != (camera.height, camera.width, 3):
            raise ValueError("Expected RGB uint8 image matching camera dimensions")
        self.initialize()
        import torch
        from utils import renderer_builder
        if self.mesh_path != mesh.source_path:
            self._release_renderer()
            self.model.renderer = renderer_builder.build(
                renderer_type=renderer_builder.RendererType.PYRENDER_RASTERIZER,
                model_path=str(mesh.source_path))
            self.model.object_vertices = {1: adapter.crop_vertices(mesh.vertices_mm)}
            self.mesh_path = mesh.source_path
        self.output.mkdir(parents=True, exist_ok=True)
        self.model.result_dir = self.output
        inputs = adapter.make_inputs(image, adapter.make_camera(camera), initial_pose.T_cam_from_object)
        start = time.perf_counter()
        with torch.inference_mode(), correspondence_selection(self.correspondence_selector), capture_quality() as quality:
            outputs = self.model.forward_pipeline(inputs, batch_idx=frame_index)
        pose = Pose(outputs["objects"].poses_cam_from_model[0].detach().cpu().numpy().astype(np.float32))
        score = float(outputs["objects"].pose_scores[0].detach().cpu())
        dt, dr = pose_delta(initial_pose, pose)
        # Masks are transported but intentionally not consumed by the baseline.
        return FrameResult(frame_index, pose, score, dt, dr, FrameStatus.TRACKED,
                           time.perf_counter() - start,
                           {"frame_quality": quality[-1] if quality else {},
                            "inference_runtime": float(outputs["run_time"]),
                            "correspondence_selector": self.correspondence_selector,
                            "object_mask_used": False, "occlusion_mask_used": False})

    def track_range(self, frames, mesh, camera, initial_pose, frame_range,
                    masks=None, progress_callback=None, cancel_token=None):
        """Yield after each frame; caller persists before requesting the next."""
        current = initial_pose
        previous = None
        for index in frame_range.indices():
            if cancel_token is not None and cancel_token.is_set():
                break
            frame_masks = masks(index) if masks else None
            result = self.refine_frame(
                frames(index), mesh, camera, current, index,
                object_mask=frame_masks.object_mask if frame_masks else None,
                occlusion_mask=frame_masks.occlusion_mask if frame_masks else None)
            result.translation_delta_mm, result.rotation_delta_deg = tracking_delta(previous, result.pose)
            previous = current = result.pose
            if progress_callback:
                progress_callback(result)
            yield result
