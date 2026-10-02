"""Exact single-object input construction from track_sequence.py."""

import numpy as np


def crop_vertices(vertices):
    vertices = np.asarray(vertices, dtype=np.float32)
    if len(vertices) > 1000:
        ids = np.random.default_rng(12345).choice(len(vertices), 1000, replace=False)
        vertices = vertices[ids]
    return vertices


def make_camera(camera):
    from utils import structs
    if camera.distortion is not None and any(camera.distortion):
        raise ValueError("GoTrack baseline requires an undistorted pinhole camera")
    return structs.PinholePlaneCameraModel(
        width=camera.width, height=camera.height, f=(camera.fx, camera.fy),
        c=(camera.cx, camera.cy), T_world_from_eye=np.eye(4, dtype=np.float32))


def make_inputs(image_rgb, camera, pose):
    import torch
    from utils import structs
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    images = structs.Collection()
    images.bitmaps = tensor.unsqueeze(0)
    images.cameras = [camera]
    images.scene_ids = torch.tensor([0], dtype=torch.int32)
    images.im_ids = torch.tensor([0], dtype=torch.int32)
    images.times = torch.tensor([0.0], dtype=torch.float32)
    objects = structs.Collection()
    objects.labels = torch.tensor([1], dtype=torch.int32)
    objects.frame_ids = torch.tensor([0], dtype=torch.int32)
    objects.inst_ids = torch.tensor([0], dtype=torch.int32)
    pose_tensor = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0)
    objects.poses_cam_from_model = pose_tensor.clone()
    objects.poses_world_from_model = pose_tensor.clone()
    return {"images": images, "objects": objects}
