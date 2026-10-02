"""CPU display overlays, independent of the tracking renderer."""
import cv2
import numpy as np
from .camera import project_points
from .types import Pose, CameraIntrinsics
from .masks import frame_masks


def draw_pose(image, mesh, matrix, camera, color, mode='contour', opacity=0.5):
    pose = Pose(np.asarray(matrix))
    vertices = mesh.vertices_mm
    uv, valid = project_points(vertices, pose, camera)
    h, w = image.shape[:2]
    reasonable = (uv[:, 0] > -w) & (uv[:, 0] < 2*w) & (uv[:, 1] > -h) & (uv[:, 1] < 2*h)
    points = np.round(uv[reasonable]).astype(np.int32)
    if mode in ('wireframe', 'shaded'):
        all_uv = np.zeros((len(vertices), 2))
        all_uv[valid] = uv
        acceptable = np.zeros(len(vertices), dtype=bool)
        acceptable[np.flatnonzero(valid)] = reasonable
        faces = mesh.faces[::max(1, len(mesh.faces)//4000)]
        faces = faces[np.all(acceptable[faces], axis=1)]
        polygons = np.round(all_uv[faces]).astype(np.int32)
        layer = image.copy()
        if mode == 'shaded':
            z = (vertices @ pose.T_cam_from_object[2, :3] + pose.T_cam_from_object[2, 3])
            for i in np.argsort(z[faces].mean(axis=1))[::-1]:
                cv2.fillConvexPoly(layer, polygons[i], color, cv2.LINE_AA)
        else:
            cv2.polylines(layer, list(polygons), True, color, 1, cv2.LINE_AA)
        cv2.addWeighted(layer, opacity, image, 1-opacity, 0, image)
    if len(points) >= 3:
        cv2.polylines(image, [cv2.convexHull(points)], True, color, 2, cv2.LINE_AA)


def render(project, mesh, index, initial=True, refined=True, masks=True, mode='contour', opacity=0.5, object_mask=True, occlusion_mask=True):
    image = cv2.cvtColor(project.sequence.rgb(index), cv2.COLOR_RGB2BGR)
    camera = CameraIntrinsics(**project.state['camera'])
    if masks:
        layers = frame_masks(project, index)
        for mask, color in ((layers.object_mask if object_mask else None, (255, 180, 0)), (layers.occlusion_mask if occlusion_mask else None, (200, 0, 255))):
            if mask is not None:
                image[mask] = (image[mask]*(1-opacity) + np.array(color)*opacity).astype(np.uint8)
    key = str(index)
    if initial:
        try:
            entry = project.state['refinements'].get(key) or project.state['poses'].get(key)
            if key in project.state['drafts']:
                matrix = project.state['drafts'][key]['matrix']
            else:
                matrix = (entry or {}).get('initial_matrix')
                if matrix is None:
                    matrix = project.pose(index).T_cam_from_object
            draw_pose(image, mesh, matrix, camera, (0, 255, 255), mode, opacity)
        except ValueError:
            pass
    if refined:
        result = project.state['refinements'].get(key) or project.state['poses'].get(key)
        if result:
            draw_pose(image, mesh, result['matrix'], camera, (0, 255, 0), mode, opacity)
    return image
