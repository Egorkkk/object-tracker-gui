"""CPU display overlays, independent of the tracking renderer."""
import cv2
import numpy as np
from .camera import project_points
from .types import Pose, CameraIntrinsics
from .masks import frame_masks


def _clip_polygon(points, axis, boundary, greater):
    """Sutherland-Hodgman clipping, including edges crossing the viewport."""
    output = []
    if not len(points):
        return np.empty((0, len(points[0]) if len(points) else 2))
    previous = points[-1]
    inside_previous = (previous[axis] >= boundary) if greater else (previous[axis] <= boundary)
    for point in points:
        inside = (point[axis] >= boundary) if greater else (point[axis] <= boundary)
        if inside != inside_previous:
            t = (boundary - previous[axis]) / (point[axis] - previous[axis])
            output.append(previous + t * (point - previous))
        if inside:
            output.append(point)
        previous, inside_previous = point, inside
    return np.asarray(output)


def _clip_viewport(points, width, height):
    for axis, boundary, greater in ((0, 0, True), (0, width-1, False), (1, 0, True), (1, height-1, False)):
        if not len(points):
            break
        points = _clip_polygon(points, axis, boundary, greater)
    return points


def draw_pose(image, mesh, matrix, camera, color, mode='contour', opacity=0.5):
    pose = Pose(np.asarray(matrix))
    vertices = mesh.vertices_mm
    uv, valid = project_points(vertices, pose, camera)
    h, w = image.shape[:2]
    camera_points = vertices @ pose.T_cam_from_object[:3, :3].T + pose.T_cam_from_object[:3, 3]
    if mode in ('wireframe', 'shaded'):
        faces = mesh.faces[::max(1, len(mesh.faces)//4000)]
        faces = faces[np.any(camera_points[faces, 2] > 1, axis=1)]
        if mode == 'shaded':
            faces = faces[np.argsort(camera_points[faces, 2].mean(axis=1))[::-1]]
        layer = image.copy()
        for face in faces:
            points = _clip_polygon(camera_points[face], 2, 1.000001, True)
            if len(points) < 3:
                continue
            projected = points[:, :2] / points[:, 2:3] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
            polygon = _clip_viewport(projected, w, h)
            if len(polygon) < 3:
                continue
            polygon = np.round(polygon).astype(np.int32)
            if mode == 'shaded':
                cv2.fillConvexPoly(layer, polygon, color, cv2.LINE_AA)
            else:
                cv2.polylines(layer, [polygon], True, color, 1, cv2.LINE_AA)
        cv2.addWeighted(layer, opacity, image, 1-opacity, 0, image)
    if valid.any() and not valid.all():
        # Include near-plane edge intersections in the silhouette.
        crossing = mesh.faces[np.any(valid[mesh.faces], axis=1) & ~np.all(valid[mesh.faces], axis=1)]
        if len(crossing):
            clipped = np.concatenate([_clip_polygon(camera_points[face], 2, 1.000001, True) for face in crossing])
            extra = clipped[:, :2] / clipped[:, 2:3] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
            uv = np.concatenate([uv, extra])
    if len(uv) >= 3:
        hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
        hull = _clip_viewport(hull, w, h)
        if len(hull) >= 3:
            cv2.polylines(image, [np.round(hull).astype(np.int32)], True, color, 2, cv2.LINE_AA)


def render(project, mesh, index, initial=True, refined=True, masks=True, mode='contour', opacity=0.5, object_mask=True, occlusion_mask=True):
    image = cv2.cvtColor(project.sequence.rgb(index), cv2.COLOR_RGB2BGR)
    camera = CameraIntrinsics(**project.state['camera'])
    if masks:
        layers = frame_masks(project, index)
        for mask, color in ((layers.object_mask if object_mask else None, (255, 180, 0)), (layers.occlusion_mask if occlusion_mask else None, (200, 0, 255))):
            if mask is not None:
                image[mask] = (image[mask]*(1-opacity) + np.array(color)*opacity).astype(np.uint8)
    key = str(index)
    if initial and mesh is not None:
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
    if refined and mesh is not None:
        result = project.state['refinements'].get(key) or project.state['poses'].get(key)
        if result:
            draw_pose(image, mesh, result['matrix'], camera, (0, 255, 0), mode, opacity)
    return image
