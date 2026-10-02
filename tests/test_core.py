"""CPU regression against unchanged prototype functions, without ML imports."""

import ast
from pathlib import Path
import unittest

import numpy as np
import trimesh

from object_tracker.core.camera import project_points
from object_tracker.core.diagnostics import pose_delta, rotation_delta_deg, tracking_delta
from object_tracker.core.mesh import load_mesh
from object_tracker.core.pose import make_pose
from object_tracker.core.types import CameraIntrinsics, FrameMasks, Pose, TrackingRange


ROOT = Path(__file__).resolve().parents[1]


def prototype_functions(filename, names):
    # Load the actual reference bodies, without executing their import-time
    # EGL/sys.path/torch setup or CLI. No copied expected implementation.
    path = ROOT / "prototypes" / filename
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"Missing reference functions in {path}")
    module = ast.Module(body=functions, type_ignores=[])
    namespace = dict(np=np, trimesh=trimesh, FX=1867., FY=1867., CX=960., CY=540.)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class CoreRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.align = prototype_functions("align_phone.py", [
            "euler_to_matrix", "make_pose", "project_points"])
        cls.refine = prototype_functions("refine_one.py", ["project_vertices", "load_mesh"])
        cls.track = prototype_functions("track_sequence.py", [
            "project_vertices", "rotation_delta_deg", "load_mesh"])
        cls.camera = CameraIntrinsics(1920, 1080, 1867., 1867., 960., 540.)

    def test_euler_order_matches_alignment(self):
        for angles in [(90, 0, 0), (23, -57, 131), (-180, 90, 180), (0, 0, 0)]:
            values = (*angles, 13, -27, 700)
            expected = self.align["make_pose"](*values)
            actual = make_pose(*values).T_cam_from_object
            np.testing.assert_array_equal(actual, expected)

    def test_projection_matches_all_three_prototypes(self):
        vertices = np.random.default_rng(42).normal(size=(100, 3)) * 100
        for dtype in (np.float32, np.float64):
            vertices = vertices.astype(dtype)
            pose = Pose(make_pose(13, 27, -45, 30, -50, 70).T_cam_from_object.astype(dtype))
            actual, valid = project_points(vertices, pose, self.camera)
            matrix = pose.T_cam_from_object
            expected, expected_valid = self.align["project_points"](vertices, matrix)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(valid, expected_valid)
            for reference in (self.refine, self.track):
                expected = reference["project_vertices"](vertices, matrix, self.camera.matrix())
                np.testing.assert_array_equal(actual, expected)

    def test_near_plane_and_opencv_axes(self):
        vertices = np.array([[0, 0, -1], [0, 0, 1], [0, 0, 1.001],
                             [10, 0, 100], [0, 10, 100]], dtype=float)
        uv, valid = project_points(vertices, Pose(np.eye(4)), self.camera)
        np.testing.assert_array_equal(valid, [False, False, True, True, True])
        np.testing.assert_allclose(uv, [[960, 540], [1146.7, 540], [960, 726.7]])
        empty, valid = project_points(vertices[:2], Pose(np.eye(4)), self.camera)
        self.assertEqual(empty.shape, (0, 2))
        self.assertFalse(valid.any())

    def test_rotation_and_translation_diagnostics(self):
        previous = make_pose(0, 0, 0, 0, 0, 700)
        current = make_pose(90, 0, 0, 3, 4, 700)
        self.assertEqual(pose_delta(previous, current), (5., 90.))
        self.assertEqual(tracking_delta(None, current), (0., 0.))
        for dtype in (np.float32, np.float64):
            a = Pose(previous.T_cam_from_object.astype(dtype))
            b = Pose(current.T_cam_from_object.astype(dtype))
            expected = self.track["rotation_delta_deg"](a.T_cam_from_object, b.T_cam_from_object)
            self.assertEqual(rotation_delta_deg(a, b), expected)

    def test_pose_validation_and_copy(self):
        matrix = np.eye(4, dtype=np.float32)
        pose = Pose(matrix)
        matrix[0, 3] = 123
        self.assertEqual(pose.T_cam_from_object[0, 3], 0)
        self.assertEqual(pose.T_cam_from_object.dtype, np.float32)
        with self.assertRaises(ValueError):
            pose.T_cam_from_object[0, 3] = 1
        for invalid in [np.eye(3), np.full((4, 4), np.nan),
                        np.diag([-1, 1, 1, 1]), np.diag([2, 1, 1, 1]),
                        np.diag([1, 1, 1, 2])]:
            with self.assertRaises(ValueError):
                Pose(invalid)

    def test_camera_validation_and_distortion_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            CameraIntrinsics(1920, 1080, 0, 1, 0, 0)
        with self.assertRaises(ValueError):
            CameraIntrinsics(1920.5, 1080, 1, 1, 0, 0)
        camera = CameraIntrinsics(1920, 1080, 1, 1, 0, 0, (0.1,))
        with self.assertRaises(ValueError):
            project_points(np.array([[0., 0., 100.]]), Pose(np.eye(4)), camera)

    def test_ranges_and_distinct_masks(self):
        self.assertEqual(list(TrackingRange(3, 5).indices()), [3, 4, 5])
        self.assertEqual(list(TrackingRange(5, 3).indices()), [5, 4, 3])
        self.assertEqual(list(TrackingRange(5, 5).indices()), [5])
        with self.assertRaises(ValueError):
            TrackingRange(-1, 3)
        object_mask = np.ones((2, 2), dtype=bool)
        occlusion_mask = np.zeros((2, 2), dtype=bool)
        masks = FrameMasks(object_mask, occlusion_mask)
        self.assertIs(masks.object_mask, object_mask)
        self.assertIs(masks.occlusion_mask, occlusion_mask)

    def test_phone_mesh_and_pose_match_references(self):
        data = ROOT / "testdata" / "phone_short"
        path = data / "phone_mm.ply"
        if not path.exists():
            self.skipTest("Local phone regression dataset is unavailable")
        mesh = load_mesh(path)
        for reference in (self.refine, self.track):
            expected = reference["load_mesh"](path)
            np.testing.assert_array_equal(mesh.vertices_mm, np.asarray(expected.vertices))
            np.testing.assert_array_equal(mesh.faces, np.asarray(expected.faces))
            np.testing.assert_array_equal(mesh.bounds, expected.bounds)
            np.testing.assert_array_equal(mesh.extents, expected.extents)
        np.testing.assert_allclose(mesh.extents, [75, 11, 160], atol=1e-4)
        np.testing.assert_array_equal(mesh.source_transform, np.eye(4))
        self.assertFalse(mesh.vertices_mm.flags.writeable)
        for name in ("initial_pose.npy", "refined_pose.npy"):
            matrix = np.load(data / name)
            pose = Pose(matrix)
            np.testing.assert_array_equal(pose.T_cam_from_object, matrix)
            vertices = mesh.vertices_mm.astype(np.float32)
            actual, _ = project_points(vertices, pose, self.camera)
            expected = self.track["project_vertices"](vertices, matrix, self.camera.matrix())
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
