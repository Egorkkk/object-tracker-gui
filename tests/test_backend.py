import threading
import unittest
import numpy as np
from object_tracker.backends.gotrack.adapter import crop_vertices
from object_tracker.backends.gotrack.backend import GoTrackBackend
from object_tracker.core.types import Pose, TrackingRange, FrameResult, FrameStatus


class BackendContractTests(unittest.TestCase):
    def test_crop_subset_matches_prototype(self):
        vertices = np.arange(6000, dtype=np.float64).reshape(-1, 3)
        expected = vertices.astype(np.float32)[np.random.default_rng(12345).choice(2000, 1000, replace=False)]
        np.testing.assert_array_equal(crop_vertices(vertices), expected)
        self.assertEqual(crop_vertices(vertices[:10]).dtype, np.float32)

    def test_propagation_backward_cancel_and_first_delta(self):
        backend = object.__new__(GoTrackBackend)
        seen = []
        token = threading.Event()
        def refine(image, mesh, camera, initial, index, **kwargs):
            seen.append((index, initial.T_cam_from_object[0, 3]))
            matrix = initial.T_cam_from_object.copy()
            matrix[0, 3] += 2
            return FrameResult(index, Pose(matrix), 1., status=FrameStatus.TRACKED)
        backend.refine_frame = refine
        results = backend.track_range(lambda index: None, None, None, Pose(np.eye(4)),
                                      TrackingRange(5, 2), cancel_token=token)
        first = next(results)
        second = next(results)
        token.set()
        self.assertEqual(list(results), [])
        self.assertEqual(seen, [(5, 0.), (4, 2.)])
        self.assertEqual(first.translation_delta_mm, 0.)
        self.assertEqual(second.translation_delta_mm, 2.)
        self.assertFalse(backend.capabilities.supports_object_mask)
        self.assertFalse(backend.capabilities.supports_occlusion_mask)
