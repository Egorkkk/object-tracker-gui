import copy
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from object_tracker.temporal import TemporalParameters as Parameters, filter_poses, FrameQuality, reliability
from object_tracker.temporal.pose_filter import continuous_quaternions
from object_tracker.temporal.state import defaults, ensure, invalidate, selected_entries
from object_tracker.core.project import Project
from object_tracker.core.types import Pose


def poses(positions, angles=None):
    result = {}
    for i, position in enumerate(positions):
        m = np.eye(4)
        m[:3, 3] = position
        if angles is not None:
            m[:3, :3] = Rotation.from_euler('z', angles[i], degrees=True).as_matrix()
        result[i] = m
    return result


class TemporalTests(unittest.TestCase):
    def test_static_jitter_and_rigidity_and_raw_immutable(self):
        rng = np.random.default_rng(4)
        raw = poses(rng.normal(0, 1, (200, 3)), rng.normal(0, 1, 200))
        before = copy.deepcopy(raw)
        filtered, diagnostics = filter_poses(raw, Parameters(enabled=True, adaptive=False))
        raw_t = np.array([m[:3, 3] for m in raw.values()])
        filtered_t = np.array([m[:3, 3] for m in filtered.values()])
        self.assertLess(np.std(filtered_t), np.std(raw_t)*.6)
        raw_r = Rotation.from_matrix([m[:3, :3] for m in raw.values()]).magnitude()
        filtered_r = Rotation.from_matrix([m[:3, :3] for m in filtered.values()]).magnitude()
        self.assertLess(np.std(filtered_r), np.std(raw_r)*.6)
        for i, m in filtered.items():
            Pose(m)
            np.testing.assert_array_equal(raw[i], before[i])
            np.testing.assert_allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(m[:3, :3]), 1.)
            self.assertIsNot(m, raw[i])
            self.assertGreaterEqual(diagnostics[i]['confidence'], 0.)

    def test_wrap_and_quaternion_sign_ambiguity(self):
        angles = np.linspace(160, 200, 81)
        angles = (angles+180) % 360-180
        raw = poses(np.zeros((81, 3)), angles)
        q = Rotation.from_matrix([m[:3, :3] for m in raw.values()]).as_quat()
        q[::2] *= -1
        continuous = continuous_quaternions(q)
        self.assertTrue(np.all(np.sum(continuous[:-1]*continuous[1:], axis=1) > 0))
        filtered, _ = filter_poses(raw, Parameters(enabled=True, adaptive=False))
        r = Rotation.from_matrix([m[:3, :3] for m in filtered.values()])
        steps = (r[:-1].inv()*r[1:]).magnitude()*180/np.pi
        np.testing.assert_allclose(steps, .5, atol=1e-6)

    def test_constant_velocity_boundaries(self):
        raw = poses(np.column_stack([np.arange(40)*2, np.zeros((40, 2))]))
        filtered, _ = filter_poses(raw, Parameters(enabled=True, adaptive=False))
        for i in raw:
            np.testing.assert_allclose(filtered[i], raw[i], atol=1e-10)

    def test_fast_motion_has_no_delay(self):
        positions = np.zeros((61, 3)); positions[30:, 0] = 100
        raw = poses(positions, np.r_[np.zeros(30), np.full(31, 60)])
        adaptive, _ = filter_poses(raw, Parameters(enabled=True))
        fixed, _ = filter_poses(raw, Parameters(enabled=True, adaptive=False))
        self.assertLess(abs(adaptive[30][0, 3]-100), 1.)
        self.assertGreater(abs(fixed[30][0, 3]-100), 10.)
        self.assertLess((Rotation.from_matrix(adaptive[30][:3,:3]).inv()*Rotation.from_euler('z',60,degrees=True)).magnitude(), .01)

    def test_low_confidence_outlier_is_weak_measurement(self):
        positions = np.zeros((61, 3)); positions[30, 0] = 100
        angles = np.zeros(61); angles[30] = 80
        raw = poses(positions, angles)
        quality = {i: FrameQuality(confidence=0. if i == 30 else 1.) for i in raw}
        filtered, diagnostics = filter_poses(raw, Parameters(enabled=True), quality)
        ordinary, _ = filter_poses(raw, Parameters(enabled=True, use_confidence=False), quality)
        self.assertLess(abs(filtered[30][0, 3]), .01)
        self.assertLess(Rotation.from_matrix(filtered[30][:3,:3]).magnitude(), .001)
        self.assertGreater(ordinary[30][0, 3], 10)
        self.assertEqual(diagnostics[30]['translation_velocity'], 0.)
        self.assertEqual(diagnostics[30]['confidence'], 0.)

    def test_range_gaps_singletons_and_disable(self):
        raw = poses(np.random.default_rng(3).normal(size=(31, 3)))
        filtered, _ = filter_poses(raw, Parameters(enabled=True, adaptive=False, start_frame=10,end_frame=20))
        for i in list(range(11))+list(range(20,31)):
            np.testing.assert_array_equal(filtered[i], raw[i])
        a = np.eye(4); b = a.copy(); b[0,3] = 100
        for parameters in (Parameters(enabled=True), Parameters(), Parameters(enabled=True,strength=0)):
            filtered, _ = filter_poses({0:a,2:b}, parameters)
            np.testing.assert_array_equal(filtered[0], a)
            np.testing.assert_array_equal(filtered[2], b)
        self.assertEqual(filter_poses({}), ({}, {}))

    def test_missing_metrics_and_modular_weights(self):
        self.assertEqual(reliability(FrameQuality()), 1.)
        self.assertEqual(reliability(FrameQuality(failed=True)), 0.)
        self.assertEqual(reliability(FrameQuality(confidence=float('nan'))), 1.)
        quality = FrameQuality(confidence=.8, inlier_ratio=.2, valid_correspondences=100, reprojection_error=4)
        self.assertAlmostEqual(reliability(quality, {'confidence':1,'inlier_ratio':1}), .5)
        self.assertEqual(reliability(quality, {'reprojection_error':1}), .5)
        for parameters in ({'strength':-1},{'angular_speed':0},{'start_frame':3}, {'enabled':1}):
            with self.assertRaises(ValueError): Parameters(**parameters)

    def test_cache_save_reload_invalidation_and_export_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Project.create(directory)
            p.state['poses'] = {str(i):dict(matrix=m.tolist(),status='TRACKED') for i,m in poses(np.random.default_rng(8).normal(size=(20,3))).items()}
            original = copy.deepcopy(p.state['poses'])
            p.state['temporal'] = defaults()
            p.state['temporal']['parameters']['enabled'] = True
            self.assertTrue(ensure(p.state)); self.assertFalse(ensure(p.state))
            self.assertNotEqual(selected_entries(p,'filtered')['5']['matrix'], original['5']['matrix'])
            p.save(); reopened = Project.open(directory)
            self.assertEqual(reopened.state['temporal'], p.state['temporal'])
            self.assertEqual(p.state['poses'], original)
            p.state['poses']['5']['matrix'][0][3] += 5
            invalidate(p.state); self.assertEqual(p.state['temporal']['filtered_poses'], {})
            self.assertTrue(ensure(p.state))
            p.state['source']['fps'] = 48
            self.assertTrue(ensure(p.state))
            p.state['temporal']['parameters']['strength'] = .2
            self.assertTrue(ensure(p.state))
            p.state['poses'].pop('5'); self.assertTrue(ensure(p.state))
            self.assertNotIn('5',p.state['temporal']['filtered_poses'])


if __name__ == '__main__': unittest.main()
