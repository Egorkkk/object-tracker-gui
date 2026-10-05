import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from pose_sensitivity_metrics import (depth_split, displacement, perturb, project,
                                      relative_delta, rotation_modes, series_stats, split_pose)


class PoseSensitivityTests(unittest.TestCase):
    def pose(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec([.3, -.5, .2]).as_matrix()
        pose[:3, 3] = [10., 20., 400.]
        return pose

    def test_camera_and_local_composition_and_pivot(self):
        pose = self.pose(); original = pose.copy()
        delta = Rotation.from_rotvec([np.deg2rad(.1), 0, 0]).as_matrix()
        camera = perturb(pose, 'rotation', 0, .1, 'camera')
        local = perturb(pose, 'rotation', 0, .1, 'local')
        np.testing.assert_allclose(camera[:3, :3], delta @ pose[:3, :3], atol=1e-14)
        np.testing.assert_allclose(local[:3, :3], pose[:3, :3] @ delta, atol=1e-14)
        self.assertFalse(np.allclose(camera[:3, :3], local[:3, :3]))
        np.testing.assert_array_equal(camera[:3, 3], pose[:3, 3])
        np.testing.assert_array_equal(local[:3, 3], pose[:3, 3])
        np.testing.assert_array_equal(pose, original)

    def test_axis_basis_conversion_and_relative_rotation(self):
        first = self.pose(); vector = np.array([.04, -.12, .07])
        second = first.copy()
        second[:3, :3] = first[:3, :3] @ Rotation.from_rotvec(np.deg2rad(vector)).as_matrix()
        second[:3, 3] += [1, -2, 3]
        dt, camera, local = relative_delta(first, second)
        np.testing.assert_allclose(dt, [1, -2, 3], atol=1e-13)
        np.testing.assert_allclose(local, vector, atol=1e-12)
        np.testing.assert_allclose(camera, first[:3, :3] @ vector, atol=1e-12)
        zero = relative_delta(first, first)
        np.testing.assert_allclose(zero, np.zeros((3,3)), atol=1e-12)

    def test_relative_rotation_wraparound(self):
        first, second = np.eye(4), np.eye(4)
        first[:3, :3] = Rotation.from_rotvec([0,0,np.deg2rad(179.)]).as_matrix()
        second[:3, :3] = Rotation.from_rotvec([0,0,np.deg2rad(-179.)]).as_matrix()
        _, camera, local = relative_delta(first, second)
        np.testing.assert_allclose(camera, [0,0,2.], atol=1e-12)
        np.testing.assert_allclose(local, [0,0,2.], atol=1e-12)

    def test_projection_and_pixel_displacement(self):
        first = np.eye(4); first[2,3] = 1000
        points = np.array([[0.,0.,0.],[10.,20.,0.]])
        camera = dict(fx=2000, fy=1000, cx=960, cy=540)
        uv = project(points, first, camera)
        np.testing.assert_allclose(uv, [[960,540],[980,560]])
        projected = project(points, perturb(first, 'translation', 0, 1.), camera)
        result = displacement(uv, projected, [0], [1])
        for key in ('mean_px','rms_px','p95_px','max_px','near_rms_px','far_rms_px'):
            self.assertAlmostEqual(result[key], 2.)
        self.assertAlmostEqual(result['near_far_ratio'], 1.)
        with self.assertRaises(ValueError): project(points, np.eye(4), camera)

    def test_known_rotation_screen_motion(self):
        pose=np.eye(4);pose[2,3]=1000
        points=np.array([[100.,0.,0.]])
        camera=dict(fx=1000,fy=1000,cx=0,cy=0)
        uv=project(points,pose,camera)
        shifted=project(points,perturb(pose,'rotation',2,.1),camera)
        expected=200*np.sin(np.deg2rad(.1)/2)
        self.assertAlmostEqual(displacement(uv,shifted)['rms_px'],expected,places=10)

    def test_near_far_split_by_camera_depth_and_ties(self):
        pose=np.eye(4);pose[2,3]=100
        points=np.array([[0,0,4],[0,0,1],[0,0,3],[0,0,2]])
        near,far,depth=depth_split(points,pose,.25)
        np.testing.assert_array_equal(near,[1]);np.testing.assert_array_equal(far,[0])
        np.testing.assert_array_equal(depth,[104,101,103,102])
        pose[:3,:3]=Rotation.from_rotvec([0,np.pi,0]).as_matrix()
        near,far,_=depth_split(points,pose,.25)
        np.testing.assert_array_equal(near,[0]);np.testing.assert_array_equal(far,[1])
        near,far,_=depth_split(np.zeros((4,3)),pose,.5)
        self.assertFalse(set(near)&set(far))
        with self.assertRaises(ValueError):depth_split(points,pose,.6)

    def test_translation_rotation_only_composition(self):
        first=self.pose();second=perturb(first,'rotation',1,2.,'local')
        second[:3,3]+=[3,4,5];original=first.copy()
        translation,rotation=split_pose(first,second)
        np.testing.assert_array_equal(translation[:3,:3],first[:3,:3])
        np.testing.assert_array_equal(translation[:3,3],second[:3,3])
        np.testing.assert_array_equal(rotation[:3,:3],second[:3,:3])
        np.testing.assert_array_equal(rotation[:3,3],first[:3,3])
        np.testing.assert_array_equal(first,original)
        translation[0,3]+=100
        np.testing.assert_array_equal(second[:3,3],[13,24,405])

    def test_rotation_covariance_detects_one_mode_without_axis_bias(self):
        axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
        vectors=np.array([-2.,-1.,0.,1.,2.])[:,None]*axis
        result=rotation_modes(vectors)
        np.testing.assert_allclose(result['variance_fractions'],[1.,0.,0.],atol=1e-14)
        np.testing.assert_allclose(result['principal_axes_xyz'][0],axis,atol=1e-14)
        self.assertAlmostEqual(result['eigenvalues_deg2'][0],2.)

    def test_signed_alternating_statistics_and_constant_series(self):
        stats=series_stats([1,-1,1,-1,1,-1])
        self.assertEqual(stats['mean'],0);self.assertEqual(stats['rms'],1)
        self.assertEqual(stats['sign_change_rate'],1)
        self.assertAlmostEqual(stats['lag1_autocorrelation'],-1)
        self.assertEqual(series_stats([0,0,0])['lag1_autocorrelation'],None)
        self.assertEqual(series_stats([0,0,0])['sign_change_rate'],None)
        self.assertEqual(series_stats([1,0,-1])['sign_change_pairs'],0)


if __name__=='__main__': unittest.main()
