import sys
from pathlib import Path
import unittest
from collections import namedtuple
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import numpy as np

from object_tracker.backends.gotrack.experimental import select_correspondences, experiment
from object_tracker.core.types import CameraIntrinsics
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from jitter_metrics import dispersion


class SelectorTests(unittest.TestCase):
    def test_ordering_count_and_ties(self):
        p=np.array([[0,0],[1,0],[2,0],[3,0]])
        c=np.array([.3,.9,.9,.1])
        np.testing.assert_array_equal(select_correspondences(p,c,2,'top_confidence'),[1,2])
        self.assertEqual(len(select_correspondences(p,c,10,'top_confidence')),4)

    def test_spatial_coverage_and_determinism(self):
        p=np.array([[0,0],[1,0],[100,0],[101,0],[0,100],[1,100],[100,100],[101,100]])
        c=np.array([.99,.98,.6,.5,.4,.3,.2,.1])
        a=select_correspondences(p,c,4,'spatial_confidence',2)
        np.testing.assert_array_equal(a,[0,2,4,6])
        np.random.seed(1)
        np.testing.assert_array_equal(a,select_correspondences(p,c,4,'spatial_confidence',2))
        self.assertEqual(len(set(select_correspondences(p,c,7,'spatial_confidence',2))),7)

    def test_empty_small_random(self):
        for s in ['random','top_confidence','spatial_confidence']:
            self.assertEqual(len(select_correspondences(np.empty((0,2)),np.empty(0),5,s)),0)
            np.testing.assert_array_equal(select_correspondences([[1,2]],[.5],4,s),[0])
        np.random.seed(4);a=select_correspondences(np.zeros((30,2)),np.ones(30),10)
        np.random.seed(4);b=np.random.choice(30,10,replace=False)
        np.testing.assert_array_equal(a,b)

    def test_invalid(self):
        with self.assertRaises(ValueError):select_correspondences([[1,2]],[.5],0)
        with self.assertRaises(ValueError):select_correspondences([[1,2]],[.5],1,'unknown')


class MetricsTests(unittest.TestCase):
    def test_known_translation_and_projection(self):
        poses=np.tile(np.eye(4),(3,1,1));poses[:,2,3]=1000;poses[:,0,3]=[-1,0,1]
        camera=CameraIntrinsics(1000,1000,1000,1000,500,500)
        metrics,series=dispersion(poses,np.array([[0,0,0],[10,10,0]]),camera)
        self.assertAlmostEqual(metrics['translation']['rms'],np.sqrt(2/3))
        self.assertAlmostEqual(metrics['screen']['rms'],np.sqrt(2/3))
        self.assertEqual(metrics['rotation']['rms'],0)

    def test_rotation_wraparound(self):
        from scipy.spatial.transform import Rotation
        poses=np.tile(np.eye(4),(2,1,1));poses[:,2,3]=1000
        poses[:,:3,:3]=Rotation.from_euler('z',[179,-179],degrees=True).as_matrix()
        metrics,_=dispersion(poses,np.array([[1,0,0]]),CameraIntrinsics(1000,1000,1000,1000,500,500))
        self.assertAlmostEqual(metrics['rotation']['rms'],1)


class HookSafetyTests(unittest.TestCase):
    def fixture(self):
        opts_type=namedtuple('Opts','num_iterations_test crop_size pnp_opts')
        pnp_type=namedtuple('PnPOpts','max_num_corresps',defaults=[10000])
        def original_pnp():
            return {}
        forward=lambda *args,**kwargs: {}
        pnp=ModuleType('utils.pnp_util');pnp.poses_from_correspondences=original_pnp
        utils=ModuleType('utils');utils.pnp_util=pnp
        config=ModuleType('utils.config');config.PnPOpts=pnp_type
        model=SimpleNamespace(opts=opts_type(5,(280,280),None),forward_pipeline=forward)
        return SimpleNamespace(model=model),pnp,{'utils':utils,'utils.config':config},forward,original_pnp

    def test_restores_options_and_functions_after_exception(self):
        backend,pnp,modules,forward,original=self.fixture();opts=backend.model.opts
        with patch.dict(sys.modules,modules):
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with experiment(backend,{'num_iterations_test':2,'crop_size':[336,336]},{}):
                    self.assertEqual(backend.model.opts.num_iterations_test,2)
                    self.assertEqual(backend.model.opts.crop_size,(336,336))
                    self.assertIsNot(pnp.poses_from_correspondences,original)
                    raise RuntimeError('intentional')
        self.assertIs(backend.model.opts,opts)
        self.assertIs(backend.model.forward_pipeline,forward)
        self.assertIs(pnp.poses_from_correspondences,original)

    def test_changed_upstream_sampling_fails_closed_and_restores(self):
        backend,pnp,modules,forward,original=self.fixture();opts=backend.model.opts
        with patch.dict(sys.modules,modules):
            with self.assertRaisesRegex(RuntimeError,'Upstream PnP changed'):
                with experiment(backend,{'correspondence':'top_confidence','num_iterations_test':2},{}):
                    self.fail('Unexpectedly patched unknown upstream source')
        self.assertIs(backend.model.opts,opts)
        self.assertIs(pnp.poses_from_correspondences,original)
