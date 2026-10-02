import tempfile
import threading
import time
from pathlib import Path
import unittest
import cv2
import numpy as np
import trimesh
from object_tracker.app.service import Application
from object_tracker.backends.base import BackendCapabilities
from object_tracker.core.types import FrameResult, FrameStatus, Pose


class FakeBackend:
    backend_id = 'test'
    version = '1'
    capabilities = BackendCapabilities()
    def __init__(self):
        self.calls = []
        self.after_frame = None
    def initialize(self): pass
    def shutdown(self): pass
    def refine_frame(self, image, mesh, camera, initial, index, *masks):
        self.calls.append(index)
        matrix = initial.T_cam_from_object.copy(); matrix[0,3] += 1
        if self.after_frame: self.after_frame(index)
        return FrameResult(index, Pose(matrix), .9, 1., 0., FrameStatus.TRACKED, .01)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)
        source = root / 'source'; source.mkdir()
        for i in range(5): cv2.imwrite(str(source / f'{i:04d}.png'), np.zeros((48,64,3), np.uint8))
        mesh = root / 'mesh.ply'; trimesh.creation.box().export(mesh)
        self.backend = FakeBackend(); self.app = Application(lambda output: self.backend)
        self.app.load(dict(directory=root/'project', source=source, mesh=mesh), create=True)
        self.app.change_pose(dict(index=0, values=[0,0,0,0,0,700], anchor=True))
    def tearDown(self):
        self.app.close(); self.tmp.cleanup()
    def wait(self):
        deadline = time.monotonic() + 5
        while self.app.snapshot()['busy'] and time.monotonic()<deadline: time.sleep(.01)
        self.assertFalse(self.app.snapshot()['busy'])
        self.assertIsNone(self.app.snapshot()['error'])
    def test_refine_requires_accept_and_reject_preserves_pose(self):
        self.app.refine(0); self.wait()
        p=self.app.project.state
        self.assertEqual(p['poses']['0']['matrix'][0][3], 0)
        self.assertEqual(p['refinements']['0']['matrix'][0][3], 1)
        self.app.accept(0, False)
        self.assertEqual(p['poses']['0']['matrix'][0][3], 0)
        self.app.refine(0); self.wait(); self.app.accept(0)
        self.assertEqual(p['anchors']['0']['matrix'][0][3], 1)
        self.assertEqual(p['poses']['0']['matrix'][0][3], 1)
    def test_cancel_reopen_resume_retrack_only_range(self):
        self.backend.after_frame=lambda index:self.app.cancel.set()
        self.app.track(dict(start=0,end=4)); self.wait()
        self.assertEqual(self.backend.calls,[0])
        self.assertEqual(self.app.project.state['job']['status'],'cancelled')
        self.app.load(dict(directory=self.app.project.directory))
        self.backend.after_frame=None
        self.app.track({},resume=True);self.wait()
        self.assertEqual(self.backend.calls,[0,1,2,3,4])
        self.assertEqual(self.app.project.state['poses']['1']['translation_delta_mm'],1)
        before=self.app.project.state['poses']['4'].copy()
        self.app.change_pose(dict(index=1,values=[0,0,0,20,0,700],anchor=True))
        self.app.track(dict(start=1,end=2));self.wait()
        self.assertEqual(self.app.project.state['poses']['4'],before)
        self.assertEqual(self.app.project.state['poses']['2']['matrix'][0][3],22)
        files=list((self.app.project.directory/'poses').glob('*/*.json'))
        self.assertGreater(len(files),5)
    def test_export_preserves_manual_pose_precision(self):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = 0.123456789012345
        matrix[2, 3] = 700
        self.app.change_pose(dict(index=0, matrix=matrix.tolist(), anchor=True))
        exported = self.app.export()
        actual = np.load(Path(exported['directory']) / 'poses.npy')
        np.testing.assert_array_equal(actual[0], matrix)

    def test_failure_is_persisted_and_can_resume(self):
        def fail(index):
            if index==1: raise RuntimeError('test failure')
        self.backend.after_frame=fail
        self.app.track(dict(start=0,end=3))
        deadline=time.monotonic()+5
        while self.app.snapshot()['busy'] and time.monotonic()<deadline:time.sleep(.01)
        self.assertEqual(self.app.project.state['job']['completed'],[0])
        self.assertEqual(self.app.project.state['job']['status'],'failed')
        self.backend.after_frame=None
        self.app.track({},resume=True);self.wait()
        self.assertEqual(self.app.project.state['job']['completed'],[0,1,2,3])
