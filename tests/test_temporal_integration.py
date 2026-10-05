"""Exercise application persistence, export and viewer using accepted raw records."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import trimesh
from object_tracker.app.service import Application
from object_tracker.core.types import FrameResult, FrameStatus, Pose
from object_tracker.core.viewer import render


class TemporalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        frames = root/'frames'; frames.mkdir()
        for i in range(21):
            cv2.imwrite(str(frames/f'{i:04}.png'), np.zeros((48,64,3),np.uint8))
        mesh = root/'mesh.ply'; trimesh.creation.box().export(mesh)
        self.app = Application(lambda _: self.fail('Temporal processing must not load the tracker'))
        self.app.load(dict(directory=root/'project',source=frames,mesh=mesh),create=True)
        for i in range(21):
            m = np.eye(4); m[:3,3] = [(-1.)**i,0,700]
            self.app.project.store_result(FrameResult(i,Pose(m),.9,status=FrameStatus.TRACKED), 'test')
        self.raw = copy.deepcopy(self.app.project.state['poses'])
        self.app.project.save()

    def tearDown(self):
        self.app.close(); self.tmp.cleanup()

    def test_recompute_without_tracker_exports_and_reload(self):
        state = self.app.temporal(dict(parameters={'enabled':True},preview='filtered'))['project']
        filtered = copy.deepcopy(state['temporal']['filtered_poses'])
        self.assertEqual(state['poses'], self.raw)
        raw_export = Path(self.app.export()['directory'])
        filtered_export = Path(self.app.export({'pose_source':'filtered'})['directory'])
        np.testing.assert_array_equal(np.load(raw_export/'poses.npy'),[e['matrix'] for e in self.raw.values()])
        np.testing.assert_array_equal(np.load(filtered_export/'poses.npy'),list(filtered.values()))
        self.assertEqual(json.loads((filtered_export/'poses.json').read_text())['pose_source'],'filtered')
        package = Path(self.app.export_nuke({'pose_source':'filtered'})['directory'])
        metadata = json.loads((package/'animation.json').read_text())
        self.assertEqual(metadata['pose_source'], 'filtered')
        self.assertEqual(len(metadata['samples']),21)
        self.app.load(dict(directory=self.app.project.directory))
        self.assertEqual(self.app.snapshot()['project']['temporal']['filtered_poses'],filtered)
        self.app.temporal(dict(parameters={'strength':.1}))
        self.assertNotEqual(self.app.project.state['temporal']['filtered_poses'],filtered)
        self.assertEqual(self.app.project.state['poses'],self.raw)
        self.assertIsNone(self.app.backend)

    def test_viewer_raw_filtered_and_manual_edits_invalidate(self):
        self.app.temporal(dict(parameters={'enabled':True}))
        with patch('object_tracker.core.viewer.draw_pose') as draw:
            render(self.app.project,self.app.mesh,10,initial=False,pose_source='filtered')
            np.testing.assert_array_equal(draw.call_args.args[2],self.app.project.state['temporal']['filtered_poses']['10'])
            render(self.app.project,self.app.mesh,10,initial=False)
            np.testing.assert_array_equal(draw.call_args.args[2],self.raw['10']['matrix'])
        self.app.change_pose(dict(index=10,values=[0,0,0,10,0,700],anchor=True))
        state = self.app.snapshot()['project']
        self.assertEqual(state['poses']['10']['matrix'][0][3],10)
        self.assertNotEqual(state['temporal']['filtered_poses']['10'],self.raw['10']['matrix'])
        self.app.temporal(dict(parameters={'enabled':False}))
        with self.assertRaises(ValueError): self.app.export({'pose_source':'filtered'})
        self.app.export()  # Existing raw export still works.

    def test_invalid_settings_leave_state_intact(self):
        self.app.temporal(dict(parameters={'enabled':True}))
        before = copy.deepcopy(self.app.project.state)
        with self.assertRaises(ValueError): self.app.temporal(dict(parameters={'strength':-1}))
        self.assertEqual(before,self.app.project.state)
