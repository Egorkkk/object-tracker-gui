import io
import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
import trimesh
from object_tracker.app.service import Application
from object_tracker.app.resources import InvalidationRequired


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.a=self.root/'frames_a'; self.b=self.root/'frames_b'
        for directory,size in ((self.a,(64,48)),(self.b,(80,60))):
            directory.mkdir()
            for i in range(3):
                cv2.imwrite(str(directory/f'shot_{1001+i}.png'),np.zeros((size[1],size[0],3),np.uint8))
        self.mesh=self.root/'mesh.ply'; trimesh.creation.box(extents=[75,11,160]).export(self.mesh)
        self.app=Application(lambda output:None)
        self.app.load(dict(directory=self.root/'project',name='empty'),create=True)
    def tearDown(self):
        self.app.close();self.temp.cleanup()
    def test_empty_then_assign_and_reopen(self):
        self.assertIsNone(self.app.project.state['camera'])
        self.app.load(dict(directory=self.root/'project'))
        self.app.replace_resource(dict(kind='source',path=str(self.a)))
        self.app.replace_resource(dict(kind='mesh',path=str(self.mesh)))
        self.assertEqual(self.app.project.state['camera']['width'],64)
        self.assertIsNotNone(self.app.mesh)
    def test_invalidation_requires_explicit_choice_and_solution_restores_context(self):
        self.app.replace_resource(dict(kind='source',path=str(self.a)))
        self.app.replace_resource(dict(kind='mesh',path=str(self.mesh)))
        self.app.change_pose(dict(index=0,values=[0,0,0,2,3,700],anchor=True))
        before=self.app.snapshot()['project']
        with self.assertRaises(InvalidationRequired):
            self.app.replace_resource(dict(kind='source',path=str(self.b)))
        self.assertEqual(self.app.snapshot()['project'],before)
        self.app.replace_resource(dict(kind='source',path=str(self.b),invalidation='preserve'))
        state=self.app.project.state
        self.assertEqual(state['camera']['width'],80)
        self.assertFalse(state['anchors']);self.assertFalse(state['poses'])
        identity=state['solutions'][0]['id']
        self.app.activate_solution(identity)
        self.assertEqual(self.app.project.state['camera']['width'],64)
        self.assertEqual(self.app.project.state['poses'],before['poses'])
        self.assertEqual(self.app.project.state['source'],before['source'])
    def test_failed_replacement_does_not_clear_results(self):
        self.app.replace_resource(dict(kind='source',path=str(self.a)))
        self.app.change_pose(dict(index=0,values=[0,0,0,2,3,700],anchor=True))
        before=self.app.snapshot()['project']
        with self.assertRaises(Exception):
            self.app.replace_resource(dict(kind='mesh',path=str(self.root/'missing.ply'),invalidation='clear'))
        self.assertEqual(self.app.snapshot()['project'],before)
    def test_upload_is_project_local_unique_and_atomic(self):
        batch=self.app.begin_upload('mesh');data=b'fake mesh bytes'
        result=self.app.upload_file(batch['id'],'mesh.ply',io.BytesIO(data),len(data))
        self.assertTrue(Path(result['path']).is_relative_to(self.app.project.directory/'cache/assets'))
        self.assertEqual(Path(result['path']).read_bytes(),data)
        with self.assertRaises(ValueError):
            self.app.upload_file(batch['id'],'mesh.ply',io.BytesIO(b'new'),3)
        with self.assertRaises(ValueError):
            self.app.upload_file(batch['id'],'../escape',io.BytesIO(b'x'),1)
        with self.assertRaises(ValueError):
            self.app.upload_file(batch['id'],'partial.ply',io.BytesIO(b'x'),10)
        self.assertFalse((Path(batch['directory'])/'partial.ply').exists())
        self.assertFalse(self.app.busy)
        second=self.app.begin_upload('mesh')
        result2=self.app.upload_file(second['id'],'mesh.ply',io.BytesIO(b'new'),3)
        self.assertNotEqual(result['path'],result2['path'])
