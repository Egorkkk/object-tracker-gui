import hashlib
import tempfile
from pathlib import Path
import unittest
import subprocess
import shutil
import cv2
import numpy as np
import trimesh
from object_tracker.core.project import Project, prepare_mesh
from object_tracker.core.types import Pose, FrameResult, FrameStatus
from object_tracker.core.storage import local_path
from object_tracker.core.masks import match_masks


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.frames = self.root / 'source'; self.frames.mkdir()
        for name in ('shot_1002.png', 'shot_1001.png', 'shot_1010.png'):
            cv2.imwrite(str(self.frames / name), np.zeros((48, 64, 3), np.uint8))
        self.mesh = self.root / 'source.ply'
        trimesh.creation.box(extents=[75, 11, 160]).export(self.mesh)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_and_source_safety(self):
        before = self.mesh.read_bytes()
        project = Project.create(self.root / 'project', self.frames, self.mesh)
        self.assertEqual(self.mesh.read_bytes(), before)
        self.assertEqual(Path(project.state['mesh']['prepared']).read_bytes(), before)
        project.state['job'] = dict(status='running')
        project.save()
        reopened = Project.open(project.directory)
        self.assertEqual(reopened.state['job']['status'], 'interrupted')
        self.assertEqual([f['source_number'] for f in reopened.sequence.frames], [1001, 1002, 1010])
        self.assertEqual(reopened.sequence.rgb(0).shape, (48, 64, 3))
        with self.assertRaises(ValueError):
            Project.create(self.frames / 'output', self.frames, self.mesh)
        with self.assertRaises(ValueError):
            Project.create(project.directory, self.frames, self.mesh)

    def test_recover_raw_record_committed_before_project_state(self):
        project = Project.create(self.root / 'project', self.frames, self.mesh)
        initial = np.eye(4); initial[2, 3] = 700
        project.state['job'] = dict(id='abc123', kind='track', status='running',
                                    resumable=True, indices=[0, 1], completed=[], elapsed=0.,
                                    current_pose=initial.tolist(), previous_pose=None)
        project.save()
        result_matrix = initial.copy(); result_matrix[0, 3] = 5
        project.store_result(FrameResult(0, Pose(result_matrix), .9, 0., 0.,
                                         FrameStatus.TRACKED, 1.5), 'abc123')
        # Simulate a crash before the project's next save.
        reopened = Project.open(project.directory)
        self.assertEqual(reopened.state['job']['completed'], [0])
        self.assertEqual(reopened.state['job']['elapsed'], 1.5)
        self.assertEqual(reopened.state['job']['current_pose'][0][3], 5)
        self.assertEqual(reopened.state['poses']['0']['matrix'][0][3], 5)

    def test_preparation_applies_scene_nodes_and_scales_only_copy(self):
        scene = trimesh.Scene()
        matrix = np.eye(4); matrix[0, 3] = 10
        scene.add_geometry(trimesh.creation.box(), transform=matrix)
        path = self.root / 'scene.glb'; scene.export(path)
        prepared = prepare_mesh(path, self.root / 'cache', [20, 30, 40], True)
        result = trimesh.load(prepared['prepared'])
        np.testing.assert_allclose(result.extents, [20, 30, 40])
        np.testing.assert_allclose(result.bounds.mean(axis=0), 0)
        self.assertGreater(trimesh.load(path).bounds.mean(axis=0)[0], 9)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg is unavailable')
    def test_video_extraction_stays_in_project(self):
        video = self.root / 'source.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                        'color=c=blue:s=64x48:r=12', '-frames:v', '4', '-pix_fmt', 'yuv420p', str(video)], check=True)
        before = video.read_bytes()
        project = Project.create(self.root / 'video-project', video, self.mesh)
        self.assertEqual(project.state['source']['fps'], 12.)
        self.assertEqual(len(project.sequence.frames), 4)
        self.assertTrue(all(Path(frame['path']).is_relative_to(project.directory / 'cache')
                            for frame in project.sequence.frames))
        self.assertEqual(video.read_bytes(), before)

    def test_mask_matching_and_windows_path(self):
        project = Project.create(self.root / 'project', self.frames, self.mesh)
        masks = self.root / 'masks'; masks.mkdir()
        cv2.imwrite(str(masks / 'object_1002.png'), np.ones((48, 64), np.uint8))
        mapping = match_masks(project.sequence.frames, masks, 'number')
        self.assertEqual(list(mapping), ['1'])
        self.assertEqual(str(local_path(r'E:\project\frames')), '/mnt/e/project/frames')
