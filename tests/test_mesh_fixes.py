import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from object_tracker.core.project import prepare_mesh
from object_tracker.core.mesh import load_mesh
from object_tracker.core.viewer import draw_pose
from object_tracker.core.types import CameraIntrinsics


class MeshFixTests(unittest.TestCase):
    def test_units_relative_scale_and_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = trimesh.creation.box(extents=[.075, .011, .160])
            mesh.apply_translation([.2, .3, .4])
            source = root/'phone.glb'; source.write_bytes(trimesh.Scene(mesh).export(file_type='glb'))
            before = source.read_bytes()
            first = prepare_mesh(source, root/'cache', center=True)
            np.testing.assert_allclose(first['dimensions_mm'], [75,11,160], atol=1e-4)
            np.testing.assert_allclose(first['bounds_mm'][0], -np.asarray(first['bounds_mm'][1]), atol=1e-5)
            second = prepare_mesh(source, root/'cache', scale=10, base=first)
            third = prepare_mesh(source, root/'cache', scale=10, base=second)
            np.testing.assert_allclose(third['dimensions_mm'], [7500,1100,16000], atol=.1)
            np.testing.assert_allclose(third['source_transform'], np.diag([10,10,10,1]) @ second['source_transform'])
            self.assertEqual(source.read_bytes(),before)

    def test_large_partially_visible_triangles_do_not_disappear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'large.ply'
            trimesh.Trimesh(vertices=[[-1000,-1000,0],[1000,-1000,0],[0,1000,0]], faces=[[0,1,2]], process=False).export(path)
            mesh=load_mesh(path); camera=CameraIntrinsics(64,48,60,60,32,24)
            pose=np.eye(4);pose[2,3]=100
            for mode in ['contour','wireframe','shaded']:
                image=np.zeros((48,64,3),np.uint8)
                draw_pose(image,mesh,pose,camera,(0,255,255),mode)
                self.assertGreater(np.count_nonzero(image),0,mode)
            pose[0,3]=450
            image=np.zeros((48,64,3),np.uint8)
            draw_pose(image,mesh,pose,camera,(0,255,255),'shaded')
            self.assertGreater(np.count_nonzero(image),0)

    def test_contour_clips_triangle_crossing_near_plane(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'near.ply'
            trimesh.Trimesh(vertices=[[-10,-10,-2],[10,-10,-2],[0,10,10]],faces=[[0,1,2]],process=False).export(path)
            image=np.zeros((48,64,3),np.uint8)
            draw_pose(image,load_mesh(path),np.eye(4),CameraIntrinsics(64,48,60,60,32,24),(0,255,255))
            self.assertGreater(np.count_nonzero(image),0)
