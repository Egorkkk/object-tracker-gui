import tempfile
import unittest
import json
import zipfile
from pathlib import Path
import numpy as np
import cv2
import trimesh
from object_tracker.core.project import Project
from object_tracker.core.pose import make_pose
from object_tracker.core.types import CameraIntrinsics
from object_tracker.core.export_nuke import export_nuke, nuke_matrix, BASIS


class NukeExportTests(unittest.TestCase):
    def test_scaled_geometry_and_animation_preserve_camera_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frames=root/'frames';frames.mkdir()
            for i in range(3):cv2.imwrite(str(frames/f'{i}.png'),np.zeros((48,64,3),np.uint8))
            source=root/'box.ply';trimesh.creation.box(extents=[75,11,160]).export(source)
            p=Project.create(root/'project',frames,source,camera=dict(width=64,height=48,fx=60,fy=63,cx=29,cy=26))
            matrices=[make_pose(10,20,30,15,25,700).T_cam_from_object,make_pose(-15,40,-60,-5,10,720).T_cam_from_object]
            for key,m in zip(['0','2'],matrices):p.state['poses'][key]=dict(matrix=m.tolist(),status='TRACKED')
            before=json.dumps(p.state,sort_keys=True);original=source.read_bytes()
            folder=export_nuke(p,.001,1001)
            data=json.loads((folder/'animation.json').read_text())
            self.assertEqual([s['frame'] for s in data['samples']],[1001,1003])
            mesh=trimesh.load(folder/'mesh.obj');np.testing.assert_allclose(mesh.extents,[.075,.011,.160])
            vertices=trimesh.load(source).vertices
            for m in matrices:
                canonical=vertices@m[:3,:3].T+m[:3,3]
                nm=nuke_matrix(m,.001)
                obj=(vertices@BASIS[:3,:3].T)*.001
                converted=obj@nm[:3,:3].T+nm[:3,3]
                np.testing.assert_allclose(converted,(canonical@BASIS[:3,:3].T)*.001,atol=1e-12)
                np.testing.assert_allclose(converted[:,0]/-converted[:,2],canonical[:,0]/canonical[:,2])
                np.testing.assert_allclose(converted[:,1]/converted[:,2],canonical[:,1]/canonical[:,2])
            self.assertEqual(json.dumps(p.state,sort_keys=True),before);self.assertEqual(source.read_bytes(),original)
            with zipfile.ZipFile(folder/'nuke_package.zip') as archive:
                self.assertIn('scene.nk',archive.namelist());self.assertIn('mesh.obj',archive.namelist())
            compile((folder/'import_nuke.py').read_text(), 'import_nuke.py','exec')
            for scale in [0,-1,float('nan'),float('inf')]:
                with self.assertRaises(ValueError):export_nuke(p,scale)
            with self.assertRaises(ValueError):export_nuke(p,1,1.5)
            p.state['poses']={}
            with self.assertRaises(ValueError):export_nuke(p)
