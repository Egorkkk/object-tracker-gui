"""Portable classic Nuke 3D scene export; core poses and assets remain unchanged."""
import json
import math
import uuid
import zipfile
import numpy as np
import trimesh
from .storage import atomic_json
from .types import CameraIntrinsics, Pose

BASIS = np.diag([1., -1., -1., 1.])


def nuke_matrix(matrix, relative_scale):
    converted = BASIS @ Pose(np.asarray(matrix)).T_cam_from_object @ BASIS
    converted[:3, 3] *= relative_scale
    return converted


def _number(value):
    return format(float(value), '.17g')


def _matrix_curves(samples):
    # Matrix knobs serialize four row vectors, unlike nuke.math.Matrix4 indexing.
    rows = []
    for row in range(4):
        components = []
        for col in range(4):
            keys = ' '.join('x%d %s' % (frame, _number(matrix[row, col])) for frame, matrix in samples)
            components.append('{curve L ' + keys + '}')
        rows.append('{' + ' '.join(components) + '}')
    return ' '.join(rows)


def export_nuke(project, relative_scale=1., first_frame=1):
    relative_scale = float(relative_scale)
    if not math.isfinite(relative_scale) or relative_scale <= 0:
        raise ValueError('Relative scale должен быть положительным конечным числом')
    if isinstance(first_frame, bool) or int(first_frame) != first_frame:
        raise ValueError('Первый кадр Nuke должен быть целым числом')
    first_frame = int(first_frame)
    state = project.state
    if not state.get('mesh') or not state.get('camera'):
        raise ValueError('Для Nuke export нужны mesh и camera')
    if not state['poses']:
        raise ValueError('Нет принятых poses для экспорта. Примите Refine или сохраните Anchor.')
    camera = CameraIntrinsics(**state['camera'])
    if camera.distortion and any(camera.distortion):
        raise ValueError('Nuke export поддерживает pinhole camera без distortion')
    entries = sorted(state['poses'].items(), key=lambda item: int(item[0]))
    samples = [(first_frame+project.check_index(int(key)), nuke_matrix(entry['matrix'], relative_scale)) for key, entry in entries]
    mesh = trimesh.load(state['mesh']['prepared'])
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError('Нужен подготовленный triangle mesh')
    mesh.apply_transform(np.diag([relative_scale, -relative_scale, -relative_scale, 1.]))
    folder = project.directory / 'exports' / ('nuke-' + uuid.uuid4().hex)
    folder.mkdir(parents=True)
    mesh.export(folder/'mesh.obj', include_texture=False)
    width, height = camera.width, camera.height
    focal = 50.
    # Nuke film-window offsets use horizontal film width, including Y.
    # win_scale compensates unequal fx/fy for a square-pixel output format.
    wx = 1-2*camera.cx/width
    wy = (2*camera.cy-height)/width * camera.fx/camera.fy
    scene = f'''# Object Tracker: OpenCV mm -> Nuke +Y up, -Z forward.
Root {{
 first_frame {first_frame}
 last_frame {first_frame+len(state['source']['frames'])-1}
 fps {_number(state['source']['fps'])}
 format "{width} {height} 0 0 {width} {height} 1 ObjectTracker"
}}
Axis2 {{
 inputs 0
 name ObjectTracker_Axis
 useMatrix true
 matrix {{{_matrix_curves(samples)}}}
 xpos 0
 ypos 0
}}
set tracker_axis [stack 0]
ReadGeo2 {{
 inputs 0
 file "[file join [file dirname [value root.name]] mesh.obj]"
 name ObjectTracker_Mesh
 xpos 0
 ypos 100
}}
set tracker_mesh [stack 0]
push $tracker_axis
push $tracker_mesh
TransformGeo {{
 inputs 2
 name ObjectTracker_Transform
 xpos 0
 ypos 150
}}
Scene {{
 inputs 1
 name ObjectTracker_Scene
 xpos 0
 ypos 200
}}
set tracker_scene [stack 0]
Camera2 {{
 inputs 0
 name ObjectTracker_Camera
 focal {_number(focal)}
 haperture {_number(focal*width/camera.fx)}
 vaperture {_number(focal*height/camera.fy)}
 win_translate {{{_number(wx)} {_number(wy)}}}
 win_scale {{1 {_number(camera.fx/camera.fy)}}}
 near {_number(max(relative_scale*.01,1e-7))}
 far {_number(max(1000000*relative_scale,1))}
 xpos 250
 ypos 200
}}
set tracker_camera [stack 0]
push $tracker_camera
push $tracker_scene
push 0
ScanlineRender {{
 inputs 3
 name ObjectTracker_Render
 xpos 0
 ypos 300
}}
'''
    (folder/'scene.nk').write_text(scene, encoding='utf-8')
    atomic_json(folder/'animation.json', dict(version=1, relative_scale=relative_scale,
        first_frame=first_frame, fps=state['source']['fps'], camera=state['camera'],
        convention='T_nuke_camera_from_object; +X right, +Y up, -Z forward',
        source_indices=[int(key) for key, _ in entries],
        samples=[dict(frame=frame,matrix=matrix.tolist()) for frame,matrix in samples]))
    (folder/'import_nuke.py').write_text('''"""Run this file in Nuke Script Editor to append the exported scene."""
import os
import nuke

def import_tracking(folder):
    folder = os.path.abspath(folder)
    before = set(nuke.allNodes())
    nuke.nodePaste(os.path.join(folder, 'scene.nk'))
    nodes = [node for node in nuke.allNodes() if node not in before]
    for node in nodes:
        if node.Class() == 'ReadGeo2':
            node['file'].setValue(os.path.join(folder, 'mesh.obj').replace('\\\\', '/'))
    return nodes

if __name__ == '__main__':
    import_tracking(os.path.dirname(os.path.abspath(__file__)))
''', encoding='utf-8')
    (folder/'README.txt').write_text('''Extract the ZIP into a single folder, then open scene.nk in Nuke.
To append nodes to an existing script, execute import_nuke.py in Nuke's Script
Editor with __file__ set to its path, or import its import_tracking(folder).
The root frame range/FPS/format apply when opening scene.nk; when appending nodes,
set the destination script format/FPS/frame range to animation.json metadata.
Relative scale multiplies both prepared mesh vertices and pose translation.
1 means millimeter scene units; 0.001 means meter scene units.
Frames map as Nuke frame = sequence index + first_frame; gaps are preserved in
key numbering and linearly interpolated by Nuke. Only accepted/manual poses are
exported; pending refinements and unsaved drafts are excluded.
This is object motion relative to a fixed camera, not a solved world camera.
OBJ exports geometry only; original material/texture reconstruction is excluded.
Classic Axis2/ReadGeo2/TransformGeo/Camera2/ScanlineRender nodes are used.
''', encoding='utf-8')
    with zipfile.ZipFile(folder/'nuke_package.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.suffix != '.zip':
                archive.write(path, path.name)
    return folder
