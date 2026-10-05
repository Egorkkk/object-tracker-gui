"""Versioned project state; source assets are read-only, derived files are local."""
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import uuid
import hashlib
import numpy as np
import trimesh

from .storage import atomic_json, local_path, separate_output
from .sequence import ImageSequence, discover, extract_video, VIDEO_SUFFIXES
from .types import CameraIntrinsics, Pose


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_mesh(source, cache, dimensions=None, center=False, scale=None, units=None, base=None):
    source = local_path(source)
    cache = separate_output(cache, [source])
    cache.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(base['prepared'] if base else source)
    detected_units = base.get('detected_units') if base else mesh.units
    source_units = units or (base.get('source_units') if base else None) or detected_units or ('meters' if source.suffix.lower() in ('.glb', '.gltf') else 'millimeters')
    unit_factor = 1.0 if base else trimesh.units.unit_conversion(source_units, 'millimeters')
    # New scene preparation is separate from the legacy baseline loader.
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('Нужен непустой треугольный mesh')
    original_extents = mesh.extents.copy()
    transform = np.eye(4)
    if center:
        transform[:3, 3] = -mesh.bounds.mean(axis=0)
    factors = np.full(3, unit_factor)
    if dimensions is not None:
        dimensions = np.asarray(dimensions, dtype=float)
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0) or np.any(original_extents <= 0):
            raise ValueError('Размеры mesh должны быть тремя положительными числами')
        factors = dimensions / original_extents
    elif scale is not None:
        factors *= np.broadcast_to(np.asarray(scale, dtype=float), (3,))
        if not np.isfinite(factors).all() or np.any(factors <= 0):
            raise ValueError('Масштаб должен быть положительным')
    transform[:3, :] *= factors[:, None]
    identity = np.array_equal(transform, np.eye(4))
    source_transform = transform @ np.asarray(base['source_transform']) if base else transform
    target = cache / (uuid.uuid4().hex + '.ply')
    if source.suffix.lower() == '.ply' and identity and base is None:
        shutil.copy2(source, target)  # Preserve verified geometry and appearance byte-for-byte.
    else:
        mesh.apply_transform(transform)
        if mesh.visual.kind == 'texture':
            mesh.visual = mesh.visual.to_color()
        mesh.export(target)
    return dict(source=str(source), prepared=str(target), dimensions_mm=mesh.extents.tolist(),
                source_transform=source_transform.tolist(), bounds_mm=mesh.bounds.tolist(), detected_units=detected_units,
                source_units=source_units, prepared_units='millimeters', vertices=len(mesh.vertices), faces=len(mesh.faces),
                source_sha256=file_identity(source), prepared_sha256=file_identity(target))


class Project:
    def __init__(self, directory, state):
        self.directory = Path(directory)
        self.state = state
        state.setdefault('solutions', [])
        state.setdefault('exports', [])
        self.sequence = ImageSequence(state['source']['frames'])

    @classmethod
    def create(cls, directory, source=None, mesh=None, name='Untitled', camera=None, fps=24.,
               dimensions=None, center=False, scale=None):
        sources = [local_path(value) for value in (source, mesh) if value]
        directory = separate_output(directory, sources)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Для нового проекта выберите пустой каталог')
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('FPS должен быть положительным')
        if source and not local_path(source).exists():
            raise ValueError('Input должен существовать')
        if mesh and not local_path(mesh).is_file():
            raise ValueError('Mesh должен существовать')
        directory.mkdir(parents=True, exist_ok=True)
        for sub in ('cache', 'poses', 'overlays', 'diagnostics', 'previews', 'exports', 'logs'):
            (directory / sub).mkdir(exist_ok=True)
        source_data = dict(type='none', path='', frames=[], fps=fps)
        intrinsics = None
        if source:
            source = local_path(source)
            if source.is_dir():
                frames, source_type = discover(source), 'image_sequence'
            elif source.suffix.lower() in VIDEO_SUFFIXES:
                frames, fps = extract_video(source, directory / 'cache/frames')
                source_type = 'video'
            else:
                raise ValueError('Выберите каталог кадров либо MP4/MOV/MKV')
            h, w = ImageSequence(frames).rgb(0).shape[:2]
            camera = camera or dict(width=w, height=h, fx=w*35/36, fy=w*35/36, cx=w/2, cy=h/2)
            intrinsics = CameraIntrinsics(**camera)
            if (intrinsics.width, intrinsics.height) != (w, h):
                raise ValueError('Размер камеры не совпадает с кадром')
            source_data = dict(type=source_type, path=str(source), frames=frames, fps=fps)
        prepared = prepare_mesh(mesh, directory / 'cache/meshes', dimensions, center, scale) if mesh else None
        state = dict(version=1, application_version='0.2.0', name=name, source=source_data,
                     mesh=prepared, camera=asdict(intrinsics) if intrinsics else None, backend=dict(id='gotrack'),
                     current_frame=0, poses={}, anchors={}, drafts={}, refinements={}, job=None,
                     masks=dict(object_mask={}, occlusion_mask={}), mask_sources={}, solutions=[], exports=[],
                     thresholds=dict(min_score=0.1, max_translation_mm=50., max_rotation_deg=30.))
        project = cls(directory, state)
        project.save()
        return project

    @classmethod
    def open(cls, directory):
        directory = local_path(directory)
        if directory.name == 'project.json':
            directory = directory.parent
        state = json.loads((directory / 'project.json').read_text())
        if state.get('version') != 1:
            raise ValueError('Неподдерживаемая версия проекта')
        if state.get('camera'):
            CameraIntrinsics(**state['camera'])
        job = state.get('job')
        if job and job['status'] in ('running', 'cancelling', 'queued'):
            job['status'] = 'interrupted'
        if job and job.get('kind') == 'track' and job.get('resumable'):
            # A per-frame record may have committed just before project.json.
            # Recover the contiguous journal prefix without re-running inference.
            attempt = str(job['id'])
            if not attempt.isalnum():
                raise ValueError('Недопустимый ID tracking job')
            for index in job['indices']:
                if index in job['completed']:
                    continue
                record = directory / 'poses' / attempt / f'{index:06d}.json'
                if not record.is_file():
                    break
                entry = json.loads(record.read_text())
                Pose(np.asarray(entry['matrix']))
                state['poses'][str(index)] = entry
                job['completed'].append(index)
                job['current_pose'] = entry['matrix']
                job['previous_pose'] = entry['matrix']
                job['current_frame'] = index
                job['score'] = entry['score']
                job['elapsed'] += entry.get('runtime') or 0.
            if len(job['completed']) == len(job['indices']):
                job['status'] = 'completed'
                job['resumable'] = False
        return cls(directory, state)

    def save(self):
        from object_tracker.temporal.state import invalidate
        invalidate(self.state)
        atomic_json(self.directory / 'project.json', self.state)

    def check_index(self, index):
        index = int(index)
        if not 0 <= index < len(self.state['source']['frames']):
            raise ValueError('Кадр вне диапазона')
        return index

    def pose(self, index):
        key = str(self.check_index(index))
        for field in ('drafts', 'anchors', 'poses'):
            if key in self.state[field]:
                return Pose(np.array(self.state[field][key]['matrix']))
        raise ValueError('Сначала задайте pose выбранного кадра')

    def store_result(self, result, attempt, initial=None):
        entry = dict(matrix=result.pose.T_cam_from_object.tolist(), score=result.score,
                     translation_delta_mm=result.translation_delta_mm,
                     rotation_delta_deg=result.rotation_delta_deg, status=result.status.value,
                     runtime=result.runtime, backend_diagnostics=result.backend_diagnostics,
                     attempt=attempt, initial_matrix=initial.T_cam_from_object.tolist() if initial else None)
        folder = self.directory / 'poses' / attempt
        folder.mkdir(parents=True, exist_ok=True)
        # Per-frame JSON is the durable raw record. Re-tracking gets a new attempt.
        atomic_json(folder / f'{result.frame_index:06d}.json', entry)
        self.state['poses'][str(result.frame_index)] = entry
        return entry
