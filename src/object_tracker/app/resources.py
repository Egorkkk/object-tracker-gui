"""Resource lifecycle and uploads; deliberately independent of inference code."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import uuid
import numpy as np

from object_tracker.core.project import Project, prepare_mesh
from object_tracker.core.mesh import load_mesh
from object_tracker.core.sequence import discover, extract_video, ImageSequence, VIDEO_SUFFIXES
from object_tracker.core.storage import atomic_json, local_path
from object_tracker.core.types import CameraIntrinsics


class InvalidationRequired(ValueError):
    pass


class ResourceOperations:
    def _dependency_choice(self, kind, decision):
        state = self.required().state
        fields = ('poses', 'refinements', 'anchors', 'drafts') if kind != 'camera' else ('poses', 'refinements')
        affected = any(state.get(field) for field in fields)
        if kind == 'source':
            affected = affected or bool(state.get('camera')) or any(state['masks'].values())
        if affected and decision not in ('preserve', 'clear'):
            raise InvalidationRequired('Изменение ресурса затрагивает существующие данные. Выберите Preserve или Clear.')
        return affected

    def _keep_solution(self, reason):
        p = self.required()
        identity = uuid.uuid4().hex
        relative = 'solutions/' + identity + '.json'
        atomic_json(p.directory / relative, deepcopy(p.state))
        entry = dict(id=identity, name=reason, path=relative,
                     created=datetime.now(timezone.utc).isoformat(), frames=len(p.state['poses']),
                     source=p.state['source']['path'], mesh=(p.state.get('mesh') or {}).get('source'))
        p.state.setdefault('solutions', []).append(entry)
        return entry

    def replace_resource(self, body):
        with self.lock:
            self.idle(); p = self.required()
            kind = body['kind']
            if kind not in ('source', 'mesh', 'camera'):
                raise ValueError('Неизвестный ресурс')
            affected = self._dependency_choice(kind, body.get('invalidation'))
            # Prepare and validate first. No active state is changed on failure.
            if kind == 'source':
                source = local_path(body['path'])
                if source.is_dir():
                    frames = discover(source)
                    fps = float(body.get('fps', p.state['source']['fps']))
                    source_type = 'image_sequence'
                elif source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
                    frames, fps = extract_video(source, p.directory / 'cache' / ('frames-' + uuid.uuid4().hex))
                    source_type = 'video'
                else:
                    raise ValueError('Выберите каталог кадров или видео MP4/MOV/MKV')
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError('FPS должен быть положительным')
                sequence = ImageSequence(frames)
                h, w = sequence.rgb(0).shape[:2]
                camera = CameraIntrinsics(w, h, w*35/36, w*35/36, w/2, h/2)
                prepared_source = dict(type=source_type, path=str(source), frames=frames, fps=fps)
            elif kind == 'mesh':
                options = body.get('options', {})
                source = body.get('path') or (p.state.get('mesh') or {}).get('source')
                if not source:
                    raise ValueError('Сначала выберите mesh')
                current = p.state.get('mesh')
                base = current if current and str(local_path(source)) == current['source'] and not options.get('units') and ('scale' in options or 'dimensions' in options) else None
                prepared_mesh = prepare_mesh(source, p.directory / 'cache/meshes', options.get('dimensions'),
                                             options.get('center', base is None), options.get('scale'),
                                             units=options.get('units'), base=base)
                mesh = load_mesh(prepared_mesh['prepared'])
            else:
                camera = CameraIntrinsics(**body['camera'])
                if not p.sequence.frames:
                    raise ValueError('Сначала выберите source')
                h, w = p.sequence.rgb(0).shape[:2]
                if (camera.width, camera.height) != (w, h):
                    raise ValueError('Размер камеры не совпадает с кадрами')
            if affected:
                if body.get('invalidation') == 'preserve':
                    self._keep_solution('До изменения ' + kind)
                else:
                    self._archive(p)
            for field in ('poses', 'refinements'):
                p.state[field] = {}
            p.state['job'] = None
            if kind in ('mesh', 'source'):
                p.state['anchors'] = {}; p.state['drafts'] = {}
            if kind == 'source':
                p.state['source'] = prepared_source
                p.state['camera'] = asdict(camera)
                p.state['camera_approximate'] = True
                p.state['masks'] = dict(object_mask={}, occlusion_mask={})
                p.state['mask_sources'] = {}
                p.state['current_frame'] = 0
                p.sequence = sequence
            elif kind == 'mesh':
                p.state['mesh'] = prepared_mesh
                self.mesh = mesh
            else:
                p.state['camera'] = asdict(camera)
                p.state['camera_approximate'] = False
            if kind in ('mesh', 'source') and p.state.get('mesh') and p.state.get('camera') and p.sequence.frames:
                bounds = np.asarray(p.state['mesh']['bounds_mm'])
                center = bounds.mean(axis=0)
                extents = bounds[1] - bounds[0]
                c = p.state['camera']
                distance = max(10., extents[0]*c['fx']/(c['width']*.45), extents[1]*c['fy']/(c['height']*.45)) + extents[2]/2
                matrix = np.eye(4); matrix[:3, 3] = [-center[0], -center[1], distance-center[2]]
                key = str(p.state.get('current_frame', 0))
                p.state['drafts'][key] = dict(matrix=matrix.tolist(), source='mesh_alignment')
            p.save()
        return self.snapshot()

    def activate_solution(self, identity):
        with self.lock:
            self.idle(); p = self.required()
            entry = next((s for s in p.state['solutions'] if s['id'] == identity), None)
            if entry is None:
                raise ValueError('Solution не найден')
            saved = json.loads((p.directory / entry['path']).read_text())
            mesh = load_mesh(saved['mesh']['prepared']) if saved.get('mesh') else None
            self._keep_solution('Предыдущий active solution')
            saved['solutions'] = deepcopy(p.state['solutions'])
            saved['exports'] = deepcopy(p.state.get('exports', []))
            # Historical jobs cannot be resumed against switched contexts implicitly.
            saved['job'] = None
            self.project = Project(p.directory, saved)
            self.mesh = mesh
            self.project.save()
        return self.snapshot()

    def begin_upload(self, kind):
        with self.lock:
            self.idle(); p = self.required()
            if kind not in ('source', 'mesh', 'object_mask', 'occlusion_mask', 'pose'):
                raise ValueError('Неизвестный тип upload')
            identity = uuid.uuid4().hex
            folder = p.directory / 'cache/assets' / kind / identity
            folder.mkdir(parents=True)
            if not hasattr(self, '_uploads'):
                self._uploads = {}
            self._uploads[identity] = dict(folder=folder, project=p.directory, kind=kind)
            return dict(id=identity, directory=str(folder))

    def upload_file(self, identity, name, stream, length):
        with self.lock:
            self.idle(); p = self.required()
            record = getattr(self, '_uploads', {}).get(identity)
            if not record or record['project'] != p.directory:
                raise ValueError('Upload session не найдена для этого проекта')
            if not name or name in ('.', '..') or '/' in name or '\\' in name or '\x00' in name:
                raise ValueError('Недопустимое имя файла')
            if length <= 0 or length > 8 * 1024**3:
                raise ValueError('Допустимый размер файла: от 1 байта до 8 GiB')
            target = record['folder'] / name
            temporary = record['folder'] / ('.' + uuid.uuid4().hex + '.part')
            if target.exists():
                raise ValueError('Файл с таким именем уже загружен; перезапись запрещена')
            self.busy = True; self.activity = 'Upload ' + name
        try:
            with temporary.open('xb') as output:
                remaining = length
                while remaining:
                    data = stream.read(min(1024 * 1024, remaining))
                    if not data:
                        raise ValueError('Передача файла прервана')
                    output.write(data); remaining -= len(data)
                output.flush(); os.fsync(output.fileno())
            # Atomic publication without overwriting an existing same-name file.
            os.link(temporary, target)
            return dict(path=str(target), directory=str(record['folder']), size=length)
        finally:
            temporary.unlink(missing_ok=True)
            with self.lock:
                self.busy = False; self.activity = None
