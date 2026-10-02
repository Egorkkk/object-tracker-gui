"""One GPU worker, atomic project updates and resumable tracking jobs."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import logging
from pathlib import Path
import threading
import time
import uuid
import numpy as np

from object_tracker.core.project import Project, prepare_mesh
from object_tracker.core.mesh import load_mesh
from object_tracker.core.pose import make_pose
from object_tracker.core.types import CameraIntrinsics, Pose, FrameStatus, TrackingRange, FrameResult
from object_tracker.core.diagnostics import tracking_delta
from object_tracker.core.storage import atomic_json, local_path
from object_tracker.core.masks import match_masks, frame_masks
from object_tracker.core.export import export_poses, preview_video

log = logging.getLogger(__name__)


class Application:
    def __init__(self, backend_factory):
        self.backend_factory = backend_factory
        self.backend = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tracking')
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.project = None
        self.mesh = None
        self.busy = False
        self.error = None
        self.activity = None
        self.progress = None

    def snapshot(self):
        with self.lock:
            return dict(project=deepcopy(self.project.state) if self.project else None,
                        directory=str(self.project.directory) if self.project else None,
                        busy=self.busy, error=self.error, activity=self.activity, progress=deepcopy(self.progress))

    def idle(self):
        if self.busy:
            raise ValueError('Дождитесь завершения операции или отмените tracking')

    def required(self):
        if not self.project:
            raise ValueError('Сначала создайте или откройте проект')
        return self.project

    def load(self, body, create=False):
        with self.lock:
            self.idle()
            self.busy = True
            self.activity = 'Создание проекта' if create else 'Открытие проекта'
            self.error = None
        try:
            project = Project.create(**body) if create else Project.open(body['directory'])
            mesh = load_mesh(project.state['mesh']['prepared'])
            with self.lock:
                self.project, self.mesh = project, mesh
                project.save()
        finally:
            with self.lock:
                self.busy = False
                self.activity = None
        return self.snapshot()

    def change_pose(self, body):
        with self.lock:
            self.idle()
            p = self.required()
            index = p.check_index(body['index'])
            pose = Pose(np.asarray(body['matrix'])) if 'matrix' in body else make_pose(*body['values'])
            record = dict(matrix=pose.T_cam_from_object.tolist(), source='manual')
            p.state['drafts'][str(index)] = record
            p.state['current_frame'] = index
            if body.get('anchor'):
                p.state['anchors'][str(index)] = record
                result = FrameResult(index, pose, status=FrameStatus.MANUAL)
                p.store_result(result, 'manual-' + uuid.uuid4().hex)
            p.state['refinements'].pop(str(index), None)
            # Editing a starting pose invalidates an old continuation intent.
            if p.state.get('job') and p.state['job'].get('kind') == 'track':
                p.state['job']['resumable'] = False
            p.save()
        return self.snapshot()

    def accept(self, index, accept=True):
        with self.lock:
            self.idle()
            p = self.required(); key = str(p.check_index(index))
            result = p.state['refinements'].pop(key, None)
            if result is None:
                raise ValueError('Нет refinement для принятия/отклонения')
            if accept:
                p.state['anchors'][key] = dict(matrix=result['matrix'], source='refined')
                p.state['drafts'].pop(key, None)
                p.state['poses'][key] = result
                if p.state.get('job'):
                    p.state['job']['resumable'] = False
            p.save()
        return self.snapshot()

    def settings(self, body):
        with self.lock:
            self.idle(); p = self.required()
            if 'thresholds' in body:
                values = body['thresholds']
                if any(not np.isfinite(x) or x < 0 for x in values.values()):
                    raise ValueError('Пороги должны быть конечными неотрицательными числами')
                p.state['thresholds'].update(values)
            if 'camera' in body:
                camera = CameraIntrinsics(**body['camera'])
                h, w = p.sequence.rgb(0).shape[:2]
                if (camera.width, camera.height) != (w, h):
                    raise ValueError('Размер камеры не совпадает с кадрами')
                self._archive(p)
                p.state['camera'] = asdict(camera)
                p.state['poses'] = {}; p.state['refinements'] = {}; p.state['job'] = None
            if 'mesh' in body:
                options = body['mesh']
                prepared = prepare_mesh(options.get('source', p.state['mesh']['source']),
                                        p.directory / 'cache/meshes', options.get('dimensions'),
                                        options.get('center', False), options.get('scale'))
                self._archive(p)
                p.state['mesh'] = prepared
                self.mesh = load_mesh(prepared['prepared'])
                for field in ('poses', 'anchors', 'drafts', 'refinements'):
                    p.state[field] = {}
                p.state['job'] = None
            p.save()
        return self.snapshot()

    def _archive(self, p):
        atomic_json(p.directory / 'diagnostics' / ('state-' + uuid.uuid4().hex + '.json'), p.state)

    def masks(self, body):
        with self.lock:
            self.idle(); p = self.required()
            kind = body['kind']
            if kind not in ('object_mask', 'occlusion_mask'):
                raise ValueError('Неизвестный тип маски')
            directory = body.get('directory')
            mapping = match_masks(p.sequence.frames, local_path(directory), body.get('mode', 'filename')) if directory else {}
            p.state.setdefault('mask_sources', {})[kind] = dict(directory=str(local_path(directory)) if directory else '', mode=body.get('mode', 'filename'))
            p.state['masks'][kind] = mapping
            p.save()
        return self.snapshot()

    def _submit(self, activity, function):
        self.idle()
        self.required()
        self.busy = True; self.error = None; self.activity = activity; self.progress = None
        self.cancel.clear()
        self.executor.submit(self._run, function)

    def _run(self, function):
        try:
            function()
        except Exception as exc:
            log.exception('Operation failed: %s', self.activity)
            with self.lock:
                self.error = f'Операция не выполнена: {exc}'
                job = self.project.state.get('job') if self.project else None
                if job and job.get('status') in ('running', 'queued', 'cancelling'):
                    job['status'] = 'failed'; job['error'] = self.error
                    self.project.save()
        finally:
            with self.lock:
                self.busy = False; self.activity = None

    def _backend(self, p, attempt):
        if self.backend is None:
            self.backend = self.backend_factory(p.directory / 'cache/gotrack_internal' / attempt)
        self.backend.output = p.directory / 'cache/gotrack_internal' / attempt
        self.backend.initialize()
        with self.lock:
            p.state['backend'] = dict(id=self.backend.backend_id, version=self.backend.version,
                                     capabilities=asdict(self.backend.capabilities),
                                     settings=dict(masks_enabled=False, pose_units='mm'),
                                     metadata=getattr(self.backend, 'metadata', {}))
            p.save()
        return self.backend

    def refine(self, index):
        with self.lock:
            self.idle(); p = self.required(); index = p.check_index(index)
            initial = p.pose(index); mesh = self.mesh
            attempt = 'refine-' + uuid.uuid4().hex
            def work():
                backend = self._backend(p, attempt)
                masks = frame_masks(p, index)
                result = backend.refine_frame(p.sequence.rgb(index).copy(), mesh, CameraIntrinsics(**p.state['camera']),
                                              initial, index, masks.object_mask, masks.occlusion_mask)
                with self.lock:
                    old = p.state['poses'].get(str(index))
                    entry = p.store_result(result, attempt, initial)
                    if old is None:
                        p.state['poses'].pop(str(index))
                    else:
                        p.state['poses'][str(index)] = old
                    p.state['refinements'][str(index)] = entry
                    p.save()
            self._submit('Refine кадра', work)
        return self.snapshot()

    def track(self, body, resume=False):
        with self.lock:
            self.idle(); p = self.required(); mesh = self.mesh
            if resume:
                job = p.state.get('job')
                if not job or job.get('kind') != 'track' or not job.get('resumable') or job['status'] not in ('cancelled', 'interrupted', 'failed'):
                    raise ValueError('Нет диапазона для продолжения')
            else:
                start, end = p.check_index(body['start']), p.check_index(body['end'])
                initial = p.pose(start)
                job = dict(id=uuid.uuid4().hex, kind='track', start=start, end=end,
                           indices=list(TrackingRange(start, end).indices()), completed=[],
                           current_pose=initial.T_cam_from_object.tolist(), previous_pose=None,
                           resumable=True, respect_anchors=bool(body.get('respect_anchors', True)), elapsed=0.)
                if p.state.get('job'):
                    atomic_json(p.directory / 'diagnostics' / ('job-' + uuid.uuid4().hex + '.json'), p.state['job'])
                p.state['job'] = job
            job['status'] = 'queued'
            job.pop('error', None)
            p.save()
            def work():
                backend = self._backend(p, job['id'])
                camera = CameraIntrinsics(**p.state['camera'])
                current = Pose(np.array(job['current_pose'], np.float32))
                previous = Pose(np.array(job['previous_pose'], np.float32)) if job['previous_pose'] else None
                with self.lock:
                    job['status'] = 'running'; p.save()
                for index in job['indices']:
                    if index in job['completed']:
                        continue
                    if self.cancel.is_set():
                        break
                    anchor = p.state['anchors'].get(str(index))
                    if job['respect_anchors'] and index != job['start'] and anchor:
                        current = Pose(np.array(anchor['matrix'], np.float32))
                    masks = frame_masks(p, index)
                    result = backend.refine_frame(p.sequence.rgb(index).copy(), mesh, camera, current,
                                                  index, masks.object_mask, masks.occlusion_mask)
                    result.translation_delta_mm, result.rotation_delta_deg = tracking_delta(previous, result.pose)
                    thresholds = p.state['thresholds']
                    if (result.score < thresholds['min_score'] or result.translation_delta_mm > thresholds['max_translation_mm']
                            or result.rotation_delta_deg > thresholds['max_rotation_deg']):
                        result.status = FrameStatus.WARNING
                    propagated = current
                    previous = current = result.pose
                    with self.lock:
                        p.store_result(result, job['id'], propagated)
                        p.state['refinements'].pop(str(index), None)
                        job['completed'].append(index)
                        job['current_pose'] = current.T_cam_from_object.tolist()
                        job['previous_pose'] = previous.T_cam_from_object.tolist()
                        job['elapsed'] += result.runtime
                        job['current_frame'] = index; job['score'] = result.score
                        p.save()
                with self.lock:
                    job['status'] = 'completed' if len(job['completed']) == len(job['indices']) else 'cancelled'
                    job['resumable'] = job['status'] != 'completed'
                    p.save()
            self._submit('Tracking диапазона', work)
        return self.snapshot()

    def stop(self):
        self.cancel.set()
        with self.lock:
            p = self.required(); job = p.state.get('job')
            if self.busy and job and job.get('status') in ('running', 'queued'):
                job['status'] = 'cancelling'; p.save()
        return self.snapshot()

    def export(self):
        with self.lock:
            self.idle()
            folder = export_poses(self.required())
            return dict(directory=str(folder), files=[str(p.relative_to(self.project.directory)) for p in folder.iterdir()])

    def preview(self):
        with self.lock:
            self.idle(); p = self.required(); mesh = self.mesh
            def work():
                def progress(count):
                    with self.lock:
                        self.progress = dict(completed=count, total=len(p.sequence.frames))
                path = preview_video(p, mesh, progress, self.cancel)
                with self.lock:
                    p.state['last_preview'] = str(path.relative_to(p.directory)); p.save()
            self._submit('Генерация preview', work)
        return self.snapshot()

    def close(self):
        self.cancel.set()
        # EGL resources must be released on their owning worker thread.
        if self.backend is not None:
            self.executor.submit(self.backend.shutdown)
        self.executor.shutdown(wait=True)
