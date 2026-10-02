import csv
from pathlib import Path
import uuid
import numpy as np
import cv2
from .storage import atomic_json
from .viewer import render


def export_poses(project):
    folder = project.directory / 'exports' / uuid.uuid4().hex
    folder.mkdir(parents=True)
    entries = sorted(project.state['poses'].items(), key=lambda item: int(item[0]))
    frames = np.array([int(key) for key, _ in entries], dtype=np.int64)
    matrices = np.array([entry['matrix'] for _, entry in entries], dtype=np.float64).reshape(-1, 4, 4)
    np.save(folder / 'poses.npy', matrices)
    np.savez(folder / 'poses.npz', frame_indices=frames, T_cam_from_object=matrices)
    atomic_json(folder / 'poses.json', dict(convention='T_cam_from_object', camera_axes='OpenCV', units='mm',
                                          camera=project.state['camera'], poses=dict(entries)))
    with (folder / 'poses.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'source_number', 'filename', 'score', 'tx', 'ty', 'tz',
                         *[f'r{i}{j}' for i in range(3) for j in range(3)],
                         'translation_delta_mm', 'rotation_delta_deg', 'status'])
        for key, entry in entries:
            frame = project.state['source']['frames'][int(key)]
            matrix = np.asarray(entry['matrix'])
            writer.writerow([key, frame['source_number'], frame['filename'], entry.get('score'),
                             *matrix[:3, 3], *matrix[:3, :3].flatten(), entry.get('translation_delta_mm'),
                             entry.get('rotation_delta_deg'), entry['status']])
    return folder


def preview_video(project, mesh, progress=None, cancel=None):
    target = project.directory / 'previews' / (uuid.uuid4().hex + '.mp4')
    camera = project.state['camera']
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'),
                             project.state['source']['fps'], (camera['width'], camera['height']))
    if not writer.isOpened():
        raise RuntimeError('Не удалось создать preview video')
    try:
        for index in range(len(project.state['source']['frames'])):
            if cancel is not None and cancel.is_set():
                break
            image = render(project, mesh, index, initial=False, masks=False)
            entry = project.state['poses'].get(str(index), {})
            text = f"Frame {index}  score {entry.get('score')}  {entry.get('status', 'UNTRACKED')}"
            cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
            writer.write(image)
            if progress:
                progress(index + 1)
    finally:
        writer.release()
    return target
