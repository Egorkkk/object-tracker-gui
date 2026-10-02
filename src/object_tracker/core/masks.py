from pathlib import Path
import cv2
import numpy as np
from .sequence import discover
from .types import FrameMasks


def match_masks(frames, directory, mode='filename'):
    candidates = discover(directory)
    if mode not in ('filename', 'number'):
        raise ValueError('Сопоставление масок: filename или number')
    lookup = {}
    for item in candidates:
        key = Path(item['filename']).stem if mode == 'filename' else item['source_number']
        if key is None:
            continue
        if key in lookup:
            raise ValueError('Неоднозначное сопоставление масок: ' + str(key))
        lookup[key] = item['path']
    matched = {}
    for frame in frames:
        key = Path(frame['filename']).stem if mode == 'filename' else frame['source_number']
        if key in lookup:
            matched[str(frame['index'])] = lookup[key]
    return matched


def load_mask(path, shape):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        raise ValueError('Маска отсутствует либо её размер не совпадает с кадром: ' + str(path))
    return mask > 0


def frame_masks(project, index):
    shape = (project.state['camera']['height'], project.state['camera']['width'])
    values = {}
    for kind in ('object_mask', 'occlusion_mask'):
        path = project.state['masks'][kind].get(str(index))
        values[kind] = load_mask(path, shape) if path else None
    return FrameMasks(**values)
