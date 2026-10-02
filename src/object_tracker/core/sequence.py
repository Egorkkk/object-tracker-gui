"""Bounded, on-demand input with explicit sequence and source frame indices."""
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import json
import cv2

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv'}


def natural_key(path):
    return tuple((0, int(s)) if s.isdigit() else (1, s.lower())
                 for s in re.split(r'(\d+)', Path(path).name))


def discover(directory):
    paths = sorted((p for p in Path(directory).iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
    if not paths:
        raise ValueError('В каталоге нет PNG/JPG кадров')
    result = []
    for index, path in enumerate(paths):
        numbers = re.findall(r'\d+', path.stem)
        result.append(dict(index=index, source_number=int(numbers[-1]) if numbers else None,
                           filename=path.name, path=str(path.resolve())))
    return result


def extract_video(source, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=False)
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=avg_frame_rate', '-of', 'json', str(source)], text=True))
    rate = info['streams'][0]['avg_frame_rate'].split('/')
    fps = float(rate[0]) / float(rate[1]) if float(rate[1]) else 24.
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
                    '-vsync', '0', '-start_number', '0', str(cache / '%08d.png')], check=True)
    return discover(cache), fps


class ImageSequence:
    def __init__(self, frames):
        self.frames = frames

    @lru_cache(maxsize=3)
    def rgb(self, index):
        if index < 0 or index >= len(self.frames):
            raise IndexError('Кадр вне диапазона')
        image = cv2.imread(self.frames[index]['path'])
        if image is None:
            raise ValueError('Не удалось прочитать кадр: ' + self.frames[index]['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.setflags(write=False)
        return image
