"""Runtime checks inside the GPU worker, before loading expensive model weights."""
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import tempfile


def checkpoint_identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), size=Path(path).stat().st_size, sha256=digest.hexdigest())


def check_runtime(output):
    import torch
    import pyrender
    import numpy as np
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна; проверьте WSL GPU и LD_LIBRARY_PATH')
    ctypes.CDLL('libcuda.so')
    weights = Path(torch.hub.get_dir()) / 'checkpoints/dinov2_vits14_reg4_pretrain.pth'
    if not weights.is_file():
        raise FileNotFoundError('Локальные DINOv2 weights не найдены: ' + str(weights))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=output) as stream:
        stream.write(b'write-check')
    renderer = pyrender.OffscreenRenderer(16, 16)
    renderer.delete()
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, egl=True, libcuda=True,
                dino_weights=str(weights), ffmpeg=shutil.which('ffmpeg'), ffprobe=shutil.which('ffprobe'))
