"""Atomic project writes and Windows/WSL path normalization."""
import json
import os
from pathlib import Path
import re
import tempfile


def local_path(value):
    text = str(value).strip()
    match = re.match(r'^([A-Za-z]):[\\/](.*)$', text)
    if match:
        text = '/mnt/' + match[1].lower() + '/' + match[2].replace('\\', '/')
    return Path(text).expanduser().resolve()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def separate_output(output, sources):
    output = local_path(output)
    for source in sources:
        source = local_path(source)
        if output == source or (source.is_dir() and source in output.parents):
            raise ValueError('Каталог проекта должен находиться отдельно от исходных данных')
    return output
