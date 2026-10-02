"""Prepare a patched runtime copy; never modify the upstream checkout."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def prepare_runtime(source: Path, output: Path) -> Path:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Runtime output must be outside upstream source")
    files = sorted(p for folder in ("model", "utils", "configs", "dataloader")
                   for p in (source / folder).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".yaml"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    digest.update(b"disable-second-dinov2-hub-load-v1")
    target = output / digest.hexdigest()[:20]
    if (target / ".ready").exists():
        return target
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="runtime-", dir=output))
    try:
        for path in files:
            dest = staging / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        dino = staging / "utils/dinov2_util.py"
        code = dino.read_text()
        line = '        self.model = torch.hub.load("facebookresearch/dinov2", self.model_base_name)'
        if line not in code.splitlines():
            raise RuntimeError("Unexpected GoTrack DINOv2 source; review compatibility patch")
        dino.write_text(code.replace(line, "        # " + line.strip(), 1))
        (staging / ".ready").write_text(digest.hexdigest())
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def activate_runtime(runtime: Path):
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    for name in ("model", "utils"):
        module = sys.modules.get(name)
        if module and not all(Path(p).resolve().is_relative_to(runtime)
                              for p in module.__path__):
            raise RuntimeError(f"Conflicting {name} module already imported; restart worker")
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))


def backend_version(source: Path) -> str:
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                   text=True).strip() + "+dinov2-local"
