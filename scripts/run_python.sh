#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tracker_python="${TRACKER_PYTHON:-$HOME/miniconda3/envs/gotrack/bin/python}"
tracker_env="$(dirname -- "$(dirname -- "$tracker_python")")"
export LD_LIBRARY_PATH="$HOME/wsl-libs:/usr/lib/wsl/lib:$tracker_env/lib:${LD_LIBRARY_PATH:-}"
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "$tracker_python" "$@"
