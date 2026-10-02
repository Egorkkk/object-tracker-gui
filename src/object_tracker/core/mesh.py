"""Read-only loading of already prepared millimeter meshes."""

from pathlib import Path

import numpy as np
import trimesh

from .types import MeshData


def load_mesh(path: str | Path) -> MeshData:
    """Preserve prototype trimesh defaults, vertex ordering and scene flattening.

    No unit inference, scaling, centering, decimation or output writes occur here.
    Legacy scene flattening concatenates geometries without scene-node transforms.
    Full scene-aware preparation must be a separate, explicitly tested operation.
    Rendering should retain the original asset's visual data, not rebuild a
    colorless mesh from these arrays.
    """
    source_path = Path(path).resolve(strict=True)
    mesh = trimesh.load(source_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError("Expected a nonempty triangle mesh")
    vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
    faces = np.array(mesh.faces, dtype=np.int64, copy=True)
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite")
    transform = np.eye(4, dtype=np.float64)
    for array in (vertices, faces, transform):
        array.setflags(write=False)
    return MeshData(source_path, vertices, faces, transform)
