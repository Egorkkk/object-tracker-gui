# Object Tracker GUI

A local browser GUI for known-mesh 6DoF tracking. The first implementation uses
GoTrack while project state, camera, pose, masks, diagnostics and exports remain
backend-independent. No ComfyUI nodes are included.

## Run in the existing environment

From the repository root inside WSL2:

```bash
bash scripts/run_server.sh
```

Open **http://localhost:8765** in your browser. The server binds to loopback by
default. To access it through your LAN/Tailscale interface, set its bind address:

```bash
bash scripts/run_server.sh --host 0.0.0.0 --port 8765
```

The application is a single-user local tool with server filesystem access, not
a public multi-user service. Select paths on the WSL server; Windows drive paths
such as `E:\shot\frames` are converted to `/mnt/e/shot/frames`.

The launcher uses `~/miniconda3/envs/gotrack/bin/python` by default and preserves
the documented EGL/WSL library setup. Override the interpreter without upgrading
its dependencies:

```bash
TRACKER_PYTHON=/path/to/gotrack/bin/python bash scripts/run_server.sh
```

Optional server flags: `--project /path/to/project`, `--gotrack /path/to/checkout`,
`--checkpoint /path/to/gotrack_checkpoint.pt`. Existing local weights are required;
there is no automatic dependency or model download.

## Browser workflow

1. Use **File → New Project** to choose a parent directory and project name.
   Source and Mesh are optional at creation. **File → Open Project** and recent
   projects reopen existing `project.json` files.
2. Select **Source** in Project Tree. **Browse Server** opens an Explorer-style
   browser with Places, breadcrumbs, history, filters and sortable details.
   Choose an image folder or a video; video frames are extracted into project
   cache. **Upload** uses the browser's native picker for one video or multiple
   images. Set sequence FPS in Inspector.
3. Select **Mesh** and browse/upload a mesh. Check dimensions in millimeters;
   apply X/Y/Z dimensions, uniform scale or centering in Inspector. Preparation
   creates a separate asset. GLTF/OBJ uploads can include sibling dependencies.
4. In **ALIGN**, check camera intrinsics. Initial focal values are a labeled
   approximate 35 mm full-frame equivalent, not calibrated intrinsics.
5. Choose a frame and align using XYZ translation handles or rotation rings.
   **Local / Global** changes gizmo axes, not the canonical pose convention.
   Numeric RX/RY/RZ and TX/TY/TZ remain available in every Inspector context.
   Shift gives fine adjustment; edits autosave. Import NPY or set a pose anchor.
6. **Refine**, compare yellow initial and green refined overlays, then accept
   or reject. Unaccepted refinement does not replace the active tracked pose.
7. In **TRACK**, drag the Frames track or In/Out edges to select an inclusive
   range. Track forward/backward or retrack it. Anchors reinitialize refinement
   when enabled; baseline pose propagation is unchanged.
8. Cancel between frames, reopen and **Resume**. Saved results are retained.
   Correct a pose and retrack only the affected range.
9. In **REVIEW**, scrub/play, inspect score/ΔT/ΔR and click diagnostic graphs to
   navigate. Viewer supports fit, 1:1, wheel zoom, middle-button pan, independent
   overlays, wireframe and shaded preview. Resize panels with splitters; sizes
   persist in the browser.
10. Select each mask layer to browse a server folder or upload multiple masks.
    Match by filename stem or source frame number. Object and occlusion masks
    remain separate; GoTrack baseline does not use them for refinement.
11. Select **Exports** for NPY, indexed NPZ, JSON, CSV and an overlay MP4 at source
    FPS. Download links remain in the project.

**Replace** is always available in each resource Inspector. Changes affecting
existing poses offer **Keep as Separate Solution**, **Clear Active Tracking** or
**Cancel**. Open a preserved solution under **Other Solutions** to restore its
source, mesh, camera, masks and poses. Uploads use unique project-local folders
and never overwrite same-named files. Browse Server references existing files
without copying them.

Shortcuts: **Left/Right** frames, **Space** playback, **Ctrl+S** save, **W/E**
translate/rotate, **L/G** local/global. Navigation shortcuts are inactive while
editing fields or using dialogs. Sequence indices, filenames and source numbers
are stored separately.

## Storage and behavior

- Canonical pose: `T_cam_from_object`, OpenCV (+X right, +Y down, +Z forward), mm.
- `project.json` holds versioned state and is atomically replaced on changes.
- `poses/<attempt>/<index>.json` holds a durable raw per-frame record.
  Tracking progress and continuation pose are saved after every frame.
- `cache/meshes/`, `cache/frames*/` and `cache/assets/<kind>/<batch>/` contain
  prepared/extracted/uploaded assets, never source edits.
- `solutions/` contains preserved resource contexts and raw pose references.
- `diagnostics/` retains previous settings/job snapshots; `exports/` and
  `previews/` hold derived deliverables.
- `outputs/logs/application.log` and `backend_gotrack.log` retain full exceptions.
- Source/checkpoint/config/runtime identity and environment versions are saved
  under project backend metadata after model initialization.
- Model/CUDA/EGL access is confined to one worker thread. The model stays resident
  across frames and project changes; renderer resources are released on that
  same thread when switching meshes or shutting down.

Changing camera settings requires an explicit invalidation choice when results
exist and retains manual anchors for new refinement. Mesh/source changes clear
active poses and anchors after that choice; old contexts can be preserved as
solutions. Source changes also reset camera assumptions and mask mappings. Thresholds mark
new results as warnings without modifying the predicted matrices.

## GoTrack compatibility

The upstream submodule stays unchanged. The adapter makes a content-addressed
runtime copy of GoTrack Python/config files under `outputs/runtime/` and disables
only the duplicate DINOv2 `torch.hub.load`, matching the existing working patch
in `patches/gotrack.patch`. No dependency versions are changed.

Before loading weights, the worker checks CUDA/GPU, `libcuda.so`, local DINOv2
weights, EGL context creation and output writability; ffmpeg/ffprobe availability
is recorded. Renderer unit/convention conversions remain inside GoTrack.

## Validation

```bash
bash scripts/run_python.sh -m unittest discover -s tests -v
node tests/pose_math.mjs
bash scripts/run_python.sh scripts/regression_gotrack.py --frames 60 --output outputs/regression_new
bash scripts/run_python.sh scripts/recovery_smoke.py
```

GPU regression runs the unchanged `refine_one.py` and `track_sequence.py` in
separate processes, with identical data/config/weights and controlled Python,
NumPy, torch and OpenCV RNG seeds. Only paths and frame count are overridden in
the prototype harness. Outputs include input/checkpoint hashes, environments,
raw matrices, score comparisons and prototype overlays.

The browser acceptance script uses an existing Playwright installation:

```bash
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/ui_redesign_smoke.mjs
# Full 60-frame variant:
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/browser_smoke.mjs
```

Run the server first (`TRACKER_URL` defaults to `http://127.0.0.1:8765`). The
script creates its own project and checks browser dialogs, native uploads, gizmos,
GPU refine/tracking, cancel/reopen/resume, resource replacement/solutions, masks,
backward tracking, export and preview. Default range is eight frames; the
compatibility entry point `browser_smoke.mjs` checks 60 frames.
See [validation results](docs/VALIDATION.md).

## Current limits

- Folder upload and drag/drop are deferred; multiple-file uploads and server
  folder selection are supported. Mesh dependencies must be sibling files.
- One project and one GPU job at a time; cancellation finishes the current frame.
- RGB pinhole cameras; nonzero distortion is rejected by the baseline backend.
- Sequence discovery sorts all supported image files naturally. Separate mixed
  shots into directories; pattern selection and EXR decoding are not implemented.
- Wireframe/shaded display uses a sampled set of faces and is an alignment preview,
  not a photorealistic renderer. Tracking always uses the full prepared asset.
- GLB/GLTF scene transforms are baked during preparation; textures are converted
  to vertex colors when exporting prepared PLY. This path has geometry tests but
  does not have the phone baseline's full tracking quality validation.
- No smoothing, automatic relocalization, mask inference or mask-driven GoTrack
  refinement. Raw tracking is always preserved.
- Resume preserves already saved results; subsequent poses are not promised to
  be bit-identical to an uninterrupted run because PnP uses random sampling.
- Preview playback is capped at 12 fps; exported video uses source/project FPS.

See [SPEC.md](SPEC.md), [KNOWN_WORKING.md](KNOWN_WORKING.md), and the
[initial audit](docs/INITIAL_AUDIT.md) for the original requirements and baseline.
