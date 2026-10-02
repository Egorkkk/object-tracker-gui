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

1. Select an image directory (PNG/JPG/JPEG) or MP4/MOV/MKV, source mesh, and a new
   empty project directory. Video frames are extracted automatically by ffmpeg
   into the project cache. For image sequences, set source FPS.
2. Check mesh dimensions in millimeters. Apply X/Y/Z dimensions, uniform scale,
   or centering as needed; preparation always creates a new cache asset.
3. Set `fx/fy/cx/cy`. Initial focal values use a labeled approximate 35 mm
   full-frame equivalent, not calibrated intrinsics.
4. Choose a start frame, edit RX/RY/RZ and TX/TY/TZ, or import a 4x4 NPY pose.
   Manual edits autosave. Save an anchor when the pose is ready.
5. Run **Refine**, compare yellow initial and green refined overlays, then accept
   or reject. Unaccepted refinement does not replace the active tracked pose.
6. Set inclusive start/end indices and run **Track / Retrack**. End < start tracks
   backward. By default anchors inside the range reinitialize refinement there.
   Without additional anchors, baseline refined-pose propagation is unchanged.
7. Cancel between frames, reopen the project and resume. A stopped process is
   detected on reopen; already committed frame results are kept. A pose or
   calibration edit invalidates the previous resume intent.
8. Scrub/play the timeline, inspect score/dT/dR, click graphs to navigate, correct
   a pose and retrack only the selected range. Raw files from earlier attempts
   remain in separate attempt directories.
9. Load object/occlusion mask directories, matched by filename stem or source
   number. Layers have independent display controls. Missing frame masks are
   reported. GoTrack receives both inputs but does **not** consume either in its
   correspondence/refinement pipeline.
10. Export NPY, indexed NPZ, JSON and CSV, or generate an overlay MP4 at source FPS.
    Browser download links are provided after export and preview generation.

The file browser enumerates the server filesystem. To create a new output
subdirectory, select its parent and append the new directory name in the field.
Source sequences may start at arbitrary numbers; sequence index, filename and
source frame number are stored separately.

## Storage and behavior

- Canonical pose: `T_cam_from_object`, OpenCV (+X right, +Y down, +Z forward), mm.
- `project.json` holds versioned state and is atomically replaced on changes.
- `poses/<attempt>/<index>.json` holds a durable raw per-frame record.
  Tracking progress and continuation pose are saved after every frame.
- `cache/meshes/` and `cache/frames/` contain prepared assets, never source edits.
- `diagnostics/` retains previous settings/job snapshots; `exports/` and
  `previews/` hold derived deliverables.
- `outputs/logs/application.log` and `backend_gotrack.log` retain full exceptions.
- Source/checkpoint/config/runtime identity and environment versions are saved
  under project backend metadata after model initialization.
- Model/CUDA/EGL access is confined to one worker thread. The model stays resident
  across frames and project changes; renderer resources are released on that
  same thread when switching meshes or shutting down.

Changing camera settings archives active results and retains manual anchors for
new refinement. Changing mesh preparation archives results and clears active
poses/anchors because the object coordinate frame may change. Thresholds mark
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
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/browser_smoke.mjs
```

Run the server on port 8765 first. The script creates its own project and verifies
60-frame GPU tracking, cancel/reopen/resume, range retracking, export and preview.
See [validation results](docs/VALIDATION.md).

## Current limits

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
