# Object Tracker GUI — Specification

## 1. Purpose

Build a local browser-based application for 6DoF rigid object tracking in RGB image sequences or video using a known 3D mesh.

The first tracking backend is **GoTrack**, but the application architecture must allow other tracking / pose-refinement backends to be added later without rewriting the GUI, project format, camera model, pose representation, or export layer.

The application is intended primarily for VFX / matchmove workflows.

Do not implement ComfyUI nodes in this project phase.

---

# 2. Primary Workflow

The user should be able to:

1. Create or open a project.
2. Select an image sequence or video source.
3. Select a 3D mesh.
4. Define or adjust mesh dimensions.
5. Define camera intrinsics.
6. Select a starting frame.
7. Manually align the mesh to the object.
8. Refine the pose on the current frame using GoTrack.
9. Accept or reject the refined pose.
10. Select a frame range.
11. Track forward or backward through that range.
12. Inspect pose, score, motion deltas, masks, and overlays.
13. Correct pose on an arbitrary frame.
14. Re-track only a selected range.
15. Save and reopen the project.
16. Export poses and diagnostics.

The user must not need to manually edit Python files, copy pose matrices, rename frames, or invoke helper scripts during normal use.

---

# 3. Existing Working Prototypes

The repository contains prototype scripts in:

```text
prototypes/
```

Expected scripts:

```text
align_phone.py
refine_one.py
track_sequence.py
```

These scripts are known-working reference implementations.

They must not be treated as disposable examples.

Development should proceed by extracting and refactoring their working logic into reusable core/backend code.

Keep the prototype scripts unchanged whenever possible so they remain useful as regression references.

---

# 4. Application Type

Use a browser-based GUI.

Recommended architecture:

```text
Browser
   |
   v
FastAPI server
   |
   v
Application/Core
   |
   v
TrackingBackend
   |
   +-- GoTrackBackend
   +-- future backends
```

The server is expected to run inside WSL2.

The browser may run:

- on the same Windows machine;
- from another machine over LAN;
- through Tailscale.

Avoid unnecessary frontend complexity.

Plain HTML/CSS/JavaScript is acceptable.

Do not use Gradio as the main UI framework.

React/Vue/etc. should only be introduced if clearly justified by UI complexity.

---

# 5. Repository Architecture

Recommended structure:

```text
object-tracker-gui/
├── SPEC.md
├── KNOWN_WORKING.md
├── README.md
│
├── prototypes/
│   ├── align_phone.py
│   ├── refine_one.py
│   └── track_sequence.py
│
├── src/
│   └── object_tracker/
│       ├── app/
│       │   ├── server.py
│       │   └── api.py
│       │
│       ├── core/
│       │   ├── types.py
│       │   ├── camera.py
│       │   ├── mesh.py
│       │   ├── sequence.py
│       │   ├── project.py
│       │   ├── masks.py
│       │   ├── tracking.py
│       │   ├── diagnostics.py
│       │   └── export.py
│       │
│       └── backends/
│           ├── base.py
│           └── gotrack/
│               ├── backend.py
│               ├── adapter.py
│               └── environment.py
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── external/
│   └── gotrack/
│
├── patches/
│   └── gotrack.patch
│
├── scripts/
│   ├── run_server.sh
│   └── check_environment.py
│
└── tests/
```

The exact structure may be adjusted if there is a good technical reason, but backend-specific code must remain isolated from the application core.

---

# 6. Backend Independence

The GUI and application core must not directly import GoTrack internals.

Do not couple GUI code to:

```text
model.gotrack
dinov2
pyrender
GoTrack utils
```

All GoTrack-specific behavior belongs inside the GoTrack backend adapter.

Conceptually:

```text
GUI
 |
 v
Application services
 |
 v
TrackingBackend API
 |
 +-- GoTrackBackend
 +-- FutureBackend
```

The first implementation should remain simple and practical.

Do not build an unnecessarily generic tracking framework before a second backend exists.

---

# 7. Internal Pose Convention

Use one canonical pose representation throughout the application:

```text
T_cam_from_object
```

Equivalent wording:

```text
model/object coordinates -> camera coordinates
```

Matrix:

```text
4x4 homogeneous transform
```

Camera coordinate convention:

```text
OpenCV

+X = right
+Y = down
+Z = forward
```

Translation unit:

```text
millimeters
```

This convention must be used by:

- project storage;
- viewer;
- diagnostics;
- export layer;
- tracking results;
- pose keyframes.

Backends using different conventions must convert internally in their adapter.

Examples:

- meters -> millimeters;
- OpenGL -> OpenCV;
- `T_object_from_cam` -> `T_cam_from_object`.

Do not leak backend coordinate conventions into the application core.

---

# 8. Core Data Types

Create backend-independent structures.

## CameraIntrinsics

Required fields:

```text
width
height
fx
fy
cx
cy
distortion optional
```

---

## MeshData

Required concepts:

```text
source_path
vertices_mm
faces
bounds
extents
source_transform
```

Core mesh coordinates are expressed in millimeters.

Backend adapters may convert units as needed.

---

## Pose

Contains:

```text
T_cam_from_object
```

Do not use Euler angles as the primary stored representation.

Euler angles may be used only for UI editing or export.

---

## FrameResult

Conceptually:

```text
frame_index
pose
score
translation_delta_mm
rotation_delta_deg
status
runtime
backend_diagnostics
```

---

## TrackingResult

Conceptually:

```text
range
frame_results
backend_id
backend_version
camera
mesh_reference
```

---

# 9. Tracking Backend Interface

Provide a minimal backend abstraction.

Conceptual API:

```python
class TrackingBackend:
    backend_id: str
    capabilities: BackendCapabilities

    def initialize(self):
        ...

    def shutdown(self):
        ...

    def refine_frame(
        self,
        image,
        mesh,
        camera,
        initial_pose,
        object_mask=None,
        occlusion_mask=None,
    ):
        ...

    def track_range(
        self,
        sequence,
        mesh,
        camera,
        initial_pose,
        start_frame,
        end_frame,
        direction,
        masks=None,
        progress_callback=None,
        cancel_token=None,
    ):
        ...
```

The exact Python API may differ if a cleaner design is found.

---

# 10. Backend Capabilities

Backends should expose capabilities such as:

```text
requires_mesh
requires_initial_pose
requires_depth

supports_refine
supports_tracking

supports_object_mask
supports_occlusion_mask
supports_relocalization
```

The GUI should use these capabilities to enable or disable controls.

---

# 11. GoTrack Backend

The first backend is GoTrack.

It must encapsulate:

```text
GoTrack model
DINOv2
checkpoint loading
pyrender
EGL
CUDA
GoTrack input construction
pose conversion
GoTrack compatibility workarounds
```

Load the model once and keep it resident in GPU memory until explicitly unloaded or the application exits.

Do not reload the checkpoint for every frame.

---

# 12. Input Sources

Support two source types.

## Image sequence

At minimum:

```text
PNG
JPG/JPEG
```

Future-friendly support for EXR is desirable.

The user selects an input directory.

The application must detect frame sequences automatically when possible.

Support filenames such as:

```text
0000.png
0001.png

shot_0100.png
shot_0101.png
```

Do not require manual renaming.

---

## Video

At minimum:

```text
MP4
MOV
MKV
```

For the MVP it is acceptable to extract frames automatically with ffmpeg into a project-local cache.

The user must not need to invoke ffmpeg manually.

---

# 13. Frame Numbering

Separate:

```text
sequence index
source frame number
filename
```

Do not assume sequences always start at frame zero.

Support:

```text
start frame
padding
prefix
```

Example:

```text
shot_1001.png
shot_1002.png
```

---

# 14. Output and Project Paths

All paths must be configurable.

Do not hardcode:

```text
custom/
track60/
~/gotrack/
```

The user chooses:

```text
input
mesh
output directory
project name
track name
object name
```

Recommended structure:

```text
output/
└── track_name/
    ├── project.json
    ├── cache/
    ├── poses/
    ├── overlays/
    ├── diagnostics/
    ├── previews/
    └── exports/
```

Source files must never be overwritten.

---

# 15. Windows / WSL Path Handling

Because the browser may run on Windows while the backend runs inside WSL2, support Windows-style paths.

Example:

```text
E:\project\frames
```

should be convertible to:

```text
/mnt/e/project/frames
```

Do not require the user to manually convert paths when avoidable.

---

# 16. Mesh Loading

Support at minimum:

```text
PLY
OBJ
GLB
GLTF
```

using trimesh or equivalent.

On load display:

```text
vertex count
face count
bounds
extents
detected units if available
```

---

# 17. Mesh Preparation

Allow:

```text
uniform scale
independent X/Y/Z scale
centering at origin
```

The user should be able to define known real-world dimensions in millimeters.

Example:

```text
75 x 11 x 160 mm
```

Never modify the original mesh.

Store a prepared version in the project cache.

---

# 18. Camera Intrinsics

Provide direct editing of:

```text
fx
fy
cx
cy
```

Defaults:

```text
cx = width / 2
cy = height / 2
```

Also provide an approximate focal-length helper using full-frame equivalent focal length.

Example:

```text
35 mm equivalent
```

The UI must clearly indicate that this conversion is approximate.

Future architecture should allow importing intrinsics from:

```text
COLMAP
OpenCV calibration
camera metadata
Nuke
```

These importers are not required for MVP.

---

# 19. Viewer

The main viewer should support:

```text
RGB
initial pose
refined pose
wireframe mesh
shaded mesh
object mask
occlusion mask
```

Useful combinations:

```text
RGB only
RGB + initial
RGB + refined
RGB + initial + refined
```

Provide overlay opacity control.

A lightweight display mesh may be decimated independently of the tracking mesh.

---

# 20. Timeline

Provide:

```text
previous frame
next frame
play
pause
jump to frame
range start
range end
```

Timeline state should visually distinguish:

```text
untracked
tracked
manual/key pose
warning
failed
```

---

# 21. Manual Initial Pose

Port the functionality from `align_phone.py`.

Controls:

```text
RX
RY
RZ
TX
TY
TZ
```

Provide numeric fields.

Sliders are useful but should not be the only editing method.

Future mouse manipulation is desirable:

```text
drag -> XY
wheel -> depth
modified drag -> rotation
```

Mouse manipulation is not mandatory for the first functional MVP.

---

# 22. Refine Current Frame

Provide a button:

```text
Refine Current Frame
```

Input:

```text
current RGB frame
mesh
camera
current initial pose
optional masks
```

Output:

```text
refined pose
score
translation difference
rotation difference
runtime
```

Display both initial and refined poses simultaneously.

The user must be able to:

```text
Accept Refined Pose
Reject Refined Pose
```

---

# 23. Tracking

The user selects:

```text
start frame
end frame
direction
```

Forward tracking is mandatory.

Backward tracking should be supported if practical and at minimum must be architecturally possible.

GoTrack tracking currently uses pose propagation:

```text
pose N
  |
  v
initial pose N+1
  |
  v
GoTrack refinement
  |
  v
pose N+1
```

This behavior has already been validated experimentally.

---

# 24. Tracking Jobs

Tracking must not block the UI/server event loop.

Provide a simple worker/job system.

Display:

```text
current frame
frames completed
total frames
runtime
seconds per frame
current score
```

Provide:

```text
Cancel
```

Do not require a distributed job queue.

---

# 25. Resume

Tracking results must be persisted incrementally.

If tracking is cancelled or the process stops, already-computed poses remain available.

Support:

```text
Resume
```

without recomputing the entire sequence.

---

# 26. Pose Keys

Allow arbitrary frames to be marked as pose anchors.

Possible sources:

```text
manual pose
accepted refined pose
```

A tracking range may start from any pose anchor.

Do not assume one initial pose for the entire project.

---

# 27. Retracking

Provide:

```text
Retrack Range
```

The user should be able to replace tracking results only between selected frames.

This is a core VFX workflow requirement.

---

# 28. Diagnostics

Store per-frame:

```text
score
tx
ty
tz
translation_delta_mm
rotation_delta_deg
runtime
status
```

Statuses may include:

```text
UNTRACKED
TRACKED
MANUAL
WARNING
FAILED
```

---

# 29. Diagnostic Graphs

Display timeline graphs for:

```text
score
translation delta
rotation delta
```

Clicking a graph point should navigate the viewer to that frame.

---

# 30. Suspicious Frame Detection

Provide configurable warning thresholds:

```text
minimum score
maximum translation jump
maximum rotation jump
```

These thresholds should mark suspicious frames.

Do not automatically modify poses based only on these heuristics.

---

# 31. Pose Smoothing

The tested GoTrack workflow shows small pose jitter on a stationary object.

Architecture should support optional post-processing.

Possible MVP filters:

```text
translation smoothing
rotation smoothing
```

Requirements:

- raw poses must never be overwritten;
- smoothing is disabled by default;
- smoothed poses are stored/exported separately;
- smoothing is not part of the tracking algorithm itself.

---

# 32. Object Masks

Provide an independent per-frame object-mask layer.

Potential sources:

```text
PNG sequence
SAM/SAM3
manual masks
external software
```

The application does not need to generate SAM masks in this phase.

Mask sequence matching should support:

```text
filename matching
frame-number matching
```

Show missing masks.

---

# 33. Object Mask Uses

Potential uses include:

```text
bounding box hints
crop hints
visibility diagnostics
quality evaluation
relocalization
backend input
```

Whether a backend directly consumes object masks is determined through backend capabilities.

---

# 34. Occlusion Masks

Provide a second, separate mask type:

```text
occlusion_mask
```

This represents foreground elements covering the tracked object, e.g.:

```text
hand
person
foreground prop
```

Never merge the semantic meaning of:

```text
object_mask
occlusion_mask
```

They must remain distinct throughout the project format and backend API.

---

# 35. Masks and GoTrack

Current testing shows GoTrack can survive significant hand occlusion without explicit masks.

Therefore:

- do not alter GoTrack correspondence behavior by default;
- support loading, storage, display, and transport of masks;
- experimental use of occlusion masks inside GoTrack should remain optional;
- any behavior-changing mask integration must be A/B tested against the current known-working behavior.

---

# 36. Project Format

Store project state in a human-readable format, preferably JSON.

Example concepts:

```json
{
  "version": 1,
  "name": "phone_take01",

  "source": {
    "type": "image_sequence",
    "path": "...",
    "pattern": "..."
  },

  "mesh": {
    "source": "...",
    "prepared": "...",
    "dimensions_mm": [75, 11, 160]
  },

  "camera": {
    "width": 1920,
    "height": 1080,
    "fx": 1867,
    "fy": 1867,
    "cx": 960,
    "cy": 540
  },

  "backend": {
    "id": "gotrack",
    "settings": {}
  }
}
```

Large binary data must not be embedded directly in JSON.

---

# 37. Multiple Tracking Solutions

The data model should not prevent multiple tracking attempts/solutions from existing in one project.

Example future use:

```text
GoTrack attempt A
GoTrack attempt B
FutureBackend attempt
Manually corrected solution
```

The MVP may expose only one active solution in the GUI.

---

# 38. Autosave and Recovery

Autosave after important state changes:

```text
camera edit
mesh preparation
manual pose
accepted refinement
tracking progress
mask setup
```

A crash should not destroy an already-computed tracking result.

---

# 39. Export

For MVP support:

```text
4x4 pose matrices
NPY/NPZ
JSON
CSV
```

CSV should include at minimum:

```text
frame
score
tx
ty
tz
rotation representation
translation_delta_mm
rotation_delta_deg
status
```

Architecture should allow future exporters for:

```text
Nuke
Blender
Maya
```

DCC-specific coordinate conversion must live in exporters, not tracking core.

---

# 40. Preview Video

Provide optional preview generation.

Overlay may include:

```text
mesh contour
frame number
score
warning status
```

Use source/project FPS.

---

# 41. Environment Isolation

The application architecture should allow ML backends to run in separate processes/environments in the future.

Desired future topology:

```text
GUI/Core
   |
   | local RPC / HTTP / IPC
   v
GoTrack worker
```

This matters because GoTrack uses an old environment while future backends may require much newer CUDA/PyTorch versions.

For the MVP it is acceptable for GUI and GoTrack to run in the same `gotrack` environment if this substantially reduces development time.

Do not create architectural dependencies that make future process isolation difficult.

---

# 42. Startup Diagnostics

Before loading the model, check:

```text
CUDA available
GPU visible
GoTrack checkpoint available
DINOv2 weights available
libcuda available
EGL renderer available
ffmpeg available
required directories writable
```

Show useful user-facing errors.

Do not expose a raw Python traceback as the only error information.

The full traceback should still be written to logs.

---

# 43. Logging

Maintain logs such as:

```text
logs/application.log
logs/backend_gotrack.log
```

Include context where useful:

```text
frame
backend
GPU
project
exception
settings
```

---

# 44. Reproducibility

Store with project metadata:

```text
application version
backend ID
backend version/git commit
checkpoint identity
camera intrinsics
prepared mesh dimensions
relevant backend settings
```

---

# 45. Performance

Do not load entire long image sequences into RAM.

Use bounded caching, e.g. LRU frame cache.

Avoid repeatedly loading:

```text
GoTrack model
DINOv2
mesh
```

Do not use the full high-resolution mesh merely for GUI projection when a display representation can be used.

---

# 46. Source Safety

Never modify or overwrite:

```text
source frames
source video
source mesh
source masks
```

Derived assets belong in the project/output directory.

---

# 47. Development Order

Recommended order:

## Phase 1

Extract reusable logic from prototypes.

## Phase 2

Implement `GoTrackBackend`.

## Phase 3

Implement project/source/mesh/camera layers.

## Phase 4

Implement basic browser GUI and viewer.

## Phase 5

Implement manual alignment and single-frame refine.

## Phase 6

Implement range tracking, progress, cancel/resume.

## Phase 7

Implement diagnostics and graphs.

## Phase 8

Implement masks.

## Phase 9

Implement pose anchors and retracking.

## Phase 10

Polish exports and preview generation.

---

# 48. Regression Requirement

Do not replace working prototype behavior without comparison.

For identical:

```text
input frames
mesh
camera intrinsics
initial pose
checkpoint
```

compare:

```text
prototypes/track_sequence.py
```

against:

```text
new GoTrackBackend
```

The resulting poses should match within reasonable numerical tolerance.

Unexpected significant deviation is a regression until explained.

---

# 49. MVP Acceptance Test

The following must be possible entirely from the browser GUI:

1. Create project.
2. Select approximately 60 RGB frames.
3. Select phone mesh.
4. Set real mesh dimensions.
5. Set camera intrinsics.
6. Choose starting frame.
7. Manually align mesh.
8. Refine current frame with GoTrack.
9. Accept refined pose.
10. Select a 60-frame range.
11. Track forward.
12. Observe a sequence containing:
    - stationary phone;
    - hand entering;
    - partial hand occlusion;
    - phone pickup;
    - significant object motion;
    - partial exit from image boundary.
13. Inspect tracking overlays.
14. Inspect score / translation delta / rotation delta.
15. Correct an arbitrary frame.
16. Re-track a selected range.
17. Save project.
18. Close application.
19. Reopen project.
20. Continue working.
21. Export raw poses and CSV.
22. Generate preview video.

No manual editing of Python scripts or pose files should be required.

---

# 50. Out of Scope for Current MVP

Do not implement yet:

```text
ComfyUI nodes
SAM3 inference
Depth Anything
automatic object recognition
global automatic relocalization
Nuke plugin
Blender addon
Maya plugin
cloud execution
multi-user operation
multi-GPU scheduling
```

The architecture should not unnecessarily prevent these features later.

---

# 51. Priority Order

When tradeoffs arise, prioritize:

```text
1. Correct pose mathematics
2. Preserve known-working tracking behavior
3. Reliable project persistence
4. Reliable GoTrack execution
5. Manual alignment workflow
6. Retracking and correction workflow
7. Diagnostics
8. UI polish
```

Do not sacrifice correctness for visual polish.

---

# 52. Development Rule

Before making major architectural changes:

1. inspect the prototype scripts;
2. understand why they currently work;
3. preserve the tested conventions and units;
4. make the smallest useful refactor;
5. run regression tests.

Do not "clean up" unusual compatibility code simply because it looks redundant until its purpose has been verified.