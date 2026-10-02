# Known Working State

This document records the currently verified working GoTrack setup.

Treat this file as factual reference data.

Do not change versions, conventions, patches, or environment behavior without a specific reason and regression testing.

---

# 1. Platform

Verified environment:

```text
OS layer: WSL2 / Ubuntu
Host: Windows
Python environment: conda
Conda env name: gotrack
Python: 3.10
```

GoTrack inference runs on NVIDIA CUDA GPU from WSL2.

---

# 2. GoTrack Source

Working GoTrack repository:

```text
~/gotrack
```

The new application should use a pinned copy/submodule under:

```text
external/gotrack
```

Record the exact tested commit here:

```text
GOTRACK_COMMIT=<fill from: git rev-parse HEAD>
```

Do not silently update GoTrack upstream while debugging application code.

---

# 3. Checkpoint

Working GoTrack checkpoint:

```text
gotrack_checkpoint.pt
```

Approximate size:

```text
~1.6 GB
```

It is loaded using:

```python
net_util.load_checkpoint(
    model=model,
    checkpoint_path="gotrack_checkpoint.pt",
    checkpoint_key="model_state_dict",
    prefix="models.1.",
)
```

The model loads successfully with:

```text
Pretrained: 532
Loaded: 532
Cannot loaded: 0
```

Do not commit checkpoint files to this repository.

---

# 4. DINOv2

Working DINOv2 model:

```text
dinov2_vits14_reg
```

Cached weight file:

```text
~/.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth
```

Approximate size:

```text
~84 MB
```

---

# 5. Required DINOv2 Compatibility Patch

In GoTrack:

```text
utils/dinov2_util.py
```

the working code constructs the backbone through the locally installed/pinned DINOv2 package:

```python
self.model = dinov2_backbones.__dict__[self.model_base_name](
    pretrained=True
)
```

The following second model load must remain disabled:

```python
# self.model = torch.hub.load(
#     "facebookresearch/dinov2",
#     self.model_base_name
# )
```

Reason:

Current upstream DINOv2 `main` expects modules that do not exist in the DINOv2 revision installed with this GoTrack environment.

Observed failure before patch:

```text
ModuleNotFoundError:
No module named 'dinov2.hub.cell_dino'
```

Do not restore this line unless the DINOv2 dependency strategy is intentionally changed and tested.

---

# 6. setuptools Compatibility

Old PyTorch Lightning code used by this environment requires `pkg_resources`.

Working version:

```text
setuptools==80.9.0
```

Observed failure with incompatible setup:

```text
ModuleNotFoundError:
No module named 'pkg_resources'
```

The deprecation warning from `pkg_resources` is currently expected and harmless.

---

# 7. WSL2 CUDA Library Workaround

Observed runtime failure:

```text
Could not load library libcudnn_cnn_infer.so.8.
Error:
libcuda.so: cannot open shared object file
```

WSL2 provides:

```text
/usr/lib/wsl/lib/libcuda.so.1
```

A local compatibility symlink was created:

```text
~/wsl-libs/libcuda.so
    ->
/usr/lib/wsl/lib/libcuda.so.1
```

Working launch environment:

```bash
LD_LIBRARY_PATH=$HOME/wsl-libs:/usr/lib/wsl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Do not install an NVIDIA Linux kernel driver inside WSL to fix this.

The GPU driver is supplied by the Windows host.

---

# 8. EGL

GoTrack/pyrender runs headlessly using EGL.

Required environment:

```text
PYOPENGL_PLATFORM=egl
```

Set this before importing pyrender/OpenGL-dependent code.

---

# 9. Known Working Smoke Test

The following model construction/loading sequence is verified:

```python
import torch

from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils import net_util

cfg = OmegaConf.load("configs/model/gotrack.yaml")

model = instantiate(cfg)

net_util.load_checkpoint(
    model=model,
    checkpoint_path="gotrack_checkpoint.pt",
    checkpoint_key="model_state_dict",
    prefix="models.1.",
)

model = model.cuda().eval()
```

---

# 10. Current Test Object

Test object:

```text
smartphone
```

Original mesh was generated using SAM3D Objects.

Original approximate extents:

```text
X = 0.498
Y = 0.149
Z = 0.999
```

The mesh was rescaled non-uniformly to approximately:

```text
75 x 11 x 160 mm
```

Current tested prepared mesh:

```text
phone_mm.ply
```

Observed mesh size:

```text
vertices: 174534
extents: [75, 11, 160] mm
```

Approximate axis interpretation:

```text
X = phone width
Y = phone thickness
Z = phone height
```

---

# 11. Camera Used in Testing

Input resolution:

```text
1920 x 1080
```

No calibrated camera solution was available.

Approximate lens assumption:

```text
~35 mm full-frame equivalent
```

Current approximate intrinsics:

```text
fx = 1867
fy = 1867
cx = 960
cy = 540
```

Intrinsic matrix:

```text
[1867    0  960]
[   0 1867  540]
[   0    0    1]
```

These values are approximate, not calibrated.

Do not treat them as generally correct for other footage.

---

# 12. Canonical Pose Convention

All current prototypes use:

```text
T_cam_from_model
```

Meaning:

```text
model coordinates -> camera coordinates
```

Camera system:

```text
OpenCV

+X right
+Y down
+Z forward
```

Translation:

```text
millimeters
```

This convention must be preserved in the application core.

---

# 13. Initial Alignment Prototype

Working script:

```text
prototypes/align_phone.py
```

Original working location:

```text
~/gotrack/custom/align_phone.py
```

Functions:

```text
load frame
load mesh
manual RX/RY/RZ
manual TX/TY/TZ
project mesh
save initial pose
```

Outputs:

```text
initial_pose.npy
initial_pose.txt
overlay.png
```

Typical initial orientation for the phone mesh was around a 90-degree X rotation, but actual pose depends on the shot.

---

# 14. Single-Frame Refinement Prototype

Working script:

```text
prototypes/refine_one.py
```

Original working location:

```text
~/gotrack/custom/refine_one.py
```

Required launch from GoTrack root:

```bash
PYTHONPATH=. python custom/refine_one.py
```

or equivalent Python path setup.

Full working launch includes the WSL CUDA library path:

```bash
LD_LIBRARY_PATH=$HOME/wsl-libs:/usr/lib/wsl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
PYTHONPATH=. \
python custom/refine_one.py
```

Verified result:

GoTrack significantly improves an approximately manually aligned pose on the test phone frame.

The refined projected contour was substantially closer to the real object than the initial manual contour.

It was not pixel-perfect.

Likely remaining contributors include:

```text
approximate focal length
approximate SAM3D geometry
reflective / low-feature phone surfaces
```

---

# 15. Sequence Tracking Prototype

Working script:

```text
prototypes/track_sequence.py
```

Original working location:

```text
~/gotrack/custom/track_sequence.py
```

Current tracking strategy:

```text
initial pose frame N
        |
        v
GoTrack refine
        |
        v
refined pose frame N
        |
        v
initial pose frame N+1
```

This is pose propagation through the sequence.

---

# 16. Verified Tracking Behaviour

A test of approximately 50–60 consecutive frames was run successfully.

Sequence content included:

```text
phone stationary
hand enters
hand partially occludes phone
hand grabs phone
phone begins moving
phone is lifted
phone partially leaves image boundaries
```

Observed result:

```text
tracking quality: visually very good
```

Specific observations:

```text
small pose jitter exists while phone is stationary

hand occlusion was handled without explicit masks

partial object exit outside frame was handled successfully

tracking remained stable through the important pickup motion
```

This is the current regression baseline.

---

# 17. Masks

No masks were required for the successful sequence test.

Therefore current GoTrack behavior without masks is the reference behavior.

Future object-mask or occlusion-mask integration must not change this default path unless explicitly enabled.

Object masks and occlusion masks should remain separate concepts.

---

# 18. GoTrack Mesh Units

GoTrack/BOP pose translations operate in millimeter-scale coordinates.

The renderer internally converts mesh geometry to meters for pyrender and converts depth back to millimeters.

Do not pre-convert the application pose translation to meters before passing through the current GoTrack path.

---

# 19. Current Prototype Outputs

Sequence prototype produces:

```text
poses/
    0000.npy
    0000.txt
    ...

overlays/
    0000.png
    ...

tracking.csv

tracking_preview.mp4
```

Current CSV diagnostics include:

```text
frame
score
translation_delta_mm
rotation_delta_deg
tx
ty
tz
```

---

# 20. Known Non-Issues / Expected Warnings

The following messages are currently expected:

```text
pkg_resources deprecation warning
xFormers is available
```

These are not considered failures.

---

# 21. Known Risks

## Approximate camera intrinsics

Current `fx/fy=1867` is estimated.

Systematic pose/scale error may come from incorrect intrinsics rather than the tracker.

---

## Approximate mesh geometry

The phone mesh originated from single-view reconstruction.

Invisible geometry may be approximate.

The mesh was manually rescaled to known-like dimensions.

---

## Reflective object

A smartphone contains:

```text
reflective surfaces
feature-poor glass
screen content
specular highlights
```

Despite this, current tests worked well.

---

## Tracking drift

Frame-to-frame propagation can accumulate errors.

Pose anchors and range retracking are therefore important application features.

---

# 22. Regression Dataset

Recommended local regression dataset:

```text
testdata_local/phone/
```

Suggested contents:

```text
frames/
phone_mm.ply
initial_pose.npy
camera.json
```

Keep real production footage out of git unless explicitly allowed.

Add:

```text
testdata_local/
```

to `.gitignore`.

A short 10–15 frame subset may be used for fast development regression tests.

The full 50–60 frame sequence should remain available for more complete integration testing.

---

# 23. Regression Rule

Before and after major refactoring, compare the new backend against the known-working prototype using identical:

```text
frames
mesh
camera intrinsics
initial pose
checkpoint
```

Unexpected large pose differences should be investigated.

Do not assume the rewritten implementation is correct merely because it runs.

---

# 24. License Note

The current GoTrack repository/checkpoint uses a non-commercial license.

Before using GoTrack itself in commercial production, verify licensing requirements separately.

The application architecture should make replacing GoTrack with another backend possible.

---

# 25. Important Instruction for Codex

When something in the current setup looks unusual, assume first that it may exist because of an already-discovered compatibility issue.

Before removing or replacing:

```text
environment variables
patches
unit conversions
coordinate conversions
PYTHONPATH handling
renderer setup
```

check this document and the prototype scripts.

Preserving the known-working tracking pipeline is more important than making the code look cleaner.