# Initial audit and extraction boundary

Audit date: 2026-10-02. Specification: `SPEC.md`, `KNOWN_WORKING.md`.

## Repository and environment

- Initial git status: `.gitmodules` and `external/gotrack` staged;
  specification, known-working notes, prototypes and testdata untracked.
  Existing index entries and source files are preserved.
- Both `external/gotrack` and the working `~/gotrack` are at
  `68f76055755f2a4a8967e13ece834f975f008bdf`.
- The submodule is clean, but lacks the local working DINOv2 fix.
  `patches/gotrack.patch` records the exact diff from the working checkout;
  it has NOT been applied to the submodule.
- Existing conda interpreter: `~/miniconda3/envs/gotrack/bin/python`.
  Observed NumPy 1.23.5, trimesh 5.1.0, PyTorch 2.0.1+cu117,
  setuptools 80.9.0. CUDA available, NVIDIA GeForce RTX 4060 Ti.
- Checkpoint is present under `external/gotrack`, DINOv2 weights in the
  documented torch cache, and the WSL `libcuda.so` symlink exists.
  Presence and CUDA visibility are not a full model/EGL smoke test.
- `testdata/phone_short` contains 61 PNGs, prepared `phone_mm.ply`, original
  `phone.glb`, initial and refined poses. No camera JSON is provided;
  the baseline intrinsics are explicitly available in both prototypes.

## First implementation stage

Extract only pure geometry and backend-independent data contracts:

1. `Pose`: `T_cam_from_object`, OpenCV axes, millimeters, preserve input float
   precision and reject invalid rigid transforms without correcting them.
2. Euler editing helper (`Rz @ Ry @ Rx`), pinhole camera and projection.
3. Read-only mesh loading with the same trimesh defaults as the prototypes.
   Already prepared mesh coordinates are interpreted as mm without rescaling.
4. Raw dT/dR diagnostics, result/status/range types and separate mask fields.
5. CPU reference comparisons using function bodies read from the unchanged
   prototypes; include the supplied prepared phone mesh and both saved poses.

This stage does not introduce a server, GUI, model loader, dependency install,
mesh preparation/export or a new tracking implementation. It does not establish
GPU tracking equivalence. Core does not import torch, pyrender or GoTrack.

## Ready for subsequent extraction

The following existing blocks can be moved into a small GoTrack adapter:

- `make_inputs`: uint8 RGB -> CHW float32 / 255; batch of one; exact int32
  labels/frame/instance IDs; clone pose for camera and world transforms.
- Fixed-seed subset: cast mesh vertices to float32 first, sample 1000 without
  replacement using `default_rng(12345)` only when more than 1000 exist.
  This is a GoTrack crop choice, not a core mesh abstraction.
- Camera adapter: identity float32 `T_world_from_eye`, explicit dimensions and K.
- Model initialization: YAML configuration, debug false, checkpoint key
  `model_state_dict`, prefix `models.1.`, `.cuda().eval()`, resident model.
- Renderer from the original/prepared asset path, preserving material/colors;
  `object_vertices={1: crop_vertices}`, required output directory and filename.
- Inference-mode forward call; extraction of raw pose/score; incremental
  per-frame persistence; propagated refined pose as next initialization.

Implement the minimal backend interface alongside this concrete adapter,
with capabilities indicating that both semantic mask inputs are unsupported
by the baseline. Do not silently apply them to correspondence generation.

## Behavior and risks requiring regression

- Preserve vertex ordering, trimesh processing defaults and crop sampling.
  Changing ordering changes the selected 1000 vertices even with the same seed.
- Preserve RGB conversion, dtype, tensor layout, identity world camera and IDs.
- Renderer divides mesh vertices and camera translation by 1000, then restores
  rendered depth to mm. OpenCV/OpenGL conversion is confined to the renderer.
  Application poses must NOT be preconverted to meters.
- Preserve YAML and option defaults: five test iterations, 280x280 crop,
  gray background, re-crop each iteration, padding, SSAA and PnP settings.
- PnP uses NumPy random correspondence sampling as well as OpenCV RANSAC.
  Mesh seed alone does not make the complete inference deterministic.
- `forward_pipeline` writes internal results and invokes visualization even
  with `debug=False`. Sequence prototype disables `get_vis_tiles`; refinement
  prototype does not. Preserve each reference path during comparison.
- First sequence dT/dR are zero; subsequent values compare adjacent refined
  poses. Single-frame refinement deltas compare refined versus initial pose.
- A failed PnP can return identity/zero score. Do not silently add fallback,
  smoothing, threshold rejection or relocalization to the baseline.
- Legacy Scene loading concatenates geometries without node transforms.
  This does not affect the verified PLY. Full GLB/GLTF scene preparation needs
  separate tests; this stage deliberately preserves the reference loader.
- Both prototypes need GoTrack imports and config/checkpoint paths resolved
  from a GoTrack checkout. `track_sequence.py` has hardcoded paths and its
  repo-relative `ROOT` no longer points at GoTrack in this layout. Use an
  external regression harness to set its module globals and working directory;
  leave the reference source unchanged.
- Keep EGL setup before OpenGL imports, existing LD_LIBRARY_PATH and DINOv2
  patch. Do not update dependencies or install a Linux NVIDIA driver.

## Next stage and GPU regression gate

1. Reproduce the existing DINOv2 patch via a documented patched runtime copy
   or the separate patch; preserve upstream/app separation.
2. Implement lazy resident `GoTrackBackend` with explicit source/checkpoint/
   output paths, lifecycle cleanup and the exact input-construction path.
3. Run original `refine_one.py` and the adapter with identical first image,
   prepared mesh, K, initial pose and checkpoint. Record seeds, hashes,
   versions, pose differences, score differences and runtime.
4. Run original `track_sequence.py` through a harness overriding only paths,
   intrinsics and frame limit, then the adapter on the same 10–15 frames.
   Compare every raw pose; control RNG identically, quantify dT and relative
   rotation, inspect overlays if deviations occur. Repeat on the full sequence
   before claiming preservation of occlusion/pickup behavior.
5. Do not proceed to project/server/viewer while significant differences remain
   unexplained. CPU geometry checks are not a substitute for this gate.

## Subsequent implementation

The first-stage boundary above is historical. The resident backend, regression
harness, project/input layer, local HTTP server, browser workflow, masks, exports
and recovery are now implemented. See [validation](VALIDATION.md) and
[README](../README.md) for current behavior and remaining limits.
