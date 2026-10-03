# Validation — 2026-10-02

## GPU regression against unchanged prototypes

Pinned GoTrack commit: `68f76055755f2a4a8967e13ece834f975f008bdf`.
Only the documented DINOv2 duplicate hub-load patch is applied, in an output-local
runtime copy. Upstream and prototype files remain unchanged.

Checkpoint SHA-256:
`f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086`.

Inputs: `testdata/phone_short`, prepared phone PLY, original initial pose,
1920x1080 frames, K = (1867, 1867, 960, 540). Four independent processes run
prototype refine, adapter refine, prototype sequence and adapter sequence.
Python/NumPy/torch/OpenCV seeds are 12345; crop seed remains the prototype's 12345.

| Comparison | Frames | Maximum matrix difference | Maximum score difference |
|---|---:|---:|---:|
| Single-frame prototype / adapter | 1 | 0 | 0 |
| Full sequence prototype / adapter | 60 | 0 | 0 |
| Recheck after startup diagnostics/metadata | 12 + refine | 0 | 0 |

No tolerance was needed for observed matrix equality. Harness failure thresholds
are 0.1 mm, 0.1 degrees and 0.001 score. This establishes parity with the reference,
not absolute accuracy of the approximate camera or reconstructed mesh.

Detailed local artifacts:

- `outputs/regression_full/report.json`, stage logs, environment JSON and overlays.
- `outputs/regression_final/report.json` for the final initialization path.
- [Compact machine-readable summary](validation_summary.json).

The first runtime attempt exposed a missing `dataloader` copy and a namespace
package check using `__file__`; both were fixed before inference comparison.
The report reader also needed CSV parsing: GoTrack writes BOP CSV despite the
internal file's `.json` extension. Neither fix changed upstream inference.

## Browser acceptance with real GPU

A headless Chromium/Playwright run completed the following entirely through the UI:

- Create a project, configure camera, import initial NPY pose.
- Refine and explicitly accept the result.
- Start a 60-frame range; cancel after four committed frames.
- Reopen and resume to all 60 frames without recomputing saved frames.
- Retrack frames 2–3; frame 59 remains byte-equivalent in project state.
- Export NPY/NPZ/JSON/CSV, generate MP4, save and reopen.
- Navigate to frame 45 and inspect overlay and diagnostics.

No JavaScript page errors occurred. Screenshot and report:
`outputs/browser-smoke-1790929250015/`.

Follow-up browser checks passed for manual-pose autosave during frame navigation,
independent object/occlusion mask matching and rendering, wireframe and shaded
preview. Artifacts: `outputs/ui_checks/`. Synthetic masks were removed from the
active project mapping after the check. Sampled shaded display is not a full
mesh rasterizer and is not used by tracking.

## Process interruption and recovery

`recovery_smoke.py` launched a separate server, started real tracking, terminated
that process during the range, then launched a new process with the same project.
The new process recognized the interrupted job, resumed through frame 7, and
preserved all previously committed entries exactly. CUDA/libcuda/EGL diagnostics
and the checkpoint identity were verified in persisted backend metadata.

Artifacts: `outputs/recovery-1790929539953289847/`.

An additional unit test covers the narrower crash window after a raw per-frame
record commits but before project.json commits. Project reopen reconciles the
contiguous raw-record journal prefix so those poses are not recomputed.

## CPU / service checks

19 unittest tests pass, covering:

- Canonical transforms, axes, projection and dT/dR versus prototype functions.
- Prepared phone geometry/pose loading, crop sampling and pose propagation.
- Backward iteration, cancellation and first-frame delta semantics.
- Project roundtrip, natural numbering, Windows paths and source protection.
- Scene-node transforms, mesh preparation and mask matching.
- Automatic video extraction: four synthetic frames at 12 fps; source MP4 unchanged.
- Refine accept/reject, cancel/reopen/resume, failure persistence and range isolation.
- Crash recovery from committed raw records not yet indexed by project.json.
- Export preserves float64 manual-pose precision.

The failure-persistence test deliberately injects an exception; its logged
traceback is expected. Existing ffmpeg runs may print ncurses/libtinfo version
warnings inherited from the conda library path; extraction completed successfully.

## Remaining boundaries

Nonzero camera distortion, EXR, mixed-shot pattern selection, smoothing,
relocalization and masks influencing GoTrack are not implemented. GLB preparation
has geometry tests, but phone tracking equivalence is established on the original
prepared PLY path. The local server supports one active project/job and should be
used on the local machine or trusted LAN/Tailscale network.

## Browser UI redesign — 2026-10-02

The redesigned interface passed 21 Playwright checks with real GoTrack GPU inference.
The run created a project through File > New, browsed server image sequences,
uploaded a mesh, edited pose using Global/Local translation handles and a rotation
ring, refined/accepted, tracked, cancelled, reopened and resumed without changing
committed poses, and retracked a backward range without changing outside poses.

It also covered explicit invalidation on source/mesh replacement, preserved
solution restoration, multiple native uploads for object/occlusion masks and source
frames, independent mask visibility, pose exports, MP4 preview, keyboard shortcuts,
resizable panels, and reload at 1440x900 and 1920x1080. No browser page errors.
Artifacts: `outputs/ui-redesign-1790965183910/` (report and screenshots).

23 Python unit tests pass, including four new resource lifecycle/upload tests.
`node tests/pose_math.mjs` passes Local/Global transform, Euler and perspective
axis-constraint checks. JavaScript syntax and `git diff --check` pass. Inference
adapter, unchanged prototypes and upstream GoTrack have no source diff in this
redesign. The earlier 60-frame parity result above remains the inference baseline.

The port 8765 server was restarted with the user's existing `0370_test1` project.
A separate Chromium check loaded its 1920-pixel frame with no page errors.

## Mesh units and Viewer fixes — 2026-10-03

26 Python tests pass. New checks cover meters-to-mm GLB conversion, centered
geometry, cumulative relative scale, unchanged source bytes, and contour/wireframe/
shaded display of triangles crossing the viewport when all vertices lie outside, including near-plane crossings.
Prepared PLY with no requested transformation still copies byte-for-byte.

Chromium checks passed on the actual phone GLB: dimensions become approximately
498 × 149 × 999 mm, initial rotation is identity, two ×10 operations accumulate
to ×100, and numeric pose edits remain synchronized under delayed frame responses.
Artifact: `outputs/mesh-ui-1791045092495/`.

The physical dimensions of this GLB are not inferred to be a real phone: its
normalized model coordinates multiplied by the declared meters still need known
real-world dimensions supplied by the user. Existing project assets are not
automatically rescaled; Rebuild with Units provides explicit correction with the
normal dependency invalidation options. Tracking/refinement source is unchanged.

## Nuke geometry and animation export — 2026-10-03

27 Python tests pass. The new exporter test verifies coordinate conversion,
consistent mesh/translation scaling, nonconsecutive frame offsets, read-only
source/project state, archive contents, and rejection of invalid scale or empty
accepted poses. Browser export at scale 0.001 / first frame 1001 passed and the ZIP
was downloaded successfully (`outputs/nuke-ui-report.json`).

Actual installed Nuke 17.0v1 opened the exported .nk without parser errors.
Eight real tracking keys matched native Axis2 world_matrix values with maximum
component difference 1.32e-8. A separate triangle render with fx=1867, fy=1600,
cx=1100, cy=400 at 1920x1080 matched expected OpenCV silhouette bounds within
1.86 pixels (edge rasterization). Artifacts: `outputs/nuke-acceptance/`.
This also verified file-path resolution and Axis2 -> TransformGeo wiring.

Initial test exports exposed missing outer braces around serialized Matrix knobs
and the vertical film-window offset's normalization by horizontal film width.
Both were corrected before these validation results. No inference, project pose
conventions or source geometry were modified. OBJ material/texture reconstruction,
world-camera solving and quaternion interpolation at subframes are outside this
exporter's scope.

## Automatic export downloads — 2026-10-03

Chromium verified exactly one automatic download for each export button:
`poses_package.zip`, `nuke_package.zip`, and the completed preview MP4. Further
polling and page reload triggered no additional downloads. Pose ZIP contains the
four NPY/NPZ/JSON/CSV files; its NPY data retains manual-pose precision.
27 Python tests pass. Artifact: `outputs/auto-download-report.json`.
