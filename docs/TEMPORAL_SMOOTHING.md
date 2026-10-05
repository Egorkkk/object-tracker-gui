# Temporal pose smoothing

Temporal smoothing is a CPU-only, non-destructive post-processing stage. Raw
accepted tracking remains in `poses` and its per-attempt journal. Filtering does
not call GoTrack, mutate anchors, or modify tracker propagation. Old projects
start with filtering disabled and existing exports continue to default to raw.

## Methods

Translation uses a Gaussian-weighted local linear fit centered on each frame.
The offline window uses past and future samples. Unlike a box average, the fit
reproduces constant-velocity translation even at the ends of a track. At full
strength the Gaussian standard deviation is 3.25 frames; at the default strength
of 0.6 it is 2.05 frames. Windows extend three standard deviations in each
direction and never cross missing frame indices.

Rotation matrices are converted to normalized XYZW quaternions. Quaternion
signs are made temporally consistent. Each local fit works on shortest-path
rotation vectors relative to a pilot orientation (the SO(3) logarithm), then
maps the fitted vector back through the exponential and a normalized quaternion
to a rotation matrix. Euler angles are only used by the existing inspector and
export presentation. Matrices and Euler channels are never averaged.

Both outputs remain rigid transforms. A zero component slider leaves that
component raw; disabling smoothing or setting overall strength to zero leaves
the complete pose raw.

## GoTrack quality and confidence

The adapter preserves scalar metrics from the final refinement iteration under
`backend_diagnostics.frame_quality`. Capture is scoped to the single inference
worker, leaves solver inputs and correspondence sampling unchanged, and restores
all hooks on exit. The ordinary stored `score` is GoTrack's weighted inlier
quality, rather than a separate feature-confidence probability.

| Input | Normalization | Default weight |
| --- | --- | --- |
| Weighted inlier quality / legacy score | Clamp to 0–1 | 1 |
| Mean confidence of valid correspondences | Clamp to 0–1 | 0.5 |
| Evaluated inlier ratio | Clamp to 0–1 | 1 |
| Number of valid correspondences | Clamp count / 100 to 0–1 | 0.25 |
| Median reprojection error, crop-camera pixels | 1 / (1 + (error / 4)²) | 0.5 |

Reliability is the weighted mean of available normalized inputs. Missing or
non-finite inputs are omitted and the remaining weights are renormalized. A
known failure gives reliability zero. With no usable quality inputs, reliability
is neutral (1), giving normal mathematical smoothing. The generic `FrameQuality`
and `reliability(..., weights=...)` API allows inputs to be disabled/reweighted.
Reliability is a heuristic, not a calibrated probability.

The adapter also saves evaluated inlier count, selected correspondence count,
and the too-few-correspondences / PnP failure flag. GoTrack evaluates inliers over
all valid correspondences, not just the selected RANSAC subset. Its upstream
failure arrays can contain duplicate entries, and too-few correspondences may
report `failed=False`; capture accounts for both. Median reprojection error is
recovered from the same point residual calculation because upstream returns an
RGB error visualization instead of its scalar median. This small CPU calculation
adds no image analysis or inference.

## Adaptive behavior

Translation speed is measured in mm/s and angular speed in degrees/s, using the
project FPS. With confidence weighting enabled, unreliable pilot poses are
interpolated between measurements with reliability at least 0.35 (translation
interpolation and quaternion SLERP), preventing a bad pose from being interpreted
as deliberate motion. This pilot is only used to estimate motion and select the
rotation reference; the final fit still uses the raw measurements.

Adaptive smoothing scales the window by `1 / (1 + (speed / threshold)²)`.
Translation and rotation have separate speed thresholds, defaulting to 100 mm/s
and 90 degrees/s. Static sections get broad windows and high-speed sections get
narrow windows, without introducing a causal delay. Confidence independently
sets a lower bound of `1 - reliability` on the window factor, so low-confidence
measurements remain broadly smoothed even if their apparent motion is high.
Measurement weights are multiplied by `max(reliability, 0.001)²`. Confidence
weighting and motion adaptation can each be disabled independently.

## UI and manual shot check

1. Open an existing project with accepted tracked poses. Select **REVIEW** or
   **Tracking → Temporal Smoothing** in the Project Tree.
2. Enable smoothing. Adjust **Strength**, **Translation**, and **Rotation**.
   Changes recompute on control release/change; **Recompute** also forces a
   fresh cache. Confirm that no tracking job starts.
3. Choose **Filtered** in the viewer's **Pose** selector or smoothing inspector.
   Scrub/play, then switch to **Raw** to compare the green result overlay.
   ALIGN uses raw poses; filtered poses cannot be edited through the gizmo or
   Current Pose fields. The yellow Initial overlay retains its existing meaning.
4. Open **Diagnostics** below the timeline. Reliability and translation
   correction are added by default. Enable translation speed, angular speed,
   and rotation correction as needed. Each graph shows its units, maximum and
   current playhead; clicking a graph selects the corresponding frame. Per-frame
   reliability and correction are also shown in the viewer status bar.
5. Toggle **Adaptive to motion** and **Use tracking confidence** independently.
   Under **Advanced**, adjust the two motion thresholds if the shot's scale
   or speed requires it.
6. Optionally set timeline IN / OUT, enable **Use timeline IN / OUT**, and click
   **Recompute**. Outside the saved range poses remain raw. Corrections smoothly
   ramp from zero at each range endpoint over three frames. Editing IN / OUT
   alone does not silently change the saved filter range; Recompute applies it.
7. Use **Reset Defaults** to restore default filtering parameters and full range.
   Disable filtering to return to raw preview.
8. In **Exports**, explicitly choose **Raw** or **Filtered**, then export poses,
   Nuke package or MP4. JSON and Nuke animation metadata identify the source and
   include filter parameters for filtered output. Raw diagnostic fields retain
   their meaning as tracking measurements.
9. Save and reopen. Parameters, selected preview, filtered matrices and
   diagnostics survive reload. Changing accepted poses, FPS, settings or tracked
   frame coverage invalidates the cache and recomputes it when idle or needed.

## Persistence and reusable API

`temporal` contains `parameters`, `preview`, `filtered_poses`, `diagnostics` and
`source_signature`. The signature covers raw pose records (including quality),
FPS, frame coverage and parameters. Raw journals remain untouched. Cache
invalidation uses the existing project save path, and idle snapshots lazily
refresh derived results. Preserved solutions keep their temporal state through
the existing complete-state serialization. Unaccepted drafts/refinements do not
become filter inputs until accepted or stored as an anchor.

`filter_poses({frame_index: matrix}, TemporalParameters(...), quality, fps)`
returns independent pose arrays and per-frame diagnostics without accessing a
project, tracker, UI, GPU or source images. `temporal/state.py` provides the
project adapter; `/api/temporal` is the application entry point.

## Validation and limitations

Run:

```bash
scripts/run_python.sh -m unittest discover -s tests -q
node tests/pose_math.mjs
scripts/run_python.sh scripts/validate_temporal_pnp.py
```

The numerical tests cover translation/rotation jitter reduction, ±180-degree
wraparound, quaternion sign ambiguity, constant-velocity boundaries, intentional
fast motion, low-confidence outliers, rigidity, missing metrics, gaps, singleton
tracks, selected ranges and disabled/zero-strength behavior. Integration tests
cover raw immutability, recomputation without loading a tracker, cache
invalidation, save/reload, viewer source and raw/filtered exports. The optional
PnP regression checks actual CPU solver outputs for random and top-confidence
selection, plus the upstream too-few-points fallback.

Validation completed: 49 Python tests, JavaScript pose checks, and eight browser
workflow checks with no JavaScript exceptions. Synthetic 3,000-frame filtering
completed in approximately 0.6 seconds on the local CPU. Browser artifacts are
in `outputs/temporal-validation/` (ignored generated output).

Existing shots retain their legacy score but cannot recover discarded quality
metrics without tracking again. No independent true visibility ratio is exposed
by this adapter; valid correspondence confidence is not labeled as visibility.
There is no optical flow analysis, blur detection, silhouette analysis, new CV
network, or image-aware filtering. Missing poses are not synthesized across gaps.
Very short selected ranges have little or no interior correction. Windowed
local-linear fits can overshoot near abrupt motion when adaptation is disabled;
rotation motion exceeding 180 degrees between samples is inherently ambiguous.
The full GPU tracking pipeline was not rerun for this change; actual PnP capture
and the existing tracking service tests verify the changed tracking boundary.

## Files changed for this feature

Added:
- `src/object_tracker/temporal/{__init__,confidence,pose_filter,state}.py`
- `src/object_tracker/backends/gotrack/quality.py`
- `tests/test_temporal.py`, `tests/test_temporal_integration.py`
- `scripts/validate_temporal_pnp.py`, this document

Modified:
- `frontend/app.js`, `frontend/index.html`, `frontend/style.css`
- `src/object_tracker/app/service.py`, `src/object_tracker/app/server.py`
- `src/object_tracker/backends/gotrack/backend.py`
- `src/object_tracker/core/project.py`, `src/object_tracker/core/viewer.py`
- `src/object_tracker/core/export.py`, `src/object_tracker/core/export_nuke.py`
- `README.md`

Pre-existing changes in jitter research, correspondence selection and service
selection tests were retained.
