# Object Tracker GUI — UI/UX Redesign Specification

## 1. Purpose

This document specifies the redesign of the existing Object Tracker browser GUI.

The application backend and basic tracking workflow are already functional.

The goal of this phase is **not to rewrite the tracking application**, but to turn the current technical/debug-oriented interface into a compact professional VFX tracking workspace.

Visual and interaction references:

- Nuke
- DaVinci Resolve
- Mocha Pro

Do **not** use SynthEyes or Houdini as UI references.

The references are conceptual only.

Do not copy any application literally.

---

# 2. Current State

The existing browser GUI already supports the core application workflow.

The current interface is functional but has several UX problems:

- project creation controls occupy permanent workspace;
- source and mesh paths behave too much like configuration fields;
- replacing project assets is cumbersome;
- file/folder browser dialogs are custom and inconvenient;
- viewer controls are not organized like a professional tracking application;
- timeline is too primitive;
- project data, tracking controls and setup parameters are mixed together;
- there is no clear application mode/context;
- manual pose manipulation needs a proper viewer interaction model.

The redesign should preserve all working backend functionality.

---

# 3. Platform Decision

Keep the current:

```text
browser frontend
+
FastAPI/backend in WSL2
```

architecture.

Do **not** migrate to a native Windows GUI at this stage.

The browser platform is considered sufficient for:

- source browsing;
- uploads;
- mesh replacement;
- viewer interaction;
- 3D gizmos;
- timeline;
- diagnostic graphs;
- keyboard shortcuts;
- drag-and-drop;
- tracking progress.

Do not introduce Electron or another desktop shell.

If a genuine browser limitation is discovered that prevents a required workflow, document the limitation and discuss it before changing application architecture.

---

# 4. Overall Visual Direction

Target appearance:

```text
modern Resolve-like structure
+
Nuke/Mocha-like information density
```

The application should feel like a professional VFX utility, not a generic web dashboard.

Characteristics:

- dark theme only;
- compact controls;
- restrained spacing;
- high information density;
- clear panel hierarchy;
- small/medium text;
- subtle borders;
- minimal decorative UI;
- color used mainly for state/status.

Avoid:

- oversized web buttons;
- card-based SaaS/dashboard appearance;
- large empty margins;
- rounded “mobile app” controls;
- excessive animations;
- very large typography.

---

# 5. Main Workspace Layout

Use the following primary layout:

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ Application Bar / Project / Backend Status                               │
├───────────────────────────────────────────────────────────────────────────┤
│ Main Toolbar                                                              │
├───────────────┬───────────────────────────────────────────┬───────────────┤
│               │                                           │               │
│ PROJECT TREE  │                  VIEWER                   │   INSPECTOR   │
│               │                                           │               │
│               │                                           │               │
├───────────────┴───────────────────────────────────────────┴───────────────┤
│ Playback / Frame Controls                                                 │
├───────────────────────────────────────────────────────────────────────────┤
│ TIMELINE                                                                  │
├───────────────────────────────────────────────────────────────────────────┤
│ DIAGNOSTICS — collapsible                                                 │
└───────────────────────────────────────────────────────────────────────────┘
```

---

# 6. Panel Resizing

The three main areas:

```text
Project Tree
Viewer
Inspector
```

must be resizable using draggable splitters.

The timeline height should also be resizable.

Requirements:

- no full docking system;
- no floating windows;
- no arbitrary panel rearrangement;
- panel sizes should persist in browser/local project preferences.

This phase does not require collapsible left/right panels.

---

# 7. Application Bar

The top application bar should be compact.

Suggested contents:

```text
Object Tracker
Project Name

File
Edit
View

Backend: GoTrack
GPU/status indicator

Save
```

Project save state should be visible.

Example:

```text
Object Tracker | 0370_test1                    GoTrack ● Connected    Save
```

Avoid putting tracking controls in the application bar.

---

# 8. Main Toolbar

Below the application bar, provide a compact toolbar for frequent operations.

Suggested structure:

```text
[Source ▼] [Mesh ▼]

[Align]
[Refine]

[Track →]
[← Track]

[Export]
```

Exact icons/text may be adjusted.

The toolbar should expose frequent actions without duplicating every Inspector control.

---

# 9. Application Modes

Use three clear high-level working modes:

```text
ALIGN
TRACK
REVIEW
```

Do not introduce separate Setup/Refine/etc. modes.

The active mode may be represented using toolbar tabs/buttons.

---

## ALIGN

Focus:

- camera setup;
- mesh pose;
- manual alignment;
- initial pose refinement.

---

## TRACK

Focus:

- tracking range;
- anchors;
- track/retrack;
- progress;
- masks.

---

## REVIEW

Focus:

- playback;
- overlays;
- diagnostics;
- warnings;
- pose inspection;
- export preparation.

Changing mode should primarily change available tools and Inspector context.

It should not replace the entire page.

---

# 10. Project Tree

The left panel becomes a persistent project hierarchy.

Keep all currently relevant project entities visible.

Suggested structure:

```text
PROJECT

▼ Media
    Source

▼ Geometry
    Mesh

▼ Camera
    Camera

▼ Masks
    Object Mask
    Occlusion Mask

▼ Tracking
    Active Solution
    Other Solutions

▼ Pose Anchors
    Frame 0010
    Frame 0043
    ...

▼ Exports
```

The structure can evolve later but all these categories should currently remain available.

---

# 11. Project Tree Selection

The Project Tree is the primary context selection mechanism.

Selecting:

```text
Source
```

shows source properties in Inspector.

Selecting:

```text
Mesh
```

shows mesh properties.

Selecting:

```text
Camera
```

shows camera intrinsics.

Selecting:

```text
Object Mask
```

shows mask source settings.

Selecting a pose anchor shows anchor details.

Selecting a tracking solution shows tracking settings/status.

---

# 12. Inspector

The right Inspector is context-sensitive.

However, a compact current-pose section must remain permanently available.

The Inspector therefore has:

```text
Context Properties

--------------------

Current Pose
RX
RY
RZ

TX
TY
TZ

Coordinate Space:
[Local | Global]
```

Pose controls remain visible regardless of which project item is selected.

---

# 13. Pose Controls

Pose editing must support:

```text
RX
RY
RZ
TX
TY
TZ
```

Use compact numeric controls.

Sliders may be available but should not dominate the Inspector.

Numeric fields should support:

- direct typing;
- click/drag numeric adjustment if practical;
- keyboard stepping;
- modifier-based fine adjustment.

---

# 14. Coordinate Space

Manual manipulation must support:

```text
LOCAL
GLOBAL
```

coordinate systems.

Meaning:

### Local

Transform gizmo follows the current object orientation.

### Global

Transform gizmo remains aligned with the application/camera world axes.

A visible selector must exist:

```text
Coordinate Space
[Local] [Global]
```

This affects interactive gizmo manipulation.

It must not alter the underlying canonical pose convention.

---

# 15. Viewer

The viewer is the primary workspace and should receive the majority of screen area.

Do not surround it with configuration forms.

Provide a compact viewer toolbar directly above it.

Example:

```text
View: [RGB ▼]

Overlay:
[Initial ●]
[Refined ●]
[Masks ▼]

Display:
[Wireframe ▼]

Opacity ─────○────

[Fit] [1:1]
```

---

# 16. Viewer Overlay Modes

Support:

```text
RGB only

Initial Pose
Refined Pose
Initial + Refined

Object Mask
Occlusion Mask

Wireframe
Shaded Mesh
```

Mask controls can use one dropdown while retaining independent Object/Occlusion visibility.

---

# 17. Viewer Navigation

Support:

```text
Fit
100% / 1:1
Zoom
Pan
```

Mouse wheel may zoom unless reserved by the active transform tool.

Viewer navigation should follow conventions familiar from VFX applications where practical.

---

# 18. Interactive Pose Manipulation

Manual alignment should no longer depend primarily on sliders.

Implement a 3D transform gizmo in the viewer.

Required:

```text
translation gizmo
rotation gizmo
```

The UI may use separate transform tools or a combined gizmo.

Minimum requirement:

- visual XYZ axes;
- axis-constrained translation;
- axis-constrained rotation;
- mouse interaction;
- synchronized Inspector numeric values.

---

# 19. Gizmo Coordinate Modes

The transform gizmo must support:

```text
Local
Global
```

mode as defined above.

Changing the mode must update gizmo orientation immediately.

---

# 20. Transform Interaction

The exact mouse mapping may be chosen based on implementation quality.

Preferred concepts:

```text
drag axis handle -> constrained movement
drag plane handle -> 2D plane movement
drag rotation ring -> constrained rotation
```

Avoid hidden gesture-only manipulation when a visible gizmo can communicate the operation.

Direct viewer manipulation should coexist with numeric Inspector editing.

---

# 21. Playback Bar

Directly below Viewer:

```text
|◀   ◀   ▶/Pause   ▶   ▶|

Frame [ 42 ]

42 / 134
```

Exact button set may be simplified.

Required:

- previous frame;
- next frame;
- play/pause;
- direct frame entry;
- current/total frame display.

---

# 22. Timeline

Replace the current single status strip with a compact multi-track timeline.

Do not turn this into a full nonlinear editor.

Required tracks:

```text
Frames
Tracking
Anchors
Object Mask
Occlusion Mask
```

Warnings/failures may appear on the Tracking track.

---

# 23. Timeline Status Colors

Keep the current state-color semantics:

```text
gray   = untracked
green  = tracked
blue   = pose anchor
yellow = warning
red    = failed
```

Colors should remain clearly readable.

No need to intentionally desaturate them in this phase.

---

# 24. Timeline Range Selection

The tracking range should be selectable directly on the timeline.

Required concepts:

```text
range start
range end
selected region
```

Numeric Start/End controls may still exist as secondary controls.

The timeline should become the preferred selection method.

---

# 25. Timeline Controls

Provide:

```text
Start
End

Track Forward
Track Backward
Retrack Range

Use Anchors
```

These should be compact and visually associated with the timeline/tracking area.

Avoid scattering them through the page.

---

# 26. Timeline Interaction

Minimum required:

- click to change frame;
- visually select active frame;
- select tracking range;
- click an anchor;
- click warning/failed frames.

Nice-to-have later:

- zoom;
- horizontal pan.

Full advanced timeline navigation is not required for the first UI redesign.

---

# 27. Diagnostics Panel

Below the timeline, add a resizable/collapsible Diagnostics section.

Collapsed by default is acceptable.

It should provide graphs for:

```text
Score
ΔTranslation
ΔRotation
```

Allow switching graph visibility.

Example:

```text
Diagnostics ▼

[x] Score
[x] ΔT
[x] ΔR
```

Graph clicks should navigate to the associated frame.

---

# 28. Project Creation

Remove permanent project-creation controls from the main workspace.

Creating a project should happen through:

```text
File > New Project
```

or an equivalent initial dialog.

The dialog should collect only necessary initial information.

Possible fields:

```text
Project Name
Project Directory
Source optional
Mesh optional
FPS if needed
```

Source and Mesh can also be assigned later.

---

# 29. Opening Projects

Use:

```text
File > Open Project
```

and optionally:

```text
Recent Projects
```

Do not keep an “Open project” path entry permanently visible in the workspace.

---

# 30. Resource Replacement

Source, Mesh, Object Mask and Occlusion Mask must be replaceable after project creation.

Each relevant Inspector should have a permanently visible:

```text
Replace...
```

button.

Examples:

```text
Source
0370_plate
/mnt/e/...

[Replace Source...]
```

and:

```text
Mesh
phone_mm.ply
75 x 11 x 160 mm

[Replace Mesh...]
```

---

# 31. Resource Selection: Two Distinct Workflows

All relevant resource selectors should provide two explicit buttons:

```text
[Browse Server...]
[Upload...]
```

Do not merge them into one hidden menu.

Their meaning must remain distinct.

---

# 32. Browse Server

`Browse Server` selects data that already exists in the filesystem visible to the backend.

Examples:

```text
/home/...
/mnt/c/...
/mnt/d/...
/mnt/e/...
```

No files are copied.

This is the preferred method for large image sequences already stored on the workstation.

---

# 33. Upload

`Upload` selects files from the computer running the web browser and transfers them to the tracking server/project.

Examples:

- remote laptop through Tailscale;
- local Windows browser uploading a newly downloaded mesh;
- external masks.

Uploaded resources should go into a predictable project-local asset/cache location.

Never silently overwrite a same-named existing file.

---

# 34. Folder Upload

Folder upload for image/mask sequences is useful but is not mandatory for the immediate redesign.

Priority:

```text
1. Server folder browser
2. Single/multiple file upload
3. Folder upload
4. Drag/drop
```

---

# 35. Server File Browser — Visual Design

Replace the current custom path selection dialogs with a much more conventional Explorer-like dialog.

Reference concept:

```text
Windows Explorer
```

Do not attempt to open the actual native Windows Explorer dialog from the server.

Implement an in-app server filesystem browser with conventional layout.

---

# 36. File Browser Layout

Suggested layout:

```text
┌──────────────────────────────────────────────────────────────┐
│ Select Mesh                                                  │
├──────────────────────────────────────────────────────────────┤
│ ←  →  ↑   /mnt/e/POL/ASSETS/EP08/0370/                     │
├──────────────┬───────────────────────────────────────────────┤
│ Places       │ Name             Type     Size      Modified │
│              │                                               │
│ Home         │ 📁 frames                                   │
│ Project      │ 📁 masks                                    │
│ /mnt/c       │ phone.glb        GLB      4.2 MB    ...      │
│ /mnt/d       │ phone.ply        PLY      11 MB     ...      │
│ /mnt/e       │                                               │
│ Recent       │                                               │
├──────────────┴───────────────────────────────────────────────┤
│ File name: [ phone.glb                                  ]   │
│                                  [Cancel] [Open]             │
└──────────────────────────────────────────────────────────────┘
```

---

# 37. Server Browser Places

Left sidebar should include useful shortcuts:

```text
Home
Project
Recent

/mnt/c
/mnt/d
/mnt/e
```

Only show drives/paths that actually exist.

Optional:

```text
Favorites
```

may be added later.

---

# 38. File Browser Breadcrumb

Provide a standard editable or clickable breadcrumb/path bar.

Example:

```text
/mnt/e/POL/ASSETS/EP08/0370/
```

The user should be able to:

- click parent path segments;
- type/paste an absolute path;
- go to parent directory.

---

# 39. File Browser View

Use a details table by default:

```text
Name
Type
Size
Modified
```

Support sorting where practical.

Do not use icon-thumbnail mode as the default.

Image thumbnail browsing can be added later if it becomes useful.

---

# 40. Folder Selection

When selecting an image sequence directory, the browser enters Folder Selection mode.

The dialog should make it explicit that the selected object is a folder:

```text
[Select Folder]
```

not:

```text
[Open]
```

---

# 41. File Filters

Dialogs should apply context-specific filters.

Examples:

Mesh:

```text
*.ply
*.obj
*.glb
*.gltf
```

Video:

```text
*.mp4
*.mov
*.mkv
```

Project:

```text
project.json
```

Masks/images:

```text
*.png
*.jpg
*.jpeg
*.exr
```

Allow:

```text
All Files
```

where useful.

---

# 42. Upload Dialog

Uploads should use the browser's native file-selection dialog.

The application does not need to reproduce the OS file picker for client-side uploads.

After selection show:

```text
filename
size
upload progress
destination
```

---

# 43. Drag and Drop

Drag-and-drop is desirable but not mandatory for the immediate implementation.

Design APIs/UI so it can be added later.

Likely future targets:

```text
drop video/sequence -> Source
drop GLB/PLY -> Mesh
drop masks -> corresponding mask resource
```

Do not delay the main redesign waiting for drag-and-drop.

---

# 44. Source Replacement Behaviour

When replacing Source:

1. load/index the new source;
2. update resolution;
3. update frame count;
4. update source FPS where applicable;
5. refresh Viewer;
6. refresh Timeline.

Existing tracking results may become invalid.

Do not silently delete them.

---

# 45. Mesh Replacement Behaviour

When replacing Mesh:

1. load the new mesh;
2. show dimensions/statistics;
3. prepare display representation;
4. refresh viewer;
5. keep source mesh unchanged.

Existing tracking poses may correspond to the old mesh.

Do not silently treat them as valid for the new mesh.

---

# 46. Dependency Invalidation

Resource changes must explicitly invalidate dependent data.

Examples:

### Source changed

Potentially invalid:

```text
camera assumptions
tracking results
anchors
masks
```

depending on sequence compatibility.

### Mesh changed

Potentially invalid:

```text
tracking results
manual poses
anchors
```

### Camera intrinsics changed

Potentially invalid:

```text
tracking results
refined poses
```

---

# 47. Invalidation Dialog

When a change affects existing data, show a clear dialog.

Example:

```text
The mesh has changed.

Existing tracking results were calculated using:
phone_old.ply

Choose what to do:

[Keep old tracking as separate solution]
[Clear active tracking]
[Cancel]
```

Prefer preserving results rather than destructive automatic cleanup.

---

# 48. Avoid Excess Confirmation

Do not show confirmation dialogs for harmless changes.

Confirmation/invalidation is needed only when data may become semantically incompatible or be destroyed.

---

# 49. Compact Controls

Target UI sizing should be closer to Nuke/Mocha than a typical website.

Approximate guidance, not strict requirements:

```text
normal font:        ~12–13 px
small metadata:     ~11 px
control height:     ~26–30 px
toolbar height:     ~30–34 px
```

Use consistent spacing.

Do not mix large and small button styles arbitrarily.

---

# 50. Buttons

Use semantic hierarchy.

Primary actions:

```text
Refine
Track
Retrack
Accept
```

can have emphasized styling.

Secondary controls:

```text
Replace
Browse
Upload
Cancel
```

should be more restrained.

Avoid bright blue on every clickable element.

---

# 51. Forms

Properties should use compact two-column layouts:

```text
FX        [1867.000]
FY        [1867.000]
CX        [960.000]
CY        [540.000]
```

not vertically stacked web-form cards.

---

# 52. Metadata

Read-only metadata should not look like editable controls.

Example:

```text
Resolution     1920 × 1080
Frames         134
FPS            25
```

Avoid putting read-only values inside textbox-looking elements unless they are intentionally editable.

---

# 53. Visual Hierarchy

Primary hierarchy:

```text
Viewer
Timeline
Inspector
Project Tree
Diagnostics
```

The viewer should remain visually dominant.

Project-management UI should not compete with the image.

---

# 54. Color Theme

Dark theme only.

Use neutral dark grays.

Suggested conceptual layers:

```text
application background
panel background
control background
selected/hover
borders
```

Do not make the interface pure black.

Maintain enough luminance separation between panels to make the structure obvious.

---

# 55. Tracking Colors

Keep:

```text
tracked = green
anchor = blue
warning = yellow
failed = red
untracked = gray
```

Use the same semantic colors consistently across:

- timeline;
- diagnostics;
- status labels;
- viewer markers.

---

# 56. Tooltips

Compact icon controls should have tooltips.

Especially:

```text
viewer overlays
playback controls
gizmo modes
timeline actions
```

Do not require tooltips to understand primary text-labelled controls.

---

# 57. Keyboard Shortcuts

Implement basic useful shortcuts where safe.

Suggested initial set:

```text
Left / Right      previous / next frame
Space             play / pause
Ctrl+S            save project
```

Optional later:

```text
W                 translate tool
E                 rotate tool
L                 Local coordinates
G                 Global coordinates
```

Do not conflict with browser shortcuts without reason.

---

# 58. Responsive Behaviour

This is a desktop professional tool.

Do not optimize for mobile.

Target:

```text
1920x1080 and larger
```

It should remain usable around:

```text
1440x900
```

but desktop widescreen is the priority.

---

# 59. Do Not Rewrite Backend Logic

This phase is primarily frontend/interaction redesign.

Do not unnecessarily change:

```text
GoTrackBackend
pose mathematics
tracking propagation
project persistence
working inference code
```

unless required to support new resource replacement APIs.

Changes to backend functionality should remain small and independently testable.

---

# 60. Migration of Current UI

Current functional elements must be mapped into the new layout, not dropped.

Examples:

Current:

```text
project path fields
```

become:

```text
New/Open Project dialogs
Project Tree
Inspector
```

Current:

```text
mesh dimensions section
```

becomes:

```text
Mesh selected -> Inspector
```

Current:

```text
camera parameters
```

become:

```text
Camera selected -> Inspector
```

Current:

```text
Initial / Refined / Mask checkboxes
```

become:

```text
Viewer Toolbar
```

Current:

```text
range start/end
```

become:

```text
Timeline range + secondary numeric controls
```

---

# 61. Recommended Implementation Order

Do not redesign everything in one large change.

## Phase A — Layout

Implement:

```text
Application Bar
Toolbar
Project Tree
Viewer
Inspector
Timeline area
Diagnostics placeholder
resizable splitters
```

Reuse existing controls temporarily where needed.

---

## Phase B — Resource Management

Implement:

```text
Browse Server dialog
Upload
Replace Source
Replace Mesh
Replace Masks
invalidation logic
```

This is high priority.

---

## Phase C — Inspector Refactor

Move:

```text
mesh settings
camera settings
source settings
mask settings
tracking solution settings
```

to context-sensitive Inspector.

Add persistent Pose controls.

---

## Phase D — Viewer

Implement/refine:

```text
viewer toolbar
overlay selection
fit / zoom / pan
```

---

## Phase E — Interactive Alignment

Implement:

```text
translation gizmo
rotation gizmo
Local / Global coordinates
synchronization with pose numeric fields
```

---

## Phase F — Timeline

Implement multi-track timeline:

```text
Tracking
Anchors
Object Mask
Occlusion Mask
```

and range selection.

---

## Phase G — Diagnostics

Implement collapsible graphs:

```text
Score
ΔT
ΔR
```

---

## Phase H — Polish

Only after the workflows are stable:

```text
spacing
icons
hover states
shortcuts
minor animations
drag/drop
```

---

# 62. Acceptance Criteria

The redesigned UI is successful if the following workflow feels coherent without navigating through raw path fields.

### Project

1. Open application.
2. Create/open a project using conventional dialogs.
3. Main workspace appears.

### Source

4. Select Source in Project Tree.
5. Click `Browse Server`.
6. Use Explorer-like filesystem browser.
7. Select image sequence directory.
8. Viewer and timeline update.
9. Click `Replace Source`.
10. Select another directory.
11. Source updates without recreating project.

### Mesh

12. Select Mesh.
13. Use `Browse Server` or `Upload`.
14. Mesh loads.
15. Mesh properties appear in Inspector.
16. Click `Replace Mesh`.
17. Choose another mesh.
18. Viewer updates.
19. Existing dependent tracking is not silently destroyed.

### Alignment

20. Enter ALIGN mode.
21. Manipulate pose using viewer gizmo.
22. Switch between Local and Global coordinate space.
23. Numeric Pose fields update simultaneously.
24. Run Refine.
25. Accept refined pose.

### Tracking

26. Enter TRACK mode.
27. Select range on timeline.
28. Run tracking.
29. Track state appears on timeline.
30. Anchors/warnings/failures are visually distinct.

### Review

31. Enter REVIEW mode.
32. Play sequence.
33. Change overlays using Viewer toolbar.
34. Open Diagnostics panel.
35. Click a graph point to navigate to frame.

No manual filesystem path editing should be necessary for ordinary workflow.