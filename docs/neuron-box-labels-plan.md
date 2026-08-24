# Plan: Display Selected Neuron Box Labels

> **Historical plan:** The Points text-overlay work remains supported, but all
> references below to controlled napari `Labels` layers describe the former
> mask integration. Plugin-managed Labels support was removed on 2026-08-22;
> Image layers now provide all spatial metadata.

**Generated**: 2026-07-29
**Estimated Complexity**: Medium

## Overview

Add an optional text overlay for checked neuron ROI boxes. The existing box
geometry remains in the two managed `Vectors` layers. A separate managed,
transparent `Points` layer provides exactly one text anchor per visible
checked neuron because napari 0.8 `Vectors` layers do not support text and
attaching text to vector-edge features would repeat it four times in 2D or
twelve times in 3D.

The displayed value is:

```python
biological_name.strip() or str(neuron_id)
```

The biological name comes from the annotation table's second, `biological`,
column. The third, `annotation`, column is not displayed. The numeric fallback
uses the repository's zero-based ROI `neuron_id`, not the one-based Labels
value.

## Confirmed Product Decisions

- Add one optional `Show selected box labels` checkbox to Neuron Selection.
- Keep the option off by default.
- Treat "selected" as `checked_ids`, not only `active_id`.
- Render one label for each checked neuron whose box is currently rendered.
- Prefer the second annotation-table column, `biological`.
- Fall back to the zero-based numeric `neuron_id` when `biological` is empty
  or contains only whitespace.
- Do not use the third, `annotation`, column for box text.
- Do not duplicate text for the active box.
- Checked neurons missing at the current time remain checked but have neither
  box geometry nor text at that time.
- In an individual Z-layer view, show text only for boxes assigned to that
  range by the existing center-Z rule.
- Keep source Labels and ROI arrays read-only.
- Add a session-only text color selector, defaulting to white.

## Implementation Defaults and Non-Goals

- Use centered, 12-pixel text with a user-selectable color that defaults to
  white.
- Keep the Points marker fully transparent; only its text is visible.
- Do not add font size, label-mode, or per-neuron visibility controls.
- Do not persist the checkbox state between application sessions.
- Do not truncate biological names or alter their stored table/Excel values.
  Only leading and trailing whitespace is removed for display and fallback.
- Do not migrate, rewrite, or add columns to existing Excel workbooks.
- Do not replace the current Vectors boxes with Shapes.
- Do not display IDs when no ROI NPY is loaded because there are no box
  centers to anchor the text.

## Prerequisites and Constraints

- Keep napari support at `>=0.8,<0.9`.
- Do not add a runtime dependency.
- Use the napari 0.8 `Points` text API already present in the project
  environment:
  - point features contain `neuron_id` and `display_text`;
  - text uses the `{display_text}` feature template.
- Copy `scale`, `translate`, `axis_labels`, and `units` from the controlled
  Labels layer so Points and Vectors share one coordinate system.
- Keep all layer and QWidget mutations on the Qt GUI thread.
- Identify the Points layer by metadata, never by its user-visible name.
- Preserve the existing global `checked_ids` and `active_id` behavior.
- Preserve existing 2D/3D Vectors edge counts and neuron ID features.

## Display Contract

### Checkbox Off

- The managed text layer is hidden and contains no display points.
- Checked/active state and both Vectors layers behave exactly as before.
- Editing or loading annotation data does not affect the current view until
  the option is enabled.

### Checkbox On

- Each visible checked box contributes one point and one string.
- Each Points feature contains the zero-based `neuron_id`.
- `display_text` contains the stripped biological name or numeric fallback.
- The active box does not create an additional point.

### 2D Display

- Include a neuron only when the existing 2D box geometry intersects the
  current half-open Z slice.
- Anchor the text at `(current_z, rendered_center_y, rendered_center_x)`.
- Use the center of the clipped, rendered rectangle so an ROI crossing an
  image boundary cannot place its text outside the visible box.

### 3D Display

- Anchor the text at the center of the clipped, rendered 3D box.
- For 4D data, prepend the current viewer time to produce `(t,z,y,x)`.
- Let napari render the text as a camera-facing Points label.

### Z-Layer Display

- All follows the existing behavior and includes every current-time checked
  box that is otherwise visible.
- Layer k includes only checked boxes whose `center_z` belongs to Layer k.
- Do not clip or duplicate a label at a Z-layer cut.
- Preserve checked identities outside Layer k without rendering their text.

## Sprint 1: Pure Label-Anchor Geometry

**Goal**: Compute a single text anchor that exactly matches the rendered box
without changing Qt or napari layer behavior.

**Demo/Validation**:

- Run focused `_roi.py` unit tests.
- Verify anchors match the center of normal, clipped, and partially
  out-of-bounds boxes in 2D and 3D.

### Task 1.1: Add a 3D rendered-box center helper

- **Location**:
  - `src/napari_worm_neuron_annotator/_roi.py`
  - `src/napari_worm_neuron_annotator/_tests/test_roi.py`
- **Description**:
  - Add a small pure helper that reuses `_clipped_bounds()`.
  - Return the midpoint of each clipped `(z,y,x)` bound.
  - Return `None` when the box does not intersect the supplied volume shape.
  - Preserve float coordinates.
- **Dependencies**: None.
- **Acceptance Criteria**:
  - A fully in-bounds box returns its existing center.
  - A boundary-crossing box returns the center of the clipped geometry.
  - A fully out-of-bounds box returns `None`.
- **Validation**:
  - Test normal, fractional, boundary-crossing, and outside-volume boxes.

### Task 1.2: Add a 2D rendered-rectangle center helper

- **Location**:
  - `src/napari_worm_neuron_annotator/_roi.py`
  - `src/napari_worm_neuron_annotator/_tests/test_roi.py`
- **Description**:
  - Reuse `_clipped_bounds()` and the same half-open Z intersection rule as
    `box_vectors_2d()`.
  - Return `(current_z, center_y, center_x)` for an intersecting box.
  - Return `None` when the box does not intersect the current slice or volume.
- **Dependencies**: Task 1.1.
- **Acceptance Criteria**:
  - A label anchor exists exactly when 2D vector geometry exists.
  - A Z index equal to `z_max` is excluded.
  - Clipped y/x coordinates match the displayed rectangle center.
- **Validation**:
  - Test slices below, inside, and at the upper half-open boundary.

### Task 1.3: Protect existing geometry behavior

- **Location**:
  - `src/napari_worm_neuron_annotator/_tests/test_roi.py`
- **Description**:
  - Retain explicit assertions for four 2D edges and twelve 3D edges while
    adding anchor tests.
  - Assert the new helpers do not mutate the frozen `NeuronBox` or source
    arrays.
- **Dependencies**: Tasks 1.1 and 1.2.
- **Acceptance Criteria**:
  - Existing Vectors geometry tests pass unchanged.
  - Anchor computation allocates only small coordinate arrays.
- **Validation**:
  - Run `pixi run pytest src/napari_worm_neuron_annotator/_tests/test_roi.py -q`.

## Sprint 2: Optional Managed Points Text Layer

**Goal**: Users can toggle one correctly labeled point per currently rendered
checked box.

**Demo/Validation**:

- Load a small ROI fixture.
- Enable the checkbox and verify numeric fallback text.
- Enter biological names and verify immediate replacement.
- Disable the checkbox and verify all text disappears without changing boxes.

### Task 2.1: Add the compact visibility control

- **Location**:
  - `src/napari_worm_neuron_annotator/_widget.py`
  - `src/napari_worm_neuron_annotator/_tests/test_widget.py`
- **Description**:
  - Add `Show selected box labels` below the existing All/None controls in the
    Neuron Selection group.
  - Add a compact `Text color` button that opens the Qt color selector.
  - Default it to unchecked.
  - Default the text color to white and keep the chosen RGB color for the
    current widget session.
  - Connect `toggled` to the ROI overlay refresh path.
  - Keep the layout responsive in the existing narrow, vertically scrollable
    dock.
- **Dependencies**: Sprint 1.
- **Acceptance Criteria**:
  - The control is visible without widening the dock.
  - Toggling it never changes `checked_ids` or `active_id`.
- **Validation**:
  - Assert default state and selection-state preservation with pytest-qt.

### Task 2.2: Define and create the managed Points layer

- **Location**:
  - `src/napari_worm_neuron_annotator/_widget.py`
  - `src/napari_worm_neuron_annotator/_tests/test_widget.py`
- **Description**:
  - Import `napari.layers.Points`.
  - Add a dedicated metadata role such as `roi_box_labels`.
  - Generalize the existing ensure/remove naming from Vectors-only to managed
    ROI overlays so lifecycle intent remains clear.
  - Create one empty Points layer alongside the managed Vectors layers when a
    valid ROI and compatible Labels layer are active.
  - Configure transparent point face/border and centered text using
    `{display_text}` and the current session color.
  - Copy dimension count and Labels transforms, axes, and units.
- **Dependencies**: Task 2.1.
- **Acceptance Criteria**:
  - Exactly one managed Points layer exists per widget/ROI session.
  - The layer is identified by metadata even if napari changes its name.
  - Its marker is invisible and its text configuration is deterministic.
  - No new package dependency is introduced.
- **Validation**:
  - Assert layer type, role, ndim, transform, text template, and transparent
    marker properties.

### Task 2.3: Build one label record per rendered checked box

- **Location**:
  - `src/napari_worm_neuron_annotator/_widget.py`
  - `src/napari_worm_neuron_annotator/_tests/test_widget.py`
- **Description**:
  - During the existing ROI iteration, collect one anchor per checked neuron
    after applying current-time validity, active Z-range membership, supported
    view-axis checks, and 2D/3D visibility.
  - Use Sprint 1 helpers for anchor coordinates.
  - For 4D layers, prepend current viewer time.
  - Store `neuron_id` and `display_text` as Points features.
  - Use `biological.strip() or str(neuron_id)` for text.
  - When the checkbox is off, set empty data/features and hide the layer.
- **Dependencies**: Tasks 2.1 and 2.2.
- **Acceptance Criteria**:
  - N checked visible boxes produce N points, not `4N` or `12N`.
  - Active styling does not duplicate a point.
  - Numeric fallback is zero-based.
  - Duplicate biological names remain separate points with distinct
    `neuron_id` features.
- **Validation**:
  - Test one/multiple selections, All, None, active changes, biological names,
    whitespace-only names, and duplicate names.

### Task 2.4: Refresh text after annotation changes

- **Location**:
  - `src/napari_worm_neuron_annotator/_widget.py`
  - `src/napari_worm_neuron_annotator/_tests/test_widget.py`
- **Description**:
  - Refresh the Points data after a second-column edit.
  - Refresh after `_set_annotation_rows()` because it blocks item signals
    during Current IDs and Excel loads.
  - Refresh after a digital-column edit as well because it can change the
    row-to-identity mapping under the current table behavior.
  - Ignore third-column edits for box text.
  - Do not mutate the stored biological or annotation strings.
- **Dependencies**: Task 2.3.
- **Acceptance Criteria**:
  - Editing `biological` updates visible text immediately.
  - Clearing it immediately restores the numeric ID.
  - Editing only `annotation` leaves the box text unchanged.
  - Excel-loaded biological names appear without another selection change.
- **Validation**:
  - Add direct table-edit and `_set_annotation_rows()` regression tests.

## Sprint 3: Full View Integration and Lifecycle

**Goal**: Text stays synchronized across time, 2D/3D, Z-layer views, source
changes, unload, and shutdown.

**Demo/Validation**:

- Exercise time and view changes with a missing ROI observation.
- Switch among All and individual Z layers.
- Switch controlled Labels layers, unload ROI, and close the widget.

### Task 3.1: Synchronize dimensions and Z-layer filtering

- **Location**:
  - `src/napari_worm_neuron_annotator/_widget.py`
  - `src/napari_worm_neuron_annotator/_tests/test_widget.py`
  - `src/napari_worm_neuron_annotator/_tests/test_z_layer_widget.py`
- **Description**:
  - Include Points refresh in the existing dimensions and Z-view refresh
    paths.
  - Preserve the existing supported-axis guard; unsupported views produce
    empty Points data together with empty Vectors data.
  - Reuse the same current-time and center-Z predicates used for box geometry.
- **Dependencies**: Sprint 2.
- **Acceptance Criteria**:
  - 2D points follow the current Z slice and appear only for intersecting
    rectangles.
  - 3D points use box centers.
  - Time changes remove missing observations and add newly valid ones without
    changing global checks.
  - Layer k shows only labels belonging to Layer k; All restores all valid
    checked labels.
- **Validation**:
  - Test 3D and 4D data, 2D/3D display changes, missing observations, exact
    Z-cut membership, and All/Layer k transitions.

### Task 3.2: Complete managed-layer cleanup

- **Location**:
  - `src/napari_worm_neuron_annotator/_widget.py`
  - `src/napari_worm_neuron_annotator/_tests/test_widget.py`
- **Description**:
  - Include the Points metadata role in managed ROI overlay discovery and
    cleanup.
  - Remove it on ROI unload, controlled Labels replacement, incompatible
    dimensions, and widget shutdown.
  - Ignore manager-owned removal callbacks just as for managed Vectors.
  - Recreate it with the correct ndim when switching between compatible 3D
    and 4D Labels layers.
- **Dependencies**: Task 3.1.
- **Acceptance Criteria**:
  - No managed Points layer survives unload or shutdown.
  - Switching 3D/4D sources leaves exactly two Vectors layers and one Points
    layer with matching ndim.
  - Repeated cleanup is harmless.
- **Validation**:
  - Extend existing managed-layer lifecycle tests with Points role assertions.

### Task 3.3: Add focused regression coverage

- **Location**:
  - `src/napari_worm_neuron_annotator/_tests/test_widget.py`
  - `src/napari_worm_neuron_annotator/_tests/test_z_layer_widget.py`
- **Description**:
  - Assert unchanged selected/active Vectors edge counts and features while
    text is enabled and disabled.
  - Assert checkbox state never changes selection state.
  - Assert annotation selection, Q/W, All/None, Labels opacity, and Z-layer
    behavior remain independent of text visibility.
- **Dependencies**: Tasks 3.1 and 3.2.
- **Acceptance Criteria**:
  - New tests assert exact point coordinates, text values, features, and
    counts rather than only layer existence.
  - All existing tests continue to pass.
- **Validation**:
  - Run focused widget and Z-layer widget suites.

### Task 3.4: Document the user-facing behavior

- **Location**:
  - `README.md`
- **Description**:
  - Document the optional checkbox and `biological`-then-ID fallback.
  - State that only current visible checked boxes receive text.
  - State that numeric IDs remain zero-based and annotation column three is
    not used for box labels.
- **Dependencies**: Functional behavior complete.
- **Acceptance Criteria**:
  - Documentation uses the exact GUI label.
  - Documentation matches tested 2D/3D, time, and Z-layer behavior.
- **Validation**:
  - Review README steps against the manual acceptance workflow.

## Testing Strategy

### Pure unit tests

- Normal and clipped 3D centers.
- 2D half-open slice intersection and rectangle centers.
- Fractional ROI bounds and fully out-of-volume boxes.
- No mutation of source ROI data.

### Widget tests

- Checkbox default, toggle behavior, and selection-state independence.
- Managed Points type, metadata, transforms, axes, units, and transparent
  marker.
- Exact one-point-per-checked-box behavior.
- Zero-based numeric fallback.
- Second-column preference and third-column independence.
- Direct edits, Current IDs synchronization, and Excel-loaded names.
- All/None, Q/W, active changes, and duplicate biological names.
- 2D/3D, 3D/4D, time changes, and missing observations.
- All/Layer k filtering and exact Z-cut behavior.
- ROI unload, Labels switch, layer removal, and shutdown cleanup.
- Unchanged selected/active Vectors edge counts and color features.

### Automated validation

Run from the repository root:

```powershell
pixi run pytest src/napari_worm_neuron_annotator/_tests/test_roi.py -q
pixi run pytest src/napari_worm_neuron_annotator/_tests/test_widget.py -q
pixi run pytest src/napari_worm_neuron_annotator/_tests/test_z_layer_widget.py -q
pixi run pytest -q
pixi run -e excel pytest -q
pixi run ruff check .
$env:PYTHONUTF8='1'; pixi run npe2 validate src/napari_worm_neuron_annotator/napari.yaml
git diff --check
```

Confirm that no raw data, generated workbooks, or runtime-derived artifacts
are added to Git.

### Real-data acceptance

Run:

```powershell
pixi run launch-actual
```

Verify:

- Initial selection still produces 12 selected and 12 active 3D edges.
- The checkbox is off initially and no box text is visible.
- Enabling it shows one centered label for the initial checked neuron.
- A populated biological name replaces its numeric ID immediately.
- Clearing the biological name restores the zero-based ID.
- All at time zero preserves `120 × 12 = 1440` selected edges and produces
  exactly 120 text anchors when all 120 boxes are valid and visible.
- None empties selected, active, and text overlays.
- Q/W preserves prior checks and text follows the checked set.
- 2D slice changes show labels only for intersecting rectangles.
- 3D text remains centered while rotating the camera.
- Time changes preserve global checks while missing boxes and labels disappear.
- Z All/Layer k changes keep Image, Labels, Vectors, and text synchronized.
- Text remains readable over representative Image and Labels colormaps.
- ROI unload and dock closure remove all three managed ROI overlay layers.

## Potential Risks and Gotchas

- **Text repetition**: using Vectors edge features would render four or twelve
  copies.
  - Mitigation: use one Points row per neuron.
- **Point marker leakage**: a tiny marker could remain visible behind text.
  - Mitigation: test fully transparent face and border values in napari 0.8.
- **2D Z visibility**: a point at the ROI's center Z would disappear on other
  intersecting slices.
  - Mitigation: place 2D anchors at the current Z slice.
- **Boundary mismatch**: raw ROI centers may differ from centers of clipped
  visible boxes.
  - Mitigation: derive anchors from the same clipped bounds helper used by
    vector geometry.
- **Stale biological names**: table population blocks signals.
  - Mitigation: explicitly refresh after `_set_annotation_rows()`.
- **Layer-type assumptions**: existing cleanup currently recognizes only
  `Vectors`.
  - Mitigation: use metadata-driven managed ROI overlay cleanup that accepts
    both Vectors and Points.
- **Dense scenes**: All can create substantial visual overlap even though the
  point count and memory cost are small.
  - Mitigation: default the checkbox off and leave standard napari layer
    visibility available; font-size configuration is outside this scope.
- **3D readability and occlusion**: volume rendering and bright colormaps can
  reduce contrast.
  - Mitigation: default to white and let the user select a contrasting
    session-only text color; outline/background remains outside this scope.
- **Editable digital IDs**: changing column one can remap biological names
  under existing behavior.
  - Mitigation: refresh after digital edits but do not redesign annotation
    identity editing in this feature.

## Rollback Plan

- Remove the checkbox and its signal connection.
- Remove only the new Points metadata role and managed Points creation,
  refresh, and cleanup paths.
- Remove the new label-anchor helpers if no other code uses them.
- Remove the focused Points/text tests and README section.
- Leave both existing Vectors roles, global selection state, Labels display,
  ROI data, annotation workbook format, and Z-layer implementation unchanged.
- No data migration or workbook rollback is required.
