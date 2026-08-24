# AGENTS.md

## Project purpose

This repository provides a napari plugin for navigating and annotating
read-only neuron bounding-box ROIs on Image volumes. Its primary
responsibilities are preserving ROI identity across overlapping boxes and
deriving 2D/3D `Vectors` box overlays plus optional `Points` text overlays
without modifying source data.

Treat annotation and Excel features as secondary to correct Image + ROI
navigation and visualization. The Image layer supplies spatial metadata; the
ROI array supplies neuron identities and geometry. Ordinary napari `Labels`
layers may coexist in the viewer, but the plugin must not manage them.

## Current product behavior

The plugin has two independent neuron-selection states:

- `checked_ids`: persistent global identities selected by checkboxes. This set
  controls the `Neuron boxes – selected` Vectors layer and, when box labels
  are enabled, the `Neuron labels – selected` Points layer.
- `active_id`: the most recently activated identity. It controls the bold,
  current list row, annotation-table selection, view centering, and the yellow
  `Neuron box – active` Vectors layer.

Keep these states separate. The supported interactions are:

- Initial ROI load checks and activates only the first valid neuron.
- Clicking a row checks and activates it without clearing previous checks.
- Checking a checkbox adds and activates that identity.
- Unchecking the active identity clears `active_id`; unchecking another
  identity does not change the active neuron.
- Q/W navigates to the previous/next valid neuron, checks it, and preserves
  existing checks.
- All checks every global identity and preserves an existing active identity.
- None clears both states.
- Time changes preserve global checked and active identities. Missing boxes
  remain listed and checked but are not rendered at that time point.

The Neuron Selection control is a non-alternating `QTreeWidget`. Keep its
visible help text concise (`Q/W: last/next`). Do not reintroduce Labels
selection, opacity or colormap controls, `Show Labels context`, `Reset
Display`, random-colormap controls, text ID entry, Labels-only identity
discovery, or Ctrl-click selection semantics.

`Show selected box labels` is optional and off by default. When enabled:

- render exactly one centered text label per currently rendered checked box;
- use the stripped `biological` value from annotation-table column two,
  falling back to the zero-based `neuron_id` when it is empty;
- never use annotation-table column three as box text;
- do not duplicate text for the active box;
- keep missing or outside-Z-range checked identities global but omit their
  text together with their box geometry;
- use the session-only Text color control, defaulting to white.

## Z-layer display behavior

The compact Z Layers panel coordinates one Image layer and the runtime Neuron
Boxes overlays:

- Support `(z,y,x)` and `(t,z,y,x)` Image layers.
- Reject RGB, multiscale, direct Zarr, non-axis-aligned, plane-depiction, and
  clipped Image sources. Users may wrap Zarr data as Dask before splitting.
- Users provide explicit, strictly increasing Z cuts. Ranges are half-open;
  for cuts `4,10`, the ranges are `[0,4)`, `[4,10)`, and `[10,Z)`.
- Derived Image layers must use view-preserving NumPy/memory-map slices or
  lazy Dask slices. Never allocate full-size zero-filled copies or modify the
  source array.
- Every derived Image layer uses `additive` blending. Do not change the source
  Image layer's blending setting.
- All/Layer k manage only derived Images, checked/active Vectors, and enabled
  Points text.
- Layer k shows whole boxes and enabled text whose `center_z` belongs to that
  range. Do not clip a crossing box or duplicate it or its text across ranges.
- `checked_ids` and `active_id` remain global. In Layer k mode, Q/W navigates
  only current-time boxes assigned to that range. The neuron tree still lists
  all IDs and grays IDs outside the active range without disabling checkboxes.
- Clear removes all runtime Z layers and restores the source Image visibility
  captured before splitting. Z configuration is session-only.

Never inspect, bind, modify, hide, slice, or synchronize Labels layers.

The Z profile counts, for each slice at the current time point, pixels with
intensity strictly greater than an editable threshold. The default threshold
is `170`. Compute exactly one y-x plane at a time, do not count NaN values,
and reject complex-valued images. Changing the threshold refreshes the
profile; changing time marks it stale until Refresh is clicked. Clicking the
profile toggles a Z cut.

The complete navigator is vertically scrollable. Keep wide control groups
responsive in a narrow napari dock; do not hide actions beyond the viewport.

## Engineering style

- Prefer small, explicit, maintainable changes.
- Fix root causes without adding unnecessary abstractions or dependencies.
- Preserve backward-compatible plugin behavior unless a change is documented.
- Keep UI code, ROI geometry, and file I/O behavior independently testable
  where practical.

## Repository layout

- `src/napari_worm_neuron_annotator/_roi.py`: read-only NPY parsing, ID mapping, pure
  2D/3D box geometry, and clipped text-anchor coordinates.
- `src/napari_worm_neuron_annotator/_z_layers.py`: pure Z-range, slicing, membership,
  translation, and threshold-profile helpers.
- `src/napari_worm_neuron_annotator/_z_profile.py`: compact Qt Z-profile plot and
  click-to-cut interaction.
- `src/napari_worm_neuron_annotator/_widget.py`: widget state, Image selection,
  navigation, optional box text/color, Z display, and managed-layer lifecycles.
- `src/napari_worm_neuron_annotator/napari.yaml`: npe2 plugin manifest.
- `src/napari_worm_neuron_annotator/_tests/`: pytest and pytest-qt tests.
- `docs/napari-0.8-migration.md`: supported Python/napari/Qt policy.
- `scripts/launch_20260304_w3_immobile.py`: read-only, memory-mapped real-data napari
  launcher used for manual acceptance.
- `pyproject.toml`: package metadata, dependencies, Ruff, Pixi, and test setup.

## Qt and concurrency

- All QWidget access must occur on the Qt GUI thread.
- Do not update widgets from `threading.Thread`.
- Return worker results through Qt signals or a Qt-managed worker mechanism.
- Capture the target layer and request generation before starting background
  work. Discard results if the user has switched layers or started a newer
  request.
- Keep background tasks bounded and ensure closing the widget cannot leave a
  worker updating deleted Qt objects.
- Avoid modal dialogs in core computation functions; return structured errors
  that the UI layer can display.

## Annotation and Excel behavior

- Use the zero-based ROI `neuron_id` across the neuron list, table,
  preservation logic, and Excel files.
- Populate the annotation table from ROI Current IDs after the ROI source
  changes. Preserve matching biological names and text by identity.
- Treat column two, `biological`, as the display name used by the neuron list
  and optional box text. Strip it only for display/fallback decisions; do not
  rewrite the stored table or workbook value.
- Keep column three, `annotation`, as free-form text and never use it for box
  labels.
- Keep the manual Current IDs refresh, Load, and Save actions. Do not
  reintroduce Start, End, or Fill controls.
- Reloading current IDs must preserve annotations by neuron identity, not row
  position.
- Saving and loading an unchanged workbook must preserve digital IDs and text.
- Advertise only file formats that the installed reader can actually load.
- Handle invalid workbooks, missing sheets, permission errors, and unsupported
  formats without crashing the Qt callback.
- Keep Excel dependencies optional unless they become a required plugin
  feature; expose them through a clearly named optional dependency group.

## ROI overlay behavior

- Treat `(T,N,K)` NPY data as read-only; load with memory mapping and
  `allow_pickle=False`.
- Interpret the first six columns as
  `[x, y, z_scaled, width, height, depth_scaled]`; ignore later columns.
- Keep ROI `neuron_id` zero-based throughout the plugin.
- Preserve float box bounds and half-open `[min,max)` slice semantics.
- Store overlapping box identity in per-vector `neuron_id` features; never
  rasterize multiple identities back into one dense Labels value.
- `Neuron boxes – selected` uses metadata role `roi_boxes_selected` and contains
  only `checked_ids ∩ valid_ids` for the current time and view.
- Give each selected box a deterministic color derived from its zero-based
  `neuron_id`. Keep that color stable across time and selection changes.
- `Neuron box – active` uses metadata role `roi_box_active`, contains only the
  current valid `active_id`, and is always rendered as a thick yellow outline.
- `Neuron labels – selected` is a transparent, non-editable Points layer with
  metadata role `roi_box_labels`. It contains one point per rendered checked
  neuron, with `neuron_id` and `display_text` features.
- Remove the legacy `roi_boxes_all` layer when encountered; do not create it.
- In 2D, generate four edges only for boxes intersecting the current z slice.
  Place box text at `(current_z, clipped_center_y, clipped_center_x)`.
- In 3D, generate 12 edges per included box and place text at the center of
  the clipped rendered box.
- Managed Vectors and Points layers are runtime-derived artifacts identified
  by metadata. Remove them when unloading ROI, changing the source Image, or
  closing the widget, and disconnect viewer events during shutdown.
- Support only explicit `(z,y,x)` and `(t,z,y,x)` source Image layouts.
- Copy scale, translation, axis labels, and units from the source Image layer.
- Do not save Vectors or Points as an annotation format or write ROI changes
  back to NPY.

## Dependency and packaging rules

- Do not add a specific Qt binding such as PyQt5 or PySide to runtime
  dependencies. The napari host owns the Qt backend.
- Target Python 3.11–3.14 and napari 0.8.x unless a documented compatibility
  migration changes this range.
- Keep pytest, pytest-qt, coverage, and lint tools in development/testing
  dependencies only.
- Keep `NeuronAnnotatorWidget` as the public widget name and the only widget
  shown in the npe2 manifest. Preserve `LabelManager` as a Python and command
  compatibility alias until a documented migration removes it.
- Keep `napari.yaml`, entry points, package exports, and metadata tests
  synchronized when renaming widgets or commands.
- Use the setuptools-scm generated version as the single version source.

## Testing expectations

Every behavior change should include a focused regression test. Prefer small
arrays with explicit expected values.

At minimum, test:

- widget creation with zero, one, and multiple pre-existing Image layers, and
  with unrelated Labels layers present;
- layer insertion, removal, rename, reorder, and selection;
- Image + ROI loading, selection, rendering, annotation, and Z display;
- initial checked/active state, row clicks, checkbox changes, Q/W wrap,
  All/None, and active cancellation;
- 2D four-edge rectangles, 3D 12-edge boxes, overlapping identity features,
  time changes, and missing observations;
- deterministic selected-box colors that remain stable across time and do not
  depend on other layers;
- selected and active Vectors edge counts and managed-layer cleanup;
- one Points text anchor per visible checked box, biological-name fallback,
  exact 2D/3D/4D coordinates, missing observations, text color selection and
  cancellation, read-only behavior, and cleanup;
- Z-cut parsing, half-open membership, 3D/4D view-preserving slices, shifted
  translation, and per-Z threshold pixel counts;
- All/Layer k synchronization across Image, Vectors, Points, Q/W navigation,
  gray outside-range IDs, additive blending, and cleanup on source changes;
- unrelated Labels insertion, data/geometry changes, and removal do not affect
  plugin state, source visibility, ROI overlays, or Z-layer sessions;
- editable threshold refresh, click-to-cut behavior, and vertical scrolling;
- annotation preservation and Excel save/load round trips.

Tests must assert behavior and values, not only that widgets exist or buttons
can be clicked.

Automated tests are regression checks, not the primary product acceptance.
For GUI changes, exercise the real napari Image + ROI workflow and verify the
selection list, Q/W, All/None, 2D/3D switching, time navigation, box text
fallback/color, layer visibility, Z threshold profile, additive blending, and
dock scrolling.

## Validation commands

Run from the repository root:

```powershell
pixi run pytest -q
pixi run -e excel pytest -q
pixi run ruff check .
$env:PYTHONUTF8='1'; pixi run npe2 validate src/napari_worm_neuron_annotator/napari.yaml
pixi run launch-actual
```

`launch-actual` expects the git-ignored local `20260304_w3_immobile_npy`
dataset referenced by the
launcher. It memory-maps only the Image and ROI arrays and must never modify
either source array.
At time zero in 3D, the current reference data has 120 valid boxes: initial
selection produces 12 selected and 12 active edges, All produces
`120 × 12 = 1440` selected edges and 120 text anchors when box labels are
enabled, and None empties both box layers plus the text layer.

For packaging changes, also build the package and verify that napari discovers
the npe2 contribution.

Before finishing:

- inspect `git diff --check`;
- confirm no raw data or generated workbooks were added;
- report tests that were run and any checks that could not be run.
