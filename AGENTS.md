# AGENTS.md

## Project purpose

This repository provides a napari plugin for navigating read-only neuron
bounding-box ROIs and managing `Labels` visibility. Its primary
responsibilities are preserving ROI identity across overlapping boxes,
changing checked/unchecked label alpha without changing RGB, and deriving
2D/3D `Vectors` overlays without modifying source data.

Treat annotation and Excel features as secondary to correct label
visualization.

## Current product behavior

The plugin has two independent neuron-selection states:

- `checked_ids`: persistent global identities selected by checkboxes. This set
  controls Labels checked/unchecked alpha and the
  `Neuron boxes – selected` Vectors layer.
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
visible help text concise (`Q/W: last/next`). Labels opacity controls belong
inside the Labels Layer panel. Do not reintroduce `Show Labels context`,
`Reset Display`, random-colormap controls, text ID entry, or Ctrl-click
selection semantics.

## Z-layer display behavior

The compact Z Layers panel coordinates one compatible Image layer, the
controlled Labels layer, and the runtime Neuron Boxes overlays:

- Support only matching `(z,y,x)` and `(t,z,y,x)` Image/Labels pairs with
  matching shapes, axis labels, scale, translation, and units.
- Reject RGB, multiscale, direct Zarr, non-axis-aligned, plane-depiction, and
  clipped Image sources. Users may wrap Zarr data as Dask before splitting.
- Users provide explicit, strictly increasing Z cuts. Ranges are half-open;
  for cuts `4,10`, the ranges are `[0,4)`, `[4,10)`, and `[10,Z)`.
- Derived Image layers must use view-preserving NumPy/memory-map slices or
  lazy Dask slices. Never allocate full-size zero-filled copies or modify the
  source array.
- Every derived Image layer uses `additive` blending. Do not change the source
  Image layer's blending setting.
- All shows every derived Image layer, the complete Labels layer, and all
  currently valid checked/active boxes.
- Layer k shows only its derived Image, a read-only Labels slice, and whole
  boxes whose `center_z` belongs to that range. Do not clip a crossing box or
  duplicate it across ranges.
- `checked_ids` and `active_id` remain global. In Layer k mode, Q/W navigates
  only current-time boxes assigned to that range. The neuron tree still lists
  all IDs and grays IDs outside the active range without disabling checkboxes.
- Clear removes all runtime Z layers and restores the source Image and Labels
  visibility captured before splitting. Z configuration is session-only.

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
- Do not modify raw label arrays when a display-only colormap change is enough.
- Keep UI code, label-statistics code, and file I/O behavior independently
  testable where practical.

## Repository layout

- `src/napari_label_manager/_roi.py`: read-only NPY parsing, ID mapping, and
  pure 2D/3D box geometry.
- `src/napari_label_manager/_z_layers.py`: pure Z-range, slicing, membership,
  translation, and threshold-profile helpers.
- `src/napari_label_manager/_z_profile.py`: compact Qt Z-profile plot and
  click-to-cut interaction.
- `src/napari_label_manager/_widget.py`: widget state, layer selection,
  colormap/opacity control, navigation, Z display, and managed-layer
  lifecycles.
- `src/napari_label_manager/napari.yaml`: npe2 plugin manifest.
- `src/napari_label_manager/_tests/`: pytest and pytest-qt tests.
- `docs/napari-0.8-migration.md`: supported Python/napari/Qt policy.
- `scripts/launch_20260417_w2.py`: read-only, memory-mapped real-data napari
  launcher used for manual acceptance.
- `pyproject.toml`: package metadata, dependencies, Ruff, Pixi, and test setup.

## Napari Labels requirements

### Colormaps and opacity

- Support the napari colormap types actually received from `Labels`, including
  cyclic and direct label colormaps.
- Never assume that `enumerate(colormap.colors)` maps directly to label IDs.
  Verify the mapping through napari's colormap semantics.
- Opacity-only operations must preserve each label's RGB color exactly.
- Preserve `background_value`; do not hard-code zero as background when the
  active colormap defines another value.
- Keep unknown or out-of-range label behavior consistent with the source
  colormap. Do not accidentally make such labels transparent.
- Validate label IDs before applying changes. Malformed input must not be
  silently interpreted as unrelated valid IDs.
- Do not mutate `layer.data` to implement visibility or opacity controls.

### Layer lifecycle and caching

- Track only `napari.layers.Labels` instances in the selector.
- Connect layer-specific events when a layer is selected and disconnect them
  when it is replaced, removed, or the widget is destroyed.
- Invalidate cached IDs and counts when label data changes, including painting,
  undo/redo, replacement of `layer.data`, and relevant dimensional navigation.
- Cache entries must be tied to both the layer and the data/view state used to
  compute them. Never return stale IDs after an edit or layer switch.
- Avoid duplicate event connections and duplicate selector instances.

### Dimensions and large data

- Do not infer that axis 0 is time merely because an array has three or more
  dimensions. A 3D array is commonly a volumetric label image.
- If current-view or current-time statistics are intended, obtain navigation
  state from `viewer.dims` and explicitly define which axes are being sliced.
- State whether an operation covers the full label volume or only the current
  displayed slice/time point.
- Preserve integer label IDs exactly; do not cast them through floating point.
- For large NumPy, Dask, Zarr, or memory-mapped arrays, avoid unconditional
  full-array masks, copies, flattening, or `np.unique` calls.
- Sampling may support approximate status information, but actions named
  "All Current IDs" must not silently omit labels. Mark approximations clearly
  and do not use them for destructive or authoritative operations.

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

- Use one documented ID convention across the label layer, table, preservation
  logic, and Excel files. If a displayed ID is offset, convert at explicit
  boundaries and test round trips.
- Populate the annotation table from Current IDs automatically when the source
  changes. Preserve matching biological names and text by identity.
- Keep the manual Current IDs refresh, Load, and Save actions. Do not
  reintroduce Start, End, or Fill controls.
- Reloading current IDs must preserve annotations by label identity, not row
  position.
- Saving and loading an unchanged workbook must preserve digital IDs and text.
- Advertise only file formats that the installed reader can actually load.
- Handle invalid workbooks, missing sheets, permission errors, and unsupported
  formats without crashing the Qt callback.
- Keep Excel dependencies optional unless they become a required plugin
  feature; expose them through a clearly named optional dependency group.

## ROI and Vectors behavior

- Treat `(T,N,K)` NPY data as read-only; load with memory mapping and
  `allow_pickle=False`.
- Interpret the first six columns as
  `[x, y, z_scaled, width, height, depth_scaled]`; ignore later columns.
- Keep ROI `neuron_id` zero-based and convert to a Labels value only at the
  explicit `label_value = neuron_id + 1` boundary.
- Preserve float box bounds and half-open `[min,max)` slice semantics.
- Store overlapping box identity in per-vector `neuron_id` features; never
  rasterize multiple identities back into one dense Labels value.
- `Neuron boxes – selected` uses metadata role `roi_boxes_selected` and contains
  only `checked_ids ∩ valid_ids` for the current time and view.
- `Neuron box – active` uses metadata role `roi_box_active`, contains only the
  current valid `active_id`, and is always rendered as a thick yellow outline.
- Remove the legacy `roi_boxes_all` layer when encountered; do not create it.
- In 2D, generate four edges only for boxes intersecting the current z slice.
  In 3D, generate 12 edges per included box.
- Managed Vectors layers are runtime-derived artifacts identified by metadata.
  Remove them when unloading ROI or closing the widget, and disconnect viewer
  events during shutdown.
- Support only explicit `(z,y,x)` and `(t,z,y,x)` controlled Labels layouts.
- Copy scale, translation, and axis labels from the controlled Labels layer.
- Do not save Vectors as an annotation format or write ROI changes back to NPY.

## Dependency and packaging rules

- Do not add a specific Qt binding such as PyQt5 or PySide to runtime
  dependencies. The napari host owns the Qt backend.
- Target Python 3.11–3.14 and napari 0.8.x unless a documented compatibility
  migration changes this range.
- Keep pytest, pytest-qt, coverage, and lint tools in development/testing
  dependencies only.
- Keep `napari.yaml`, entry points, and package exports synchronized when
  renaming widgets or commands.
- Use the setuptools-scm generated version as the single version source.

## Testing expectations

Every behavior change should include a focused regression test. Prefer small
arrays with explicit expected values.

At minimum, test:

- widget creation with zero, one, and multiple pre-existing Labels layers;
- layer insertion, removal, rename, reorder, and selection;
- initial checked/active state, row clicks, checkbox changes, Q/W wrap,
  All/None, and active cancellation;
- opacity changes preserving RGB values for cyclic and direct colormaps;
- checked, unchecked, background, and unknown-label opacity behavior;
- cache invalidation after painting and `layer.data` replacement;
- 2D four-edge rectangles, 3D 12-edge boxes, overlapping identity features,
  time changes, and missing observations;
- selected and active Vectors edge counts and managed-layer cleanup;
- Z-cut parsing, half-open membership, 3D/4D view-preserving slices, shifted
  translation, and per-Z threshold pixel counts;
- All/Layer k synchronization across Image, Labels, Vectors, Q/W navigation,
  gray outside-range IDs, additive blending, and cleanup on source changes;
- editable threshold refresh, click-to-cut behavior, and vertical scrolling;
- annotation preservation and Excel save/load round trips.

Tests must assert behavior and values, not only that widgets exist or buttons
can be clicked.

Automated tests are regression checks, not the primary product acceptance.
For GUI changes, also exercise the real napari workflow and verify the
selection list, Q/W, All/None, 2D/3D switching, time navigation, Labels alpha,
layer visibility, Z threshold profile, additive blending, and dock scrolling.

## Validation commands

Run from the repository root:

```powershell
pixi run pytest -q
pixi run -e excel pytest -q
pixi run ruff check .
$env:PYTHONUTF8='1'; pixi run npe2 validate src/napari_label_manager/napari.yaml
pixi run launch-actual
```

`launch-actual` expects the local `20260417_w2_npy` dataset referenced by the
launcher. It memory-maps image and Labels arrays and must never modify them.
At time zero in 3D, the current reference data has 138 valid boxes: initial
selection produces 12 selected and 12 active edges, All produces
`138 × 12 = 1656` selected edges, and None empties both managed layers.

For packaging changes, also build the package and verify that napari discovers
the npe2 contribution.

Before finishing:

- inspect `git diff --check`;
- confirm no raw data or generated workbooks were added;
- report tests that were run and any checks that could not be run.
