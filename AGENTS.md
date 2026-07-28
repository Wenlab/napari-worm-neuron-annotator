# AGENTS.md

## Project purpose

This repository provides a napari plugin for navigating read-only neuron
bounding-box ROIs and managing `Labels` visibility. Its primary
responsibilities are preserving ROI identity across overlapping boxes,
changing checked/unchecked label alpha without changing RGB, and deriving
2D/3D `Vectors` overlays without modifying source data.

Treat annotation and Excel features as secondary to correct label
visualization.

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
- `src/napari_label_manager/_widget.py`: widget state, layer selection,
  colormap/opacity control, navigation, and managed Vectors lifecycle.
- `src/napari_label_manager/napari.yaml`: npe2 plugin manifest.
- `src/napari_label_manager/_tests/`: pytest and pytest-qt tests.
- `docs/napari-0.8-migration.md`: supported Python/napari/Qt policy.
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
- Keep ROI `neuron_id` zero-based and convert to a Labels value only at the
  explicit `label_value = neuron_id + 1` boundary.
- Preserve float box bounds and half-open `[min,max)` slice semantics.
- Store overlapping box identity in per-vector `neuron_id` features; never
  rasterize multiple identities back into one dense Labels value.
- Managed Vectors layers are runtime-derived artifacts identified by metadata.
  Remove them and disconnect dims events when unloading or closing.
- Support only explicit `(z,y,x)` and `(t,z,y,x)` controlled Labels layouts.

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
- single IDs, ranges, duplicates, malformed input, and background IDs;
- opacity changes preserving RGB values for cyclic and direct colormaps;
- checked, unchecked, background, and unknown-label opacity behavior;
- cache invalidation after painting and `layer.data` replacement;
- 2D images, 3D volumes, and explicitly defined time-series data;
- stale asynchronous results after a rapid layer switch;
- annotation preservation and Excel save/load round trips.

Tests must assert behavior and values, not only that widgets exist or buttons
can be clicked.

## Validation commands

Run from the repository root:

```powershell
pixi run pytest -q
pixi run -e excel pytest -q
pixi run ruff check .
```

For packaging or manifest changes, also build the package and verify that
napari discovers the npe2 contribution.

Before finishing:

- inspect `git diff --check`;
- confirm no raw data or generated workbooks were added;
- report tests that were run and any checks that could not be run.
