# napari-worm-neuron-annotator

`napari-worm-neuron-annotator` is a napari plugin for navigating and annotating
read-only neuron bounding-box ROIs on 3D or 4D Image volumes.

The plugin keeps each data source in a separate role:

- the Image layer supplies the spatial axes, world transform, and Z-navigation
  context;
- the ROI array supplies neuron identity and box geometry;
- runtime `Vectors` and `Points` layers show boxes and optional text without
  writing them into a dense mask;
- an explicitly selected `Labels` layer can add a compatible mask overlay and
  checked/unchecked alpha control.

The plugin does not modify Image, ROI, or optional Labels source data.

> **Roadmap:** Plugin-managed Labels are planned for deprecation. The primary
> workflow now uses Image + ROI-derived Vectors/Points, while an explicitly
> selected Labels layer remains available for compatibility.
> See [the deferred deprecation decision](#deferred-deprecation-of-plugin-managed-labels).

## Features

- Image + ROI operation without a Labels layer.
- Read-only loading of `(T,N,K)` ROI NPY arrays.
- Dynamic 2D bounding rectangles and 3D 12-edge wireframes.
- Stable per-neuron box colors derived from zero-based ROI identity.
- Checkable neuron list with a separate active neuron.
- All/None controls and cumulative Q/W navigation.
- Active-neuron highlighting and view centering.
- View-preserving Z-layer display synchronized across Image and ROI overlays.
- Zero-based ROI annotation with optional Excel import/export.
- Optional Labels overlay with independent checked/unchecked alpha and exact
  RGB preservation.
- Automatic restoration of the original Labels colormap and opacity when the
  widget closes or switches to another Labels layer.

## Installation

Install into an existing napari environment:

```text
pip install napari-worm-neuron-annotator
```

For a fresh environment, install napari with its default Qt 6 backend:

```text
pip install "napari-worm-neuron-annotator[all]"
```

Excel support alone is available through:

```text
pip install "napari-worm-neuron-annotator[excel]"
```

The base plugin does not install or select a Qt binding. The napari
environment owns the Qt backend; the convenience `all` extra delegates that
choice to napari.

This release targets Python 3.11–3.14 and napari 0.8.x. See the
[napari 0.8 migration notes](docs/napari-0.8-migration.md) for the dependency
and compatibility decisions.

## Image + ROI workflow

1. Open an Image layer with `(z,y,x)` or `(t,z,y,x)` axes.
2. Open `Plugins > Worm Neuron Annotator`.
3. Select the Image layer, then load the ROI NPY file.
4. Click a neuron row to check and activate that ID.
5. Use the checkbox column to add or remove IDs from the persistent set.
6. Use Q and W to check and activate the previous or next valid ID without
   clearing IDs already checked.
7. Use **All** or **None** to check every ID or clear all checked/active IDs.

The active row is bold and remains the current row. Unchecking the active ID
clears the active state; unchecking another ID does not change the active
neuron.

The ROI array defines the neuron list. You do not need a Labels layer for
selection, box rendering, annotation, centering, time navigation, or Z-layer
display.

## Optional Labels overlay

Select a Labels layer only when you need the mask overlay. Its shape, axes,
scale, translation, and units must match the selected Image layer. The ROI
array remains the identity source, and the Image layer remains the spatial
authority.

The **Labels Layer** panel controls checked and unchecked label alpha. The
defaults are `0.50` and `0.00`. Alpha changes preserve each label's RGB value,
and the widget restores the original colormap and opacity when it closes or
switches to another Labels layer. The napari layer-list eye icon can hide the
whole mask.

## Deferred deprecation of plugin-managed Labels

The current release keeps the optional Labels integration and the legacy
`LabelManager` Python and command aliases for compatibility. The public widget
name is `NeuronAnnotatorWidget`, and napari lists only that widget.

A future compatibility migration may remove plugin-managed Labels after users
have had time to move their Image + ROI workflows. No removal version is set.
Source Labels data remains read-only during the migration period.

## Z-layer display

The compact **Z Layers** panel separates a 3D volume along Z so individual
depth ranges can be rendered without the other ranges obscuring them:

1. Select a compatible Image layer.
2. Inspect the current-time curve showing how many pixels in each Z slice are
   strictly above the editable threshold (default `170`). Click the curve to
   add or remove a cut, or enter explicit cuts such as `4,10`.
3. Click **Split**.
4. Use **Show** to select `All` or one generated layer.

Cuts use half-open Python ranges. For a volume with 18 Z slices, `4,10`
creates `[0,4)`, `[4,10)`, and `[10,18)`. A boundary slice belongs to the
following layer.

**All** displays every generated Image layer using additive blending and all
currently valid checked/active ROI overlays. **Layer k** displays only that
Image range and the overlays whose box center Z belongs to the range. A box
that crosses a cut is shown whole in the one layer containing its center.

When you explicitly bind compatible Labels, **Split** creates read-only view
or lazy-slice proxies. **All** shows the complete source Labels and hides the
proxies. **Layer k** hides the source and shows its matching proxy slice. The
plugin never allocates a dense Labels copy.

Checked and active neuron identities remain global. In an individual layer,
Q/W navigates only neurons assigned to that layer; other neurons remain in
the list, are shown in gray, and keep operable checkboxes. Activating a gray
row does not move the view outside the selected Z layer.

Generated Image layers use slices of NumPy arrays, memory maps, or Dask
arrays rather than full-size zero-filled copies. Direct Zarr arrays should
be wrapped as Dask arrays before splitting. The Image source must have
`(z,y,x)` or `(t,z,y,x)` axes, axis-aligned volume depiction, and no clipping
planes. Optional Labels must match its shape and spatial metadata. **Clear**
removes generated Image and Labels layers and restores the source visibility
captured before splitting. Normal napari eye icons may still override
visibility until the next **Show** selection.

### Launch the validated 20260304_w3_immobile dataset

The repository includes a ready-to-use launcher for the git-ignored local
dataset at `data/20260304_w3_immobile_npy`:

```text
pixi run launch-actual
```

It memory-maps `volumes.npy` and `neuron_point_tuple.npy`, opens napari, docks
the navigator, and loads all 120 ROI identities. The launcher leaves its
`LOAD_OPTIONAL_LABELS` validation toggle disabled by default. Enable it to
memory-map `neuron_mask.npy` and exercise the optional Labels integration.

## ROI input format

The ROI loader accepts a numeric NumPy array with shape:

```text
(T, N, K), K >= 6
```

The first six fields are:

```text
x_center, y_center, z_scaled, width, height, depth_scaled
```

Additional fields are ignored. The neuron ID is the index on the `N` axis:

```text
neuron_id = 0 ... N - 1
label_value = neuron_id + 1
background label_value = 0
```

NaN, infinite values, non-positive sizes, and time points outside the source
array are treated as missing observations. Missing neurons remain in the
checkable list so that their global identity is stable, but Q/W skips them at
the current time point.

### Coordinate and time mapping

The plugin requires the selected Image layer to have one of these axis orders:

```text
(z, y, x)
(t, z, y, x)
```

`z_divisor`, defaulting to 5, converts source z and depth coordinates:

```text
z_index = z_scaled / z_divisor
depth_in_slices = depth_scaled / z_divisor
```

For a viewer that displays a cropped or strided time range:

```text
source_t = volume_start + viewer_t * volume_stride
```

Configure `volume_start` and `volume_stride` before loading the NPY.

The derived ROI overlay layers copy scale, translation, axis labels, and
units from the Image layer. Do not apply the z scale a second time in the ROI
coordinates. Optional Labels never supply ROI colors or spatial metadata.

## 2D and 3D box display

Two managed Vectors layers are created after loading an ROI file:

- `Neuron boxes – selected`: checked, currently valid boxes colored by their
  stable `neuron_id` palette;
- `Neuron box – active`: the active box with a thick yellow outline.

Initially only the first valid neuron is checked and active. **All** includes
all IDs in the selected layer, while **None** empties both box layers and the
optional text overlay. Checked identities that are missing at the current time
remain checked but are temporarily omitted from the geometry.

Enable **Show selected box labels** to place one text label at the center of
each currently rendered checked box. The text uses the annotation table's
`biological` value when present and otherwise falls back to the zero-based
`neuron_id`. The third `annotation` column is not used for box labels. The
option is off by default. Use **Text color** to choose a session-only label
color that contrasts with the current Image colormap.

In 2D mode, the plugin draws four rectangle edges only when the current z
slice intersects the box's half-open z range.

In 3D mode, each box is represented by 12 vector edges. Overlapping boxes
remain independent vector records with a `neuron_id` feature. They may overlap
visually, but one box does not erase the identity of another.

Vectors and the transparent Points text layer are derived display data. They
are removed when the ROI is unloaded or the widget closes and are not saved
as a separate annotation format.

## Annotation

The `digital` column stores the zero-based ROI `neuron_id`.

The table follows the loaded ROI identities. Existing biological names and
annotation text are preserved by identity when the ROI source changes.
Activating a neuron selects its annotation row. Selecting a table row checks
and activates the corresponding neuron without clearing other checked IDs.
The `biological` value is also displayed in the neuron list.

The complete navigator is vertically scrollable when the napari dock is
shorter than its controls.

Excel operations support `.xlsx` workbooks. Saving and loading do not apply an
implicit `+1` or `-1` conversion.

## Development

Run tests and lint from the repository root:

```text
pixi run pytest -q
pixi run -e excel pytest -q
pixi run ruff check .
```

The repository uses a `src` layout. Pure ROI parsing and geometry live in
`src/napari_worm_neuron_annotator/_roi.py`; Qt and napari lifecycle behavior
live in `src/napari_worm_neuron_annotator/_widget.py`.

## License

Distributed under the BSD-3-Clause license.
