# napari-label-manager

`napari-label-manager` is a napari plugin for navigating neuron bounding-box
ROIs and controlling the visibility of checked IDs in a `Labels` layer.

The plugin treats the two representations differently:

- the ROI array is the authoritative neuron identity and box geometry;
- the `Labels` layer is a display context in which label value
  `neuron_id + 1` represents a zero-based neuron ID;
- napari `Vectors` layers are generated at runtime to show overlapping boxes
  without writing them into a dense integer mask.

Original ROI arrays and Labels data are never modified.

## Features

- Checkable neuron list with a distinct active neuron.
- All/None controls and cumulative Q/W navigation.
- Independent opacity for checked and unchecked Labels.
- Exact RGB preservation when changing alpha.
- Read-only loading of `neuron_pt_tuple.npy`.
- Dynamic 2D bounding rectangles on the current z slice.
- Dynamic 3D 12-edge wireframes for the current volume.
- Active-neuron highlighting and view centering.
- Zero-copy Z-layer display synchronized across Image, Labels, and boxes.
- Zero-based ROI annotation with optional Excel import/export.
- Automatic restoration of the original Labels colormap and opacity when the
  widget closes or switches to another Labels layer.

## Installation

Install into an existing napari environment:

```text
pip install napari-label-manager
```

For a fresh environment, install napari with its default Qt 6 backend:

```text
pip install "napari-label-manager[all]"
```

Excel support alone is available through:

```text
pip install "napari-label-manager[excel]"
```

The base plugin does not install or select a Qt binding. The napari
environment owns the Qt backend; the convenience `all` extra delegates that
choice to napari.

This release targets Python 3.11–3.14 and napari 0.8.x. See the
[napari 0.8 migration notes](docs/napari-0.8-migration.md) for the dependency
and compatibility decisions.

## Basic Labels workflow

1. Open a `Labels` layer.
2. Open `Plugins > Neuron ROI Navigator`.
3. Select the Labels layer in the widget.
4. Click a row to check and activate that ID.
5. Use the checkbox column to add or remove IDs from the persistent set.
6. Use Q and W to check and activate the previous or next valid ID without
   clearing IDs already checked.
7. Use **All** or **None** to check every ID or clear all checked/active IDs.

The active row is bold and remains the current row. Unchecking the active ID
clears the active state; unchecking another ID does not change the active
neuron.

The **Labels Layer** panel also controls checked and unchecked label alpha.
The defaults are `0.50` for checked labels and `0.00` for unchecked labels.
Hide the entire mask with the normal eye icon in napari's layer list. The
original Labels colormap and opacity are restored automatically when the
widget closes or switches to another Labels layer.

For small in-memory arrays, IDs are discovered exactly from the Labels data.
The plugin intentionally does not scan out-of-core arrays or arrays larger
than 10 million voxels. Load an ROI NPY file in those cases.

Labels-only discovery cannot recover an identity that has already been
completely overwritten in a dense mask.

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

**All** displays every generated Image layer using additive blending, the
complete Labels layer, and all currently valid checked boxes. **Layer k**
displays only that Image
range, the corresponding read-only Labels slice, and boxes whose center Z
belongs to the range. A box that crosses a cut is shown whole in the one
layer containing its center.

Checked and active neuron identities remain global. In an individual layer,
Q/W navigates only neurons assigned to that layer; other neurons remain in
the list, are shown in gray, and keep operable checkboxes. Activating a gray
row does not move the view outside the selected Z layer.

Generated Image layers use slices of NumPy arrays, memory maps, or Dask
arrays rather than full-size zero-filled copies. Direct Zarr arrays should
be wrapped as Dask arrays before splitting. The Image and Labels sources
must have matching `(z,y,x)` or `(t,z,y,x)` shape, axis labels, scale,
translation, and units, with axis-aligned volume depiction and no clipping
planes. **Clear** removes all generated Z layers and restores the original
source visibility. Normal napari eye icons may still override visibility
until the next **Show** selection.

### Launch the validated 20260417_w2 dataset

The repository includes a ready-to-use launcher for the local validation
dataset:

```text
pixi run launch-actual
```

It memory-maps `volumes.npy`, `neuron_mask.npy`, and
`neuron_point_tuple.npy`, opens napari, docks the navigator, and loads all
138 ROI identities automatically.

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

The plugin assumes the controlled Labels layer has one of these axis orders:

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
units from the controlled Labels layer. Do not apply the z scale a second
time in the ROI coordinates.

## 2D and 3D box display

Two managed Vectors layers are created after loading an ROI file:

- `Neuron boxes – selected`: checked, currently valid boxes at low opacity;
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

In ROI mode, the `digital` column always stores the zero-based `neuron_id`.
In Labels-only mode, it stores the raw Labels value.

The table automatically follows the current identities. Existing biological
names and annotation text are preserved by identity when the source changes.
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
`src/napari_label_manager/_roi.py`; Qt and napari lifecycle behavior live in
`src/napari_label_manager/_widget.py`.

## License

Distributed under the BSD-3-Clause license.
