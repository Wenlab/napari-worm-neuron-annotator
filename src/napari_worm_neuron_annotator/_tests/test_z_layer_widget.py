import numpy as np
import pytest
from napari.layers import Labels, Points, Vectors
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
)

from napari_worm_neuron_annotator._widget import (
    ROLE_ACTIVE,
    ROLE_BOX_LABELS,
    ROLE_KEY,
    ROLE_SELECTED,
    LabelManager,
    NeuronAnnotatorWidget,
)

ROLE_Z_IMAGE = "z_layer_image"
ROLE_Z_LABELS = "z_layer_labels"


def _managed_layers(viewer, role):
    return [
        layer
        for layer in viewer.layers
        if layer.metadata.get(ROLE_KEY) == role
    ]


def _managed_vector(viewer, role):
    return next(
        layer
        for layer in viewer.layers
        if isinstance(layer, Vectors) and layer.metadata.get(ROLE_KEY) == role
    )


def _managed_box_labels(viewer):
    return next(
        layer
        for layer in viewer.layers
        if isinstance(layer, Points)
        and layer.metadata.get(ROLE_KEY) == ROLE_BOX_LABELS
    )


def _combo_texts(combo):
    return [combo.itemText(index) for index in range(combo.count())]


def _ancestor_group(widget):
    parent = widget.parentWidget()
    while parent is not None and not isinstance(parent, QGroupBox):
        parent = parent.parentWidget()
    return parent


def _add_matching_layers(
    viewer,
    shape,
    *,
    image_visible=True,
    labels_visible=True,
):
    ndim = len(shape)
    if ndim == 3:
        axis_labels = ("z", "y", "x")
        scale = (2.0, 1.0, 1.0)
        translate = (10.0, 20.0, 30.0)
    else:
        axis_labels = ("t", "z", "y", "x")
        scale = (3.0, 2.0, 1.0, 1.0)
        translate = (5.0, 10.0, 20.0, 30.0)

    image_data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    labels_data = np.zeros(shape, dtype=np.int32)
    labels_data[(0,) * ndim] = 1
    image = viewer.add_image(
        image_data,
        name=f"image-{ndim}d",
        axis_labels=axis_labels,
        scale=scale,
        translate=translate,
        visible=image_visible,
    )
    labels = viewer.add_labels(
        labels_data,
        name=f"labels-{ndim}d",
        axis_labels=axis_labels,
        scale=scale,
        translate=translate,
        visible=labels_visible,
    )
    return image, labels, image_data, labels_data


def _split(widget, qtbot, cuts):
    widget.z_cuts_input.setText(cuts)
    qtbot.mouseClick(widget.split_z_btn, Qt.LeftButton)


def _bind_labels(widget, labels):
    widget.labels_combo.setCurrentText(labels.name)
    assert widget.current_labels is labels


def _select_z_layer(widget, number):
    prefix = f"Layer {number} "
    index = next(
        index
        for index, text in enumerate(_combo_texts(widget.z_view_combo))
        if text.startswith(prefix)
    )
    widget.z_view_combo.setCurrentIndex(index)


def _roi_data_by_z_center():
    data = np.full((1, 3, 6), np.nan, dtype=np.float32)
    # Z is stored scaled by the widget's default divisor of five.
    data[0, 0] = [4, 4, 5, 2, 2, 5]  # center z=1, Layer 1
    data[0, 1] = [4, 4, 15, 2, 2, 20]  # center z=3, crosses cut
    data[0, 2] = [4, 4, 25, 2, 2, 5]  # center z=5, Layer 2
    return data


def test_split_image_without_labels_creates_no_labels_proxy(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    source = viewer.add_image(
        np.arange(6 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5),
        name="image",
    )
    widget = NeuronAnnotatorWidget(viewer)

    _split(widget, qtbot, "2,4")

    assert widget.current_image is source
    assert len(_managed_layers(viewer, ROLE_Z_IMAGE)) == 3
    assert _managed_layers(viewer, ROLE_Z_LABELS) == []
    assert not source.visible


def test_switching_image_during_split_preserves_roi_state(
    make_napari_viewer, qtbot, tmp_path
):
    viewer = make_napari_viewer()
    image_a = viewer.add_image(np.zeros((6, 8, 8)), name="image-a")
    image_b = viewer.add_image(
        np.zeros((6, 8, 8)), name="image-b", translate=(3, 4, 5)
    )
    widget = NeuronAnnotatorWidget(viewer)
    widget.image_combo.setCurrentText(image_a.name)
    roi_path = tmp_path / "centers.npy"
    np.save(roi_path, _roi_data_by_z_center())
    widget.load_roi_path(roi_path)
    widget.check_all()
    widget.annotation_table.item(0, 1).setText("AVA")
    active_before = widget.active_id

    _split(widget, qtbot, "3")
    widget.image_combo.setCurrentText(image_b.name)

    assert widget.current_image is image_b
    assert widget.image_combo.currentText() == image_b.name
    assert not _managed_layers(viewer, ROLE_Z_IMAGE)
    assert widget.checked_ids == {0, 1, 2}
    assert widget.active_id == active_before
    assert widget.annotation_table.item(0, 1).text() == "AVA"
    np.testing.assert_allclose(
        _managed_vector(viewer, ROLE_SELECTED).translate, image_b.translate
    )


def test_z_layer_controls_are_compact_and_filter_image_sources(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((5, 5)), name="image-2d")
    viewer.add_image(np.zeros((6, 5, 5, 3)), name="image-rgb", rgb=True)
    viewer.add_image(
        [np.zeros((6, 5, 5)), np.zeros((3, 3, 3))],
        name="image-multiscale",
        multiscale=True,
    )
    viewer.add_image(np.zeros((6, 5, 5)), name="image-3d")
    viewer.add_image(np.zeros((2, 6, 5, 5)), name="image-4d")
    viewer.add_labels(np.zeros((6, 5, 5), dtype=int), name="labels")

    widget = LabelManager(viewer)

    assert isinstance(widget.z_image_combo, QComboBox)
    assert isinstance(widget.z_cuts_input, QLineEdit)
    assert isinstance(widget.split_z_btn, QPushButton)
    assert isinstance(widget.z_view_combo, QComboBox)
    assert isinstance(widget.clear_z_btn, QPushButton)
    assert widget.scroll_area.widget() is widget.scroll_content
    assert widget.scroll_area.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded
    group = _ancestor_group(widget.z_cuts_input)
    assert group is not None
    assert group.title() == "Z Layers"
    assert all(
        _ancestor_group(control) is group
        for control in (
            widget.z_cuts_input,
            widget.split_z_btn,
            widget.z_view_combo,
            widget.clear_z_btn,
            widget.z_profile,
            widget.z_profile_refresh_btn,
        )
    )
    descriptions = [label.text() for label in group.findChildren(QLabel)]
    assert "Split Image by z; sync optional Labels and boxes." in descriptions
    assert _combo_texts(widget.z_image_combo) == ["image-3d", "image-4d"]
    assert _combo_texts(widget.z_view_combo) == ["All"]
    assert not widget.z_view_combo.isEnabled()
    assert not widget.clear_z_btn.isEnabled()
    assert widget.z_profile_threshold_spin.value() == 170
    assert widget.z_profile._values.shape == (6,)

    widget.z_profile.cutToggled.emit(3)
    assert widget.z_cuts_input.text() == "3"
    widget.z_profile.cutToggled.emit(3)
    assert widget.z_cuts_input.text() == ""


def test_layer_selection_clamps_only_the_2d_z_position(
    make_napari_viewer,
    qtbot,
):
    viewer = make_napari_viewer()
    _add_matching_layers(viewer, (2, 6, 4, 5))
    widget = LabelManager(viewer)
    viewer.dims.current_step = (1, 0, 0, 0)

    _split(widget, qtbot, "3")
    _select_z_layer(widget, 2)

    assert viewer.dims.current_step == (1, 4, 0, 0)

    viewer.dims.ndisplay = 3
    viewer.dims.current_step = (1, 1, 0, 0)
    widget.z_view_combo.setCurrentIndex(0)
    _select_z_layer(widget, 2)

    assert viewer.dims.current_step == (1, 1, 0, 0)


def test_z_profile_refresh_uses_the_current_4d_time(
    make_napari_viewer,
    qtbot,
):
    viewer = make_napari_viewer()
    image_data = np.zeros((2, 3, 2, 2), dtype=np.uint16)
    image_data[0, :, 0, 0] = [171, 50, 200]
    image_data[1, :, 1, 1] = [10, 180, 300]
    viewer.add_image(image_data, name="image")
    viewer.add_labels(
        np.zeros_like(image_data, dtype=np.int16),
        name="labels",
    )

    widget = LabelManager(viewer)

    np.testing.assert_array_equal(widget.z_profile._values, [1, 0, 1])
    assert widget.z_profile_threshold_spin.value() == 170
    assert widget.z_profile_state_label.text() == "t=0"

    viewer.dims.current_step = (1, 0, 0, 0)
    qtbot.waitUntil(
        lambda: "Refresh" in widget.z_profile_state_label.text(),
        timeout=1000,
    )
    qtbot.mouseClick(widget.z_profile_refresh_btn, Qt.LeftButton)

    np.testing.assert_array_equal(widget.z_profile._values, [0, 1, 1])
    assert widget.z_profile_state_label.text() == "t=1"

    widget.z_profile_threshold_spin.setValue(190)

    np.testing.assert_array_equal(widget.z_profile._values, [0, 0, 1])
    assert widget.z_profile_state_label.text() == "t=1"


@pytest.mark.parametrize(
    "shape",
    [
        (6, 4, 5),
        (2, 6, 4, 5),
    ],
)
def test_split_3d_and_4d_synchronizes_image_and_labels(
    make_napari_viewer,
    qtbot,
    shape,
):
    viewer = make_napari_viewer()
    source, labels, image_data, labels_data = _add_matching_layers(
        viewer, shape
    )
    labels.contour = 1
    widget = LabelManager(viewer)
    _bind_labels(widget, labels)
    labels_before = labels_data.copy()
    label_color = np.asarray(labels.get_color(1), dtype=float)

    _split(widget, qtbot, "2,4")

    derived = sorted(
        _managed_layers(viewer, ROLE_Z_IMAGE),
        key=lambda layer: float(layer.translate[-3]),
    )
    assert len(derived) == 3
    assert [layer.data.shape[-3] for layer in derived] == [2, 2, 2]
    assert all(np.shares_memory(layer.data, image_data) for layer in derived)
    assert all(layer.blending == "additive" for layer in derived)
    assert source.blending != "additive"
    assert [float(layer.translate[-3]) for layer in derived] == [
        10.0,
        14.0,
        18.0,
    ]
    assert not source.visible
    assert all(layer.visible for layer in derived)
    assert labels.visible
    assert widget.z_view_combo.isEnabled()
    assert widget.clear_z_btn.isEnabled()
    assert len(_combo_texts(widget.z_view_combo)) == 4
    assert _combo_texts(widget.z_image_combo) == [source.name]

    _select_z_layer(widget, 2)

    assert [layer.visible for layer in derived] == [False, True, False]
    assert not source.visible
    assert not labels.visible
    proxies = _managed_layers(viewer, ROLE_Z_LABELS)
    assert len(proxies) == 1
    proxy = proxies[0]
    assert isinstance(proxy, Labels)
    assert proxy.visible
    assert not proxy.editable
    assert proxy.contour == 1
    assert proxy.data.shape[-3] == 2
    assert np.shares_memory(proxy.data, labels_data)
    np.testing.assert_array_equal(proxy.data, labels_data[..., 2:4, :, :])
    assert tuple(proxy.axis_labels) == tuple(labels.axis_labels)
    np.testing.assert_allclose(proxy.scale, labels.scale)
    expected_translate = np.asarray(labels.translate, dtype=float)
    expected_translate[-3] += 2 * float(labels.scale[-3])
    np.testing.assert_allclose(proxy.translate, expected_translate)
    assert proxy.name not in _combo_texts(widget.layer_combo)

    assert not widget.selected_opacity_slider.isEnabled()
    proxy_color = np.asarray(proxy.get_color(1), dtype=float)
    np.testing.assert_allclose(proxy_color, label_color)
    np.testing.assert_array_equal(labels.data, labels_before)

    widget.z_view_combo.setCurrentIndex(0)

    assert all(layer.visible for layer in derived)
    assert labels.visible
    assert not any(
        layer.visible for layer in _managed_layers(viewer, ROLE_Z_LABELS)
    )


def test_memmap_split_layers_keep_shared_storage(
    make_napari_viewer,
    qtbot,
    tmp_path,
):
    image_path = tmp_path / "image.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(image_path, np.ones((6, 4, 5), dtype=np.float32))
    np.save(labels_path, np.ones((6, 4, 5), dtype=np.int16))
    image_data = np.load(image_path, mmap_mode="r", allow_pickle=False)
    labels_data = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    viewer = make_napari_viewer()
    viewer.add_image(
        image_data,
        name="memmap-image",
        axis_labels=("z", "y", "x"),
    )
    labels = viewer.add_labels(
        labels_data,
        name="memmap-labels",
        axis_labels=("z", "y", "x"),
    )
    widget = LabelManager(viewer)
    _bind_labels(widget, labels)

    _split(widget, qtbot, "2,4")
    _select_z_layer(widget, 2)

    assert all(
        np.shares_memory(layer.data, image_data)
        for layer in _managed_layers(viewer, ROLE_Z_IMAGE)
    )
    proxy = _managed_layers(viewer, ROLE_Z_LABELS)[0]
    assert np.shares_memory(proxy.data, labels_data)


def test_dask_split_layers_remain_lazy(
    make_napari_viewer,
    qtbot,
):
    da = pytest.importorskip("dask.array")
    image_data = da.ones((6, 4, 5), chunks=(2, 4, 5))
    labels_data = da.ones((6, 4, 5), chunks=(2, 4, 5), dtype=np.int16)
    viewer = make_napari_viewer()
    viewer.add_image(
        image_data,
        name="dask-image",
        axis_labels=("z", "y", "x"),
        contrast_limits=(0, 1),
    )
    labels = viewer.add_labels(
        labels_data,
        name="dask-labels",
        axis_labels=("z", "y", "x"),
    )
    widget = LabelManager(viewer)
    _bind_labels(widget, labels)

    _split(widget, qtbot, "2,4")
    _select_z_layer(widget, 2)

    assert all(
        isinstance(layer.data, da.Array)
        for layer in _managed_layers(viewer, ROLE_Z_IMAGE)
    )
    proxy = _managed_layers(viewer, ROLE_Z_LABELS)[0]
    assert isinstance(proxy.data, da.Array)


def test_direct_zarr_split_is_rejected_before_materialization(
    make_napari_viewer,
):
    zarr = pytest.importorskip("zarr")
    image_data = zarr.array(np.ones((6, 4, 5), dtype=np.float32))
    labels_data = zarr.array(np.ones((6, 4, 5), dtype=np.int16))
    viewer = make_napari_viewer()
    image = viewer.add_image(
        image_data,
        name="zarr-image",
        axis_labels=("z", "y", "x"),
        contrast_limits=(0, 1),
    )
    viewer.add_labels(
        labels_data,
        name="zarr-labels",
        axis_labels=("z", "y", "x"),
    )
    widget = LabelManager(viewer)

    with pytest.raises(ValueError, match="Direct Zarr"):
        widget._validate_image_source(image)

    assert widget.current_image is None
    assert not _managed_layers(viewer, ROLE_Z_IMAGE)
    assert not _managed_layers(viewer, ROLE_Z_LABELS)


def test_layer_view_filters_whole_boxes_by_center_and_navigation(
    make_napari_viewer,
    qtbot,
    tmp_path,
):
    viewer = make_napari_viewer()
    source, labels, _, _ = _add_matching_layers(viewer, (6, 8, 8))
    del source, labels
    widget = LabelManager(viewer)
    roi_path = tmp_path / "centers.npy"
    np.save(roi_path, _roi_data_by_z_center())
    widget.load_roi_path(roi_path)
    viewer.dims.ndisplay = 3
    widget.check_all()
    widget.show_box_labels_checkbox.setChecked(True)

    _split(widget, qtbot, "3")
    box_labels = _managed_box_labels(viewer)
    assert set(box_labels.features["neuron_id"]) == {0, 1, 2}
    _select_z_layer(widget, 1)

    selected = _managed_vector(viewer, ROLE_SELECTED)
    active = _managed_vector(viewer, ROLE_ACTIVE)
    assert set(selected.features["neuron_id"]) == {0}
    assert set(active.features["neuron_id"]) == {0}
    assert set(box_labels.features["neuron_id"]) == {0}

    _select_z_layer(widget, 2)

    assert widget.checked_ids == {0, 1, 2}
    assert widget.active_id == 0
    assert set(selected.features["neuron_id"]) == {1, 2}
    assert set(box_labels.features["neuron_id"]) == {1, 2}
    assert len(active.data) == 0

    crossing = np.asarray(selected.data)[
        np.asarray(selected.features["neuron_id"]) == 1
    ]
    crossing_start_z = crossing[:, 0, 0]
    crossing_stop_z = crossing_start_z + crossing[:, 1, 0]
    assert min(crossing_start_z.min(), crossing_stop_z.min()) < 3
    assert max(crossing_start_z.max(), crossing_stop_z.max()) > 3

    outside = widget._selection_items[0]
    inside = widget._selection_items[1]
    gray = QColor("#777777")
    assert outside.foreground(1).color() == gray
    assert inside.foreground(1).color() != gray
    assert "(missing)" not in outside.text(1)
    assert outside.font(1).bold()
    assert outside.flags() & Qt.ItemIsEnabled
    assert outside.checkState(0) == Qt.Checked

    widget.navigate(1)
    assert widget.active_id == 1
    assert widget.checked_ids == {0, 1, 2}
    widget.navigate(1)
    assert widget.active_id == 2
    widget.navigate(1)
    assert widget.active_id == 1
    widget.navigate(-1)
    assert widget.active_id == 2

    outside.setCheckState(0, Qt.Unchecked)
    assert widget.checked_ids == {1, 2}
    assert widget.active_id == 2
    outside.setCheckState(0, Qt.Checked)
    assert widget.checked_ids == {0, 1, 2}
    assert widget.active_id == 0
    assert outside.font(1).bold()
    assert len(active.data) == 0
    widget.navigate(1)
    assert widget.active_id == 1


def test_4d_time_change_recomputes_center_membership(
    make_napari_viewer,
    qtbot,
    tmp_path,
):
    viewer = make_napari_viewer()
    _add_matching_layers(viewer, (2, 6, 8, 8))
    widget = LabelManager(viewer)
    roi = np.full((2, 2, 6), np.nan, dtype=np.float32)
    roi[0, 0] = [4, 4, 5, 2, 2, 5]
    roi[0, 1] = [4, 4, 25, 2, 2, 5]
    roi[1, 0] = [4, 4, 25, 2, 2, 5]
    roi[1, 1] = [4, 4, 5, 2, 2, 5]
    roi_path = tmp_path / "time-centers.npy"
    np.save(roi_path, roi)
    widget.load_roi_path(roi_path)
    viewer.dims.ndisplay = 3
    widget.check_all()
    widget.show_box_labels_checkbox.setChecked(True)
    _split(widget, qtbot, "3")
    _select_z_layer(widget, 1)
    selected = _managed_vector(viewer, ROLE_SELECTED)
    box_labels = _managed_box_labels(viewer)

    assert set(selected.features["neuron_id"]) == {0}
    assert set(box_labels.features["neuron_id"]) == {0}

    viewer.dims.current_step = (1, 0, 0, 0)
    qtbot.waitUntil(
        lambda: set(selected.features["neuron_id"]) == {1},
        timeout=1000,
    )
    assert widget.checked_ids == {0, 1}
    assert widget.active_id == 0
    assert set(box_labels.features["neuron_id"]) == {1}

    widget.navigate(1)

    assert widget.active_id == 1
    assert widget.checked_ids == {0, 1}


def test_clear_restores_sources_and_preserves_global_selection(
    make_napari_viewer,
    qtbot,
    tmp_path,
):
    viewer = make_napari_viewer()
    image, labels, _, _ = _add_matching_layers(
        viewer,
        (6, 8, 8),
        image_visible=False,
        labels_visible=False,
    )
    widget = LabelManager(viewer)
    _bind_labels(widget, labels)
    roi_path = tmp_path / "centers.npy"
    np.save(roi_path, _roi_data_by_z_center())
    widget.load_roi_path(roi_path)
    widget.check_all()
    expected_checked = set(widget.checked_ids)
    expected_active = widget.active_id

    _split(widget, qtbot, "3")

    assert labels.visible
    assert any(
        layer.visible for layer in _managed_layers(viewer, ROLE_Z_IMAGE)
    )
    qtbot.mouseClick(widget.clear_z_btn, Qt.LeftButton)

    assert not _managed_layers(viewer, ROLE_Z_IMAGE)
    assert not _managed_layers(viewer, ROLE_Z_LABELS)
    assert not image.visible
    assert not labels.visible
    assert widget.checked_ids == expected_checked
    assert widget.active_id == expected_active
    assert widget.z_view_combo.currentText() == "All"
    assert not widget.z_view_combo.isEnabled()
    assert not widget.clear_z_btn.isEnabled()

    widget.clear_z_layers()
    assert widget.checked_ids == expected_checked
    assert widget.active_id == expected_active

    _split(widget, qtbot, "2,4")
    widget.shutdown()
    assert not _managed_layers(viewer, ROLE_Z_IMAGE)
    assert not _managed_layers(viewer, ROLE_Z_LABELS)
    assert not image.visible
    assert not labels.visible


def test_resplit_replaces_the_existing_managed_layer_set(
    make_napari_viewer,
    qtbot,
):
    viewer = make_napari_viewer()
    _add_matching_layers(viewer, (6, 4, 5))
    widget = LabelManager(viewer)
    _split(widget, qtbot, "3")
    old_layers = [
        *_managed_layers(viewer, ROLE_Z_IMAGE),
        *_managed_layers(viewer, ROLE_Z_LABELS),
    ]
    _select_z_layer(widget, 2)

    _split(widget, qtbot, "2,4")

    assert all(layer not in viewer.layers for layer in old_layers)
    new_images = _managed_layers(viewer, ROLE_Z_IMAGE)
    assert len(new_images) == 3
    assert [layer.data.shape[-3] for layer in new_images] == [2, 2, 2]
    assert all(layer.visible for layer in new_images)
    assert widget.z_view_combo.currentText() == "All"


def test_switching_controlled_labels_clears_then_allows_new_split(
    make_napari_viewer,
    qtbot,
):
    viewer = make_napari_viewer()
    image, labels_a, _, _ = _add_matching_layers(viewer, (6, 4, 5))
    labels_b = viewer.add_labels(
        np.zeros((6, 4, 5), dtype=np.int16),
        name="labels-b",
        axis_labels=tuple(labels_a.axis_labels),
        scale=tuple(labels_a.scale),
        translate=tuple(labels_a.translate),
    )
    widget = LabelManager(viewer)
    _bind_labels(widget, labels_a)
    _split(widget, qtbot, "3")

    widget.layer_combo.setCurrentText(labels_b.name)

    assert widget.current_layer is labels_b
    assert not _managed_layers(viewer, ROLE_Z_IMAGE)
    assert not _managed_layers(viewer, ROLE_Z_LABELS)
    assert image.visible
    assert labels_a.visible

    _split(widget, qtbot, "2,4")

    assert len(_managed_layers(viewer, ROLE_Z_IMAGE)) == 3
    assert widget.current_layer is labels_b


def test_split_rejects_transform_mismatch_without_partial_layers(
    make_napari_viewer,
    qtbot,
    monkeypatch,
):
    viewer = make_napari_viewer()
    image = viewer.add_image(
        np.zeros((6, 4, 5)),
        name="image",
        scale=(2, 1, 1),
    )
    labels = viewer.add_labels(
        np.zeros((6, 4, 5), dtype=int),
        name="labels",
        scale=(3, 1, 1),
    )
    widget = LabelManager(viewer)
    widget.labels_combo.setCurrentText(labels.name)
    del qtbot, monkeypatch

    with pytest.raises(ValueError, match="scale"):
        widget._validate_labels_binding(image, labels)

    assert widget.current_labels is None
    assert not _managed_layers(viewer, ROLE_Z_IMAGE)
    assert not _managed_layers(viewer, ROLE_Z_LABELS)
    assert image.visible
    assert labels.visible
    assert not widget.z_view_combo.isEnabled()
    assert "scale" in widget.status_label.text().lower()


def test_split_rejects_plane_depiction(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    image, _, _, _ = _add_matching_layers(viewer, (6, 4, 5))
    image.depiction = "plane"
    widget = LabelManager(viewer)

    with pytest.raises(ValueError, match="volume depiction"):
        widget._validate_image_source(image)

    assert not _managed_layers(viewer, ROLE_Z_IMAGE)
    assert not _managed_layers(viewer, ROLE_Z_LABELS)


@pytest.mark.parametrize("changed_source", ["image", "labels"])
def test_source_data_replacement_clears_split_session(
    make_napari_viewer,
    qtbot,
    changed_source,
):
    viewer = make_napari_viewer()
    image, labels, _, _ = _add_matching_layers(viewer, (6, 4, 5))
    widget = LabelManager(viewer)
    _bind_labels(widget, labels)
    _split(widget, qtbot, "3")
    assert _managed_layers(viewer, ROLE_Z_IMAGE)

    source = image if changed_source == "image" else labels
    source.data = np.array(source.data, copy=True)

    qtbot.waitUntil(
        lambda: not _managed_layers(viewer, ROLE_Z_IMAGE),
        timeout=1000,
    )
    assert not _managed_layers(viewer, ROLE_Z_LABELS)
    assert image.visible
    assert labels.visible
    assert not widget.z_view_combo.isEnabled()


def test_removing_one_derived_image_clears_the_split_session(
    make_napari_viewer,
    qtbot,
):
    viewer = make_napari_viewer()
    image, labels, _, _ = _add_matching_layers(viewer, (6, 4, 5))
    widget = LabelManager(viewer)
    _bind_labels(widget, labels)
    _split(widget, qtbot, "2,4")
    derived = _managed_layers(viewer, ROLE_Z_IMAGE)
    assert len(derived) == 3

    viewer.layers.remove(derived[1])

    assert not _managed_layers(viewer, ROLE_Z_IMAGE)
    assert not _managed_layers(viewer, ROLE_Z_LABELS)
    assert image.visible
    assert labels.visible
    assert not widget.z_view_combo.isEnabled()


@pytest.mark.parametrize("removed_role", ["image", "labels", "proxy"])
def test_removing_a_split_source_or_proxy_clears_the_session(
    make_napari_viewer,
    qtbot,
    removed_role,
):
    viewer = make_napari_viewer()
    image, labels, _, _ = _add_matching_layers(viewer, (6, 4, 5))
    widget = LabelManager(viewer)
    _bind_labels(widget, labels)
    _split(widget, qtbot, "3")
    _select_z_layer(widget, 1)
    proxy = _managed_layers(viewer, ROLE_Z_LABELS)[0]
    removed = {"image": image, "labels": labels, "proxy": proxy}[removed_role]

    viewer.layers.remove(removed)

    assert not _managed_layers(viewer, ROLE_Z_IMAGE)
    assert not _managed_layers(viewer, ROLE_Z_LABELS)
    assert not widget.z_view_combo.isEnabled()
    if image in viewer.layers:
        assert image.visible
    if labels in viewer.layers:
        assert labels.visible


def test_source_geometry_change_clears_the_split_session(
    make_napari_viewer,
    qtbot,
):
    viewer = make_napari_viewer()
    image, labels, _, _ = _add_matching_layers(viewer, (6, 4, 5))
    widget = LabelManager(viewer)
    _bind_labels(widget, labels)
    _split(widget, qtbot, "3")

    image.scale = (3, 1, 1)

    assert not _managed_layers(viewer, ROLE_Z_IMAGE)
    assert not _managed_layers(viewer, ROLE_Z_LABELS)
    assert image.visible
    assert labels.visible
    assert not widget.z_view_combo.isEnabled()


def test_labels_update_refreshes_the_proxy_without_clearing_split(
    make_napari_viewer,
    qtbot,
):
    viewer = make_napari_viewer()
    _, labels, _, _ = _add_matching_layers(viewer, (6, 4, 5))
    widget = LabelManager(viewer)
    _bind_labels(widget, labels)
    _split(widget, qtbot, "3")
    _select_z_layer(widget, 1)
    proxy = _managed_layers(viewer, ROLE_Z_LABELS)[0]

    labels.data[1, 1, 1] = 2
    labels.events.labels_update(
        data=np.asarray([[[2]]], dtype=labels.data.dtype),
        offset=(1, 1, 1),
    )

    assert proxy.data[1, 1, 1] == 2
    assert widget._available_ids == []
    assert _managed_layers(viewer, ROLE_Z_IMAGE)
    assert proxy.visible
    assert not proxy.editable
