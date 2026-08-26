import numpy as np
import pytest
from app_model.types import KeyBinding
from napari.layers import Points, Vectors
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QComboBox, QLineEdit, QMessageBox, QTreeWidget

from napari_worm_neuron_annotator._colors import neuron_color
from napari_worm_neuron_annotator._widget import (
    EXCEL_AVAILABLE,
    ROLE_ACTIVE,
    ROLE_BOX_LABELS,
    ROLE_KEY,
    ROLE_SELECTED,
    LabelManager,
    NeuronAnnotatorWidget,
    _match_neuron_ids,
)


def _roi_data():
    data = np.full((2, 2, 8), np.nan, dtype=np.float32)
    data[0, 0, :6] = [10, 10, 10, 8, 8, 10]
    data[0, 1, :6] = [10, 10, 10, 4, 4, 5]
    data[1, 1, :6] = [12, 12, 15, 4, 4, 5]
    return data


def _managed_layer(viewer, role):
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


def _box_label_rgba(layer):
    return np.asarray(layer.text.color.constant, dtype=float)


def _camera_orientation2d(viewer):
    return tuple(
        getattr(value, "value", value)
        for value in viewer.camera.orientation2d
    )


def _press_viewer_key(viewer, key):
    viewer.keymap[KeyBinding.from_str(key)](viewer)


def test_image_is_spatial_authority_and_roi_works_without_labels(
    make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    image = viewer.add_image(
        np.zeros((2, 6, 24, 24), dtype=np.uint16),
        name="image",
        axis_labels=("t", "z", "y", "x"),
        scale=(2, 5, 1, 1),
        translate=(3, 4, 5, 6),
        units=("s", "um", "um", "um"),
    )
    widget = NeuronAnnotatorWidget(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data())

    widget.load_roi_path(roi_path)
    viewer.dims.current_step = (0, 2, 0, 0)

    assert LabelManager is NeuronAnnotatorWidget
    assert widget.current_image is image
    assert widget._available_ids == [0, 1]
    assert widget.checked_ids == {0}
    selected = _managed_layer(viewer, ROLE_SELECTED)
    assert selected.ndim == 4
    assert selected.data.shape == (4, 2, 4)
    np.testing.assert_allclose(selected.scale, image.scale)
    np.testing.assert_allclose(selected.translate, image.translate)


def test_widget_ignores_labels_layers_when_no_image_exists(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((3, 2, 2), dtype=np.int32), name="labels")

    widget = NeuronAnnotatorWidget(viewer)

    assert widget.current_image is None
    assert not widget.load_roi_btn.isEnabled()
    assert not hasattr(widget, "labels_combo")
    assert not hasattr(widget, "labels_layer_group")
    assert not widget.orientation_group.isEnabled()


def test_orientation_controls_apply_absolute_viewer_transform_and_reset(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    image_data = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    image = viewer.add_image(image_data, name="image")
    viewer.camera.orientation2d = ("up", "left")
    widget = NeuronAnnotatorWidget(viewer)
    viewer.camera.center = (1.0, 2.0, 3.0)
    viewer.camera.zoom = 2.5
    baseline_center = tuple(viewer.camera.center)
    baseline_zoom = viewer.camera.zoom

    assert widget.orientation_group.isEnabled()
    assert not widget.orientation_reset_btn.isEnabled()
    assert widget.orientation_rotation_combo.currentData() == 0

    widget.orientation_rotation_combo.setCurrentIndex(
        widget.orientation_rotation_combo.findData(90)
    )

    assert tuple(viewer.dims.order) == (0, 2, 1)
    assert _camera_orientation2d(viewer) == ("up", "right")
    assert widget.orientation_reset_btn.isEnabled()

    widget.flip_horizontal_checkbox.setChecked(True)
    widget.flip_vertical_checkbox.setChecked(True)

    assert tuple(viewer.dims.order) == (0, 2, 1)
    assert _camera_orientation2d(viewer) == ("down", "left")
    assert tuple(viewer.camera.center) == baseline_center
    assert viewer.camera.zoom == pytest.approx(baseline_zoom)
    assert image.data is image_data

    viewer.dims.ndisplay = 3
    viewer.camera.angles = (10.0, 20.0, 30.0)
    expected_angles = tuple(viewer.camera.angles)
    widget.flip_vertical_checkbox.setChecked(False)
    np.testing.assert_allclose(viewer.camera.angles, expected_angles)
    assert widget.orientation_rotation_combo.currentData() == 90

    widget.orientation_reset_btn.click()

    assert tuple(viewer.dims.order) == (0, 1, 2)
    assert _camera_orientation2d(viewer) == ("up", "left")
    assert widget.orientation_rotation_combo.currentData() == 0
    assert not widget.flip_horizontal_checkbox.isChecked()
    assert not widget.flip_vertical_checkbox.isChecked()
    assert not widget.orientation_reset_btn.isEnabled()


def test_external_orientation_change_becomes_new_baseline(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((3, 4, 5)), name="image")
    widget = NeuronAnnotatorWidget(viewer)
    widget.orientation_rotation_combo.setCurrentIndex(
        widget.orientation_rotation_combo.findData(90)
    )
    assert tuple(viewer.dims.order) == (0, 2, 1)

    viewer.dims.transpose()

    assert tuple(viewer.dims.order) == (0, 1, 2)
    assert _camera_orientation2d(viewer) == ("down", "left")
    assert widget.orientation_rotation_combo.currentData() == 0
    assert not widget.orientation_reset_btn.isEnabled()

    viewer.camera.orientation2d = ("up", "right")
    widget.flip_horizontal_checkbox.setChecked(True)
    assert _camera_orientation2d(viewer) == ("up", "left")
    widget.reset_orientation()
    assert tuple(viewer.dims.order) == (0, 1, 2)
    assert _camera_orientation2d(viewer) == ("up", "right")


def test_shutdown_restores_orientation_baseline(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((3, 4, 5)), name="image")
    viewer.camera.orientation2d = ("up", "left")
    baseline_order = tuple(viewer.dims.order)
    widget = NeuronAnnotatorWidget(viewer)

    widget.orientation_rotation_combo.setCurrentIndex(
        widget.orientation_rotation_combo.findData(270)
    )
    widget.flip_vertical_checkbox.setChecked(True)
    assert tuple(viewer.dims.order) != baseline_order

    widget.shutdown()

    assert tuple(viewer.dims.order) == baseline_order
    assert _camera_orientation2d(viewer) == ("up", "left")


def test_viewer_ndim_change_rebases_orientation(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((3, 4, 5)), name="image-3d")
    widget = NeuronAnnotatorWidget(viewer)
    widget.orientation_rotation_combo.setCurrentIndex(
        widget.orientation_rotation_combo.findData(90)
    )
    assert widget.orientation_rotation_combo.currentData() == 90

    image_4d = viewer.add_image(np.zeros((2, 3, 4, 5)), name="image-4d")

    assert viewer.dims.ndim == 4
    assert widget._orientation_baseline_ndim == 4
    assert widget.orientation_rotation_combo.currentData() == 0
    assert not widget.flip_horizontal_checkbox.isChecked()
    assert not widget.flip_vertical_checkbox.isChecked()
    assert widget._orientation_baseline_order == tuple(viewer.dims.order)

    widget.image_combo.setCurrentText(image_4d.name)
    assert widget.current_image is image_4d
    assert widget.orientation_group.isEnabled()


def test_same_ndim_image_switch_preserves_orientation(make_napari_viewer):
    viewer = make_napari_viewer()
    image_a = viewer.add_image(np.zeros((3, 4, 5)), name="image-a")
    image_b = viewer.add_image(np.zeros((6, 7, 8)), name="image-b")
    widget = NeuronAnnotatorWidget(viewer)
    widget.image_combo.setCurrentText(image_a.name)
    widget.orientation_rotation_combo.setCurrentIndex(
        widget.orientation_rotation_combo.findData(90)
    )
    widget.flip_vertical_checkbox.setChecked(True)
    expected_order = tuple(viewer.dims.order)
    expected_camera = _camera_orientation2d(viewer)

    widget.image_combo.setCurrentText(image_b.name)

    assert widget.current_image is image_b
    assert widget.orientation_rotation_combo.currentData() == 90
    assert widget.flip_vertical_checkbox.isChecked()
    assert tuple(viewer.dims.order) == expected_order
    assert _camera_orientation2d(viewer) == expected_camera


def test_g_h_navigate_z_only_in_2d_and_unbind_on_shutdown(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 3, 2)), name="image")
    widget = NeuronAnnotatorWidget(viewer)
    viewer.dims.current_step = (2, 0, 0)

    _press_viewer_key(viewer, "G")
    assert viewer.dims.current_step == (1, 0, 0)
    _press_viewer_key(viewer, "G")
    _press_viewer_key(viewer, "G")
    assert viewer.dims.current_step == (0, 0, 0)

    _press_viewer_key(viewer, "H")
    assert viewer.dims.current_step == (1, 0, 0)
    viewer.dims.current_step = (3, 0, 0)
    _press_viewer_key(viewer, "H")
    assert viewer.dims.current_step == (3, 0, 0)

    viewer.dims.ndisplay = 3
    _press_viewer_key(viewer, "G")
    assert viewer.dims.current_step == (3, 0, 0)

    widget.shutdown()
    assert KeyBinding.from_str("G") not in viewer.keymap
    assert KeyBinding.from_str("H") not in viewer.keymap


def test_j_k_navigate_4d_time_with_shift_acceleration(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((25, 4, 3, 2)), name="image-4d")
    widget = NeuronAnnotatorWidget(viewer)
    viewer.dims.current_step = (12, 2, 0, 0)

    _press_viewer_key(viewer, "J")
    assert viewer.dims.current_step == (11, 2, 0, 0)
    _press_viewer_key(viewer, "K")
    assert viewer.dims.current_step == (12, 2, 0, 0)

    _press_viewer_key(viewer, "Shift-J")
    assert viewer.dims.current_step == (2, 2, 0, 0)
    _press_viewer_key(viewer, "Shift-J")
    assert viewer.dims.current_step == (0, 2, 0, 0)
    viewer.dims.current_step = (20, 2, 0, 0)
    _press_viewer_key(viewer, "Shift-K")
    _press_viewer_key(viewer, "Shift-K")
    assert viewer.dims.current_step == (24, 2, 0, 0)

    viewer.dims.ndisplay = 3
    _press_viewer_key(viewer, "J")
    assert viewer.dims.current_step == (23, 2, 0, 0)

    viewer.add_image(np.zeros((4, 3, 2)), name="image-3d")
    widget.image_combo.setCurrentText("image-3d")
    before = viewer.dims.current_step
    _press_viewer_key(viewer, "K")
    assert viewer.dims.current_step == before

    widget.shutdown()
    for key in ("J", "K", "Shift-J", "Shift-K"):
        assert KeyBinding.from_str(key) not in viewer.keymap


def test_widget_initializes_checkable_selection(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((3, 2, 2)), name="image")

    widget = LabelManager(viewer)

    assert isinstance(widget.selection_tree, QTreeWidget)
    assert isinstance(widget.neuron_search_input, QLineEdit)
    assert isinstance(widget.box_label_mode_combo, QComboBox)
    assert not widget.selection_tree.alternatingRowColors()
    assert widget.navigation_help_label.text() == "Q/W: last/next"
    assert "Shift+Q/W" in widget.navigation_help_label.toolTip()
    assert [
        widget.box_label_mode_combo.itemText(index)
        for index in range(widget.box_label_mode_combo.count())
    ] == ["Biological", "Digital", "Digital + biological"]
    assert widget.box_label_mode_combo.currentData() == "biological"
    assert widget.search_matches_label.text() == ""
    assert not widget.check_matches_btn.isEnabled()
    assert widget.show_box_labels_checkbox.text() == (
        "Show selected box labels"
    )
    assert not widget.show_box_labels_checkbox.isChecked()
    assert widget.box_label_color_btn.text() == "#FFFFFF"
    assert widget._available_ids == []
    assert widget.active_id is None
    assert widget.checked_ids == set()
    assert not hasattr(widget, "reset_btn")
    assert not hasattr(widget, "labels_visible_checkbox")


def test_roi_load_preserves_covered_ids_and_renders_2d_and_3d(
    make_napari_viewer, qtbot, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(
        np.zeros((2, 6, 24, 24), dtype=np.uint16),
        name="image",
        scale=(1, 5, 1, 1),
    )
    expected_rgb = {
        neuron_id: np.asarray(neuron_color(neuron_id))[:3]
        for neuron_id in (0, 1)
    }
    widget = LabelManager(viewer)

    roi_path = tmp_path / "neuron_pt_tuple.npy"
    np.save(roi_path, _roi_data())
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(roi_path), "NumPy arrays (*.npy)"),
    )

    widget.load_roi_npy()
    viewer.dims.current_step = (0, 2, 0, 0)
    qtbot.wait(10)

    assert widget._available_ids == [0, 1]
    assert set(widget._selection_items) == {0, 1}
    assert widget.checked_ids == {0}
    selected_layer = _managed_layer(viewer, ROLE_SELECTED)
    active_layer = _managed_layer(viewer, ROLE_ACTIVE)
    assert active_layer.edge_width == 2.0
    assert selected_layer.data.shape == (4, 2, 4)
    assert active_layer.data.shape == (4, 2, 4)
    assert set(selected_layer.features["neuron_id"]) == {0}
    index = list(selected_layer.features["neuron_id"]).index(0)
    np.testing.assert_allclose(
        np.asarray(selected_layer.edge_color)[index, :3],
        expected_rgb[0],
    )

    widget._selection_items[1].setCheckState(0, Qt.Checked)
    assert widget.checked_ids == {0, 1}
    assert widget.active_id == 1
    assert selected_layer.data.shape == (8, 2, 4)
    for neuron_id in (0, 1):
        index = list(selected_layer.features["neuron_id"]).index(neuron_id)
        np.testing.assert_allclose(
            np.asarray(selected_layer.edge_color)[index, :3],
            expected_rgb[neuron_id],
        )

    viewer.dims.ndisplay = 3
    qtbot.wait(10)

    assert selected_layer.data.shape == (24, 2, 4)
    assert active_layer.data.shape == (12, 2, 4)
    assert set(selected_layer.features["neuron_id"]) == {0, 1}


def test_native_yx_transpose_preserves_roi_vectors_points_and_centering(
    make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    image = viewer.add_image(
        np.zeros((2, 6, 24, 24), dtype=np.uint16),
        name="image",
        scale=(2, 5, 1, 1),
        translate=(3, 4, 5, 6),
    )
    widget = NeuronAnnotatorWidget(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data())
    widget.load_roi_path(roi_path)
    viewer.dims.current_step = (0, 2, 0, 0)
    widget.show_box_labels_checkbox.setChecked(True)

    selected = _managed_layer(viewer, ROLE_SELECTED)
    active = _managed_layer(viewer, ROLE_ACTIVE)
    box_labels = _managed_box_labels(viewer)
    expected_selected = np.asarray(selected.data).copy()
    expected_active = np.asarray(active.data).copy()
    expected_points = np.asarray(box_labels.data).copy()
    expected_checked = set(widget.checked_ids)
    expected_active_id = widget.active_id

    viewer.dims.transpose()

    assert tuple(viewer.dims.order) == (0, 1, 3, 2)
    np.testing.assert_allclose(selected.data, expected_selected)
    np.testing.assert_allclose(active.data, expected_active)
    np.testing.assert_allclose(box_labels.data, expected_points)
    assert list(selected.features["neuron_id"]) == [0] * 4
    assert list(active.features["neuron_id"]) == [0] * 4
    assert list(box_labels.features["neuron_id"]) == [0]
    assert widget.checked_ids == expected_checked
    assert widget.active_id == expected_active_id

    widget.activate_id(0)
    box = widget.roi_dataset.get_box(0, 0)
    assert box is not None
    world = image.data_to_world((0, *box.center_zyx))
    np.testing.assert_allclose(viewer.camera.center, world[-3:])

    viewer.dims.ndisplay = 3
    assert selected.data.shape == (12, 2, 4)
    assert active.data.shape == (12, 2, 4)
    assert box_labels.data.shape == (1, 4)

    viewer.dims.transpose()
    assert tuple(viewer.dims.order) == (0, 1, 2, 3)
    assert selected.data.shape == (12, 2, 4)
    assert active.data.shape == (12, 2, 4)
    assert box_labels.data.shape == (1, 4)


def test_unsupported_spatial_roll_still_hides_roi(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 24, 24), dtype=np.uint16), name="image")
    widget = NeuronAnnotatorWidget(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    widget.load_roi_path(roi_path)
    viewer.dims.current_step = (2, 0, 0)
    selected = _managed_layer(viewer, ROLE_SELECTED)
    active = _managed_layer(viewer, ROLE_ACTIVE)
    assert selected.data.shape == (4, 2, 3)

    viewer.dims.order = (1, 2, 0)

    assert selected.data.shape == (0, 2, 3)
    assert active.data.shape == (0, 2, 3)
    assert not widget.orientation_group.isEnabled()

    viewer.dims.order = (0, 1, 2)
    assert selected.data.shape == (4, 2, 3)
    assert active.data.shape == (4, 2, 3)
    assert widget.orientation_group.isEnabled()


def test_optional_box_labels_use_biological_name_with_id_fallback(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.zeros((2, 6, 24, 24), dtype=np.uint16),
        name="image",
        axis_labels=("t", "z", "y", "x"),
        scale=(2, 5, 1, 1),
        translate=(3, 4, 5, 6),
        units=("s", "um", "um", "um"),
    )
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data())
    widget.load_roi_path(roi_path)
    viewer.dims.current_step = (0, 2, 0, 0)

    box_labels = _managed_box_labels(viewer)
    assert not widget.show_box_labels_checkbox.isChecked()
    assert not box_labels.visible
    assert box_labels.data.shape == (0, 4)
    np.testing.assert_allclose(box_labels.scale, image_layer.scale)
    np.testing.assert_allclose(box_labels.translate, image_layer.translate)
    assert tuple(box_labels.axis_labels) == tuple(image_layer.axis_labels)
    assert tuple(box_labels.units) == tuple(image_layer.units)
    assert not box_labels.editable
    np.testing.assert_allclose(_box_label_rgba(box_labels), [1, 1, 1, 1])

    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )
    widget.box_label_color_btn.click()
    assert widget.box_label_color_btn.text() == "#123456"
    np.testing.assert_allclose(
        _box_label_rgba(box_labels),
        [0x12 / 255, 0x34 / 255, 0x56 / 255, 1],
    )

    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QColorDialog.getColor",
        lambda *args, **kwargs: QColor(),
    )
    widget.box_label_color_btn.click()
    assert widget.box_label_color_btn.text() == "#123456"
    np.testing.assert_allclose(
        _box_label_rgba(box_labels),
        [0x12 / 255, 0x34 / 255, 0x56 / 255, 1],
    )

    widget.show_box_labels_checkbox.setChecked(True)

    assert widget.show_box_labels_checkbox.isChecked()
    assert box_labels.visible
    np.testing.assert_allclose(box_labels.data, [[0, 2, 10, 10]])
    assert list(box_labels.features["neuron_id"]) == [0]
    assert list(box_labels.features["display_text"]) == ["0"]
    assert list(box_labels.text.values) == ["0"]
    assert np.all(np.asarray(box_labels.face_color)[:, 3] == 0)
    assert np.all(np.asarray(box_labels.border_color)[:, 3] == 0)

    widget._set_annotation_rows([(0, "AVA", "note"), (1, "", "")])
    assert list(box_labels.features["display_text"]) == ["AVA"]

    widget.annotation_table.item(0, 1).setText("  AVA  ")
    assert list(box_labels.features["display_text"]) == ["AVA"]
    assert list(box_labels.text.values) == ["AVA"]

    points_before = np.asarray(box_labels.data).copy()
    checked_before = set(widget.checked_ids)
    active_before = widget.active_id
    widget.box_label_mode_combo.setCurrentIndex(
        widget.box_label_mode_combo.findData("digital")
    )
    assert list(box_labels.features["display_text"]) == ["0"]
    widget.box_label_mode_combo.setCurrentIndex(
        widget.box_label_mode_combo.findData("digital_biological")
    )
    assert list(box_labels.features["display_text"]) == ["0 · AVA"]
    widget.box_label_mode_combo.setCurrentIndex(
        widget.box_label_mode_combo.findData("biological")
    )
    assert list(box_labels.features["display_text"]) == ["AVA"]
    np.testing.assert_allclose(box_labels.data, points_before)
    assert widget.checked_ids == checked_before
    assert widget.active_id == active_before

    widget.annotation_table.item(0, 2).setText("ignored note")
    assert list(box_labels.features["display_text"]) == ["AVA"]

    widget.annotation_table.item(0, 1).setText("   ")
    assert list(box_labels.features["display_text"]) == ["0"]

    widget.box_label_mode_combo.setCurrentIndex(
        widget.box_label_mode_combo.findData("digital_biological")
    )
    assert list(box_labels.features["display_text"]) == ["0"]

    widget._selection_items[1].setCheckState(0, Qt.Checked)
    assert set(box_labels.features["neuron_id"]) == {0, 1}
    assert len(box_labels.data) == 2

    viewer.dims.current_step = (1, 3, 0, 0)
    assert list(box_labels.features["neuron_id"]) == [1]
    np.testing.assert_allclose(box_labels.data, [[1, 3, 12, 12]])
    viewer.dims.ndisplay = 3
    np.testing.assert_allclose(box_labels.data, [[1, 3, 12, 12]])

    expected_checked = set(widget.checked_ids)
    expected_active = widget.active_id
    widget.show_box_labels_checkbox.setChecked(False)
    assert not box_labels.visible
    assert box_labels.data.shape == (0, 4)
    assert widget.checked_ids == expected_checked
    assert widget.active_id == expected_active

    widget.box_label_mode_combo.setCurrentIndex(
        widget.box_label_mode_combo.findData("digital")
    )
    assert box_labels.data.shape == (0, 4)
    widget.show_box_labels_checkbox.setChecked(True)
    assert list(box_labels.features["display_text"]) == ["1"]
    widget.show_box_labels_checkbox.setChecked(False)


def test_search_matching_uses_exact_digital_and_casefolded_biological():
    names = {1: "AVA", 10: "AVB", 12: "  ava-left  "}

    assert _match_neuron_ids([1, 10, 12], names, "1") == [1]
    assert _match_neuron_ids([1, 10, 12], names, "001") == [1]
    assert _match_neuron_ids([1, 10, 12], names, "AvA") == [1, 12]
    assert _match_neuron_ids([1, 10, 12], names, "10, ava") == [1, 10, 12]
    assert _match_neuron_ids([1, 10, 12], names, " , ") == []
    assert _match_neuron_ids([1, 10, 12], names, "9" * 5_000) == []


def test_search_marks_all_matches_and_enter_activates_one(
    make_napari_viewer, qtbot, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 24, 24), dtype=np.uint16), name="image")
    widget = LabelManager(viewer)
    roi = np.full((1, 3, 6), np.nan, dtype=np.float32)
    roi[0, 0] = [6, 6, 10, 4, 4, 5]
    roi[0, 1] = [12, 12, 10, 4, 4, 5]
    roi[0, 2] = [18, 18, 10, 4, 4, 5]
    roi_path = tmp_path / "search.npy"
    np.save(roi_path, roi)
    widget.load_roi_path(roi_path)
    viewer.dims.ndisplay = 3
    widget._set_annotation_rows(
        [(0, "AVA", "first"), (1, "AVB", ""), (2, "AVA-L", "")]
    )

    checked_before = set(widget.checked_ids)
    active_before = widget.active_id
    camera_before = tuple(viewer.camera.center)
    widget.neuron_search_input.setText("ava")

    assert widget._search_match_ids == [0, 2]
    assert widget.search_matches_label.text() == "2 matches"
    assert widget.checked_ids == checked_before
    assert widget.active_id == active_before
    assert tuple(viewer.camera.center) == camera_before
    assert widget._selection_items[0].background(1).style() != Qt.NoBrush
    assert widget._selection_items[2].background(1).style() != Qt.NoBrush
    assert widget._selection_items[1].background(1).style() == Qt.NoBrush

    qtbot.keyClick(widget.neuron_search_input, Qt.Key_Return)
    assert widget.active_id == 0
    assert widget.search_matches_label.text() == "1/2 matches"
    qtbot.keyClick(widget.neuron_search_input, Qt.Key_Return)
    assert widget.active_id == 2
    assert widget.checked_ids == {0, 2}
    assert widget.search_matches_label.text() == "2/2 matches"
    active_layer = _managed_layer(viewer, ROLE_ACTIVE)
    assert set(active_layer.features["neuron_id"]) == {2}

    widget.neuron_search_input.setText("av")
    active_before = widget.active_id
    camera_before = tuple(viewer.camera.center)
    qtbot.mouseClick(widget.check_matches_btn, Qt.LeftButton)
    assert widget.checked_ids == {0, 1, 2}
    assert widget.active_id == active_before
    assert tuple(viewer.camera.center) == camera_before

    widget.neuron_search_input.setText("new")
    assert widget._search_match_ids == []
    widget.annotation_table.item(1, 1).setText("NEW")
    assert widget._search_match_ids == [1]
    assert widget.search_matches_label.text() == "1 matches"

    widget.check_none()
    assert widget.neuron_search_input.text() == "new"
    assert widget._search_match_ids == [1]
    widget.neuron_search_input.clear()
    assert widget._search_match_ids == []
    assert widget.search_matches_label.text() == ""
    assert widget._selection_items[1].background(1).style() == Qt.NoBrush

    widget.activate_id(0, locate=False)
    widget.neuron_search_input.setFocus()
    qtbot.keyClicks(widget.neuron_search_input, "qW")
    assert widget.neuron_search_input.text() == "qW"
    assert widget.active_id == 0

    widget.neuron_search_input.setText("ava")
    widget.unload_roi()
    assert widget.neuron_search_input.text() == ""
    assert widget._search_match_ids == []


def test_shift_q_w_navigate_only_checked_ids_and_unbind(
    make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 24, 24), dtype=np.uint16), name="image")
    widget = LabelManager(viewer)
    roi = np.full((1, 4, 6), np.nan, dtype=np.float32)
    for neuron_id in range(4):
        roi[0, neuron_id] = [
            4 + 4 * neuron_id,
            4 + 4 * neuron_id,
            10,
            2,
            2,
            5,
        ]
    roi_path = tmp_path / "checked-navigation.npy"
    np.save(roi_path, roi)
    widget.load_roi_path(roi_path)
    viewer.dims.ndisplay = 3

    widget.checked_ids = {0, 2, 3}
    widget.active_id = 0
    widget._selection_changed(locate=False)
    _press_viewer_key(viewer, "Shift-W")
    assert widget.active_id == 2
    assert widget.checked_ids == {0, 2, 3}
    _press_viewer_key(viewer, "Shift-W")
    assert widget.active_id == 3
    _press_viewer_key(viewer, "Shift-W")
    assert widget.active_id == 0
    _press_viewer_key(viewer, "Shift-Q")
    assert widget.active_id == 3

    widget.active_id = 1
    widget.navigate_checked(1)
    assert widget.active_id == 2
    widget.active_id = 1
    widget.navigate_checked(-1)
    assert widget.active_id == 0
    assert widget.checked_ids == {0, 2, 3}

    widget.active_id = None
    widget.navigate_checked(1)
    assert widget.active_id == 0
    widget.active_id = None
    widget.navigate_checked(-1)
    assert widget.active_id == 3

    widget.checked_ids.clear()
    active_before = widget.active_id
    camera_before = tuple(viewer.camera.center)
    widget.navigate_checked(1)
    assert widget.active_id == active_before
    assert widget.checked_ids == set()
    assert tuple(viewer.camera.center) == camera_before
    assert "No checked neuron" in widget.status_label.text()

    widget.checked_ids = {0}
    widget.active_id = 0
    widget.navigate(1)
    assert widget.active_id == 1
    assert widget.checked_ids == {0, 1}

    widget.shutdown()
    for key in ("Q", "W", "Shift-Q", "Shift-W"):
        assert KeyBinding.from_str(key) not in viewer.keymap


def test_row_click_checkbox_all_and_none_update_vector_layers(
    make_napari_viewer, qtbot, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(
        np.zeros((1, 6, 24, 24), dtype=np.uint16),
        name="image",
    )
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    widget.load_roi_path(roi_path)
    viewer.dims.ndisplay = 3
    widget.show_box_labels_checkbox.setChecked(True)

    selected_layer = _managed_layer(viewer, ROLE_SELECTED)
    active_layer = _managed_layer(viewer, ROLE_ACTIVE)
    box_labels = _managed_box_labels(viewer)
    assert selected_layer.data.shape == (12, 2, 4)
    assert active_layer.data.shape == (12, 2, 4)
    assert list(box_labels.features["neuron_id"]) == [0]

    item_one = widget._selection_items[1]
    widget._on_selection_item_clicked(item_one, 1)
    assert widget.checked_ids == {0, 1}
    assert widget.active_id == 1
    assert item_one.font(1).bold()
    assert selected_layer.data.shape == (24, 2, 4)
    assert set(box_labels.features["neuron_id"]) == {0, 1}

    item_one.setCheckState(0, Qt.Unchecked)
    assert widget.checked_ids == {0}
    assert widget.active_id is None
    assert active_layer.data.shape == (0, 2, 4)
    assert selected_layer.data.shape == (12, 2, 4)
    assert list(box_labels.features["neuron_id"]) == [0]

    qtbot.mouseClick(widget.check_all_btn, Qt.LeftButton)
    assert widget.checked_ids == {0, 1}
    assert widget.active_id == 0
    assert selected_layer.data.shape == (24, 2, 4)
    assert active_layer.data.shape == (12, 2, 4)
    assert set(box_labels.features["neuron_id"]) == {0, 1}

    qtbot.mouseClick(widget.check_none_btn, Qt.LeftButton)
    assert widget.checked_ids == set()
    assert widget.active_id is None
    assert selected_layer.data.shape == (0, 2, 4)
    assert active_layer.data.shape == (0, 2, 4)
    assert box_labels.data.shape == (0, 4)


def test_navigation_skips_missing_roi_at_current_time(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(
        np.zeros((2, 6, 24, 24), dtype=np.uint16),
        name="image",
    )
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data())
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(roi_path), ""),
    )
    widget.load_roi_npy()

    viewer.dims.current_step = (1, 0, 0, 0)
    assert widget.active_id == 0
    assert widget.roi_dataset.get_box(1, 0) is None

    widget.navigate(1)

    assert widget.active_id == 1
    assert widget.checked_ids == {0, 1}

    widget.active_id = 0
    checked_before = set(widget.checked_ids)
    widget.navigate_checked(1)
    assert widget.active_id == 1
    assert widget.checked_ids == checked_before


def test_switching_3d_and_4d_images_recreates_managed_vectors(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    layer_4d = viewer.add_image(
        np.zeros((2, 6, 24, 24), dtype=np.uint16),
        name="image_4d",
    )
    layer_3d = viewer.add_image(
        np.zeros((6, 24, 24), dtype=np.uint16),
        name="image_3d",
    )
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data())
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(roi_path), ""),
    )
    widget.load_roi_npy()
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#abcdef"),
    )
    widget.box_label_color_btn.click()
    widget._set_annotation_rows([(0, "AVA", ""), (1, "", "")])
    widget.box_label_mode_combo.setCurrentIndex(
        widget.box_label_mode_combo.findData("digital_biological")
    )
    widget.show_box_labels_checkbox.setChecked(True)
    viewer.dims.ndisplay = 3

    widget.image_combo.setCurrentText(layer_3d.name)
    managed = [
        layer
        for layer in viewer.layers
        if isinstance(layer, Vectors)
        and layer.metadata.get(ROLE_KEY) in (ROLE_SELECTED, ROLE_ACTIVE)
    ]
    assert len(managed) == 2
    assert all(layer.ndim == 3 for layer in managed)
    assert _managed_box_labels(viewer).ndim == 3
    assert list(
        _managed_box_labels(viewer).features["display_text"]
    ) == ["0 · AVA"]
    np.testing.assert_allclose(
        _box_label_rgba(_managed_box_labels(viewer)),
        [0xAB / 255, 0xCD / 255, 0xEF / 255, 1],
    )

    widget.image_combo.setCurrentText(layer_4d.name)
    managed = [
        layer
        for layer in viewer.layers
        if isinstance(layer, Vectors)
        and layer.metadata.get(ROLE_KEY) in (ROLE_SELECTED, ROLE_ACTIVE)
    ]
    assert len(managed) == 2
    assert all(layer.ndim == 4 for layer in managed)
    assert _managed_box_labels(viewer).ndim == 4
    assert list(
        _managed_box_labels(viewer).features["display_text"]
    ) == ["0 · AVA"]
    np.testing.assert_allclose(
        _box_label_rgba(_managed_box_labels(viewer)),
        [0xAB / 255, 0xCD / 255, 0xEF / 255, 1],
    )


def test_annotation_sync_uses_zero_based_roi_ids(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(
        np.zeros((2, 6, 24, 24), dtype=np.uint16),
        name="image",
    )
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data())
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(roi_path), ""),
    )
    widget.load_roi_npy()

    assert not hasattr(widget, "annotation_start_input")
    assert not hasattr(widget, "annotation_end_input")
    assert not hasattr(widget, "fill_annotation_btn")
    assert widget.annotation_table.item(0, 0).text() == "0"
    assert widget.annotation_table.item(1, 0).text() == "1"
    widget.annotation_table.item(0, 1).setText("AVA")

    widget.activate_id(1)
    selected_rows = widget.annotation_table.selectionModel().selectedRows()
    assert selected_rows[0].row() == 1
    assert "AVA" in widget._selection_items[0].text(1)
    assert widget.checked_ids == {0, 1}

    widget.unload_roi()

    assert widget.annotation_table.rowCount() == 2
    assert widget.annotation_table.item(0, 1).text() == "AVA"
    widget.load_roi_path(roi_path)
    assert widget.annotation_table.item(0, 1).text() == "AVA"


@pytest.mark.skipif(not EXCEL_AVAILABLE, reason="openpyxl is optional")
def test_excel_round_trip_preserves_roi_identity(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    widget = LabelManager(viewer)
    widget._set_annotation_rows([(0, "AVA", "head"), (1, "AVB", "tail")])
    path = tmp_path / "annotations.xlsx"

    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(path), ""),
    )
    widget.save_annotation_to_excel()
    widget._set_annotation_rows([])
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(path), ""),
    )
    widget.load_excel_to_annotation()

    assert widget._annotation_rows() == {
        0: ("AVA", "head"),
        1: ("AVB", "tail"),
    }


def test_shutdown_leaves_labels_untouched_and_removes_managed_roi_layers(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(
        np.zeros((1, 6, 24, 24), dtype=np.uint16), name="image"
    )
    layer = viewer.add_labels(
        np.zeros((1, 6, 24, 24), dtype=np.int32),
        name="labels",
        opacity=0.4,
    )
    original_colormap = layer.colormap
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(roi_path), ""),
    )
    widget.load_roi_npy()
    assert any(isinstance(item, Vectors) for item in viewer.layers)
    assert _managed_box_labels(viewer) in viewer.layers

    widget.unload_roi()
    assert not any(
        item.metadata.get(ROLE_KEY)
        in (ROLE_SELECTED, ROLE_ACTIVE, ROLE_BOX_LABELS)
        for item in viewer.layers
    )

    widget.load_roi_path(roi_path)
    assert _managed_box_labels(viewer) in viewer.layers
    widget.shutdown()

    assert layer.colormap is original_colormap
    assert layer.opacity == pytest.approx(0.4)
    assert not any(
        isinstance(item, Vectors)
        and item.metadata.get(ROLE_KEY) in (ROLE_SELECTED, ROLE_ACTIVE)
        for item in viewer.layers
    )
    assert not any(
        isinstance(item, Points)
        and item.metadata.get(ROLE_KEY) == ROLE_BOX_LABELS
        for item in viewer.layers
    )


def test_proof_transition_cancel_preserves_loaded_roi(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 24, 24), dtype=np.uint16), name="image")
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    widget.load_roi_path(roi_path)
    assert not widget._proof_size_draft_dirty
    widget.proofread_store.set_observation_deleted(0, 0)
    assert widget.proofread_store.dirty
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QMessageBox.warning",
        lambda *args, **kwargs: QMessageBox.Cancel,
    )

    assert widget.unload_roi() is False
    assert widget.roi_dataset is not None
    assert widget.proofread_store.resolve(0, 0) is None
    assert widget.proofread_store.dirty


def test_proof_transition_discard_allows_unload(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 24, 24), dtype=np.uint16), name="image")
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    widget.load_roi_path(roi_path)
    widget.proofread_store.set_observation_deleted(0, 0)
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QMessageBox.warning",
        lambda *args, **kwargs: QMessageBox.Discard,
    )

    assert widget.unload_roi() is True
    assert widget.roi_dataset is None
    assert widget.proofread_store is None


def test_proof_transition_cancel_prevents_image_switch(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    image_a = viewer.add_image(
        np.zeros((6, 24, 24), dtype=np.uint16), name="image-a"
    )
    image_b = viewer.add_image(
        np.zeros((6, 24, 24), dtype=np.uint16), name="image-b"
    )
    widget = LabelManager(viewer)
    widget.image_combo.setCurrentText(image_a.name)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    widget.load_roi_path(roi_path)
    widget.proofread_store.set_observation_deleted(0, 0)
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QMessageBox.warning",
        lambda *args, **kwargs: QMessageBox.Cancel,
    )

    widget.image_combo.setCurrentText(image_b.name)

    assert widget.current_image is image_a
    assert widget.image_combo.currentData() is image_a
    assert widget.proofread_store.dirty


def test_clean_proof_session_rebinds_when_switching_image(
    make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    image_a = viewer.add_image(
        np.zeros((6, 24, 24), dtype=np.uint16), name="image-a"
    )
    image_b = viewer.add_image(
        np.zeros((7, 24, 24), dtype=np.uint16), name="image-b"
    )
    widget = LabelManager(viewer)
    widget.image_combo.setCurrentText(image_a.name)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    widget.load_roi_path(roi_path)
    original_store = widget.proofread_store

    assert not original_store.dirty
    assert not widget._proof_session_has_content()
    assert not widget._image_matches_proof_session(image_b)

    widget.image_combo.setCurrentText(image_b.name)

    assert widget.current_image is image_b
    assert widget.proofread_store is not original_store
    assert widget.proofread_store.dataset is widget.roi_dataset
    assert widget.proofread_store.image_signature == (
        widget._proof_image_signature(image_b)
    )
    assert widget._proof_view_allowed()
    assert widget.proofreading_toggle.isEnabled()


def test_external_image_removal_retains_dirty_detached_session(
    make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    image = viewer.add_image(
        np.zeros((6, 24, 24), dtype=np.uint16), name="image"
    )
    viewer.add_image(
        np.zeros((6, 24, 24), dtype=np.uint16), name="fallback"
    )
    widget = LabelManager(viewer)
    widget.image_combo.setCurrentText(image.name)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    widget.load_roi_path(roi_path)
    widget.proofread_store.set_observation_deleted(0, 0)

    viewer.layers.remove(image)

    assert widget.current_image is None
    assert widget._proof_detached
    assert widget.proofread_store.dirty
    assert widget.proofread_store.resolve(0, 0) is None


def test_shutdown_cancel_preserves_widget_session(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 24, 24), dtype=np.uint16), name="image")
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    widget.load_roi_path(roi_path)
    widget.proofread_store.set_observation_deleted(0, 0)
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QMessageBox.warning",
        lambda *args, **kwargs: QMessageBox.Cancel,
    )

    assert widget.shutdown() is False
    assert not widget._closed
    assert widget.current_image is not None
    assert widget.proofread_store.dirty
    assert _managed_layer(viewer, ROLE_SELECTED) in viewer.layers

    assert widget.shutdown(force=True) is True


def test_proof_transition_save_failure_keeps_session(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 24, 24), dtype=np.uint16), name="image")
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    widget.load_roi_path(roi_path)
    store = widget.proofread_store
    store.set_observation_deleted(0, 0)
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QMessageBox.warning",
        lambda *args, **kwargs: QMessageBox.Save,
    )
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "proof.json"), ""),
    )

    def fail_save(path):
        raise PermissionError(path)

    monkeypatch.setattr(store, "save", fail_save)

    assert widget.unload_roi() is False
    assert widget.proofread_store is store
    assert store.dirty
