import numpy as np
import pytest
from napari.layers import Vectors
from napari.utils import colormaps as cmap
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QTreeWidget

from napari_label_manager._widget import (
    EXCEL_AVAILABLE,
    ROLE_ACTIVE,
    ROLE_KEY,
    ROLE_SELECTED,
    LabelManager,
)


def _rgba(layer, label_value):
    return np.asarray(layer.get_color(label_value), dtype=float)


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
        if isinstance(layer, Vectors)
        and layer.metadata.get(ROLE_KEY) == role
    )


def test_widget_initializes_checkable_selection(make_napari_viewer):
    viewer = make_napari_viewer()
    labels = np.asarray([[0, 1], [2, 0]], dtype=np.int32)
    layer = viewer.add_labels(labels, name="labels")

    widget = LabelManager(viewer)

    assert isinstance(widget.layer_combo, QComboBox)
    assert isinstance(widget.selection_tree, QTreeWidget)
    assert not widget.selection_tree.alternatingRowColors()
    assert widget.navigation_help_label.text() == "Q/W: last/next"
    assert (
        widget.selected_opacity_slider.parentWidget()
        is widget.labels_layer_group
    )
    assert widget.current_layer is layer
    assert widget._available_ids == [1, 2]
    assert set(widget._selection_items) == {1, 2}
    assert widget.active_id == 1
    assert widget.checked_ids == {1}
    assert widget._selection_items[1].checkState(0) == Qt.Checked
    assert widget._selection_items[1].font(1).bold()
    assert not hasattr(widget, "reset_btn")
    assert not hasattr(widget, "labels_visible_checkbox")


def test_layer_selector_tracks_insertions_and_removals(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = LabelManager(viewer)
    assert not widget.layer_combo.isEnabled()

    first = viewer.add_labels(np.zeros((3, 3), dtype=int), name="first")
    second = viewer.add_labels(np.ones((3, 3), dtype=int), name="second")

    assert widget.layer_combo.count() == 2
    widget.layer_combo.setCurrentText("second")
    assert widget.current_layer is second

    viewer.layers.remove(second)
    assert widget.current_layer is first
    assert widget.layer_combo.currentText() == "first"


def test_selection_changes_alpha_without_changing_rgb(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    layer = viewer.add_labels(
        np.asarray([[0, 1], [2, 0]], dtype=np.int32),
        name="labels",
        opacity=0.6,
    )
    original_colormap = layer.colormap
    rgb_before = {value: _rgba(layer, value)[:3] for value in (1, 2)}

    widget = LabelManager(viewer)

    np.testing.assert_allclose(_rgba(layer, 1)[:3], rgb_before[1])
    np.testing.assert_allclose(_rgba(layer, 2)[:3], rgb_before[2])
    assert _rgba(layer, 1)[3] == pytest.approx(1.0)
    assert _rgba(layer, 2)[3] == pytest.approx(0.2)

    qtbot.mouseClick(
        widget.selection_tree.viewport(),
        Qt.LeftButton,
        pos=widget.selection_tree.visualItemRect(
            widget._selection_items[2]
        ).center(),
    )
    assert widget.checked_ids == {1, 2}
    assert widget.active_id == 2
    assert _rgba(layer, 2)[3] == pytest.approx(1.0)

    widget.shutdown()
    assert layer.colormap is original_colormap
    assert layer.opacity == pytest.approx(0.6)


def test_direct_colormap_rgb_is_preserved_and_navigation_wraps(
    make_napari_viewer
):
    viewer = make_napari_viewer()
    direct = cmap.direct_colormap(
        {
            None: (0.0, 0.0, 0.0, 0.0),
            0: (0.0, 0.0, 0.0, 0.0),
            7: (0.1, 0.3, 0.8, 1.0),
            9: (0.8, 0.2, 0.1, 1.0),
        }
    )
    direct.background_value = 0
    layer = viewer.add_labels(
        np.asarray([[0, 7, 9]], dtype=np.int32),
        name="direct",
        colormap=direct,
    )
    rgb_before = {value: _rgba(layer, value)[:3] for value in (7, 9)}

    widget = LabelManager(viewer)

    assert widget.active_id == 7
    widget.navigate(-1)
    assert widget.active_id == 9
    assert widget.checked_ids == {7, 9}
    widget.navigate(1)
    assert widget.active_id == 7
    assert widget.checked_ids == {7, 9}
    for value in (7, 9):
        np.testing.assert_allclose(_rgba(layer, value)[:3], rgb_before[value])


def test_label_data_change_invalidates_button_ids(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_labels(
        np.asarray([[0, 1], [0, 0]], dtype=np.int32), name="labels"
    )
    rgb_two = _rgba(layer, 2)[:3]
    widget = LabelManager(viewer)
    assert widget._available_ids == [1]

    layer.data = np.asarray([[0, 1], [2, 0]], dtype=np.int32)

    assert widget._available_ids == [1, 2]
    assert set(widget._selection_items) == {1, 2}
    np.testing.assert_allclose(_rgba(layer, 2)[:3], rgb_two)
    assert _rgba(layer, 2)[3] == pytest.approx(0.2)


def test_roi_load_preserves_covered_ids_and_renders_2d_and_3d(
    make_napari_viewer, qtbot, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    labels = np.zeros((2, 6, 24, 24), dtype=np.int32)
    labels[0, 1:4, 8:12, 8:12] = 2
    labels_layer = viewer.add_labels(
        labels,
        name="covered_labels",
        scale=(1, 5, 1, 1),
    )
    expected_rgb = {
        neuron_id: _rgba(labels_layer, neuron_id + 1)[:3]
        for neuron_id in (0, 1)
    }
    widget = LabelManager(viewer)

    roi_path = tmp_path / "neuron_pt_tuple.npy"
    np.save(roi_path, _roi_data())
    monkeypatch.setattr(
        "napari_label_manager._widget.QFileDialog.getOpenFileName",
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


def test_row_click_checkbox_all_and_none_update_vector_layers(
    make_napari_viewer, qtbot, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_labels(
        np.zeros((1, 6, 24, 24), dtype=np.int32),
        name="labels",
    )
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data()[:1])
    widget.load_roi_path(roi_path)
    viewer.dims.ndisplay = 3

    selected_layer = _managed_layer(viewer, ROLE_SELECTED)
    active_layer = _managed_layer(viewer, ROLE_ACTIVE)
    assert selected_layer.data.shape == (12, 2, 4)
    assert active_layer.data.shape == (12, 2, 4)

    item_one = widget._selection_items[1]
    widget._on_selection_item_clicked(item_one, 1)
    assert widget.checked_ids == {0, 1}
    assert widget.active_id == 1
    assert item_one.font(1).bold()
    assert selected_layer.data.shape == (24, 2, 4)

    item_one.setCheckState(0, Qt.Unchecked)
    assert widget.checked_ids == {0}
    assert widget.active_id is None
    assert active_layer.data.shape == (0, 2, 4)
    assert selected_layer.data.shape == (12, 2, 4)

    qtbot.mouseClick(widget.check_all_btn, Qt.LeftButton)
    assert widget.checked_ids == {0, 1}
    assert widget.active_id == 0
    assert selected_layer.data.shape == (24, 2, 4)
    assert active_layer.data.shape == (12, 2, 4)

    qtbot.mouseClick(widget.check_none_btn, Qt.LeftButton)
    assert widget.checked_ids == set()
    assert widget.active_id is None
    assert selected_layer.data.shape == (0, 2, 4)
    assert active_layer.data.shape == (0, 2, 4)


def test_navigation_skips_missing_roi_at_current_time(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_labels(
        np.zeros((2, 6, 24, 24), dtype=np.int32),
        name="labels",
    )
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data())
    monkeypatch.setattr(
        "napari_label_manager._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(roi_path), ""),
    )
    widget.load_roi_npy()

    viewer.dims.current_step = (1, 0, 0, 0)
    assert widget.active_id == 0
    assert widget.roi_dataset.get_box(1, 0) is None

    widget.navigate(1)

    assert widget.active_id == 1
    assert widget.checked_ids == {0, 1}


def test_switching_3d_and_4d_labels_recreates_managed_vectors(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    layer_4d = viewer.add_labels(
        np.zeros((2, 6, 24, 24), dtype=np.int32),
        name="labels_4d",
    )
    layer_3d = viewer.add_labels(
        np.zeros((6, 24, 24), dtype=np.int32),
        name="labels_3d",
    )
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data())
    monkeypatch.setattr(
        "napari_label_manager._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(roi_path), ""),
    )
    widget.load_roi_npy()

    widget.layer_combo.setCurrentText(layer_3d.name)
    managed = [
        layer
        for layer in viewer.layers
        if isinstance(layer, Vectors)
        and layer.metadata.get(ROLE_KEY) in (ROLE_SELECTED, ROLE_ACTIVE)
    ]
    assert len(managed) == 2
    assert all(layer.ndim == 3 for layer in managed)

    widget.layer_combo.setCurrentText(layer_4d.name)
    managed = [
        layer
        for layer in viewer.layers
        if isinstance(layer, Vectors)
        and layer.metadata.get(ROLE_KEY) in (ROLE_SELECTED, ROLE_ACTIVE)
    ]
    assert len(managed) == 2
    assert all(layer.ndim == 4 for layer in managed)


def test_annotation_sync_uses_zero_based_roi_ids(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_labels(
        np.zeros((2, 6, 24, 24), dtype=np.int32),
        name="labels",
    )
    widget = LabelManager(viewer)
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, _roi_data())
    monkeypatch.setattr(
        "napari_label_manager._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(roi_path), ""),
    )
    widget.load_roi_npy()
    widget.load_current_ids_to_annotation()

    assert widget.annotation_table.item(0, 0).text() == "0"
    assert widget.annotation_table.item(1, 0).text() == "1"
    widget.annotation_table.item(0, 1).setText("AVA")

    widget.activate_id(1)
    selected_rows = widget.annotation_table.selectionModel().selectedRows()
    assert selected_rows[0].row() == 1
    assert "AVA" in widget._selection_items[0].text(1)
    assert widget.checked_ids == {0, 1}


@pytest.mark.skipif(not EXCEL_AVAILABLE, reason="openpyxl is optional")
def test_excel_round_trip_preserves_roi_identity(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((2, 2), dtype=np.int32), name="labels")
    widget = LabelManager(viewer)
    widget._set_annotation_rows([(0, "AVA", "head"), (1, "AVB", "tail")])
    path = tmp_path / "annotations.xlsx"

    monkeypatch.setattr(
        "napari_label_manager._widget.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(path), ""),
    )
    widget.save_annotation_to_excel()
    widget._set_annotation_rows([])
    monkeypatch.setattr(
        "napari_label_manager._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(path), ""),
    )
    widget.load_excel_to_annotation()

    assert widget._annotation_rows() == {
        0: ("AVA", "head"),
        1: ("AVB", "tail"),
    }


def test_shutdown_restores_layer_and_removes_managed_vectors(
    make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
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
        "napari_label_manager._widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(roi_path), ""),
    )
    widget.load_roi_npy()
    assert any(isinstance(item, Vectors) for item in viewer.layers)

    widget.shutdown()

    assert layer.colormap is original_colormap
    assert layer.opacity == pytest.approx(0.4)
    assert not any(
        isinstance(item, Vectors)
        and item.metadata.get(ROLE_KEY) in (ROLE_SELECTED, ROLE_ACTIVE)
        for item in viewer.layers
    )
