"""Qt/napari regression tests for the 2D proofreading workflow."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from napari.layers import Vectors
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication, QLineEdit, QMessageBox

from napari_worm_neuron_annotator._proofread import (
    DELETED,
    PRESENT,
    ObservationPatch,
)
from napari_worm_neuron_annotator._widget import (
    PROOF_TARGET_EDGE_WIDTH,
    PROOF_TARGET_HALF_LENGTH,
    ROLE_ACTIVE,
    ROLE_KEY,
    ROLE_PROOF_TARGET,
    ROLE_SELECTED,
    NeuronAnnotatorWidget,
)


@pytest.fixture
def proof_widgets(qtbot, make_napari_viewer):
    """Force-close widgets so dirty-state dialogs never leak into teardown."""
    # Depending on both owning GUI fixtures guarantees this finalizer runs
    # before either pytest-qt or napari starts closing widgets/viewers.
    del qtbot, make_napari_viewer
    widgets: list[NeuronAnnotatorWidget] = []
    yield widgets
    for widget in widgets:
        widget.shutdown(force=True)


def _roi_data() -> np.ndarray:
    """Return six raw volumes addressed by Image times 0, 1, and 2."""
    data = np.full((6, 2, 6), np.nan, dtype=np.float32)
    # Widget defaults to z_divisor=5, so these boxes are centered on z=3.
    data[1, 0, :6] = (10, 10, 15, 7, 7, 15)
    data[5, 0, :6] = (12, 12, 15, 7, 7, 15)
    data[3, 1, :6] = (20, 20, 15, 5, 5, 15)
    return data


def _make_widget(
    make_napari_viewer,
    qtbot,
    tmp_path,
    proof_widgets,
    *,
    image_ndim: int = 4,
    image_kwargs: dict[str, object] | None = None,
) -> tuple[object, NeuronAnnotatorWidget]:
    viewer = make_napari_viewer()
    image_shape = (3, 8, 32, 32) if image_ndim == 4 else (8, 32, 32)
    viewer.add_image(
        np.zeros(image_shape, dtype=np.uint16),
        name="image",
        **(image_kwargs or {}),
    )
    # napari centers newly created dimensions.  Pin the intended first Image
    # time before ROI load so initial selection is based on volume_start.
    if image_ndim == 4:
        viewer.dims.current_step = (0, 3, 0, 0)
    else:
        viewer.dims.current_step = (3, 0, 0)
    widget = NeuronAnnotatorWidget(viewer)
    proof_widgets.append(widget)
    qtbot.addWidget(
        widget,
        before_close_func=lambda item: item.shutdown(force=True),
    )
    widget.show()
    widget.volume_start_spin.setValue(1)
    widget.volume_stride_spin.setValue(2)
    path = tmp_path / f"proof-{image_ndim}d.npy"
    np.save(path, _roi_data())
    widget.load_roi_path(path)
    QApplication.processEvents()
    return viewer, widget


def _managed_vectors(viewer, role: str) -> Vectors:
    return next(
        layer
        for layer in viewer.layers
        if isinstance(layer, Vectors) and layer.metadata.get(ROLE_KEY) == role
    )


def _set_cursor(viewer, *position: float) -> None:
    viewer.cursor.position = tuple(float(value) for value in position)
    viewer.cursor.viewbox = (0, 0)


def _proof_target_layer(viewer) -> Vectors | None:
    matches = [
        layer
        for layer in viewer.layers
        if isinstance(layer, Vectors)
        and layer.metadata.get(ROLE_KEY) == ROLE_PROOF_TARGET
    ]
    assert len(matches) <= 1
    return matches[0] if matches else None


def _assert_crosshair_geometry(
    layer: Vectors, center: tuple[float, ...]
) -> None:
    """Assert two orthogonal line segments centered at ``center``."""
    assert layer.vector_style == "line"
    assert layer.edge_width == pytest.approx(PROOF_TARGET_EDGE_WIDTH)
    assert layer.data.shape == (2, 2, len(center))
    expected = np.zeros_like(layer.data)
    expected[0, 0] = center
    expected[0, 0, -1] -= PROOF_TARGET_HALF_LENGTH
    expected[0, 1] = (0.0,) * len(center)
    expected[0, 1, -1] = 2.0 * PROOF_TARGET_HALF_LENGTH
    expected[1, 0] = center
    expected[1, 0, -2] -= PROOF_TARGET_HALF_LENGTH
    expected[1, 1] = (0.0,) * len(center)
    expected[1, 1, -2] = 2.0 * PROOF_TARGET_HALF_LENGTH
    np.testing.assert_allclose(layer.data, expected)


def _assert_no_proof_target(
    viewer, widget: NeuronAnnotatorWidget
) -> None:
    assert widget._proof_target_zyx is None
    layer = _proof_target_layer(viewer)
    assert layer is None or len(layer.data) == 0 or not layer.visible


def _mouse_event(
    event_type: str,
    world_position: tuple[float, ...],
    *,
    canvas_position: tuple[float, float],
    button: int = 1,
    modifiers: tuple[str, ...] = (),
    is_dragging: bool = False,
):
    return read_only_mouse_event(
        type=event_type,
        pos=canvas_position,
        position=world_position,
        button=button,
        modifiers=modifiers,
        is_dragging=is_dragging,
    )


def _click_world(
    viewer,
    *world_position: float,
    button: int = 1,
    modifiers: tuple[str, ...] = (),
    canvas_position: tuple[float, float] = (10.0, 10.0),
) -> None:
    world = tuple(float(value) for value in world_position)
    mouse_press_callbacks(
        viewer,
        _mouse_event(
            "mouse_press",
            world,
            canvas_position=canvas_position,
            button=button,
            modifiers=modifiers,
        ),
    )
    mouse_release_callbacks(
        viewer,
        _mouse_event(
            "mouse_release",
            world,
            canvas_position=canvas_position,
            button=button,
            modifiers=modifiers,
        ),
    )


def _drag_world(
    viewer,
    *world_position: float,
    start: tuple[float, float] = (10.0, 10.0),
    stop: tuple[float, float] = (20.0, 20.0),
) -> None:
    world = tuple(float(value) for value in world_position)
    mouse_press_callbacks(
        viewer,
        _mouse_event(
            "mouse_press", world, canvas_position=start, button=1
        ),
    )
    mouse_move_callbacks(
        viewer,
        _mouse_event(
            "mouse_move",
            world,
            canvas_position=stop,
            button=1,
            is_dragging=True,
        ),
    )
    mouse_release_callbacks(
        viewer,
        _mouse_event(
            "mouse_release", world, canvas_position=stop, button=1
        ),
    )


def _enable_proofreading(widget: NeuronAnnotatorWidget) -> None:
    widget.viewer.layers.selection.active = widget.current_image
    widget.proofreading_toggle.setChecked(True)
    QApplication.processEvents()
    assert widget.proofreading_enabled
    assert widget._proof_key_filter is not None


def _press_proof_key(qtbot, widget: NeuronAnnotatorWidget, key: Qt.Key) -> None:
    canvas = widget._get_proof_canvas_native()
    assert canvas is not None
    canvas.setFocus()
    QApplication.processEvents()
    event = QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier)
    assert widget._proof_key_filter.eventFilter(canvas, event)
    QApplication.processEvents()


def _annotation_ids(widget: NeuronAnnotatorWidget) -> list[int]:
    return [
        int(widget.annotation_table.item(row, 0).text())
        for row in range(widget.annotation_table.rowCount())
    ]


def test_proofreading_defaults_off_and_handlers_are_inert(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    _, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store

    assert widget.viewer.layers.selection.active is widget.current_image
    assert widget.proofreading_toggle.isEnabled()
    assert not widget.proofreading_toggle.isChecked()
    assert not widget.proofreading_enabled
    assert widget._proof_key_filter is None
    assert not widget.proof_width_spin.isEnabled()
    # Save is only actionable when there are applied edits or a draft.
    assert not widget.proof_save_btn.isEnabled()

    widget._proof_delete_current()

    assert store.resolve(1, 0) is not None
    assert store.observation_patches == {}


def test_loaded_roi_preserves_source_image_active_for_proofreading(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )

    # The overlay layers are runtime artifacts and must not steal selection
    # from the Image that supplies the spatial proofreading context.
    assert viewer.layers.selection.active is widget.current_image
    widget.proofreading_toggle.setChecked(True)
    QApplication.processEvents()

    assert widget.proofreading_enabled
    assert widget.proofreading_toggle.isChecked()
    assert widget._proof_key_filter is not None


def test_current_box_status_reports_identity_time_and_modified_observation(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )

    assert widget.proof_current_box_label.text() == (
        "Neuron 0: center (x=10.00, y=10.00, z=3.00), t=0"
    )

    _enable_proofreading(widget)
    widget.proof_width_spin.setValue(9.0)
    widget._proof_apply_size()
    assert widget.proof_current_box_label.text() == (
        "Neuron 0 (resized): center (x=10.00, y=10.00, z=3.00), t=0"
    )

    # A deleted observation has no center to show, but it is still a modified
    # observation and should retain that marker in the compact status line.
    widget._proof_delete_current()
    assert widget.proof_current_box_label.text() == (
        "Neuron 0 (deleted): no box, t=0"
    )

    # Image t=1 maps to raw volume 3 in this fixture, where neuron 0 is
    # absent; the status must retain the active identity and Image time.
    viewer.dims.set_current_step(0, 1)
    QApplication.processEvents()
    assert widget.proof_current_box_label.text() == "Neuron 0: no box, t=1"


def test_current_box_status_refreshes_for_active_neuron_and_image_time(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )

    # At Image t=1 the fixture maps to raw volume 3, where neuron 1 has a
    # valid observation.  Activating it should refresh both identity and t.
    viewer.dims.set_current_step(0, 1)
    QApplication.processEvents()
    widget.activate_id(1, locate=False)
    assert widget.proof_current_box_label.text() == (
        "Neuron 1: center (x=20.00, y=20.00, z=3.00), t=1"
    )


def test_proof_key_filter_does_not_use_deprecated_qt_viewer_accessor(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    _, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _enable_proofreading(widget)
    assert not any(
        isinstance(item.message, FutureWarning)
        and "qt_viewer" in str(item.message)
        for item in caught
    )


def test_proofreading_only_enables_in_2d_and_3d_turns_it_off(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )

    _enable_proofreading(widget)
    viewer.dims.ndisplay = 3
    QApplication.processEvents()

    assert not widget.proofreading_enabled
    assert not widget.proofreading_toggle.isChecked()
    assert widget._proof_key_filter is None
    assert not widget.proofreading_toggle.isEnabled()

    viewer.dims.ndisplay = 2
    QApplication.processEvents()
    assert widget.proofreading_toggle.isEnabled()


def test_current_volume_index_maps_4d_time_and_3d_to_fixed_volume(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )

    assert widget._current_volume_index() == 1
    viewer.dims.set_current_step(0, 1)
    assert widget._current_volume_index() == 3
    viewer.dims.set_current_step(0, 2)
    assert widget._current_volume_index() == 5

    _, widget_3d = _make_widget(
        make_napari_viewer,
        qtbot,
        tmp_path,
        proof_widgets,
        image_ndim=3,
    )
    assert widget_3d._current_volume_index() == 1


def test_f7_delete_and_f8_present_guard_restore_store_backed_vectors(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store
    selected = _managed_vectors(viewer, ROLE_SELECTED)
    active = _managed_vectors(viewer, ROLE_ACTIVE)
    _enable_proofreading(widget)
    _click_world(viewer, 0, 3, 17, 18)

    assert selected.data.shape == (4, 2, 4)
    assert active.data.shape == (4, 2, 4)
    assert widget._proof_target_zyx == pytest.approx((3, 17, 18))

    # F8 must neither move an existing observation nor consume the target.
    _press_proof_key(qtbot, widget, Qt.Key_F8)
    assert (1, 0) not in store.observation_patches
    assert store.resolve(1, 0).center_zyx == pytest.approx((3, 10, 10))
    assert widget._proof_target_zyx == pytest.approx((3, 17, 18))

    # A moving napari cursor must not alter the click-locked target.
    _set_cursor(viewer, 0, 3, 25, 26)
    assert widget._proof_target_zyx == pytest.approx((3, 17, 18))

    _press_proof_key(qtbot, widget, Qt.Key_F7)
    assert store.observation_patches[(1, 0)].state == DELETED
    assert store.resolve(1, 0) is None
    assert selected.data.shape == (0, 2, 4)
    assert active.data.shape == (0, 2, 4)

    _press_proof_key(qtbot, widget, Qt.Key_F8)
    restored = store.resolve(1, 0)
    assert store.observation_patches[(1, 0)].state == PRESENT
    assert restored.center_zyx == pytest.approx((3, 17, 18))
    assert restored.size_zyx == pytest.approx((3, 7, 7))
    assert selected.data.shape == (4, 2, 4)
    assert active.data.shape == (4, 2, 4)
    _assert_no_proof_target(viewer, widget)
    marker = _proof_target_layer(viewer)
    assert marker is not None
    assert not marker.editable


def test_f9_adds_default_sized_neuron_to_tree_annotation_and_vectors_then_f12_exits(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store
    selected = _managed_vectors(viewer, ROLE_SELECTED)
    active = _managed_vectors(viewer, ROLE_ACTIVE)
    _enable_proofreading(widget)
    _click_world(viewer, 0, 3, 22, 23)

    _press_proof_key(qtbot, widget, Qt.Key_F9)

    added_id = 2
    added = store.resolve(1, added_id)
    assert added.center_zyx == pytest.approx((3, 22, 23))
    assert added.size_zyx == pytest.approx((3, 7, 7))
    assert widget.active_id == added_id
    assert widget.checked_ids == {0, added_id}
    assert widget._available_ids == [0, 1, added_id]
    assert added_id in widget._selection_items
    assert widget.selection_tree.topLevelItemCount() == 3
    assert _annotation_ids(widget) == [0, 1, added_id]
    assert selected.data.shape == (8, 2, 4)
    assert active.data.shape == (4, 2, 4)
    _assert_no_proof_target(viewer, widget)

    _press_proof_key(qtbot, widget, Qt.Key_F12)

    assert not widget.proofreading_enabled
    assert not widget.proofreading_toggle.isChecked()
    assert widget._proof_key_filter is None
    assert store.resolve(1, added_id) is not None


def test_delete_all_confirmation_normalizes_patches_and_f8_restores_one_volume(
    make_napari_viewer, qtbot, tmp_path, monkeypatch, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store
    selected = _managed_vectors(viewer, ROLE_SELECTED)
    active = _managed_vectors(viewer, ROLE_ACTIVE)
    store.set_observation_present(
        1, 0, center_zyx=(3, 14, 14), size_zyx=(3, 7, 7)
    )
    store.set_observation_present(
        3, 0, center_zyx=(3, 15, 15), size_zyx=(3, 7, 7)
    )
    _enable_proofreading(widget)
    _click_world(viewer, 0, 3, 18, 19)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )

    widget._proof_delete_all()

    assert store.delete_all_ids == {0}
    assert not any(neuron_id == 0 for _, neuron_id in store.observation_patches)
    assert all(store.resolve(volume, 0) is None for volume in range(6))
    assert selected.data.shape == (0, 2, 4)
    assert active.data.shape == (0, 2, 4)

    _press_proof_key(qtbot, widget, Qt.Key_F8)

    assert store.delete_all_ids == {0}
    assert set(store.observation_patches) == {(1, 0)}
    assert store.resolve(1, 0).center_zyx == pytest.approx((3, 18, 19))
    assert selected.data.shape == (4, 2, 4)
    assert active.data.shape == (4, 2, 4)

    viewer.dims.set_current_step(0, 2)
    QApplication.processEvents()
    assert widget._current_volume_index() == 5
    assert store.resolve(5, 0) is None
    assert selected.data.shape == (0, 2, 4)
    assert active.data.shape == (0, 2, 4)


def test_proof_key_filter_ignores_spinbox_and_table_cell_editors(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    _, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store
    _enable_proofreading(widget)
    event_filter = widget._proof_key_filter
    canvas = widget._get_proof_canvas_native()
    assert canvas is not None

    # Explicitly activate the standalone test window before requesting focus.
    # On Windows/Qt, ``setFocus`` alone can be ignored while another napari
    # test window remains active, making this assertion timing-dependent.
    widget.activateWindow()
    widget.raise_()
    qtbot.mouseClick(widget.proof_width_spin, Qt.LeftButton)
    QApplication.processEvents()
    assert widget._proof_key_focus_blocked()
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_F7, Qt.NoModifier)
    assert not event_filter.eventFilter(canvas, event)
    assert store.resolve(1, 0) is not None

    table_item = widget.annotation_table.item(0, 1)
    widget.annotation_table.setCurrentItem(table_item)
    widget.annotation_table.editItem(table_item)
    QApplication.processEvents()
    editor = QApplication.focusWidget()
    assert isinstance(editor, QLineEdit)
    assert widget._proof_key_focus_blocked()
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_F7, Qt.NoModifier)
    assert not event_filter.eventFilter(canvas, event)
    assert store.resolve(1, 0) is not None


def test_f8_requires_click_target_and_preserves_float_restore_size(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store
    _enable_proofreading(widget)
    widget._proof_delete_current()
    store.observation_patches[(1, 0)] = ObservationPatch.deleted(
        (2.5, 6.25, 8.75)
    )
    widget._proof_size_draft_dirty = False
    widget._update_proof_size_controls()

    assert widget.proof_depth_spin.value() == pytest.approx(2.5)
    assert widget.proof_height_spin.value() == pytest.approx(6.25)
    assert widget.proof_width_spin.value() == pytest.approx(8.75)

    # The live napari cursor is deliberately not a placement fallback.
    _set_cursor(viewer, 0, 3, 17, 18)
    widget._proof_place_cursor()
    assert store.resolve(1, 0) is None

    _click_world(viewer, 0, 3, 17, 18)
    widget._proof_place_cursor()
    assert store.resolve(1, 0).size_zyx == pytest.approx(
        (2.5, 6.25, 8.75)
    )
    _assert_no_proof_target(viewer, widget)


def test_retire_added_neuron_reserves_id_and_refreshes_ui(
    make_napari_viewer,
    qtbot,
    tmp_path,
    monkeypatch,
    proof_widgets,
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store
    _enable_proofreading(widget)
    _click_world(viewer, 0, 3, 22, 23)
    widget._proof_add_neuron()
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes
    )

    widget._proof_retire_added_neuron()

    assert 2 in store.retired_ids
    assert 2 not in widget._available_ids
    assert 2 not in _annotation_ids(widget)
    assert widget.active_id is None
    assert 2 not in widget.checked_ids
    _click_world(viewer, 0, 3, 24, 25)
    widget._proof_add_neuron()
    assert widget.active_id == 3


def test_save_refreshes_empty_provisional_identity_and_info(
    make_napari_viewer,
    qtbot,
    tmp_path,
    monkeypatch,
    proof_widgets,
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store
    _enable_proofreading(widget)
    _click_world(viewer, 0, 3, 22, 23)
    widget._proof_add_neuron()
    widget._proof_delete_current()
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._widget.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "proof.json"), ""),
    )

    widget.save_proof_edits()

    assert 2 in store.retired_ids
    assert 2 not in widget._available_ids
    assert not store.dirty
    assert "Proof: On" in widget.info_text.toPlainText()
    assert "dirty=no" in widget.info_text.toPlainText()


def test_image_signature_change_pauses_proof_and_hides_overlays(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    image = widget.current_image
    _enable_proofreading(widget)

    image.scale = tuple(float(value) * 2 for value in image.scale)
    QApplication.processEvents()

    assert widget._proof_detached
    assert not widget.proofreading_enabled
    assert not widget.proofreading_toggle.isEnabled()
    assert not any(
        isinstance(layer, Vectors)
        and layer.metadata.get(ROLE_KEY) in {ROLE_SELECTED, ROLE_ACTIVE}
        for layer in viewer.layers
    )
    assert "Proof: Off (paused)" in widget.info_text.toPlainText()


def test_apply_uses_original_size_draft_owner(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store
    _enable_proofreading(widget)
    widget.proof_width_spin.setValue(13.5)
    assert widget._proof_size_draft_target == (1, 0)

    # Simulate a non-vetoable external selection transition while retaining
    # the draft. Apply must still target the neuron that owns the draft.
    widget.active_id = 1
    widget._proof_apply_size()

    assert store.resolve(1, 0).size_zyx == pytest.approx((3, 7, 13.5))
    assert store.resolve(3, 1).size_zyx == pytest.approx((3, 5, 5))


def test_f9_does_not_bypass_cancelled_size_draft(
    make_napari_viewer,
    qtbot,
    tmp_path,
    monkeypatch,
    proof_widgets,
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    _enable_proofreading(widget)
    _click_world(viewer, 0, 3, 22, 23)
    widget.proof_width_spin.setValue(13.5)

    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: None)

    widget._proof_add_neuron()

    assert widget.proofread_store.provisional_added_ids == set()
    assert widget.active_id == 0
    assert widget._proof_size_draft_dirty
    assert widget._proof_target_zyx == pytest.approx((3, 22, 23))


def test_click_locks_world_to_data_target_and_stable_4d_crosshair(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    scale = (2.0, 3.0, 4.0, 5.0)
    translate = (11.0, 13.0, 17.0, 19.0)
    viewer, widget = _make_widget(
        make_napari_viewer,
        qtbot,
        tmp_path,
        proof_widgets,
        image_kwargs={"scale": scale, "translate": translate},
    )
    _enable_proofreading(widget)
    active_before = widget.active_id
    checked_before = set(widget.checked_ids)
    assert not widget.proofread_store.dirty

    # Data coordinate (t, z, y, x) = (0, 3, 17, 18).
    world = tuple(
        translate[index] + scale[index] * value
        for index, value in enumerate((0.0, 3.0, 17.0, 18.0))
    )
    _click_world(viewer, *world)

    assert widget._proof_target_zyx == pytest.approx((3, 17, 18))
    assert widget.active_id == active_before
    assert widget.checked_ids == checked_before
    assert not widget.proofread_store.dirty
    marker = _proof_target_layer(viewer)
    assert marker is not None
    assert not marker.editable
    assert marker.visible
    _assert_crosshair_geometry(marker, (0, 3, 17, 18))
    np.testing.assert_allclose(marker.scale, scale)
    np.testing.assert_allclose(marker.translate, translate)

    # Later cursor motion cannot move a locked target or its marker.
    _set_cursor(viewer, *(value + 100 for value in world))
    assert widget._proof_target_zyx == pytest.approx((3, 17, 18))
    _assert_crosshair_geometry(marker, (0, 3, 17, 18))


@pytest.mark.parametrize("rotation", (90, 270))
def test_rotation_preserves_proof_target_and_placed_data_coordinates(
    make_napari_viewer, qtbot, tmp_path, proof_widgets, rotation
):
    scale = (2.0, 3.0, 4.0, 5.0)
    translate = (11.0, 13.0, 17.0, 19.0)
    viewer, widget = _make_widget(
        make_napari_viewer,
        qtbot,
        tmp_path,
        proof_widgets,
        image_kwargs={"scale": scale, "translate": translate},
    )
    image = widget.current_image
    assert image is not None
    _enable_proofreading(widget)

    first_world = image.data_to_world((0, 3, 17, 18))
    _click_world(viewer, *first_world)
    assert widget._proof_target_zyx == pytest.approx((3, 17, 18))

    widget.orientation_rotation_combo.setCurrentIndex(
        widget.orientation_rotation_combo.findData(rotation)
    )
    QApplication.processEvents()

    assert widget.proofreading_enabled
    assert widget._proof_target_zyx == pytest.approx((3, 17, 18))
    marker = _proof_target_layer(viewer)
    assert marker is not None
    _assert_crosshair_geometry(marker, (0, 3, 17, 18))

    second_world = image.data_to_world((0, 3, 21, 22))
    _click_world(viewer, *second_world)
    assert widget._proof_target_zyx == pytest.approx((3, 21, 22))
    _assert_crosshair_geometry(marker, (0, 3, 21, 22))

    _press_proof_key(qtbot, widget, Qt.Key_F7)
    _press_proof_key(qtbot, widget, Qt.Key_F8)
    restored = widget.proofread_store.resolve(1, 0)
    assert restored.center_zyx == pytest.approx((3, 21, 22))
    assert restored.size_zyx == pytest.approx((3, 7, 7))


def test_click_replaces_target_but_invalid_clicks_and_drag_do_not(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    _enable_proofreading(widget)
    _click_world(viewer, 0, 3, 16, 17)
    assert widget._proof_target_zyx == pytest.approx((3, 16, 17))

    _click_world(viewer, 0, 3, 18, 19)
    assert widget._proof_target_zyx == pytest.approx((3, 18, 19))

    _click_world(viewer, 0, 3, 20, 21, button=2)
    _click_world(viewer, 0, 3, 20, 21, modifiers=("Shift",))
    _click_world(viewer, 0, 3, -1, 21)
    _click_world(viewer, 0, 3, 20, 100)
    _drag_world(viewer, 0, 3, 20, 21)

    assert widget._proof_target_zyx == pytest.approx((3, 18, 19))
    marker = _proof_target_layer(viewer)
    assert marker is not None
    _assert_crosshair_geometry(marker, (0, 3, 18, 19))


def test_live_cursor_does_not_enable_f8_or_f9(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store
    _enable_proofreading(widget)
    _set_cursor(viewer, 0, 3, 22, 23)
    widget._proof_delete_current()

    _press_proof_key(qtbot, widget, Qt.Key_F8)
    _press_proof_key(qtbot, widget, Qt.Key_F9)

    assert store.resolve(1, 0) is None
    assert store.provisional_added_ids == set()
    _assert_no_proof_target(viewer, widget)


@pytest.mark.parametrize("transition", ["time", "z", "off", "3d"])
def test_target_clears_on_view_or_proof_transition(
    make_napari_viewer,
    qtbot,
    tmp_path,
    proof_widgets,
    transition,
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    _enable_proofreading(widget)
    _click_world(viewer, 0, 3, 17, 18)
    assert widget._proof_target_zyx is not None

    if transition == "time":
        viewer.dims.set_current_step(0, 1)
    elif transition == "z":
        viewer.dims.set_current_step(1, 4)
    elif transition == "off":
        widget.proofreading_toggle.setChecked(False)
    else:
        viewer.dims.ndisplay = 3
    QApplication.processEvents()

    _assert_no_proof_target(viewer, widget)


def test_proof_mouse_callback_has_bounded_lifecycle(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    before = list(viewer.mouse_drag_callbacks)

    _enable_proofreading(widget)
    assert widget._proof_mouse_callback_installed
    assert len(viewer.mouse_drag_callbacks) == len(before) + 1
    widget._set_proofreading_enabled(True)
    assert len(viewer.mouse_drag_callbacks) == len(before) + 1

    widget.proofreading_toggle.setChecked(False)
    QApplication.processEvents()

    assert not widget._proof_mouse_callback_installed
    assert viewer.mouse_drag_callbacks == before


def test_proof_actions_are_inert_after_another_layer_becomes_active(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    store = widget.proofread_store
    _enable_proofreading(widget)
    _click_world(viewer, 0, 3, 17, 18)
    user_points = viewer.add_points([[0, 3, 5, 5]], name="user points")

    assert viewer.layers.selection.active is user_points
    widget._proof_delete_current()
    widget._proof_place_cursor()
    widget._proof_add_neuron()

    assert store.resolve(1, 0) is not None
    assert store.observation_patches == {}
    assert store.provisional_added_ids == set()
    assert widget._proof_target_zyx == pytest.approx((3, 17, 18))


def test_unload_removes_target_layer_and_mouse_callback(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    before = list(viewer.mouse_drag_callbacks)
    _enable_proofreading(widget)
    _click_world(viewer, 0, 3, 17, 18)

    assert widget.unload_roi(force=True)

    assert _proof_target_layer(viewer) is None
    assert not widget._proof_mouse_callback_installed
    assert viewer.mouse_drag_callbacks == before


def test_proof_target_does_not_manage_unrelated_labels(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    labels_data = np.zeros((3, 8, 32, 32), dtype=np.uint8)
    labels = viewer.add_labels(labels_data, name="unrelated labels")
    labels.visible = True
    viewer.layers.selection.active = labels
    labels.mode = "paint"
    data_before = labels.data.copy()
    assert labels.mode == "paint"

    # Editing an unrelated Labels layer takes precedence: proof mode refuses
    # to start and must not change the Labels layer to obtain canvas clicks.
    widget.proofreading_toggle.setChecked(True)
    QApplication.processEvents()

    assert not widget.proofreading_enabled
    assert not widget.proofreading_toggle.isChecked()
    np.testing.assert_array_equal(labels.data, data_before)
    assert labels.mode == "paint"
    assert labels.visible

    # Selecting the Image naturally resets napari's former Labels tool mode.
    # Once the source Image is active, proof clicks still leave the unrelated
    # layer's current state and data untouched.
    viewer.layers.selection.active = widget.current_image
    labels_mode = labels.mode
    _enable_proofreading(widget)

    _click_world(viewer, 0, 3, 17, 18)

    np.testing.assert_array_equal(labels.data, data_before)
    assert labels.mode == labels_mode
    assert labels.visible
    assert labels in viewer.layers
    assert widget._proof_target_zyx == pytest.approx((3, 17, 18))


def test_proofreading_rejects_non_yx_display_order(
    make_napari_viewer, qtbot, tmp_path, proof_widgets
):
    viewer, widget = _make_widget(
        make_napari_viewer, qtbot, tmp_path, proof_widgets
    )
    viewer.dims.order = (0, 2, 3, 1)
    QApplication.processEvents()

    assert not widget._view_axes_supported()
    assert not widget.proofreading_toggle.isEnabled()
