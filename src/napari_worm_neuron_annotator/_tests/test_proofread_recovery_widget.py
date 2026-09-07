"""Exercise recovery lifecycle and failure handling with real Qt workers."""

from __future__ import annotations

import os
from threading import Event

import numpy as np
import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QThread
from qtpy.QtWidgets import QApplication, QMessageBox

from napari_worm_neuron_annotator import NeuronAnnotatorWidget
from napari_worm_neuron_annotator import _widget as widget_module
from napari_worm_neuron_annotator._proofread_files import (
    fingerprint_file,
    hash_file_stable,
    list_history_versions,
    read_recovery,
    recovery_directory,
)


@pytest.fixture
def make_recovery_widget(qtbot, make_napari_viewer, tmp_path):
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, np.tile([8., 8., 10., 4., 4., 10.], (2, 2, 1)))
    widgets = []

    def make():
        viewer = make_napari_viewer()
        viewer.add_image(np.zeros((2, 6, 24, 24), dtype=np.uint16))
        viewer.dims.current_step = (0, 2, 0, 0)
        widget = NeuronAnnotatorWidget(viewer)
        widgets.append(widget)
        qtbot.addWidget(widget, before_close_func=lambda w: w.shutdown(force=True))
        widget.load_roi_path(roi_path)
        widget.show()
        return widget

    yield make
    for widget in widgets:
        widget.shutdown(force=True)


def _protect(widget, qtbot):
    widget._on_recovery_timer()
    qtbot.waitUntil(
        lambda: widget._recovery_thread is None and widget._recovery_current_path is not None,
        timeout=5000,
    )
    return widget._recovery_current_path


def _prime_hash(widget):
    hashed = hash_file_stable(widget.roi_dataset.path)
    widget._recovery_source_identity = hashed.identity
    widget._recovery_raw_sha256 = hashed.sha256


def test_recovery_timer_reuses_hash_and_skips_unchanged_state(
    make_recovery_widget, qtbot, monkeypatch
):
    calls = []
    original_hash = widget_module.hash_file_stable

    def counted(*args, **kwargs):
        assert QThread.currentThread() != QApplication.instance().thread()
        calls.append(args[0])
        return original_hash(*args, **kwargs)

    monkeypatch.setattr(widget_module, "hash_file_stable", counted)
    widget = make_recovery_widget()
    assert widget._recovery_timer.interval() == 30_000
    widget.proofread_store.set_observation_deleted(0, 0)
    path = _protect(widget, qtbot)
    before = (path.stat().st_mtime_ns, widget._recovery_revision)
    widget._on_recovery_timer()
    assert widget._recovery_thread is None
    assert (path.stat().st_mtime_ns, widget._recovery_revision) == before
    widget.proofread_store.set_observation_deleted(1, 1)
    _protect(widget, qtbot)
    assert len(calls) == 1
    assert widget.proofread_store.dirty


def test_recovery_write_failure_retries_without_losing_last_snapshot(
    make_recovery_widget, qtbot, monkeypatch
):
    widget = make_recovery_widget()
    widget.proofread_store.set_observation_deleted(0, 0)
    path = _protect(widget, qtbot)
    protected = path.read_bytes()
    original_write = widget_module.write_temp_bytes

    def fail_write(*_args):
        raise PermissionError("read-only directory")

    widget.proofread_store.set_observation_deleted(1, 1)
    monkeypatch.setattr(widget_module, "write_temp_bytes", fail_write)
    widget._on_recovery_timer()
    qtbot.waitUntil(lambda: widget._recovery_thread is None)
    assert "read-only directory" in widget.proof_recovery_status_label.text()
    assert path.read_bytes() == protected
    monkeypatch.setattr(widget_module, "write_temp_bytes", original_write)
    _protect(widget, qtbot)
    assert path.read_bytes() != protected
    assert widget._recovery_failure is None


@pytest.mark.parametrize("transition", ["save", "discard", "unload", "close"])
def test_inflight_snapshot_never_returns_after_transition(
    make_recovery_widget, qtbot, tmp_path, monkeypatch, transition
):
    widget = make_recovery_widget()
    widget.proofread_store.set_observation_deleted(0, 0)
    _prime_hash(widget)
    started, release = Event(), Event()
    original_write = widget_module.write_temp_bytes

    def delayed_write(*args):
        started.set()
        assert release.wait(5)
        return original_write(*args)

    monkeypatch.setattr(widget_module, "write_temp_bytes", delayed_write)
    directory = recovery_directory(widget.roi_dataset.path)
    widget._on_recovery_timer()
    qtbot.waitUntil(started.is_set)
    worker = widget._recovery_worker
    try:
        if transition == "save":
            assert widget._save_proof_to_path(str(tmp_path / "formal.json"))
        elif transition == "discard":
            widget.discard_proof_edits()
            assert not widget.proofread_store.dirty
        elif transition == "unload":
            widget.unload_roi(force=True)
        else:
            assert widget.findChildren(QThread) == []
            widget.shutdown(force=True)
        assert worker.cancelled.is_set()
    finally:
        release.set()
    qtbot.waitUntil(lambda: worker.result is not None)
    qtbot.waitUntil(lambda: not widget_module._RECOVERY_WORKERS)
    assert not list(directory.glob("*.recovery.json"))
    assert not list(directory.glob("*.tmp"))


def test_recovery_restores_provisional_state_and_requires_save_as_if_primary_changed(
    make_recovery_widget, qtbot, tmp_path
):
    first = make_recovery_widget()
    formal = tmp_path / "formal.json"
    assert first._save_proof_to_path(str(formal))
    added = first.proofread_store.add_neuron(1, (2, 9, 9))
    first._refresh_available_ids(select_first=False)
    original = _protect(first, qtbot)
    first.shutdown(force=True)
    formal.write_text("external change", encoding="utf-8")
    second = make_recovery_widget()
    assert second.proof_recovery_btn.isEnabled()
    assert not second.proofread_store.dirty
    assert second._restore_recovery(original)
    assert second._proof_sidecar_path is None
    assert added in second.proofread_store.provisional_added_ids
    assert second.proofread_store.dirty
    qtbot.waitUntil(lambda: second._recovery_thread is None)
    assert second._recovery_current_path.exists()
    assert not original.exists()
    assert formal.read_text(encoding="utf-8") == "external change"


def test_scoped_discard_replaces_recovery_without_resurrecting_discarded_edit(
    make_recovery_widget, qtbot
):
    widget = make_recovery_widget()
    store = widget.proofread_store
    store.set_observation_deleted(0, 0)
    store.set_observation_deleted(1, 1)
    old = _protect(widget, qtbot)
    widget.discard_proof_edits()
    qtbot.waitUntil(lambda: widget._recovery_thread is None)
    payload = read_recovery(widget._recovery_current_path)
    patches = payload["working_state"]["observation_patches"]
    assert [(p["volume_index"], p["neuron_id"]) for p in patches] == [(1, 1)]
    assert old == widget._recovery_current_path
    assert store.resolve(0, 0) is not None and store.resolve(1, 1) is None


def test_invalid_history_is_rejected_before_discarding_pending_edits(
    make_recovery_widget, tmp_path, monkeypatch
):
    widget = make_recovery_widget()
    widget.proofread_store.set_observation_deleted(0, 0)
    before = widget.proofread_store.working_snapshot
    invalid = tmp_path / "bad.json"
    invalid.write_text("{", encoding="utf-8")
    monkeypatch.setattr(
        widget, "_confirm_proof_transition",
        lambda *_: pytest.fail("must validate history before prompting to discard"),
    )
    assert not widget._load_history_version(invalid)
    assert widget.proofread_store.working_snapshot == before


def test_history_load_keeps_formal_binding_and_reserves_added_identity(
    make_recovery_widget, qtbot, tmp_path
):
    widget = make_recovery_widget()
    formal = tmp_path / "formal.json"
    assert widget._save_proof_to_path(str(formal))
    added = widget.proofread_store.add_neuron(0, (2, 9, 9))
    assert widget._save_proof_to_path(str(formal))
    baseline = widget.proofread_store.saved_snapshot
    version = list_history_versions(formal)[0]
    assert widget._load_history_version(version.path)
    assert widget._proof_sidecar_path == formal
    assert widget.proofread_store.saved_snapshot == baseline
    assert added in widget.proofread_store.retired_ids
    assert widget.proofread_store.dirty
    assert widget._save_proof_to_path(str(formal))
    assert widget.proofread_store.bound_sidecar_fingerprint == fingerprint_file(formal)


def test_recovery_actions_fit_narrow_scroll_viewport(make_recovery_widget, qtbot):
    widget = make_recovery_widget()
    widget.resize(320, 650)
    QApplication.processEvents()
    viewport = widget.scroll_area.viewport()
    assert widget.scroll_content.width() <= viewport.width()
    for control in (widget.proof_save_btn, widget.proof_recovery_btn, widget.proof_history_btn):
        widget.scroll_area.ensureWidgetVisible(control)
        QApplication.processEvents()
        left = control.mapTo(viewport, control.rect().topLeft()).x()
        assert left >= 0
        assert left + control.width() <= viewport.width()


def test_saved_recovery_can_be_discarded_without_losing_other_sessions(
    make_recovery_widget, qtbot, monkeypatch
):
    first = make_recovery_widget()
    first.proofread_store.set_observation_deleted(0, 0)
    old = _protect(first, qtbot)
    first.shutdown(force=True)
    second = make_recovery_widget()
    second.proofread_store.set_observation_deleted(1, 1)
    current = _protect(second, qtbot)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: QMessageBox.Discard)
    assert second._confirm_proof_transition("test")
    assert not current.exists()
    assert old.exists()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_close_after_worker_finishes_before_gui_receives_result(make_recovery_widget, qtbot):
    widget = make_recovery_widget()
    widget.proofread_store.set_observation_deleted(0, 0)
    _prime_hash(widget)
    directory = recovery_directory(widget.roi_dataset.path)
    widget._on_recovery_timer()
    thread = widget._recovery_thread
    # Deliberately delay delivery of queued GUI slots. The worker's C++ object
    # can already be deleted even though the dock still holds its Python wrapper.
    assert thread.wait(2000)
    assert widget._recovery_thread is thread
    assert widget.shutdown(force=True)
    qtbot.waitUntil(lambda: not widget_module._RECOVERY_WORKERS)
    assert not list(directory.glob("*.recovery.json"))
    assert not list(directory.glob("*.tmp"))


def test_source_change_pauses_recovery_and_refuses_incorrect_formal_binding(
    make_recovery_widget, qtbot, tmp_path
):
    widget = make_recovery_widget()
    widget.proofread_store.set_observation_deleted(0, 0)
    recovery = _protect(widget, qtbot)
    before = recovery.read_bytes()
    source = widget.roi_dataset.path
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    widget._on_recovery_timer()
    assert widget._recovery_source_changed
    assert widget._recovery_thread is None
    assert recovery.read_bytes() == before
    formal = tmp_path / "do-not-save.json"
    assert not widget._save_proof_to_path(str(formal))
    assert not formal.exists() and widget.proofread_store.dirty


@pytest.mark.parametrize("view", ["off", "3d", "detached"])
def test_protection_continues_outside_proofreading_view(make_recovery_widget, qtbot, view):
    widget = make_recovery_widget()
    widget.proofread_store.set_observation_deleted(0, 0)
    if view == "3d":
        widget.viewer.dims.ndisplay = 3
    elif view == "detached":
        widget.viewer.layers.remove(widget.current_image)
    assert not widget.proofreading_enabled
    path = _protect(widget, qtbot)
    assert read_recovery(path)["working_state"]["observation_patches"][0]["state"] == "deleted"
