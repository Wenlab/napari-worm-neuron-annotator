"""Failure-boundary regressions for proofreading persistence."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from napari_worm_neuron_annotator._proofread import (
    ProofreadStore,
    SidecarError,
)
from napari_worm_neuron_annotator._proofread_files import (
    backup_formal_bytes,
    canonical_json_bytes,
    list_history_versions,
    read_recovery,
    trim_history,
)
from napari_worm_neuron_annotator._roi import NeuronBoxDataset


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "roi.npy"
    np.save(path, np.tile([10.0, 10.0, 15.0, 7.0, 7.0, 15.0], (2, 2, 1)))
    return NeuronBoxDataset.from_npy(path)


def test_load_io_failure_preserves_working_baseline_and_binding(
    dataset, tmp_path, monkeypatch
):
    source = ProofreadStore(dataset)
    incoming = source.save(tmp_path / "incoming.json")
    current = ProofreadStore(dataset)
    current.set_observation_deleted(0, 0)
    current.save(tmp_path / "current.json")
    current.set_observation_deleted(1, 1)
    before = (
        current.working_snapshot,
        current.saved_snapshot,
        current.bound_sidecar_path,
        current.bound_sidecar_fingerprint,
    )

    def unavailable(_path):
        raise OSError("sidecar unavailable during verification")

    monkeypatch.setattr(
        "napari_worm_neuron_annotator._proofread.fingerprint_file", unavailable
    )
    with pytest.raises(SidecarError):
        current.load(incoming)

    assert (
        current.working_snapshot,
        current.saved_snapshot,
        current.bound_sidecar_path,
        current.bound_sidecar_fingerprint,
    ) == before
    assert current.dirty


def test_load_changed_file_does_not_bind_unread_content(
    dataset, tmp_path, monkeypatch
):
    source = ProofreadStore(dataset)
    incoming = source.save(tmp_path / "incoming.json")
    current = ProofreadStore(dataset)
    current.set_observation_deleted(1, 1)
    before = current.working_snapshot
    # Simulate replacement after JSON parsing but before final verification.
    monkeypatch.setattr(
        "napari_worm_neuron_annotator._proofread.fingerprint_file",
        lambda _path: "f" * 64,
    )
    with pytest.raises(SidecarError):
        current.load(incoming)
    assert current.working_snapshot == before
    assert current.dirty
    assert current.bound_sidecar_path is None


def test_recovery_rejects_boolean_schema_version(dataset, tmp_path):
    store = ProofreadStore(dataset)
    store.set_observation_deleted(0, 0)
    payload = store.recovery_payload(
        session_uuid="d59d89c91d824243b61bd3e58e3ef722",
        revision=1,
        raw_sha256=store._raw_fingerprint(),
        utc_time="2026-09-06T13:00:00Z",
    )
    payload["recovery_schema_version"] = True
    path = tmp_path / "bad.recovery.json"
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="schema"):
        read_recovery(path)
    before = store.working_snapshot
    with pytest.raises(SidecarError, match="schema"):
        store.restore_recovery_payload(payload)
    assert store.working_snapshot == before


def test_history_retention_does_not_remove_unmanaged_json(tmp_path):
    formal = tmp_path / "proof.json"
    history = tmp_path / "proof.json.history"
    history.mkdir()
    unrelated = history / "notes.json"
    unrelated.write_text('{"note":"keep this file"}', encoding="utf-8")
    backup_formal_bytes(formal, b'{"revision": 1}')
    newest = backup_formal_bytes(formal, b'{"revision": 2}')
    assert trim_history(formal, limit=1) is None
    assert unrelated.read_text(encoding="utf-8") == '{"note":"keep this file"}'
    assert newest.exists()


def test_working_snapshot_is_detached_without_changing_live_state(dataset):
    store = ProofreadStore(dataset)
    store.set_observation_present(
        0, 0, center_zyx=(3, 12, 10), size_zyx=(3, 7, 7)
    )
    before = copy.deepcopy(store.working_snapshot)
    snapshot = store.working_snapshot
    snapshot["observation_patches"][0]["box"]["center_zyx"][1] = -50
    snapshot["committed_added_ids"].append(123)
    assert store.working_snapshot == before
    assert store.resolve(0, 0).center_zyx == (3, 12, 10)


def test_history_loaded_via_normal_load_is_still_read_only(dataset, tmp_path):
    store = ProofreadStore(dataset)
    formal = store.save(tmp_path / "proof.json")
    store.set_observation_deleted(0, 0)
    store.save()
    backup = list_history_versions(formal)[0].path
    before = backup.read_bytes()
    loaded = ProofreadStore.from_sidecar(backup, dataset)
    loaded.set_observation_deleted(1, 1)
    with pytest.raises(ValueError, match="read-only"):
        loaded.save()
    assert backup.read_bytes() == before
    assert loaded.dirty


def test_failed_backup_cannot_replace_formal_file(dataset, tmp_path, monkeypatch):
    store = ProofreadStore(dataset)
    formal = store.save(tmp_path / "proof.json")
    before = formal.read_bytes()
    store.set_observation_deleted(0, 0)

    def fail(*_args):
        raise PermissionError("backup directory denied")

    monkeypatch.setattr("napari_worm_neuron_annotator._proofread.backup_formal_bytes", fail)
    with pytest.raises(PermissionError):
        store.save()
    assert formal.read_bytes() == before
    assert store.dirty


def test_history_cleanup_failure_does_not_report_save_failure(dataset, tmp_path, monkeypatch):
    store = ProofreadStore(dataset)
    formal = store.save(tmp_path / "proof.json")
    store.set_observation_deleted(0, 0)

    def fail(*_args):
        raise PermissionError("history cleanup denied")

    monkeypatch.setattr("napari_worm_neuron_annotator._proofread.trim_history", fail)
    assert store.save() == formal
    assert "cleanup denied" in store.last_history_warning
    assert not store.dirty
    loaded = ProofreadStore.from_sidecar(formal, dataset)
    assert loaded.resolve(0, 0) is None
