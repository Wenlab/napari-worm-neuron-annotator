from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from napari_worm_neuron_annotator._proofread import (
    ExternalSidecarChangeError,
    ProofreadStore,
    SidecarError,
)
from napari_worm_neuron_annotator._proofread_files import (
    backup_formal_bytes,
    canonical_json_bytes,
    hash_file_stable,
    list_history_versions,
    read_recovery,
    recovery_path,
    trim_history,
    utc_now_text,
    write_temp_bytes,
)
from napari_worm_neuron_annotator._roi import NeuronBoxDataset


def _dataset(tmp_path: Path) -> NeuronBoxDataset:
    data = np.full((3, 2, 6), np.nan, dtype=np.float32)
    data[0, 0] = [10, 11, 15, 7, 8, 10]
    data[1, 0] = [12, 13, 20, 7, 8, 10]
    data[2, 1] = [14, 15, 25, 5, 6, 15]
    path = tmp_path / "synthetic.npy"
    np.save(path, data)
    return NeuronBoxDataset.from_npy(path, z_divisor=5)


def _recovery(store: ProofreadStore) -> dict:
    return store.recovery_payload(
        session_uuid=uuid.uuid4().hex,
        revision=1,
        raw_sha256=hash_file_stable(store.dataset.path).sha256,
        utc_time=utc_now_text(),
    )


def test_recovery_preserves_working_baseline_provisional_and_delete_all(tmp_path):
    store = ProofreadStore(_dataset(tmp_path), image_signature={"shape": [3, 8, 8]})
    store.set_observation_deleted(0, 0)
    store.save(tmp_path / "proof.json")
    added = store.add_neuron(1, (2.5, 8, 9), size_zyx=(2, 4, 6))
    store.delete_all_observations(1)
    payload = _recovery(store)

    restored = ProofreadStore(store.dataset, image_signature=store.image_signature)
    formal_path, formal_hash = restored.restore_recovery_payload(payload)

    assert restored.saved_snapshot == store.saved_snapshot
    assert restored.working_snapshot == store.working_snapshot
    assert restored.dirty
    assert added in restored.provisional_added_ids
    assert 1 in restored.delete_all_ids
    assert formal_path == tmp_path / "proof.json"
    assert formal_hash == store.bound_sidecar_fingerprint


def test_recovery_failure_is_transactional(tmp_path):
    store = ProofreadStore(_dataset(tmp_path), image_signature={"a": 1})
    store.set_observation_deleted(0, 0)
    before = store.working_snapshot
    payload = _recovery(store)
    payload["working_state"]["next_neuron_id"] = -1

    with pytest.raises(SidecarError):
        store.restore_recovery_payload(payload)

    assert store.working_snapshot == before


def test_recovery_file_rejects_corruption_and_unknown_version(tmp_path):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        read_recovery(corrupt)
    corrupt.write_text(
        json.dumps(
            {
                "recovery_schema_version": 99,
                "session_uuid": "x",
                "utc_time": "now",
                "revision": 1,
                "raw": {},
                "image_signature": None,
                "formal": {"path": None, "sha256": None},
                "working_state": {},
                "saved_state": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported"):
        read_recovery(corrupt)


def test_write_recovery_uses_one_latest_path_per_session(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    store.set_observation_deleted(0, 0)
    session = uuid.uuid4().hex
    target = recovery_path(store.dataset.path, session)
    for revision in (1, 2):
        payload = store.recovery_payload(
            session_uuid=session,
            revision=revision,
            raw_sha256=hash_file_stable(store.dataset.path).sha256,
            utc_time=utc_now_text(),
        )
        temporary = write_temp_bytes(target, canonical_json_bytes(payload))
        os.replace(temporary, target)
    assert read_recovery(target)["revision"] == 2
    assert len(list(target.parent.glob("*.recovery.json"))) == 1


def test_save_history_deduplicates_retains_ten_and_detects_external_change(
    tmp_path,
):
    store = ProofreadStore(_dataset(tmp_path))
    target = tmp_path / "proof.json"
    store.set_observation_deleted(0, 0)
    store.save(target)
    first = target.read_bytes()
    store.set_observation_present(0, 0, center_zyx=(1, 2, 3))
    store.save()
    assert any(version.path.read_bytes() == first for version in list_history_versions(target))

    # Re-saving identical state creates neither a backup nor another primary write.
    before = [version.path for version in list_history_versions(target)]
    store.save()
    assert [version.path for version in list_history_versions(target)] == before

    for index in range(12):
        backup_formal_bytes(target, f"version-{index}".encode())
    assert trim_history(target) is None
    assert len(list_history_versions(target)) == 10

    target.write_text("external", encoding="utf-8")
    store.set_observation_deleted(1, 0)
    with pytest.raises(ExternalSidecarChangeError):
        store.save()
    assert target.read_text(encoding="utf-8") == "external"
    assert store.dirty


def test_history_rollback_keeps_current_baseline_and_never_reuses_ids(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    target = tmp_path / "proof.json"
    store.set_observation_deleted(0, 0)
    store.save(target)
    old = target.read_bytes()
    added = store.add_neuron(1, (2, 3, 4))
    store.save()
    history_file = tmp_path / "old.json"
    history_file.write_bytes(old)

    assert store.load_history_as_working(history_file)

    assert store.bound_sidecar_path == target
    assert added in store.retired_ids
    assert added not in store.neuron_ids
    assert store._next_neuron_id > added
    assert store.dirty
    store.save()
    assert history_file.exists()


def test_synthetic_eight_mb_recovery_workload(tmp_path):
    source = tmp_path / "eight-mb.bin"
    source.write_bytes(bytes(8 * 1024 * 1024))
    start = time.perf_counter()
    hashed = hash_file_stable(source)
    hash_seconds = time.perf_counter() - start
    payload = {"blob": "x" * (8 * 1024 * 1024)}
    start = time.perf_counter()
    data = canonical_json_bytes(payload)
    serialize_seconds = time.perf_counter() - start
    start = time.perf_counter()
    temporary = write_temp_bytes(tmp_path / "snapshot.json", data)
    write_seconds = time.perf_counter() - start

    assert len(hashed.sha256) == 64
    assert temporary.stat().st_size >= 8 * 1024 * 1024
    # A generous regression guard; the GUI performs these steps off-thread.
    assert hash_seconds < 10
    assert serialize_seconds < 10
    assert write_seconds < 10
