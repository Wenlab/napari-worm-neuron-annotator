from __future__ import annotations

import json
import random

import numpy as np
import pytest

from napari_worm_neuron_annotator import _proofread as proofread_module
from napari_worm_neuron_annotator._proofread import (
    DELETED,
    ObservationPatch,
    ProofreadStore,
)
from napari_worm_neuron_annotator._roi import NeuronBoxDataset


def _dataset(*, time_count: int = 7, neuron_count: int = 3):
    data = np.full((time_count, neuron_count, 6), np.nan, dtype=np.float32)
    for volume_index in range(time_count):
        for neuron_id in range(neuron_count):
            if (volume_index + neuron_id) % 3 == 2:
                continue
            data[volume_index, neuron_id] = (
                10 + volume_index,
                20 + neuron_id,
                5 * (2 + volume_index),
                7 + neuron_id,
                8 + neuron_id,
                10 + 5 * neuron_id,
            )
    return NeuronBoxDataset(data, z_divisor=5)


def _reference_changed_fields(store: ProofreadStore):
    result = {}
    for key, patch in store.observation_patches.items():
        volume_index, neuron_id = key
        raw = store.dataset.get_box_at_volume_index(volume_index, neuron_id)
        if patch.state == DELETED:
            result[key] = ("presence",)
            continue
        if raw is None:
            result[key] = ("presence",)
            continue
        fields = []
        if neuron_id in store.delete_all_ids:
            fields.append("presence")
        if patch.box.center_zyx != raw.center_zyx:
            fields.append("center_zyx")
        if patch.box.size_zyx != raw.size_zyx:
            fields.append("size_zyx")
        if fields:
            result[key] = tuple(fields)
    for neuron_id in store.delete_all_ids:
        for volume_index in range(store.raw_T):
            key = (volume_index, neuron_id)
            if key in store.observation_patches:
                continue
            if store.dataset.get_box_at_volume_index(volume_index, neuron_id):
                result[key] = ("presence",)
    return result


def _assert_cache_matches_reference(store: ProofreadStore) -> None:
    fields = _reference_changed_fields(store)
    assert store.observation_change_fields == fields
    assert store.modified_observations == set(fields)
    assert store.modified_ids == {neuron_id for _, neuron_id in fields}
    assert store.status.moved == sum(
        "center_zyx" in value for value in fields.values()
    )
    assert store.status.resized == sum(
        "size_zyx" in value for value in fields.values()
    )
    assert store.status.presence == sum(
        "presence" in value for value in fields.values()
    )
    assert store.status.dirty == (
        store.working_snapshot != store.saved_snapshot
    )


def test_state_views_are_read_only_and_geometry_is_immutable():
    store = ProofreadStore(_dataset())
    store.set_observation_present(
        0,
        0,
        center_zyx=[1, 2, 3],
        size_zyx=np.asarray([4, 5, 6]),
    )

    patch = store.observation_patches[(0, 0)]
    assert patch.box.center_zyx == (1.0, 2.0, 3.0)
    assert patch.box.size_zyx == (4.0, 5.0, 6.0)
    with pytest.raises(TypeError):
        store.observation_patches[(0, 0)] = patch
    with pytest.raises(TypeError):
        store.placement_size[0] = (1.0, 2.0, 3.0)
    with pytest.raises(AttributeError):
        store.delete_all_ids.add(0)
    with pytest.raises(AttributeError):
        store.committed_added_ids.add(100)


def test_revision_status_no_op_and_exact_baseline_reversion(tmp_path):
    store = ProofreadStore(_dataset())
    raw = store.dataset.get_box_at_volume_index(0, 0)
    assert store.revision == store.baseline_revision == 0

    store.set_observation_present(0, 0, raw)
    assert store.revision == 0
    store.set_observation_deleted(0, 0)
    assert store.revision == 1
    assert store.status.dirty and store.status.presence == 1
    store.set_observation_deleted(0, 0)
    assert store.revision == 1
    store.set_observation_present(0, 0, raw)
    assert store.revision == 2
    assert not store.status.dirty and store.status.presence == 0

    store.set_placement_size(0, (3, 9, 7))
    store.save(tmp_path / "proof.json")
    assert store.baseline_revision == store.revision
    baseline_revision = store.baseline_revision
    store.set_placement_size(0, (4, 8, 6))
    store.set_placement_size(0, (3, 9, 7))
    assert store.revision == baseline_revision + 2
    assert not store.dirty

    store.set_observation_present(
        0, 0, center_zyx=(9, 8, 7), size_zyx=raw.size_zyx
    )
    store.save()
    assert not store.dirty
    assert store.status.moved == 1


def test_repeated_status_queries_do_not_canonicalize_or_read_raw(monkeypatch):
    store = ProofreadStore(_dataset())
    store.set_observation_deleted(0, 0)

    def fail(*_args, **_kwargs):
        pytest.fail("cached status must not inspect canonical or raw state")

    monkeypatch.setattr(store, "_canonical_state", fail)
    monkeypatch.setattr(store.dataset, "get_box_at_volume_index", fail)
    for _ in range(100):
        assert store.dirty
        assert store.status.dirty
        assert store.status.presence == 1


def test_single_edit_compares_only_its_raw_observation(monkeypatch):
    store = ProofreadStore(_dataset(time_count=100))
    original = store.dataset.get_box_at_volume_index
    calls = []

    def counted(volume_index, neuron_id):
        calls.append((volume_index, neuron_id))
        return original(volume_index, neuron_id)

    monkeypatch.setattr(store.dataset, "get_box_at_volume_index", counted)
    store.set_observation_present(
        0, 0, center_zyx=(9, 8, 7), size_zyx=(3, 7, 7)
    )
    assert calls == [(0, 0)]


def test_cached_status_and_dirty_match_full_reference_after_seeded_edits(
    tmp_path,
):
    randomizer = random.Random(1731)
    store = ProofreadStore(_dataset())

    for step in range(90):
        neuron_id = randomizer.choice(store.neuron_ids)
        volume_index = randomizer.randrange(store.raw_T)
        operation = randomizer.randrange(7)
        if operation == 0:
            store.set_observation_present(
                volume_index,
                neuron_id,
                center_zyx=(
                    randomizer.randrange(8),
                    randomizer.randrange(30),
                    randomizer.randrange(30),
                ),
                size_zyx=(3, 7, 7),
            )
        elif operation == 1:
            store.set_observation_deleted(volume_index, neuron_id)
        elif operation == 2:
            store.delete_all_observations(neuron_id)
        elif operation == 3:
            store.discard_observation(volume_index, neuron_id)
        elif operation == 4:
            store.set_placement_size(
                neuron_id, (3, 7 + randomizer.randrange(3), 7)
            )
        elif operation == 5 and store.resolve(volume_index, neuron_id):
            store.apply_size_at_volume_index(
                volume_index,
                neuron_id,
                (3, 7 + randomizer.randrange(3), 7),
            )
        elif operation == 6 and step % 15 == 0:
            store.add_neuron(volume_index, (2, 8, 8))
        if step in {29, 59}:
            store.save(tmp_path / "proof.json")
        _assert_cache_matches_reference(store)


def test_load_constructs_and_compares_each_patch_once(
    tmp_path, monkeypatch
):
    time_count = 36
    data = np.tile(
        np.asarray([10.0, 11.0, 15.0, 7.0, 8.0, 10.0]),
        (time_count, 1, 1),
    )
    path = tmp_path / "roi.npy"
    np.save(path, data)
    dataset = NeuronBoxDataset.from_npy(path)
    source = ProofreadStore(dataset)
    for volume_index in range(time_count):
        source.set_observation_present(
            volume_index,
            0,
            center_zyx=(3, 11, 20 + volume_index),
            size_zyx=(2, 8, 7),
        )
    sidecar = source.save(tmp_path / "proof.json")

    construction_count = 0
    original_present = ObservationPatch.present.__func__

    def counted_present(cls, box):
        nonlocal construction_count
        construction_count += 1
        return original_present(cls, box)

    raw_calls = 0
    original_raw = dataset.get_box_at_volume_index

    def counted_raw(volume_index, neuron_id):
        nonlocal raw_calls
        raw_calls += 1
        return original_raw(volume_index, neuron_id)

    def fail_canonicalization(_state):
        pytest.fail("load must install validated typed state directly")

    monkeypatch.setattr(
        ObservationPatch, "present", classmethod(counted_present)
    )
    monkeypatch.setattr(dataset, "get_box_at_volume_index", counted_raw)
    monkeypatch.setattr(
        proofread_module, "_state_to_json", fail_canonicalization
    )
    loaded = ProofreadStore(dataset)
    loaded.load(sidecar)

    assert construction_count == time_count
    assert raw_calls == time_count
    assert not loaded.dirty


def test_recovery_capture_is_shallow_isolated_and_defers_json(monkeypatch):
    store = ProofreadStore(_dataset())
    store.set_observation_deleted(0, 0)
    expected = store.working_snapshot
    patch = store.observation_patches[(0, 0)]

    def fail(_state):
        pytest.fail("capture must not build canonical JSON")

    with monkeypatch.context() as context:
        context.setattr(proofread_module, "_state_to_json", fail)
        capture = store.capture_recovery_state()

    assert capture.working_state.observation_patches[(0, 0)] is patch
    with pytest.raises(TypeError):
        capture.working_state.observation_patches[(1, 0)] = patch

    store.set_observation_deleted(1, 0)
    payload = capture.payload(
        session_uuid="session",
        revision=1,
        raw_sha256="0" * 64,
        utc_time="2026-09-13T00:00:00Z",
    )
    assert payload["working_state"] == expected
    assert payload["working_state"] != store.working_snapshot
    json.dumps(payload)
