import json

import numpy as np
import pytest

from napari_worm_neuron_annotator._proofread import (
    ABSENT,
    DELETED,
    PRESENT,
    RAW,
    ProofreadStore,
    SidecarError,
)
from napari_worm_neuron_annotator._roi import NeuronBoxDataset


def _dataset(tmp_path=None, *, dtype=np.float32, field_count=7):
    fill = np.nan if np.issubdtype(dtype, np.floating) else 0
    data = np.full((3, 2, field_count), fill, dtype=dtype)
    data[0, 0, :6] = (10, 20, 10, 7, 9, 15)
    data[1, 0, :6] = (11, 21, 15, 8, 10, 20)
    data[2, 1, :6] = (12, 22, 20, 9, 11, 25)
    if tmp_path is None:
        return NeuronBoxDataset(data, z_divisor=5)
    path = tmp_path / "raw.npy"
    np.save(path, data)
    return NeuronBoxDataset.from_npy(path, z_divisor=5)


def test_dataset_volume_index_api_is_direct_and_validated():
    dataset = _dataset()

    assert dataset.raw_shape == (3, 2, 7)
    assert dataset.raw_T == 3
    assert dataset.raw_N == 2
    assert dataset.raw_dtype == np.dtype(np.float32)
    assert dataset.get_box_at_volume_index(1, 0).center_zyx == (3, 21, 11)
    assert dataset.valid_ids_at_volume_index(2) == [1]
    with pytest.raises(TypeError):
        dataset.get_box_at_volume_index(1.0, 0)
    assert dataset.get_box_at_volume_index(3, 0) is None


def test_resolver_present_deleted_and_raw_state():
    store = ProofreadStore(_dataset())

    assert store.effective_state(0, 0) == RAW
    assert store.effective_state(2, 0) == ABSENT
    store.set_observation_present(
        0, 0, center_zyx=(7, 8, 9), size_zyx=(3, 7, 7)
    )
    assert store.effective_state(0, 0) == PRESENT
    assert store.resolve(0, 0).center_zyx == (7, 8, 9)

    store.set_observation_deleted(0, 0)
    assert store.effective_state(0, 0) == DELETED
    assert store.resolve(0, 0) is None
    assert store.size_for_placement(0, 0) == (3, 7, 7)


def test_restoring_raw_removes_redundant_patch_and_modified_status():
    store = ProofreadStore(_dataset())
    raw = store.resolve(0, 0)
    store.set_observation_present(
        0, 0, center_zyx=(1, 2, 3), size_zyx=(3, 7, 7)
    )
    assert (0, 0) in store.modified_observations

    store.set_observation_present(0, 0, raw)

    assert (0, 0) not in store.observation_patches
    assert (0, 0) not in store.modified_observations


def test_delete_all_normalizes_old_patches_then_allows_one_restore():
    store = ProofreadStore(_dataset())
    store.set_observation_present(
        0, 0, center_zyx=(9, 9, 9), size_zyx=(3, 7, 7)
    )
    store.set_observation_deleted(1, 0)

    store.delete_all_observations(0)

    assert not any(key[1] == 0 for key in store.observation_patches)
    assert store.resolve(0, 0) is None
    assert store.resolve(1, 0) is None
    store.set_observation_present(
        1, 0, center_zyx=(4, 5, 6), size_zyx=(3, 7, 7)
    )
    assert store.resolve(1, 0).center_zyx == (4, 5, 6)

    store.set_observation_deleted(1, 0)

    assert (1, 0) not in store.observation_patches
    assert store.resolve(1, 0) is None


def test_delete_all_preserves_effective_size_for_later_placement():
    store = ProofreadStore(_dataset())

    assert 0 not in store.placement_size
    assert store.resolve(0, 0).size_zyx == (3, 9, 7)

    store.delete_all_observations(0)

    assert store.placement_size[0] == (3, 9, 7)
    assert store.size_for_placement(0, 1) == (3, 9, 7)
    restored = store.set_observation_present(
        1, 0, center_zyx=(4, 5, 6)
    )
    assert restored.size_zyx == (3, 9, 7)


def test_modified_observations_uses_patches_and_raw_present_delete_all():
    store = ProofreadStore(_dataset())
    store.delete_all_observations(0)
    store.set_observation_present(
        2, 0, center_zyx=(1, 2, 3), size_zyx=(3, 7, 7)
    )

    assert store.modified_observations == {(0, 0), (1, 0), (2, 0)}
    assert store.modified_ids == {0}


def test_apply_size_updates_only_currently_valid_observations():
    store = ProofreadStore(_dataset())

    store.apply_size(0, (5, 6, 7))

    assert store.placement_size[0] == (5, 6, 7)
    assert store.resolve(0, 0).size_zyx == (5, 6, 7)
    assert store.resolve(1, 0).size_zyx == (5, 6, 7)
    assert store.resolve(2, 0) is None
    assert set(store.observation_patches) == {(0, 0), (1, 0)}


def test_add_discard_and_save_commit_identity(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    first = store.add_neuron(0, (1, 2, 3))
    assert first == 2
    assert store.resolve(0, first).size_zyx == (3, 7, 7)
    assert store.dirty

    store.discard()

    assert first not in store.neuron_ids
    assert not store.dirty
    assert store.add_neuron(0, (2, 3, 4)) == 2
    sidecar = store.save(tmp_path / "edits.json")
    assert not store.dirty
    assert store.committed_added_ids == {2}

    loaded = ProofreadStore.from_sidecar(sidecar, store.dataset)
    assert loaded.committed_added_ids == {2}
    assert loaded.add_neuron(1, (2, 3, 4)) == 3


def test_save_reserves_empty_provisional_identity(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    neuron_id = store.add_neuron(0, (1, 2, 3))
    store.retire_added_neuron(neuron_id)

    path = store.save(tmp_path / "edits.json")
    loaded = ProofreadStore.from_sidecar(path, store.dataset)

    assert neuron_id in loaded.retired_ids
    assert loaded.add_neuron(0, (1, 2, 3)) == neuron_id + 1


def test_save_retires_provisional_without_present_observations(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    neuron_id = store.add_neuron(0, (1, 2, 3))
    store.set_observation_deleted(0, neuron_id)

    assert neuron_id in store.provisional_added_ids
    path = store.save(tmp_path / "edits.json")

    assert neuron_id in store.retired_ids
    assert neuron_id not in store.provisional_added_ids
    assert not store.dirty
    loaded = ProofreadStore.from_sidecar(path, store.dataset)
    assert loaded.add_neuron(0, (1, 2, 3)) == neuron_id + 1


def test_discard_restores_identity_allocator_after_retirement(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    committed = store.add_neuron(0, (1, 2, 3))
    store.save(tmp_path / "edits.json")
    store.retire_added_neuron(committed)
    provisional = store.add_neuron(1, (4, 5, 6))

    store.discard()

    assert store.identity_state[committed] == "committed_added"
    assert provisional not in store.all_neuron_ids
    assert store.add_neuron(1, (4, 5, 6)) == provisional


def test_dirty_save_and_discard_snapshot(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    store.set_observation_deleted(0, 0)
    assert store.dirty
    store.save(tmp_path / "edits.json")
    assert not store.dirty
    store.set_observation_present(
        1, 0, center_zyx=(1, 2, 3), size_zyx=(3, 7, 7)
    )
    assert store.dirty

    store.discard()

    assert not store.dirty
    assert store.resolve(0, 0) is None
    assert store.resolve(1, 0).center_zyx == (3, 21, 11)


def test_load_failure_is_transactional(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    store.set_observation_deleted(0, 0)
    dirty_before = store.dirty
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_version": 999}', encoding="utf-8")

    with pytest.raises(SidecarError, match="schema_version"):
        store.load(invalid)

    assert store.resolve(0, 0) is None
    assert store.dirty is dirty_before


def test_duplicate_json_key_is_rejected_before_state_mutation(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    store.set_observation_deleted(0, 0)
    dirty_before = store.dirty
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        '{"schema_version":1,"schema_version":1}', encoding="utf-8"
    )

    with pytest.raises(SidecarError, match="duplicate JSON key"):
        store.load(invalid)

    assert store.resolve(0, 0) is None
    assert store.dirty is dirty_before


def test_atomic_save_failure_preserves_target_and_working_state(
    tmp_path, monkeypatch
):
    store = ProofreadStore(_dataset(tmp_path))
    store.set_observation_deleted(0, 0)
    target = tmp_path / "proof.json"
    original = b"existing sidecar bytes"
    target.write_bytes(original)

    def fail_replace(source, destination):
        raise PermissionError("simulated replace failure")

    monkeypatch.setattr(
        "napari_worm_neuron_annotator._proofread.os.replace", fail_replace
    )

    with pytest.raises(PermissionError, match="replace failure"):
        store.save(target)

    assert target.read_bytes() == original
    assert store.resolve(0, 0) is None
    assert store.dirty
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update(unknown_root=True),
            "sidecar root contains unknown",
        ),
        (
            lambda payload: payload["raw"].update(unknown_raw=True),
            "raw metadata contains unknown",
        ),
        (
            lambda payload: payload["added_neurons"].update(
                provisional=[]
            ),
            "added_neurons contains unknown",
        ),
        (
            lambda payload: payload["observation_patches"][0].update(
                unknown_patch=True
            ),
            "patch contains unknown",
        ),
        (
            lambda payload: payload["observation_patches"][0]["box"].update(
                unknown_box=True
            ),
            "patch box contains unknown",
        ),
        (
            lambda payload: payload.pop("delete_all_ids"),
            "missing required.*delete_all_ids",
        ),
    ],
)
def test_sidecar_rejects_unknown_or_missing_authoritative_fields(
    tmp_path, mutate, message
):
    store = ProofreadStore(_dataset(tmp_path))
    store.set_observation_present(
        0, 0, center_zyx=(1, 2, 3), size_zyx=(3, 7, 7)
    )
    valid = store.save(tmp_path / "valid.json")
    payload = json.loads(valid.read_text(encoding="utf-8"))
    mutate(payload)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SidecarError, match=message):
        ProofreadStore.from_sidecar(invalid, store.dataset)


@pytest.mark.parametrize(
    "invalid_key", ["+0", "00", "01", " 0", "0 ", "1.0", "True"]
)
def test_sidecar_rejects_noncanonical_placement_keys(tmp_path, invalid_key):
    store = ProofreadStore(_dataset(tmp_path))
    store.apply_size(0, (3, 9, 7))
    valid = store.save(tmp_path / "valid.json")
    payload = json.loads(valid.read_text(encoding="utf-8"))
    payload["placement_size"] = {
        invalid_key: payload["placement_size"]["0"]
    }
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SidecarError, match="placement neuron_id"):
        ProofreadStore.from_sidecar(invalid, store.dataset)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("center_zyx",), ["1", 2, 3]),
        (("size_zyx",), [True, 7, 7]),
        (("size_zyx",), {"0": 3, "1": 7, "2": 7}),
    ],
)
def test_sidecar_rejects_coercible_non_json_vector_values(
    tmp_path, path, value
):
    store = ProofreadStore(_dataset(tmp_path))
    store.set_observation_present(
        0, 0, center_zyx=(1, 2, 3), size_zyx=(3, 7, 7)
    )
    valid = store.save(tmp_path / "valid.json")
    payload = json.loads(valid.read_text(encoding="utf-8"))
    payload["observation_patches"][0]["box"][path[0]] = value
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SidecarError, match="three-number list"):
        ProofreadStore.from_sidecar(invalid, store.dataset)


@pytest.mark.parametrize("schema_version", [None, 0, 2, 1.0, "1", True])
def test_sidecar_rejects_old_unknown_or_non_integer_schema(
    tmp_path, schema_version
):
    store = ProofreadStore(_dataset(tmp_path))
    valid = store.save(tmp_path / "valid.json")
    payload = json.loads(valid.read_text(encoding="utf-8"))
    if schema_version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = schema_version
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SidecarError, match="schema_version"):
        ProofreadStore.from_sidecar(invalid, store.dataset)


def test_sidecar_freezes_z_divisor_with_exact_numeric_semantics(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    valid = store.save(tmp_path / "valid.json")
    payload = json.loads(valid.read_text(encoding="utf-8"))

    assert payload["raw"]["z_divisor"] == 5.0

    # A JSON integer and float represent the same divisor value.
    payload["raw"]["z_divisor"] = 5
    equivalent = tmp_path / "equivalent.json"
    equivalent.write_text(json.dumps(payload), encoding="utf-8")
    ProofreadStore.from_sidecar(equivalent, store.dataset)

    # Even with identical NPY bytes, a different divisor changes z geometry.
    different_divisor_dataset = NeuronBoxDataset.from_npy(
        store.dataset.path, z_divisor=5.000000000000001
    )
    with pytest.raises(SidecarError, match="z_divisor does not match"):
        ProofreadStore.from_sidecar(valid, different_divisor_dataset)


@pytest.mark.parametrize(
    "z_divisor",
    [None, "5", True, 0, -1, float("nan"), float("inf")],
)
def test_sidecar_rejects_missing_or_invalid_z_divisor(tmp_path, z_divisor):
    store = ProofreadStore(_dataset(tmp_path))
    valid = store.save(tmp_path / "valid.json")
    payload = json.loads(valid.read_text(encoding="utf-8"))
    if z_divisor is None:
        del payload["raw"]["z_divisor"]
    else:
        payload["raw"]["z_divisor"] = z_divisor
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SidecarError, match="z_divisor|non-finite"):
        ProofreadStore.from_sidecar(invalid, store.dataset)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update(schema_version=999),
            "schema_version",
        ),
        (
            lambda payload: payload["observation_patches"].append(
                dict(payload["observation_patches"][0])
            ),
            "duplicate observation",
        ),
        (
            lambda payload: payload["observation_patches"][0]["box"].update(
                center_zyx=[float("nan"), 1, 2]
            ),
            "non-finite",
        ),
    ],
)
def test_sidecar_strict_validation(tmp_path, mutate, message):
    store = ProofreadStore(_dataset(tmp_path))
    store.set_observation_present(
        0, 0, center_zyx=(1, 2, 3), size_zyx=(3, 7, 7)
    )
    valid = store.save(tmp_path / "valid.json")
    payload = json.loads(valid.read_text(encoding="utf-8"))
    mutate(payload)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SidecarError, match=message):
        ProofreadStore.from_sidecar(invalid, store.dataset)


def test_corrected_export_preserves_ids_delete_all_and_present_restore(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    store.delete_all_observations(0)
    store.set_observation_present(
        1, 0, center_zyx=(8, 9, 10), size_zyx=(3, 7, 5)
    )
    added = store.add_neuron(2, (4, 5, 6))
    store.save(tmp_path / "proof.json")

    output_path = store.export_corrected_npy(tmp_path / "corrected.npy")
    output = np.load(output_path, allow_pickle=False)

    assert added == 2
    assert output.shape == (3, 3, 7)
    assert np.isnan(output[0, 0, :6]).all()
    np.testing.assert_allclose(output[1, 0, :6], (10, 9, 40, 5, 7, 15))
    np.testing.assert_allclose(output[2, 2, :6], (6, 5, 20, 7, 7, 15))


def test_export_does_not_compact_retired_id_gap(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    retired = store.add_neuron(0, (1, 2, 3))
    store.retire_added_neuron(retired)
    committed = store.add_neuron(1, (4, 5, 6))
    store.save(tmp_path / "proof.json")

    output = np.load(
        store.export_corrected_npy(tmp_path / "corrected.npy"),
        allow_pickle=False,
    )

    assert (retired, committed) == (2, 3)
    assert output.shape[1] == 4
    assert np.isnan(output[:, retired, :]).all()
    assert np.isfinite(output[1, committed, :6]).all()


def test_trailing_retired_id_is_reserved_but_not_materialized(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    retired = store.add_neuron(0, (1, 2, 3))
    store.retire_added_neuron(retired)
    sidecar = store.save(tmp_path / "proof.json")

    output = np.load(
        store.export_corrected_npy(tmp_path / "corrected.npy"),
        allow_pickle=False,
    )
    loaded = ProofreadStore.from_sidecar(sidecar, store.dataset)

    assert output.shape[1] == store.raw_N
    assert loaded.add_neuron(0, (1, 2, 3)) == retired + 1


def test_export_preserves_trailing_fields_for_raw_observations(tmp_path):
    dataset = _dataset(tmp_path, field_count=9)
    # Give trailing fields values independent of geometry validity/state.
    writable = np.load(dataset.path, mmap_mode="r+")
    writable[:, :, 6:] = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
    writable.flush()
    del writable
    dataset = NeuronBoxDataset.from_npy(dataset.path, z_divisor=5)
    raw_trailing = np.array(dataset.raw_data[:, :, 6:], copy=True)
    store = ProofreadStore(dataset)
    store.delete_all_observations(0)
    store.set_observation_deleted(2, 1)

    output = np.load(
        store.export_corrected_npy(tmp_path / "corrected.npy"),
        allow_pickle=False,
    )

    np.testing.assert_array_equal(output[:, :, 6:], raw_trailing)
    assert np.isnan(output[:, 0, :6]).all()
    assert np.isnan(output[2, 1, :6]).all()


def test_integer_raw_exports_float_and_raw_file_is_unchanged(tmp_path):
    dataset = _dataset(tmp_path, dtype=np.int16)
    raw_before = dataset.path.read_bytes()
    store = ProofreadStore(dataset)
    store.set_observation_deleted(0, 0)

    output = np.load(
        store.export_corrected_npy(tmp_path / "corrected.npy"),
        allow_pickle=False,
    )

    assert np.issubdtype(output.dtype, np.floating)
    assert np.isnan(output[0, 0, :6]).all()
    assert dataset.path.read_bytes() == raw_before


def test_export_rejects_provisional_identity(tmp_path):
    store = ProofreadStore(_dataset(tmp_path))
    store.add_neuron(0, (1, 2, 3))

    with pytest.raises(ValueError, match="save proof edits"):
        store.export_corrected_npy(tmp_path / "corrected.npy")
