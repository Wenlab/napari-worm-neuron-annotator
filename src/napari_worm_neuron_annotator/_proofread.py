"""Pure proofreading data model and sidecar persistence.

The proofreading layer is deliberately independent of napari and Qt.  A
``NeuronBoxDataset`` remains a read-only view of the original NPY array;
``ProofreadStore`` keeps only sparse edits and resolves them over that raw
view.  All times in this module are raw NPY ``volume_index`` values.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ._proofread_files import (
    RECOVERY_SCHEMA_VERSION,
    backup_formal_bytes,
    fingerprint_file,
    is_history_version,
    sha256_bytes,
    trim_history,
    validate_recovery_envelope,
)
from ._roi import NeuronBox, NeuronBoxDataset

# Sidecars written by this module use schema v2.  The reader deliberately
# keeps accepting v1 files; v1 records did not carry the derived
# ``changed_fields`` list and are upgraded in memory on load.
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
PRESENT = "present"
DELETED = "deleted"
RAW = "raw"
ABSENT = "absent"

DEFAULT_SIZE_ZYX = (3.0, 7.0, 7.0)


@dataclass(frozen=True)
class ObservationPatch:
    """One canonical sparse observation override.

    ``state`` is either :data:`PRESENT` or :data:`DELETED`.  A deleted patch
    may retain the old size as ``restore_size_zyx`` so a subsequent placement
    can restore the same template.  The restore size is metadata only and is
    never used by the resolver while the observation is deleted.
    """

    state: Literal["present", "deleted"]
    box: NeuronBox | None = None
    restore_size_zyx: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.state not in (PRESENT, DELETED):
            raise ValueError("patch state must be 'present' or 'deleted'")
        if self.state == PRESENT:
            if self.box is None:
                raise ValueError("present patch requires a box")
            if self.restore_size_zyx is not None:
                raise ValueError(
                    "present patch cannot carry restore_size_zyx"
                )
        elif self.box is not None:
            raise ValueError("deleted patch cannot carry a box")
        if self.restore_size_zyx is not None:
            object.__setattr__(
                self,
                "restore_size_zyx",
                _validate_size(self.restore_size_zyx),
            )

    @classmethod
    def present(cls, box: NeuronBox) -> ObservationPatch:
        return cls(PRESENT, box=box)

    @classmethod
    def deleted(
        cls, restore_size_zyx: tuple[float, float, float] | None = None
    ) -> ObservationPatch:
        return cls(DELETED, restore_size_zyx=restore_size_zyx)


class SidecarError(ValueError):
    """Raised when a proof sidecar is invalid or incompatible."""


class ExternalSidecarChangeError(SidecarError):
    """Raised when a bound formal sidecar changed outside this session."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _require_int(value: Any, name: str) -> int:
    if not _is_int(value):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_size(size: Any, name: str = "size_zyx") -> tuple[float, ...]:
    try:
        values = tuple(_finite_float(v, name) for v in size)
    except TypeError as exc:
        raise ValueError(f"{name} must contain three values") from exc
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain three positive finite values")
    return values


def _validate_center(center: Any) -> tuple[float, float, float]:
    try:
        values = tuple(_finite_float(v, "center_zyx") for v in center)
    except TypeError as exc:
        raise ValueError("center_zyx must contain three values") from exc
    if len(values) != 3:
        raise ValueError("center_zyx must contain three values")
    return values


def _box_from_parts(
    neuron_id: int,
    volume_index: int,
    center_zyx: Any,
    size_zyx: Any,
) -> NeuronBox:
    return NeuronBox(
        neuron_id=int(neuron_id),
        source_t=int(volume_index),
        center_zyx=_validate_center(center_zyx),
        size_zyx=_validate_size(size_zyx),
    )


def _box_to_json(box: NeuronBox) -> dict[str, Any]:
    return {
        "center_zyx": [float(v) for v in box.center_zyx],
        "size_zyx": [float(v) for v in box.size_zyx],
    }


def _patch_to_json(
    volume_index: int,
    neuron_id: int,
    patch: ObservationPatch,
    *,
    changed_fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "volume_index": int(volume_index),
        "neuron_id": int(neuron_id),
        "state": patch.state,
    }
    if patch.state == PRESENT:
        assert patch.box is not None
        record["box"] = _box_to_json(patch.box)
    elif patch.restore_size_zyx is not None:
        record["restore_size_zyx"] = [
            float(v) for v in patch.restore_size_zyx
        ]
    if changed_fields is not None:
        record["changed_fields"] = list(changed_fields)
    return record


_CHANGED_FIELD_ORDER = ("presence", "center_zyx", "size_zyx")
_CHANGED_FIELDS = frozenset(_CHANGED_FIELD_ORDER)
_RAW_BOX_UNSET = object()


@dataclass(frozen=True)
class ProofreadStatus:
    """Cached summary of the working proofreading state."""

    dirty: bool
    moved: int
    resized: int
    presence: int


@dataclass(frozen=True)
class _StoreState:
    """Immutable-container snapshot used for baselines and worker handoff."""

    observation_patches: Mapping[tuple[int, int], ObservationPatch]
    delete_all_ids: frozenset[int]
    placement_size: Mapping[int, tuple[float, float, float]]
    committed_added_ids: frozenset[int]
    provisional_added_ids: frozenset[int]
    retired_ids: frozenset[int]
    next_neuron_id: int
    changed_fields: Mapping[tuple[int, int], tuple[str, ...]]


def _frozen_mapping(values: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(values))


def _store_state(
    *,
    observation_patches: Mapping[tuple[int, int], ObservationPatch],
    delete_all_ids: set[int] | frozenset[int],
    placement_size: Mapping[int, tuple[float, float, float]],
    committed_added_ids: set[int] | frozenset[int],
    provisional_added_ids: set[int] | frozenset[int],
    retired_ids: set[int] | frozenset[int],
    next_neuron_id: int,
    changed_fields: Mapping[tuple[int, int], tuple[str, ...]],
) -> _StoreState:
    """Own immutable top-level containers while sharing immutable records."""
    return _StoreState(
        observation_patches=_frozen_mapping(observation_patches),
        delete_all_ids=frozenset(delete_all_ids),
        placement_size=_frozen_mapping(placement_size),
        committed_added_ids=frozenset(committed_added_ids),
        provisional_added_ids=frozenset(provisional_added_ids),
        retired_ids=frozenset(retired_ids),
        next_neuron_id=int(next_neuron_id),
        changed_fields=_frozen_mapping(changed_fields),
    )


def _state_to_json(state: _StoreState) -> dict[str, Any]:
    """Return the established detached JSON-dictionary state interface."""
    return {
        "observation_patches": [
            _patch_to_json(volume_index, neuron_id, patch)
            for (volume_index, neuron_id), patch in sorted(
                state.observation_patches.items()
            )
        ],
        "delete_all_ids": sorted(state.delete_all_ids),
        "placement_size": {
            str(neuron_id): [float(value) for value in size]
            for neuron_id, size in sorted(state.placement_size.items())
        },
        "committed_added_ids": sorted(state.committed_added_ids),
        "provisional_added_ids": sorted(state.provisional_added_ids),
        "retired_ids": sorted(state.retired_ids),
        "next_neuron_id": int(state.next_neuron_id),
    }


@dataclass(frozen=True)
class _RecoveryCapture:
    """Store-independent immutable input for recovery serialization."""

    raw_shape: tuple[int, ...]
    raw_dtype: str
    z_divisor: float
    image_signature: Any
    formal_path: str | None
    formal_fingerprint: str | None
    working_state: _StoreState
    saved_state: _StoreState

    def payload(
        self,
        *,
        session_uuid: str,
        revision: int,
        raw_sha256: str,
        utc_time: str,
    ) -> dict[str, Any]:
        return {
            "recovery_schema_version": RECOVERY_SCHEMA_VERSION,
            "session_uuid": session_uuid,
            "utc_time": utc_time,
            "revision": int(revision),
            "raw": {
                "shape": list(self.raw_shape),
                "dtype": self.raw_dtype,
                "z_divisor": self.z_divisor,
                "sha256": raw_sha256,
            },
            "image_signature": copy.deepcopy(self.image_signature),
            "formal": {
                "path": self.formal_path,
                "sha256": self.formal_fingerprint,
            },
            "working_state": _state_to_json(self.working_state),
            "saved_state": _state_to_json(self.saved_state),
        }


@dataclass(frozen=True)
class _SavePlan:
    payload: dict[str, Any]
    committed_state: _StoreState


def _ordered_changed_fields(value: Any) -> tuple[str, ...]:
    """Validate and canonicalize a v2 ``changed_fields`` list."""
    if not isinstance(value, list) or not value:
        raise SidecarError("changed_fields must be a non-empty list")
    if any(not isinstance(item, str) for item in value):
        raise SidecarError("changed_fields entries must be strings")
    if len(set(value)) != len(value):
        raise SidecarError("changed_fields must not contain duplicates")
    if any(item not in _CHANGED_FIELDS for item in value):
        raise SidecarError("unknown changed_fields entry")
    expected_order = [item for item in _CHANGED_FIELD_ORDER if item in value]
    if value != expected_order:
        raise SidecarError("changed_fields are not in canonical order")
    return tuple(value)


def _reject_json_constants(value: str) -> Any:
    raise SidecarError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SidecarError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_fields(
    value: dict[str, Any],
    name: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    """Enforce one exact versioned sidecar object schema."""
    optional = set() if optional is None else optional
    missing = required - value.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise SidecarError(f"{name} is missing required field(s): {fields}")
    unknown = value.keys() - required - optional
    if unknown:
        fields = ", ".join(sorted(unknown))
        raise SidecarError(f"{name} contains unknown field(s): {fields}")


def _json_triplet(value: Any, name: str) -> tuple[float, float, float]:
    """Validate a JSON vector without coercing strings or other containers."""
    if not isinstance(value, list) or len(value) != 3:
        raise SidecarError(f"{name} must be a three-number list")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise SidecarError(f"{name} must be a three-number list")
        try:
            number = float(item)
        except OverflowError as exc:
            raise SidecarError(f"{name} values must be finite") from exc
        if not math.isfinite(number):
            raise SidecarError(f"{name} values must be finite")
        numbers.append(number)
    return (numbers[0], numbers[1], numbers[2])


def _json_size(value: Any, name: str) -> tuple[float, float, float]:
    size = _json_triplet(value, name)
    if any(number <= 0 for number in size):
        raise SidecarError(f"{name} values must be positive")
    return size


class ProofreadStore:
    """Sparse, transactional proofreading state over a raw ROI dataset."""

    def __init__(
        self,
        dataset: NeuronBoxDataset,
        *,
        image_signature: Any | None = None,
    ) -> None:
        if not isinstance(dataset, NeuronBoxDataset):
            raise TypeError("dataset must be a NeuronBoxDataset")
        self.dataset = dataset
        self.image_signature = _canonical_json_value(image_signature)
        self._observation_patches: dict[
            tuple[int, int], ObservationPatch
        ] = {}
        self._delete_all_ids: set[int] = set()
        self._placement_size: dict[int, tuple[float, float, float]] = {}
        self._committed_added_ids: set[int] = set()
        self._provisional_added_ids: set[int] = set()
        self._retired_ids: set[int] = set()
        self._next_neuron_id = self.dataset.raw_N
        self._revision = 0
        self._baseline_revision = 0

        self._changed_fields: dict[tuple[int, int], tuple[str, ...]] = {}
        self._field_observations = {
            field: set() for field in _CHANGED_FIELD_ORDER
        }
        self._changed_keys_by_neuron: dict[
            int, set[tuple[int, int]]
        ] = {}
        self._patch_keys_by_neuron: dict[
            int, set[tuple[int, int]]
        ] = {}
        self._raw_presence_cache: dict[int, frozenset[int]] = {}
        self._effective_presence_cache: dict[int, frozenset[int]] = {}

        self._dirty_patch_keys: set[tuple[int, int]] = set()
        self._dirty_delete_all_ids: set[int] = set()
        self._dirty_placement_ids: set[int] = set()
        self._dirty_committed_ids: set[int] = set()
        self._dirty_provisional_ids: set[int] = set()
        self._dirty_retired_ids: set[int] = set()
        self._allocator_dirty = False
        self._status = ProofreadStatus(False, 0, 0, 0)

        self._bound_sidecar_path: Path | None = None
        self._bound_sidecar_fingerprint: str | None = None
        self.last_history_warning: str | None = None
        self._saved_state = self._capture_state()
        self._saved_patch_keys_by_neuron: dict[
            int, set[tuple[int, int]]
        ] = {}

    # ------------------------------------------------------------------
    # Identity and raw-data helpers
    # ------------------------------------------------------------------
    @property
    def raw_T(self) -> int:
        return self.dataset.raw_T

    @property
    def raw_N(self) -> int:
        return self.dataset.raw_N

    @property
    def observation_patches(
        self,
    ) -> Mapping[tuple[int, int], ObservationPatch]:
        return MappingProxyType(self._observation_patches)

    @property
    def delete_all_ids(self) -> frozenset[int]:
        return frozenset(self._delete_all_ids)

    @property
    def placement_size(
        self,
    ) -> Mapping[int, tuple[float, float, float]]:
        return MappingProxyType(self._placement_size)

    @property
    def committed_added_ids(self) -> frozenset[int]:
        return frozenset(self._committed_added_ids)

    @property
    def provisional_added_ids(self) -> frozenset[int]:
        return frozenset(self._provisional_added_ids)

    @property
    def retired_ids(self) -> frozenset[int]:
        return frozenset(self._retired_ids)

    @property
    def next_neuron_id(self) -> int:
        return self._next_neuron_id

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def baseline_revision(self) -> int:
        return self._baseline_revision

    @property
    def status(self) -> ProofreadStatus:
        return self._status

    @property
    def neuron_ids(self) -> list[int]:
        """Current non-retired identities in deterministic order."""
        ids = set(range(self.raw_N))
        ids.update(self._committed_added_ids)
        ids.update(self._provisional_added_ids)
        ids.difference_update(self._retired_ids)
        return sorted(ids)

    @property
    def all_neuron_ids(self) -> list[int]:
        ids = set(range(self.raw_N))
        ids.update(self._committed_added_ids)
        ids.update(self._provisional_added_ids)
        ids.update(self._retired_ids)
        return sorted(ids)

    @property
    def identity_state(self) -> dict[int, str]:
        states = dict.fromkeys(range(self.raw_N), "raw")
        states.update(dict.fromkeys(self._committed_added_ids, "committed_added"))
        states.update(dict.fromkeys(self._provisional_added_ids, "provisional_added"))
        states.update(dict.fromkeys(self._retired_ids, "retired"))
        return states

    @property
    def observation_count(self) -> int:
        return sum(
            len(self.observation_volume_indices(neuron_id))
            for neuron_id in self.neuron_ids
        )

    def _check_volume(self, volume_index: Any) -> int:
        value = _require_int(volume_index, "volume_index")
        if not 0 <= value < self.raw_T:
            raise ValueError(
                f"volume_index must be in [0, {self.raw_T})"
            )
        return value

    def _check_id(self, neuron_id: Any, *, allow_retired: bool = False) -> int:
        value = _require_int(neuron_id, "neuron_id")
        if value < 0 or value not in set(self.all_neuron_ids):
            raise ValueError(f"unknown neuron_id: {value}")
        if not allow_retired and value in self._retired_ids:
            raise ValueError(f"neuron_id is retired: {value}")
        return value

    def _raw_box(self, volume_index: int, neuron_id: int) -> NeuronBox | None:
        if neuron_id >= self.raw_N:
            return None
        return self.dataset.get_box_at_volume_index(volume_index, neuron_id)

    def _capture_state(self) -> _StoreState:
        """Copy only top-level containers and share immutable patch records."""
        return _store_state(
            observation_patches=self._observation_patches,
            delete_all_ids=self._delete_all_ids,
            placement_size=self._placement_size,
            committed_added_ids=self._committed_added_ids,
            provisional_added_ids=self._provisional_added_ids,
            retired_ids=self._retired_ids,
            next_neuron_id=self._next_neuron_id,
            changed_fields=self._changed_fields,
        )

    @staticmethod
    def _patch_index(
        patches: Mapping[tuple[int, int], ObservationPatch],
    ) -> dict[int, set[tuple[int, int]]]:
        result: dict[int, set[tuple[int, int]]] = {}
        for key in patches:
            result.setdefault(key[1], set()).add(key)
        return result

    def _set_saved_state(self, state: _StoreState) -> bool:
        changed = state != self._saved_state
        self._saved_state = state
        self._saved_patch_keys_by_neuron = self._patch_index(
            state.observation_patches
        )
        if changed:
            self._baseline_revision = max(
                self._baseline_revision + 1, self._revision
            )
        return changed

    def _state_equals_working(self, state: _StoreState) -> bool:
        return bool(
            self._observation_patches == state.observation_patches
            and self._delete_all_ids == state.delete_all_ids
            and self._placement_size == state.placement_size
            and self._committed_added_ids == state.committed_added_ids
            and self._provisional_added_ids == state.provisional_added_ids
            and self._retired_ids == state.retired_ids
            and self._next_neuron_id == state.next_neuron_id
        )

    def _install_working_state(
        self, state: _StoreState, *, advance_revision: bool
    ) -> bool:
        changed = not self._state_equals_working(state)
        self._observation_patches = dict(state.observation_patches)
        self._delete_all_ids = set(state.delete_all_ids)
        self._placement_size = dict(state.placement_size)
        self._committed_added_ids = set(state.committed_added_ids)
        self._provisional_added_ids = set(state.provisional_added_ids)
        self._retired_ids = set(state.retired_ids)
        self._next_neuron_id = int(state.next_neuron_id)
        self._patch_keys_by_neuron = self._patch_index(
            self._observation_patches
        )
        self._effective_presence_cache.clear()
        self._replace_all_changed_fields(state.changed_fields)
        if changed and advance_revision:
            self._revision += 1
        return changed

    def _replace_patch(
        self,
        key: tuple[int, int],
        patch: ObservationPatch | None,
    ) -> bool:
        old = self._observation_patches.get(key)
        if old == patch:
            return False
        neuron_id = key[1]
        if patch is None:
            self._observation_patches.pop(key, None)
            keys = self._patch_keys_by_neuron.get(neuron_id)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    self._patch_keys_by_neuron.pop(neuron_id, None)
        else:
            self._observation_patches[key] = patch
            self._patch_keys_by_neuron.setdefault(neuron_id, set()).add(key)
        return True

    def _set_changed_fields(
        self, key: tuple[int, int], fields: tuple[str, ...]
    ) -> None:
        previous = self._changed_fields.get(key, ())
        if previous == fields:
            return
        for field in previous:
            self._field_observations[field].discard(key)
        neuron_keys = self._changed_keys_by_neuron.get(key[1])
        if not fields:
            self._changed_fields.pop(key, None)
            if neuron_keys is not None:
                neuron_keys.discard(key)
                if not neuron_keys:
                    self._changed_keys_by_neuron.pop(key[1], None)
            return
        self._changed_fields[key] = fields
        self._changed_keys_by_neuron.setdefault(key[1], set()).add(key)
        for field in fields:
            self._field_observations[field].add(key)

    def _replace_all_changed_fields(
        self,
        changed_fields: Mapping[tuple[int, int], tuple[str, ...]],
    ) -> None:
        self._changed_fields = dict(changed_fields)
        self._field_observations = {
            field: {
                key
                for key, fields in self._changed_fields.items()
                if field in fields
            }
            for field in _CHANGED_FIELD_ORDER
        }
        self._changed_keys_by_neuron = {}
        for key in self._changed_fields:
            self._changed_keys_by_neuron.setdefault(key[1], set()).add(key)

    def _refresh_observation_changed_fields(
        self, key: tuple[int, int]
    ) -> None:
        volume_index, neuron_id = key
        patch = self._observation_patches.get(key)
        if patch is not None:
            fields = self._changed_fields_for_patch(
                volume_index, neuron_id, patch
            )
        elif (
            neuron_id in self._delete_all_ids
            and self._raw_box(volume_index, neuron_id) is not None
        ):
            fields = ("presence",)
        else:
            fields = ()
        self._set_changed_fields(key, fields)

    def _raw_present_volume_indices(self, neuron_id: int) -> frozenset[int]:
        cached = self._raw_presence_cache.get(neuron_id)
        if cached is not None:
            return cached
        if neuron_id >= self.raw_N:
            result = frozenset()
        else:
            values = np.asarray(
                self.dataset.raw_data[:, neuron_id, :6], dtype=float
            )
            result = frozenset(
                int(value)
                for value in np.flatnonzero(
                    np.all(np.isfinite(values), axis=1)
                )
            )
        self._raw_presence_cache[neuron_id] = result
        return result

    def _refresh_neuron_changed_fields(self, neuron_id: int) -> None:
        for key in tuple(self._changed_keys_by_neuron.get(neuron_id, ())):
            self._set_changed_fields(key, ())
        for key in tuple(self._patch_keys_by_neuron.get(neuron_id, ())):
            patch = self._observation_patches[key]
            self._set_changed_fields(
                key, self._changed_fields_for_patch(*key, patch)
            )
        if neuron_id in self._delete_all_ids:
            for volume_index in self._raw_present_volume_indices(neuron_id):
                key = (volume_index, neuron_id)
                if key not in self._observation_patches:
                    self._set_changed_fields(key, ("presence",))

    @staticmethod
    def _update_difference(
        differences: set[int],
        neuron_id: int,
        different: bool,
    ) -> None:
        if different:
            differences.add(neuron_id)
        else:
            differences.discard(neuron_id)

    def _refresh_dirty_patch(self, key: tuple[int, int]) -> None:
        if self._observation_patches.get(key) == self._saved_state.observation_patches.get(key):
            self._dirty_patch_keys.discard(key)
        else:
            self._dirty_patch_keys.add(key)

    def _refresh_dirty_metadata(self, neuron_id: int) -> None:
        saved = self._saved_state
        self._update_difference(
            self._dirty_delete_all_ids,
            neuron_id,
            (neuron_id in self._delete_all_ids)
            != (neuron_id in saved.delete_all_ids),
        )
        missing = object()
        self._update_difference(
            self._dirty_placement_ids,
            neuron_id,
            self._placement_size.get(neuron_id, missing)
            != saved.placement_size.get(neuron_id, missing),
        )
        for current, baseline, differences in (
            (
                self._committed_added_ids,
                saved.committed_added_ids,
                self._dirty_committed_ids,
            ),
            (
                self._provisional_added_ids,
                saved.provisional_added_ids,
                self._dirty_provisional_ids,
            ),
            (self._retired_ids, saved.retired_ids, self._dirty_retired_ids),
        ):
            self._update_difference(
                differences,
                neuron_id,
                (neuron_id in current) != (neuron_id in baseline),
            )
        self._allocator_dirty = (
            self._next_neuron_id != saved.next_neuron_id
        )

    def _refresh_dirty_neuron(
        self,
        neuron_id: int,
        *,
        extra_patch_keys: set[tuple[int, int]] | None = None,
    ) -> None:
        keys = set(self._patch_keys_by_neuron.get(neuron_id, ()))
        keys.update(self._saved_patch_keys_by_neuron.get(neuron_id, ()))
        if extra_patch_keys:
            keys.update(extra_patch_keys)
        for key in keys:
            self._refresh_dirty_patch(key)
        self._refresh_dirty_metadata(neuron_id)

    def _rebuild_dirty_cache(self) -> None:
        saved = self._saved_state
        self._dirty_patch_keys = {
            key
            for key in set(self._observation_patches)
            | set(saved.observation_patches)
            if self._observation_patches.get(key)
            != saved.observation_patches.get(key)
        }
        self._dirty_delete_all_ids = self._delete_all_ids ^ set(
            saved.delete_all_ids
        )
        placement_ids = set(self._placement_size) | set(saved.placement_size)
        missing = object()
        self._dirty_placement_ids = {
            neuron_id
            for neuron_id in placement_ids
            if self._placement_size.get(neuron_id, missing)
            != saved.placement_size.get(neuron_id, missing)
        }
        self._dirty_committed_ids = self._committed_added_ids ^ set(
            saved.committed_added_ids
        )
        self._dirty_provisional_ids = self._provisional_added_ids ^ set(
            saved.provisional_added_ids
        )
        self._dirty_retired_ids = self._retired_ids ^ set(saved.retired_ids)
        self._allocator_dirty = (
            self._next_neuron_id != saved.next_neuron_id
        )
        self._refresh_status()

    def _is_dirty(self) -> bool:
        return bool(
            self._dirty_patch_keys
            or self._dirty_delete_all_ids
            or self._dirty_placement_ids
            or self._dirty_committed_ids
            or self._dirty_provisional_ids
            or self._dirty_retired_ids
            or self._allocator_dirty
        )

    def _refresh_status(self) -> None:
        self._status = ProofreadStatus(
            dirty=self._is_dirty(),
            moved=len(self._field_observations["center_zyx"]),
            resized=len(self._field_observations["size_zyx"]),
            presence=len(self._field_observations["presence"]),
        )

    def _finish_change(self) -> None:
        self._revision += 1
        self._refresh_status()

    def observation_volume_indices(self, neuron_id: int) -> frozenset[int]:
        """Return cached volumes where one current identity has a box."""
        neuron_id = self._check_id(neuron_id)
        cached = self._effective_presence_cache.get(neuron_id)
        if cached is not None:
            return cached
        present = (
            set()
            if neuron_id in self._delete_all_ids
            else set(self._raw_present_volume_indices(neuron_id))
        )
        for key in self._patch_keys_by_neuron.get(neuron_id, ()):
            if self._observation_patches[key].state == PRESENT:
                present.add(key[0])
            else:
                present.discard(key[0])
        result = frozenset(present)
        self._effective_presence_cache[neuron_id] = result
        return result

    def has_observations(self, neuron_id: int) -> bool:
        """Return whether an identity has any current observation."""
        return bool(self.observation_volume_indices(neuron_id))

    def set_placement_size(
        self, neuron_id: int, size_zyx: Any
    ) -> tuple[float, float, float]:
        """Set one placement template through the versioned state API."""
        neuron_id = self._check_id(neuron_id)
        size = _validate_size(size_zyx)
        if self._placement_size.get(neuron_id) == size:
            return size
        self._placement_size[neuron_id] = size
        self._refresh_dirty_metadata(neuron_id)
        self._finish_change()
        return size

    # ------------------------------------------------------------------
    # Resolver and edit operations
    # ------------------------------------------------------------------
    def resolve(self, volume_index: int, neuron_id: int) -> NeuronBox | None:
        """Resolve one observation using canonical patch precedence."""
        volume_index = self._check_volume(volume_index)
        neuron_id = self._check_id(neuron_id)
        patch = self._observation_patches.get((volume_index, neuron_id))
        if patch is not None:
            if patch.state == PRESENT:
                return patch.box
            return None
        if neuron_id in self._delete_all_ids:
            return None
        return self._raw_box(volume_index, neuron_id)

    def effective_state(self, volume_index: int, neuron_id: int) -> str:
        """Return ``present``, ``deleted``, ``raw`` or ``absent``."""
        volume_index = self._check_volume(volume_index)
        neuron_id = self._check_id(neuron_id)
        patch = self._observation_patches.get((volume_index, neuron_id))
        if patch is not None:
            return patch.state
        if neuron_id in self._delete_all_ids:
            return DELETED
        return RAW if self._raw_box(volume_index, neuron_id) is not None else ABSENT

    def _coerce_box(
        self,
        volume_index: int,
        neuron_id: int,
        box: NeuronBox | None,
        center_zyx: Any | None,
        size_zyx: Any | None,
    ) -> NeuronBox:
        if box is not None:
            if not isinstance(box, NeuronBox):
                raise TypeError("box must be a NeuronBox")
            center = box.center_zyx
            size = box.size_zyx if size_zyx is None else size_zyx
        else:
            if center_zyx is None:
                raise TypeError("center_zyx is required when box is omitted")
            center = center_zyx
            if size_zyx is None:
                size_zyx = self.size_for_placement(
                    neuron_id, volume_index
                )
            size = size_zyx
        return _box_from_parts(neuron_id, volume_index, center, size)

    def set_observation_present(
        self,
        volume_index: int,
        neuron_id: int,
        box: NeuronBox | None = None,
        *,
        center_zyx: Any | None = None,
        size_zyx: Any | None = None,
    ) -> NeuronBox:
        """Insert/replace a PRESENT patch for one observation."""
        volume_index = self._check_volume(volume_index)
        neuron_id = self._check_id(neuron_id)
        new_box = self._coerce_box(
            volume_index, neuron_id, box, center_zyx, size_zyx
        )
        raw_box = self._raw_box(volume_index, neuron_id)
        # A patch that exactly restores raw data is redundant unless it is an
        # explicit exception to a Delete-all marker.
        if (
            neuron_id not in self._delete_all_ids
            and raw_box is not None
            and raw_box.center_zyx == new_box.center_zyx
            and raw_box.size_zyx == new_box.size_zyx
        ):
            new_patch = None
        else:
            new_patch = ObservationPatch.present(new_box)
        key = (volume_index, neuron_id)
        if self._replace_patch(key, new_patch):
            self._effective_presence_cache.pop(neuron_id, None)
            fields = (
                ()
                if new_patch is None
                else self._changed_fields_for_patch(
                    volume_index,
                    neuron_id,
                    new_patch,
                    raw_box=raw_box,
                )
            )
            self._set_changed_fields(key, fields)
            self._refresh_dirty_patch(key)
            self._finish_change()
        return new_box

    # Friendly alias used by callers that avoid database terminology.
    set_present = set_observation_present

    def set_observation_deleted(
        self,
        volume_index: int,
        neuron_id: int,
    ) -> None:
        """Delete one observation, normalizing against Delete-all markers."""
        volume_index = self._check_volume(volume_index)
        neuron_id = self._check_id(neuron_id)
        key = (volume_index, neuron_id)
        if neuron_id in self._delete_all_ids:
            # A PRESENT exception is removed; no redundant DELETED patch is
            # needed because the marker already expresses the deletion.
            if self._replace_patch(key, None):
                self._effective_presence_cache.pop(neuron_id, None)
                self._refresh_observation_changed_fields(key)
                self._refresh_dirty_patch(key)
                self._finish_change()
            return
        existing = self._observation_patches.get(key)
        if existing is not None and existing.state == DELETED:
            return
        raw = (
            self._raw_box(volume_index, neuron_id)
            if existing is None
            else None
        )
        if existing is None and raw is None:
            # Deleting a naturally absent observation is a no-op.  This keeps
            # modified_observations aligned with actual proofreading effects.
            return
        restore_size = None
        if existing is not None and existing.state == PRESENT:
            assert existing.box is not None
            restore_size = existing.box.size_zyx
        else:
            if raw is not None:
                restore_size = raw.size_zyx
            elif neuron_id in self._placement_size:
                restore_size = self._placement_size[neuron_id]
        if self._replace_patch(key, ObservationPatch.deleted(restore_size)):
            self._effective_presence_cache.pop(neuron_id, None)
            self._refresh_observation_changed_fields(key)
            self._refresh_dirty_patch(key)
            self._finish_change()

    set_deleted = set_observation_deleted

    def delete_all_observations(self, neuron_id: int) -> None:
        """Logically remove an identity's observations at every volume."""
        neuron_id = self._check_id(neuron_id)
        old_keys = set(self._patch_keys_by_neuron.get(neuron_id, ()))
        changed = neuron_id not in self._delete_all_ids or bool(old_keys)
        if neuron_id not in self._placement_size:
            # Infer before clearing PRESENT patches or adding the marker;
            # afterwards the resolver would see every observation as absent.
            self._placement_size[neuron_id] = self.size_for_placement(neuron_id)
            changed = True
        for key in old_keys:
            self._replace_patch(key, None)
        self._delete_all_ids.add(neuron_id)
        if not changed:
            return
        self._effective_presence_cache.pop(neuron_id, None)
        self._refresh_neuron_changed_fields(neuron_id)
        self._refresh_dirty_neuron(
            neuron_id, extra_patch_keys=old_keys
        )
        self._finish_change()

    delete_all = delete_all_observations

    def add_neuron(
        self,
        volume_index: int,
        center_zyx: Any,
        *,
        size_zyx: Any | None = None,
    ) -> int:
        """Allocate a provisional identity and place its first observation."""
        volume_index = self._check_volume(volume_index)
        center = _validate_center(center_zyx)
        if size_zyx is None:
            size = DEFAULT_SIZE_ZYX
        else:
            size = _validate_size(size_zyx)
        neuron_id = self._next_neuron_id
        while neuron_id in set(self.all_neuron_ids):
            neuron_id += 1
        self._next_neuron_id = neuron_id + 1
        self._provisional_added_ids.add(neuron_id)
        self._placement_size[neuron_id] = size
        key = (volume_index, neuron_id)
        self._replace_patch(
            key,
            ObservationPatch.present(
                _box_from_parts(neuron_id, volume_index, center, size)
            ),
        )
        self._refresh_observation_changed_fields(key)
        self._refresh_dirty_neuron(neuron_id)
        self._finish_change()
        return neuron_id

    def retire_added_neuron(self, neuron_id: int) -> None:
        """Retire an added identity without renumbering other IDs."""
        neuron_id = self._check_id(neuron_id)
        if neuron_id < self.raw_N:
            raise ValueError("raw neuron IDs cannot be retired")
        old_keys = set(self._patch_keys_by_neuron.get(neuron_id, ()))
        self._provisional_added_ids.discard(neuron_id)
        self._committed_added_ids.discard(neuron_id)
        self._retired_ids.add(neuron_id)
        self._delete_all_ids.discard(neuron_id)
        self._placement_size.pop(neuron_id, None)
        for key in old_keys:
            self._replace_patch(key, None)
        self._effective_presence_cache.pop(neuron_id, None)
        self._refresh_neuron_changed_fields(neuron_id)
        self._refresh_dirty_neuron(
            neuron_id, extra_patch_keys=old_keys
        )
        self._finish_change()

    def size_for_placement(
        self, neuron_id: int, volume_index: int | None = None
    ) -> tuple[float, float, float]:
        """Infer a committed placement size using the documented priority."""
        neuron_id = self._check_id(neuron_id)
        if neuron_id in self._placement_size:
            return self._placement_size[neuron_id]
        if volume_index is not None:
            volume_index = self._check_volume(volume_index)
            patch = self._observation_patches.get((volume_index, neuron_id))
            if patch is not None and patch.restore_size_zyx is not None:
                return patch.restore_size_zyx
            candidates = self.observation_volume_indices(neuron_id)
            candidate = min(
                candidates,
                key=lambda value: (abs(value - volume_index), value),
                default=None,
            )
        else:
            candidate = min(
                self.observation_volume_indices(neuron_id), default=None
            )
        if candidate is not None:
            box = self.resolve(candidate, neuron_id)
            if box is not None:
                return box.size_zyx
        return DEFAULT_SIZE_ZYX

    def apply_size(
        self, neuron_id: int, size_zyx: Any
    ) -> tuple[float, float, float]:
        """Apply a size to every currently valid observation of an ID."""
        neuron_id = self._check_id(neuron_id)
        size = _validate_size(size_zyx)
        # Snapshot the resolved boxes before mutating patches.  Missing
        # observations (including Delete-all volumes) are not created.
        boxes = [
            (volume_index, self.resolve(volume_index, neuron_id))
            for volume_index in sorted(
                self.observation_volume_indices(neuron_id)
            )
        ]
        changed_keys: set[tuple[int, int]] = set()
        changed = self._placement_size.get(neuron_id) != size
        self._placement_size[neuron_id] = size
        for volume_index, box in boxes:
            if box is None:
                continue
            new_box = _box_from_parts(
                neuron_id, volume_index, box.center_zyx, size
            )
            raw_box = self._raw_box(volume_index, neuron_id)
            patch = (
                None
                if neuron_id not in self._delete_all_ids
                and raw_box is not None
                and raw_box.center_zyx == new_box.center_zyx
                and raw_box.size_zyx == new_box.size_zyx
                else ObservationPatch.present(new_box)
            )
            key = (volume_index, neuron_id)
            if self._replace_patch(key, patch):
                changed = True
                changed_keys.add(key)
        if changed:
            self._refresh_neuron_changed_fields(neuron_id)
            self._refresh_dirty_neuron(
                neuron_id, extra_patch_keys=changed_keys
            )
            self._finish_change()
        return size

    def apply_size_at_volume_index(
        self,
        volume_index: int,
        neuron_id: int,
        size_zyx: Any,
    ) -> tuple[float, float, float]:
        """Apply a size to one existing observation.

        The observation must currently resolve to a box.  Its center is
        copied verbatim and no placement template or other volume is changed.
        ``set_observation_present`` performs the normal canonicalization, so
        restoring both raw center and size removes a redundant patch.
        """
        volume_index = self._check_volume(volume_index)
        neuron_id = self._check_id(neuron_id)
        size = _validate_size(size_zyx)
        box = self.resolve(volume_index, neuron_id)
        if box is None:
            raise ValueError(
                f"observation ({volume_index}, {neuron_id}) is missing"
            )
        self.set_observation_present(
            volume_index,
            neuron_id,
            center_zyx=box.center_zyx,
            size_zyx=size,
        )
        return size

    def apply_size_to_all_existing(
        self, neuron_id: int, size_zyx: Any
    ) -> tuple[float, float, float]:
        """Apply a size to existing observations, preserving legacy scope.

        The explicit all-existing operation keeps the historical placement
        template update, but avoids introducing metadata when every existing
        observation already has the requested size (a canonical no-op).
        """
        neuron_id = self._check_id(neuron_id)
        size = _validate_size(size_zyx)
        boxes = [
            self.resolve(volume_index, neuron_id)
            for volume_index in self.observation_volume_indices(neuron_id)
        ]
        if (
            all(box is None or box.size_zyx == size for box in boxes)
            and (
                neuron_id not in self._placement_size
                or self._placement_size[neuron_id] == size
            )
        ):
            return size
        return self.apply_size(neuron_id, size)

    # ------------------------------------------------------------------
    # Derived status and snapshots
    # ------------------------------------------------------------------
    def valid_ids_at_volume_index(self, volume_index: int) -> list[int]:
        volume_index = self._check_volume(volume_index)
        return [
            neuron_id
            for neuron_id in self.neuron_ids
            if self.resolve(volume_index, neuron_id) is not None
        ]

    def valid_ids(self, volume_index: int) -> list[int]:
        return self.valid_ids_at_volume_index(volume_index)

    @property
    def modified_observations(self) -> set[tuple[int, int]]:
        return set(self._changed_fields)

    @property
    def modified_ids(self) -> set[int]:
        return set(self._changed_keys_by_neuron)

    def _changed_fields_for_patch(
        self,
        volume_index: int,
        neuron_id: int,
        patch: ObservationPatch,
        *,
        delete_all_ids: set[int] | None = None,
        raw_box: NeuronBox | None | object = _RAW_BOX_UNSET,
    ) -> tuple[str, ...]:
        """Derive v2's field list from raw geometry and a complete patch.

        The list is metadata only; resolution and NPY export continue to use
        the complete patch box.  Exact tuple equality is intentional here and
        matches the canonical redundant-patch normalization rules.
        """
        markers = (
            self._delete_all_ids
            if delete_all_ids is None
            else delete_all_ids
        )
        if patch.state == DELETED:
            return ("presence",)
        if raw_box is _RAW_BOX_UNSET:
            raw_box = self._raw_box(volume_index, neuron_id)
        assert patch.box is not None
        if raw_box is None:
            return ("presence",)
        fields: list[str] = []
        if neuron_id in markers:
            fields.append("presence")
        if patch.box.center_zyx != raw_box.center_zyx:
            fields.append("center_zyx")
        if patch.box.size_zyx != raw_box.size_zyx:
            fields.append("size_zyx")
        # An exact raw restore is not a real change.  Canonical mutators remove
        # such patches; returning an empty tuple here lets v2 validation reject
        # hand-written redundant records instead of mislabelling them as
        # ``placed``.
        return tuple(fields)

    def changed_fields_for_observation(
        self, volume_index: int, neuron_id: int
    ) -> tuple[str, ...]:
        """Return the canonical v2 field list for one observation."""
        volume_index = self._check_volume(volume_index)
        neuron_id = self._check_id(neuron_id)
        return self._changed_fields.get((volume_index, neuron_id), ())

    # ``observation_change_fields`` is a convenient mapping for UI and
    # downstream consumers; the method above remains useful for point lookup.
    @property
    def observation_change_fields(
        self,
    ) -> dict[tuple[int, int], tuple[str, ...]]:
        return dict(self._changed_fields)

    @property
    def center_changed_observations(self) -> set[tuple[int, int]]:
        return set(self._field_observations["center_zyx"])

    @property
    def size_changed_observations(self) -> set[tuple[int, int]]:
        return set(self._field_observations["size_zyx"])

    @property
    def presence_changed_observations(self) -> set[tuple[int, int]]:
        return set(self._field_observations["presence"])

    # Public/UI-friendly synonyms used by the proofreading panel.
    moved_observations = property(lambda self: self.center_changed_observations)
    resized_observations = property(lambda self: self.size_changed_observations)
    presence_observations = property(
        lambda self: self.presence_changed_observations
    )
    center_observations = property(
        lambda self: self.center_changed_observations
    )
    size_observations = property(lambda self: self.size_changed_observations)

    def classify_observation(self, volume_index: int, neuron_id: int) -> str | None:
        """Return the UI status label for one modified observation."""
        volume_index = self._check_volume(volume_index)
        neuron_id = self._check_id(neuron_id)
        fields = set(self.changed_fields_for_observation(volume_index, neuron_id))
        if not fields:
            return None
        patch = self._observation_patches.get((volume_index, neuron_id))
        has_center = "center_zyx" in fields
        has_size = "size_zyx" in fields
        # Presence plus geometry (e.g. a Delete-all local restoration) keeps
        # the meaningful geometry classification; presence alone maps to the
        # placed/deleted/added labels below.
        if "presence" in fields and not (has_center or has_size):
            # A delete-all marker without an explicit per-volume patch is a
            # presence-only deletion represented by the global operation.
            if patch is None and neuron_id in self._delete_all_ids:
                return "deleted"
            if patch is None:
                return "placed"
            if patch.state == DELETED:
                return "deleted"
            if neuron_id >= self.raw_N:
                return "added"
            return "placed"
        if has_center and has_size:
            return "moved + resized"
        if has_center:
            return "moved"
        if has_size:
            return "resized"
        return None

    observation_classification = classify_observation

    def classify_neuron(self, neuron_id: int) -> set[str]:
        """Return distinct UI status labels across an identity's observations."""
        neuron_id = self._check_id(neuron_id, allow_retired=True)
        return {
            status
            for volume_index, candidate in self._changed_keys_by_neuron.get(
                neuron_id, ()
            )
            for status in [self.classify_observation(volume_index, candidate)]
            if status is not None
        }

    def _canonical_state(self) -> dict[str, Any]:
        return _state_to_json(self._capture_state())

    @property
    def saved_snapshot(self) -> dict[str, Any]:
        return _state_to_json(self._saved_state)

    @property
    def working_snapshot(self) -> dict[str, Any]:
        """Return an independent canonical snapshot for background work."""
        return _state_to_json(self._capture_state())

    @property
    def bound_sidecar_path(self) -> Path | None:
        return self._bound_sidecar_path

    @property
    def bound_sidecar_fingerprint(self) -> str | None:
        return self._bound_sidecar_fingerprint

    def bind_sidecar_baseline(
        self, path: str | Path | None, fingerprint: str | None
    ) -> None:
        """Set a previously validated formal-file binding."""
        if path is None:
            self._bound_sidecar_path = None
            self._bound_sidecar_fingerprint = None
            return
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("a formal-file SHA256 fingerprint is required")
        self._bound_sidecar_path = Path(path)
        self._bound_sidecar_fingerprint = fingerprint

    @property
    def dirty(self) -> bool:
        return self._status.dirty

    def _restore_state(self, state: _StoreState) -> None:
        self._install_working_state(state, advance_revision=False)

    def _validated_canonical_state(
        self, state: Any, *, name: str
    ) -> _StoreState:
        """Validate a complete recovery state without mutating this store."""
        if not isinstance(state, dict):
            raise SidecarError(f"{name} must be an object")
        required = {
            "observation_patches",
            "delete_all_ids",
            "placement_size",
            "committed_added_ids",
            "provisional_added_ids",
            "retired_ids",
            "next_neuron_id",
        }
        _require_fields(state, name, required=required)
        committed = _id_set(state["committed_added_ids"], f"{name} committed")
        provisional = _id_set(
            state["provisional_added_ids"], f"{name} provisional"
        )
        retired = _id_set(state["retired_ids"], f"{name} retired")
        if (committed & provisional) or (committed & retired) or (
            provisional & retired
        ):
            raise SidecarError(f"{name} identity sets overlap")
        added = committed | provisional | retired
        if any(value < self.raw_N for value in added):
            raise SidecarError(f"{name} added IDs overlap raw IDs")
        next_id = state["next_neuron_id"]
        if not _is_int(next_id) or int(next_id) < self.raw_N:
            raise SidecarError(f"{name} next_neuron_id is invalid")
        next_id = int(next_id)
        if len(added) != next_id - self.raw_N or any(v >= next_id for v in added):
            raise SidecarError(f"{name} added neuron lineage is incomplete")
        known = set(range(self.raw_N)) | committed | provisional

        delete_all = _id_set(state["delete_all_ids"], f"{name} delete_all")
        if not delete_all <= known:
            raise SidecarError(f"{name} delete_all references unknown neuron")
        placement_raw = state["placement_size"]
        if not isinstance(placement_raw, dict):
            raise SidecarError(f"{name} placement_size must be an object")
        placement: dict[int, tuple[float, float, float]] = {}
        for key, value in placement_raw.items():
            try:
                neuron_id = int(key)
            except (TypeError, ValueError) as exc:
                raise SidecarError(f"{name} invalid placement neuron ID") from exc
            if str(neuron_id) != key or neuron_id not in known:
                raise SidecarError(f"{name} invalid placement neuron ID")
            placement[neuron_id] = _json_size(
                value, f"{name} placement size"
            )

        records = state["observation_patches"]
        if not isinstance(records, list):
            raise SidecarError(f"{name} observation_patches must be a list")
        patches: dict[tuple[int, int], ObservationPatch] = {}
        changed_fields: dict[tuple[int, int], tuple[str, ...]] = {}
        keys: set[tuple[int, int]] = set()
        for record in records:
            if not isinstance(record, dict):
                raise SidecarError(f"{name} patch must be an object")
            patch_state = record.get("state")
            if patch_state == PRESENT:
                _require_fields(
                    record,
                    f"{name} present patch",
                    required={"volume_index", "neuron_id", "state", "box"},
                )
            elif patch_state == DELETED:
                _require_fields(
                    record,
                    f"{name} deleted patch",
                    required={"volume_index", "neuron_id", "state"},
                    optional={"restore_size_zyx"},
                )
            else:
                raise SidecarError(f"{name} patch state is invalid")
            try:
                volume_index = _require_int(record["volume_index"], "volume_index")
                neuron_id = _require_int(record["neuron_id"], "neuron_id")
            except TypeError as exc:
                raise SidecarError(f"{name} patch indices are invalid") from exc
            key = (volume_index, neuron_id)
            if not 0 <= volume_index < self.raw_T or neuron_id not in known:
                raise SidecarError(f"{name} patch references invalid observation")
            if key in keys:
                raise SidecarError(f"{name} contains duplicate patch")
            if patch_state == DELETED and neuron_id in delete_all:
                raise SidecarError(f"{name} deleted patch conflicts with delete_all")
            keys.add(key)
            if patch_state == PRESENT:
                box_data = record["box"]
                if not isinstance(box_data, dict):
                    raise SidecarError(f"{name} patch box must be an object")
                _require_fields(
                    box_data,
                    f"{name} patch box",
                    required={"center_zyx", "size_zyx"},
                )
                box = _box_from_parts(
                    neuron_id,
                    volume_index,
                    _json_triplet(box_data["center_zyx"], "box center_zyx"),
                    _json_size(box_data["size_zyx"], "box size_zyx"),
                )
                patch = ObservationPatch.present(box)
            else:
                restore = (
                    _json_size(record["restore_size_zyx"], "restore size")
                    if "restore_size_zyx" in record
                    else None
                )
                patch = ObservationPatch.deleted(restore)
            patches[key] = patch
            fields = self._changed_fields_for_patch(
                volume_index,
                neuron_id,
                patch,
                delete_all_ids=delete_all,
            )
            if fields:
                changed_fields[key] = fields

        for neuron_id in delete_all:
            for volume_index in self._raw_present_volume_indices(neuron_id):
                key = (volume_index, neuron_id)
                if key not in patches:
                    changed_fields[key] = ("presence",)

        return _store_state(
            observation_patches=patches,
            delete_all_ids=delete_all,
            placement_size=placement,
            committed_added_ids=committed,
            provisional_added_ids=provisional,
            retired_ids=retired,
            next_neuron_id=next_id,
            changed_fields=changed_fields,
        )

    def capture_recovery_state(
        self,
        *,
        formal_path: str | Path | None = None,
        formal_fingerprint: str | None = None,
    ) -> _RecoveryCapture:
        """Capture immutable state for a worker without JSON construction."""
        bound_path = (
            self._bound_sidecar_path
            if formal_path is None
            else Path(formal_path)
        )
        bound_fingerprint = (
            self._bound_sidecar_fingerprint
            if formal_fingerprint is None
            else formal_fingerprint
        )
        return _RecoveryCapture(
            raw_shape=self.dataset.raw_shape,
            raw_dtype=self.dataset.raw_dtype.str,
            z_divisor=float(self.dataset.z_divisor),
            image_signature=copy.deepcopy(self.image_signature),
            formal_path=None if bound_path is None else str(bound_path),
            formal_fingerprint=bound_fingerprint,
            working_state=self._capture_state(),
            saved_state=self._saved_state,
        )

    def recovery_payload(
        self,
        *,
        session_uuid: str,
        revision: int,
        raw_sha256: str,
        utc_time: str,
        formal_path: str | Path | None = None,
        formal_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Capture complete working and baseline state for crash recovery."""
        if not isinstance(session_uuid, str) or not session_uuid:
            raise ValueError("session_uuid is required")
        if not _is_int(revision) or int(revision) < 1:
            raise ValueError("revision must be a positive integer")
        if not isinstance(raw_sha256, str) or len(raw_sha256) != 64:
            raise ValueError("raw_sha256 must be a SHA256 hex digest")
        capture = self.capture_recovery_state(
            formal_path=formal_path,
            formal_fingerprint=formal_fingerprint,
        )
        return capture.payload(
            session_uuid=session_uuid,
            revision=revision,
            raw_sha256=raw_sha256,
            utc_time=utc_time,
        )

    def restore_recovery_payload(
        self, payload: Any, *, raw_sha256: str | None = None
    ) -> tuple[Path | None, str | None]:
        """Validate a recovery completely, then replace this store's state."""
        try:
            validate_recovery_envelope(payload)
        except ValueError as exc:
            raise SidecarError(str(exc)) from exc
        raw = payload.get("raw")
        if not isinstance(raw, dict) or set(raw) != {
            "shape",
            "dtype",
            "z_divisor",
            "sha256",
        }:
            raise SidecarError("recovery raw metadata is invalid")
        if (
            not isinstance(raw["shape"], list)
            or any(not _is_int(value) for value in raw["shape"])
            or tuple(raw["shape"]) != self.dataset.raw_shape
        ):
            raise SidecarError("raw shape does not match dataset")
        if raw["dtype"] != self.dataset.raw_dtype.str:
            raise SidecarError("raw dtype does not match dataset")
        if (
            isinstance(raw["z_divisor"], bool)
            or not isinstance(raw["z_divisor"], int | float)
            or raw["z_divisor"] != self.dataset.z_divisor
        ):
            raise SidecarError("raw z_divisor does not match dataset")
        actual_hash = self._raw_fingerprint() if raw_sha256 is None else raw_sha256
        if raw["sha256"] != actual_hash:
            raise SidecarError("raw fingerprint does not match dataset")
        try:
            image_signature = _canonical_json_value(payload["image_signature"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SidecarError("invalid image signature") from exc
        if image_signature != self.image_signature:
            raise SidecarError("image signature does not match dataset")
        working = self._validated_canonical_state(
            payload.get("working_state"), name="recovery working_state"
        )
        saved = self._validated_canonical_state(
            payload.get("saved_state"), name="recovery saved_state"
        )
        saved_ids = set(saved.committed_added_ids) | set(saved.retired_ids)
        working_ids = set(working.committed_added_ids) | set(
            working.retired_ids
        )
        if (
            saved.provisional_added_ids
            or not saved_ids <= working_ids
            or not saved.retired_ids <= working.retired_ids
        ):
            raise SidecarError("recovery state changes committed identity lineage")
        formal = payload.get("formal")
        if not isinstance(formal, dict) or set(formal) != {"path", "sha256"}:
            raise SidecarError("recovery formal baseline is invalid")
        formal_path = formal["path"]
        formal_hash = formal["sha256"]
        if formal_path is not None and not isinstance(formal_path, str):
            raise SidecarError("recovery formal path is invalid")
        if formal_hash is not None and (
            not isinstance(formal_hash, str) or len(formal_hash) != 64
        ):
            raise SidecarError("recovery formal fingerprint is invalid")
        # No live mutation occurs until every field and both states validate.
        working_changed = not self._state_equals_working(working)
        self._set_saved_state(saved)
        self._install_working_state(working, advance_revision=False)
        if working_changed:
            self._revision += 1
        self._rebuild_dirty_cache()
        self._bound_sidecar_path = None
        self._bound_sidecar_fingerprint = None
        return (Path(formal_path) if formal_path is not None else None, formal_hash)

    def discard(self) -> None:
        """Restore the most recently saved/loaded canonical snapshot."""
        if self._state_equals_working(self._saved_state):
            return
        self._install_working_state(
            self._saved_state, advance_revision=True
        )
        self._rebuild_dirty_cache()

    def _state_neuron_ids(self, state: _StoreState) -> set[int]:
        ids = set(range(self.raw_N))
        ids.update(state.committed_added_ids)
        ids.update(state.provisional_added_ids)
        ids.difference_update(state.retired_ids)
        return ids

    def _resolve_state(
        self,
        state: _StoreState,
        volume_index: int,
        neuron_id: int,
    ) -> NeuronBox | None:
        patch = state.observation_patches.get((volume_index, neuron_id))
        if patch is not None:
            return patch.box if patch.state == PRESENT else None
        if neuron_id in state.delete_all_ids:
            return None
        return self._raw_box(volume_index, neuron_id)

    def _neuron_signature(self, neuron_id: int) -> tuple[Any, ...]:
        keys = self._patch_keys_by_neuron.get(neuron_id, ())
        return (
            tuple(
                sorted(
                    (key, self._observation_patches[key]) for key in keys
                )
            ),
            neuron_id in self._delete_all_ids,
            self._placement_size.get(neuron_id),
            neuron_id in self._committed_added_ids,
            neuron_id in self._provisional_added_ids,
            neuron_id in self._retired_ids,
            self._next_neuron_id,
        )

    def _expand_delete_all(
        self,
        neuron_id: int,
        *,
        saved: _StoreState | None = None,
    ) -> None:
        """Replace one delete-all marker with equivalent per-volume patches."""
        if neuron_id not in self._delete_all_ids:
            return
        resolved = [
            self.resolve(volume_index, neuron_id)
            for volume_index in range(self.raw_T)
        ]
        for key in tuple(self._patch_keys_by_neuron.get(neuron_id, ())):
            self._replace_patch(key, None)
        self._delete_all_ids.remove(neuron_id)
        saved_ids = set() if saved is None else self._state_neuron_ids(saved)
        for volume_index, box in enumerate(resolved):
            if box is not None:
                raw_box = self._raw_box(volume_index, neuron_id)
                patch = (
                    None
                    if raw_box is not None
                    and raw_box.center_zyx == box.center_zyx
                    and raw_box.size_zyx == box.size_zyx
                    else ObservationPatch.present(box)
                )
                self._replace_patch((volume_index, neuron_id), patch)
            elif self._raw_box(volume_index, neuron_id) is not None:
                self._replace_patch(
                    (volume_index, neuron_id), ObservationPatch.deleted()
                )
            elif saved is not None and neuron_id in saved_ids:
                saved_box = self._resolve_state(
                    saved, volume_index, neuron_id
                )
                if saved_box is not None:
                    # Added-neuron observations have no raw box. Keep their
                    # deletion explicit so status and sidecar output retain
                    # the unsaved presence change at every other volume.
                    self._replace_patch(
                        (volume_index, neuron_id),
                        ObservationPatch.deleted(saved_box.size_zyx),
                    )

    def discard_observation(self, volume_index: int, neuron_id: int) -> bool:
        """Restore one observation to the most recent saved/loaded snapshot.

        A working delete-all marker is expanded when the saved snapshot does
        not contain that marker. This preserves deletions at every other
        volume while allowing the requested observation to be restored.
        """
        volume_index = self._check_volume(volume_index)
        neuron_id = self._check_id(neuron_id)
        before = self._neuron_signature(neuron_id)
        old_keys = set(self._patch_keys_by_neuron.get(neuron_id, ()))
        saved = self._saved_state
        saved_known = neuron_id in self._state_neuron_ids(saved)
        saved_delete_all = saved_known and neuron_id in saved.delete_all_ids

        if neuron_id in self._delete_all_ids and not saved_delete_all:
            self._expand_delete_all(neuron_id, saved=saved)

        key = (volume_index, neuron_id)
        current_delete_all = neuron_id in self._delete_all_ids
        if saved_known and current_delete_all == saved_delete_all:
            patch = saved.observation_patches.get(key)
            self._replace_patch(key, patch)
        else:
            self._replace_patch(key, None)
            saved_box = (
                self._resolve_state(saved, volume_index, neuron_id)
                if saved_known
                else None
            )
            if saved_box is not None:
                raw_box = self._raw_box(volume_index, neuron_id)
                patch = (
                    None
                    if neuron_id not in self._delete_all_ids
                    and raw_box is not None
                    and raw_box.center_zyx == saved_box.center_zyx
                    and raw_box.size_zyx == saved_box.size_zyx
                    else ObservationPatch.present(saved_box)
                )
                self._replace_patch(key, patch)
            elif self._raw_box(volume_index, neuron_id) is not None:
                self._replace_patch(key, ObservationPatch.deleted())
        if self._neuron_signature(neuron_id) == before:
            return False
        self._effective_presence_cache.pop(neuron_id, None)
        self._refresh_neuron_changed_fields(neuron_id)
        self._refresh_dirty_neuron(
            neuron_id, extra_patch_keys=old_keys | {key}
        )
        self._finish_change()
        return True

    def discard_neuron(self, neuron_id: int) -> bool:
        """Restore all state for one neuron from the saved/loaded snapshot."""
        neuron_id = self._check_id(neuron_id, allow_retired=True)
        before = self._neuron_signature(neuron_id)
        saved = self._saved_state
        old_keys = set(self._patch_keys_by_neuron.get(neuron_id, ()))

        for key in old_keys:
            self._replace_patch(key, None)
        for key in self._saved_patch_keys_by_neuron.get(neuron_id, ()):
            self._replace_patch(key, saved.observation_patches[key])

        for current_values, saved_values in (
            (self._delete_all_ids, saved.delete_all_ids),
            (self._committed_added_ids, saved.committed_added_ids),
            (self._provisional_added_ids, saved.provisional_added_ids),
            (self._retired_ids, saved.retired_ids),
        ):
            current_values.discard(neuron_id)
            if neuron_id in saved_values:
                current_values.add(neuron_id)

        self._placement_size.pop(neuron_id, None)
        if neuron_id in saved.placement_size:
            self._placement_size[neuron_id] = saved.placement_size[neuron_id]

        self._next_neuron_id = max(
            saved.next_neuron_id,
            max(self.all_neuron_ids, default=-1) + 1,
        )
        if self._neuron_signature(neuron_id) == before:
            return False
        self._effective_presence_cache.pop(neuron_id, None)
        self._refresh_neuron_changed_fields(neuron_id)
        self._refresh_dirty_neuron(
            neuron_id, extra_patch_keys=old_keys
        )
        self._finish_change()
        return True

    # ------------------------------------------------------------------
    # Sidecar metadata and persistence
    # ------------------------------------------------------------------
    def _raw_fingerprint(self) -> str:
        if self.dataset.path is not None and self.dataset.path.exists():
            digest = hashlib.sha256()
            with self.dataset.path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.dataset.raw_data).tobytes())
        return digest.hexdigest()

    def _prepare_save(self) -> _SavePlan:
        # Provisional identities with at least one PRESENT patch become
        # committed on the successful save.  Empty provisional identities are
        # omitted from the persisted lineage.
        provisional = set(self._provisional_added_ids)
        present_ids = {
            neuron_id
            for (_volume_index, neuron_id), patch in (
                self._observation_patches.items()
            )
            if patch.state == PRESENT
        }
        committed = set(self._committed_added_ids)
        committed.update(provisional & present_ids)
        empty = provisional - present_ids
        # Saving establishes an identity lineage even if the provisional ID
        # has no remaining observation.  Reserve it as retired so a later
        # load cannot reuse the numeric ID.
        retired = set(self._retired_ids) | empty
        patches = {
            key: patch
            for key, patch in self._observation_patches.items()
            if key[1] not in empty and self._changed_fields.get(key)
        }
        delete_all = self._delete_all_ids - empty
        placement = {
            neuron_id: size
            for neuron_id, size in self._placement_size.items()
            if neuron_id not in empty
        }
        changed_fields = {
            key: fields
            for key, fields in self._changed_fields.items()
            if key[1] not in empty
        }
        next_neuron_id = (
            max(self.raw_N - 1, *committed, *retired) + 1
            if committed or retired
            else max(self.raw_N, self._next_neuron_id)
        )
        state = _store_state(
            observation_patches=patches,
            delete_all_ids=delete_all,
            placement_size=placement,
            committed_added_ids=committed,
            provisional_added_ids=set(),
            retired_ids=retired,
            next_neuron_id=next_neuron_id,
            changed_fields=changed_fields,
        )
        # v2 records carry an ordered field classification derived from the
        # raw geometry and the complete patch.  Keep the box itself as the
        # sole geometry authority; ``changed_fields`` is descriptive only.
        v2_patches = [
            _patch_to_json(
                volume_index,
                neuron_id,
                patch,
                changed_fields=state.changed_fields[(volume_index, neuron_id)],
            )
            for (volume_index, neuron_id), patch in sorted(
                state.observation_patches.items()
            )
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "raw": {
                "shape": list(self.dataset.raw_shape),
                "dtype": self.dataset.raw_dtype.str,
                "z_divisor": float(self.dataset.z_divisor),
                "sha256": self._raw_fingerprint(),
            },
            "image_signature": copy.deepcopy(self.image_signature),
            "observation_patches": v2_patches,
            "delete_all_ids": sorted(state.delete_all_ids),
            "placement_size": {
                str(neuron_id): [float(value) for value in size]
                for neuron_id, size in sorted(state.placement_size.items())
            },
            "added_neurons": {
                "committed": sorted(state.committed_added_ids),
                "retired": sorted(state.retired_ids),
            },
        }
        return _SavePlan(payload=payload, committed_state=state)

    def _payload_for_save(self) -> dict[str, Any]:
        """Return the public-compatible JSON payload used by save."""
        return self._prepare_save().payload

    def _commit_saved_state(self, state: _StoreState) -> None:
        changed = self._install_working_state(
            state, advance_revision=False
        )
        if changed:
            self._revision += 1
        self._set_saved_state(state)
        self._rebuild_dirty_cache()

    def save(self, path: str | Path | None = None) -> Path:
        """Atomically save canonical edits, preserving prior exact bytes."""
        target = self._sidecar_path(path)
        plan = self._prepare_save()
        payload = plan.payload
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        output = text.encode("utf-8")
        existing: bytes | None = None
        if target.exists():
            existing = target.read_bytes()

        same_bound = False
        if self._bound_sidecar_path is not None:
            with contextlib.suppress(OSError, ValueError):
                same_bound = (
                    target.resolve() == self._bound_sidecar_path.resolve()
                )
        if (
            same_bound
            and self._bound_sidecar_fingerprint is not None
            and (
                existing is None
                or sha256_bytes(existing) != self._bound_sidecar_fingerprint
            )
        ):
            raise ExternalSidecarChangeError(
                "bound proof sidecar changed externally; use Save As"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        if existing != output:
            # History preservation is a prerequisite for replacing existing
            # formal bytes.  A failure here leaves the primary untouched.
            if existing is not None:
                backup_formal_bytes(target, existing)
            fd, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(output)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
                raise
        self._commit_saved_state(plan.committed_state)
        self._bound_sidecar_path = target
        self._bound_sidecar_fingerprint = sha256_bytes(output)
        try:
            self.last_history_warning = trim_history(target)
        except OSError as exc:
            self.last_history_warning = f"History cleanup failed: {exc}"
        return target

    save_as = save

    def _sidecar_path(self, path: str | Path | None) -> Path:
        if path is None:
            if self._bound_sidecar_path is not None:
                target = self._bound_sidecar_path
            elif self.dataset.path is None:
                raise ValueError(
                    "a sidecar path is required for an in-memory dataset"
                )
            else:
                target = self.dataset.path.with_suffix(".proofread.json")
        else:
            target = Path(path)
        if is_history_version(target):
            raise ValueError("history snapshots are read-only; use Save As")
        if self.dataset.path is not None:
            try:
                if target.resolve() == self.dataset.path.resolve():
                    raise ValueError("sidecar cannot overwrite the raw NPY")
            except FileNotFoundError:
                pass
        return target

    @classmethod
    def from_sidecar(
        cls,
        path: str | Path,
        dataset: NeuronBoxDataset,
        *,
        image_signature: Any | None = None,
    ) -> ProofreadStore:
        store = cls(dataset, image_signature=image_signature)
        store.load(path)
        return store

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        dataset: NeuronBoxDataset,
        *,
        image_signature: Any | None = None,
    ) -> ProofreadStore:
        return cls.from_sidecar(
            path, dataset, image_signature=image_signature
        )

    def load(self, path: str | Path) -> None:
        """Replace working state from a validated sidecar transactionally."""
        source = Path(path)
        try:
            source_bytes = source.read_bytes()
            payload = json.loads(
                source_bytes.decode("utf-8"),
                parse_constant=_reject_json_constants,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except SidecarError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SidecarError(f"cannot load proof sidecar: {exc}") from exc
        state = self._validate_payload(payload)
        parsed_fingerprint = sha256_bytes(source_bytes)
        try:
            if fingerprint_file(source) != parsed_fingerprint:
                raise SidecarError("proof sidecar changed while loading")
        except OSError as exc:
            raise SidecarError(f"proof sidecar changed while loading: {exc}") from exc
        # No mutation has happened before this point.
        working_changed = not self._state_equals_working(state)
        self._install_working_state(state, advance_revision=False)
        self._set_saved_state(state)
        if working_changed:
            self._revision += 1
        self._rebuild_dirty_cache()
        self._bound_sidecar_path = source
        self._bound_sidecar_fingerprint = parsed_fingerprint

    load_sidecar = load

    def load_history_as_working(self, path: str | Path) -> bool:
        """Load an older formal version as working state over current baseline."""
        historical = ProofreadStore(
            self.dataset, image_signature=copy.deepcopy(self.image_signature)
        )
        historical.load(path)
        return self._restore_history_state(historical._capture_state())

    def restore_history_snapshot(self, snapshot: dict[str, Any]) -> bool:
        """Apply a staged history version after resolving pending GUI edits."""
        working = self._validated_canonical_state(snapshot, name="history state")
        return self._restore_history_state(working)

    def _restore_history_state(self, working: _StoreState) -> bool:
        if working.provisional_added_ids:
            raise SidecarError("history must not contain provisional identities")
        baseline = self._saved_state
        bound_path = self._bound_sidecar_path
        bound_fingerprint = self._bound_sidecar_fingerprint

        current_committed = set(baseline.committed_added_ids)
        current_retired = set(baseline.retired_ids)
        historical_committed = set(working.committed_added_ids)
        max_next = max(
            baseline.next_neuron_id, working.next_neuron_id
        )
        lineage = set(range(self.raw_N, max_next))
        committed = (current_committed & historical_committed) - current_retired
        retired = lineage - committed
        patches = {
            key: patch
            for key, patch in working.observation_patches.items()
            if key[1] not in retired
        }
        transformed = _store_state(
            observation_patches=patches,
            delete_all_ids=set(working.delete_all_ids) - retired,
            placement_size={
                neuron_id: size
                for neuron_id, size in working.placement_size.items()
                if neuron_id not in retired
            },
            committed_added_ids=committed,
            provisional_added_ids=set(),
            retired_ids=retired,
            next_neuron_id=max_next,
            changed_fields={
                key: fields
                for key, fields in working.changed_fields.items()
                if key[1] not in retired
            },
        )
        changed = self._install_working_state(
            transformed, advance_revision=False
        )
        if changed:
            self._revision += 1
        self._set_saved_state(baseline)
        self._rebuild_dirty_cache()
        self._bound_sidecar_path = bound_path
        self._bound_sidecar_fingerprint = bound_fingerprint
        return self.dirty

    def _validate_payload(self, payload: Any) -> _StoreState:
        if not isinstance(payload, dict):
            raise SidecarError("sidecar root must be an object")
        schema_version = payload.get("schema_version")
        if not _is_int(schema_version) or schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SidecarError("unsupported schema_version")
        is_v2 = int(schema_version) == 2
        _require_fields(
            payload,
            "sidecar root",
            required={
                "schema_version",
                "raw",
                "observation_patches",
                "delete_all_ids",
                "placement_size",
                "added_neurons",
            },
            optional={"image_signature"},
        )
        raw = payload.get("raw")
        if not isinstance(raw, dict):
            raise SidecarError("sidecar raw metadata is required")
        _require_fields(
            raw,
            "raw metadata",
            required={"shape", "dtype", "z_divisor", "sha256"},
        )
        raw_shape = raw["shape"]
        if (
            not isinstance(raw_shape, list)
            or any(not _is_int(value) for value in raw_shape)
            or tuple(raw_shape) != self.dataset.raw_shape
        ):
            raise SidecarError("raw shape does not match dataset")
        if raw.get("dtype") != self.dataset.raw_dtype.str:
            raise SidecarError("raw dtype does not match dataset")
        raw_z_divisor = raw.get("z_divisor")
        try:
            numeric_z_divisor = float(raw_z_divisor)
        except (TypeError, ValueError, OverflowError):
            numeric_z_divisor = math.nan
        if (
            isinstance(raw_z_divisor, bool)
            or not isinstance(raw_z_divisor, int | float)
            or not math.isfinite(numeric_z_divisor)
            or numeric_z_divisor <= 0
        ):
            raise SidecarError(
                "raw z_divisor must be a positive finite JSON number"
            )
        # JSON integers and floats describe the same numeric value.  Compare
        # their binary64 values exactly: any real divisor change alters the
        # interpreted z center/depth and must not be accepted approximately.
        if numeric_z_divisor != self.dataset.z_divisor:
            raise SidecarError("raw z_divisor does not match dataset")
        fingerprint = raw.get("sha256")
        if not isinstance(fingerprint, str) or fingerprint != self._raw_fingerprint():
            raise SidecarError("raw fingerprint does not match dataset")
        if "image_signature" in payload:
            try:
                incoming_signature = _canonical_json_value(
                    payload["image_signature"]
                )
            except (TypeError, ValueError) as exc:
                raise SidecarError("invalid image signature") from exc
            if incoming_signature != self.image_signature:
                raise SidecarError("image signature does not match dataset")

        patch_records = payload.get("observation_patches", [])
        if not isinstance(patch_records, list):
            raise SidecarError("observation_patches must be a list")
        delete_values = payload.get("delete_all_ids", [])
        if not isinstance(delete_values, list):
            raise SidecarError("delete_all_ids must be a list")
        delete_all_ids = set()
        for value in delete_values:
            try:
                value = _require_int(value, "delete_all_ids entry")
            except TypeError as exc:
                raise SidecarError("invalid delete_all neuron ID") from exc
            if value in delete_all_ids:
                raise SidecarError("duplicate delete_all neuron ID")
            delete_all_ids.add(value)

        added = payload.get("added_neurons", {})
        if not isinstance(added, dict):
            raise SidecarError("added_neurons must be an object")
        _require_fields(
            added,
            "added_neurons",
            required={"committed", "retired"},
        )
        committed = _id_set(added.get("committed", []), "committed")
        retired = _id_set(added.get("retired", []), "retired")
        if committed & retired:
            raise SidecarError("an added ID cannot be both committed and retired")
        if any(value < self.raw_N for value in committed | retired):
            raise SidecarError("added IDs overlap raw IDs")
        added_lineage = committed | retired
        if added_lineage:
            expected_lineage = set(
                range(self.raw_N, max(added_lineage) + 1)
            )
            if added_lineage != expected_lineage:
                raise SidecarError(
                    "added neuron lineage contains an unreserved ID gap"
                )

        placement_raw = payload.get("placement_size", {})
        if not isinstance(placement_raw, dict):
            raise SidecarError("placement_size must be an object")
        placement_size: dict[int, tuple[float, float, float]] = {}
        for key, value in placement_raw.items():
            try:
                neuron_id = int(key)
            except (TypeError, ValueError) as exc:
                raise SidecarError("invalid placement neuron_id") from exc
            if str(neuron_id) != key or neuron_id < 0:
                raise SidecarError("invalid placement neuron_id")
            if neuron_id not in set(range(self.raw_N)) | committed:
                raise SidecarError("placement references unknown neuron")
            if neuron_id in placement_size:
                raise SidecarError("duplicate placement neuron ID")
            placement_size[neuron_id] = _json_size(
                value, "placement size"
            )

        patches: dict[tuple[int, int], ObservationPatch] = {}
        changed_fields: dict[tuple[int, int], tuple[str, ...]] = {}
        known_ids = set(range(self.raw_N)) | committed
        if not delete_all_ids <= known_ids:
            raise SidecarError("delete_all references unknown neuron")
        for record in patch_records:
            if not isinstance(record, dict):
                raise SidecarError("patch record must be an object")
            state = record.get("state")
            if state == PRESENT:
                _require_fields(
                    record,
                    "present patch",
                    required={
                        "volume_index",
                        "neuron_id",
                        "state",
                        "box",
                        *( {"changed_fields"} if is_v2 else set() ),
                    },
                )
            elif state == DELETED:
                _require_fields(
                    record,
                    "deleted patch",
                    required={
                        "volume_index",
                        "neuron_id",
                        "state",
                        *( {"changed_fields"} if is_v2 else set() ),
                    },
                    optional={"restore_size_zyx"},
                )
            else:
                raise SidecarError("unknown observation patch state")
            try:
                volume_index = _require_int(
                    record["volume_index"], "volume_index"
                )
                neuron_id = _require_int(record["neuron_id"], "neuron_id")
            except TypeError as exc:
                raise SidecarError("invalid patch volume_index/neuron_id") from exc
            if not 0 <= volume_index < self.raw_T:
                raise SidecarError("patch volume_index out of range")
            if neuron_id not in known_ids or neuron_id in retired:
                raise SidecarError("patch references unknown/retired neuron")
            key = (volume_index, neuron_id)
            if key in patches:
                raise SidecarError("duplicate observation patch")
            if state == PRESENT:
                box_data = record.get("box")
                if not isinstance(box_data, dict):
                    raise SidecarError("present patch requires box")
                _require_fields(
                    box_data,
                    "present patch box",
                    required={"center_zyx", "size_zyx"},
                )
                center = _json_triplet(
                    box_data["center_zyx"], "box center_zyx"
                )
                size = _json_size(box_data["size_zyx"], "box size_zyx")
                box = _box_from_parts(
                    neuron_id, volume_index, center, size
                )
                patch = ObservationPatch.present(box)
                if is_v2:
                    fields = _ordered_changed_fields(record.get("changed_fields"))
                    expected = self._changed_fields_for_patch(
                        volume_index,
                        neuron_id,
                        patch,
                        delete_all_ids=delete_all_ids,
                    )
                    if fields != expected:
                        raise SidecarError(
                            "changed_fields do not match raw and patch geometry"
                        )
                    if not expected:
                        raise SidecarError(
                            "changed_fields do not match raw and patch geometry"
                        )
                else:
                    # v1 had no field metadata.  During migration, silently
                    # normalize an exact raw restore just as the mutator does
                    # so the in-memory snapshot contains no redundant patch.
                    expected = self._changed_fields_for_patch(
                        volume_index,
                        neuron_id,
                        patch,
                        delete_all_ids=delete_all_ids,
                    )
                    if not expected:
                        continue
                patches[key] = patch
                changed_fields[key] = expected
            elif state == DELETED:
                if neuron_id in delete_all_ids:
                    raise SidecarError(
                        "deleted patch conflicts with delete_all marker"
                    )
                restore_size = (
                    _json_size(
                        record["restore_size_zyx"], "restore size"
                    )
                    if "restore_size_zyx" in record
                    else None
                )
                patch = ObservationPatch.deleted(restore_size)
                if is_v2:
                    fields = _ordered_changed_fields(record.get("changed_fields"))
                    expected = self._changed_fields_for_patch(
                        volume_index,
                        neuron_id,
                        patch,
                        delete_all_ids=delete_all_ids,
                    )
                    if fields != expected:
                        raise SidecarError(
                            "changed_fields do not match raw and patch geometry"
                        )
                else:
                    expected = self._changed_fields_for_patch(
                        volume_index,
                        neuron_id,
                        patch,
                        delete_all_ids=delete_all_ids,
                    )
                patches[key] = patch
                changed_fields[key] = expected

        # Retired IDs may remain reserved but cannot be active in markers.
        if delete_all_ids & retired:
            raise SidecarError("retired ID cannot have delete_all marker")
        max_id = max(
            [self.raw_N - 1, *committed, *retired], default=self.raw_N - 1
        )
        for neuron_id in delete_all_ids:
            for volume_index in self._raw_present_volume_indices(neuron_id):
                key = (volume_index, neuron_id)
                if key not in patches:
                    changed_fields[key] = ("presence",)
        return _store_state(
            observation_patches=patches,
            delete_all_ids=delete_all_ids,
            placement_size=placement_size,
            committed_added_ids=committed,
            provisional_added_ids=set(),
            retired_ids=retired,
            next_neuron_id=max_id + 1,
            changed_fields=changed_fields,
        )

    # ------------------------------------------------------------------
    # Corrected NPY materialization
    # ------------------------------------------------------------------
    def export_corrected_npy(self, path: str | Path) -> Path:
        """Write a corrected, non-destructive NPY materialization."""
        target = Path(path)
        if self.provisional_added_ids:
            raise ValueError(
                "save proof edits before exporting provisional neuron IDs"
            )
        if self.dataset.path is not None:
            try:
                if target.resolve() == self.dataset.path.resolve():
                    raise ValueError("corrected export cannot overwrite raw NPY")
            except FileNotFoundError:
                pass
        added = sorted(self.committed_added_ids)
        # Numeric IDs are stable array indices.  Retired gaps below a later
        # committed identity remain all-NaN.  A trailing retired ID remains
        # reserved in the sidecar lineage but need not materialize an empty
        # NPY column when no committed identity follows it.
        n_export = max([self.raw_N - 1, *added], default=self.raw_N - 1) + 1
        raw_dtype = self.dataset.raw_dtype
        if np.issubdtype(raw_dtype, np.floating):
            output_dtype = raw_dtype
        else:
            output_dtype = np.dtype(np.float64)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(fd)
        result: np.memmap | None = None
        try:
            result = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=output_dtype,
                shape=(
                    self.raw_T,
                    n_export,
                    self.dataset.raw_shape[2],
                ),
            )
            # Copy one complete volume at a time.  This bounds working memory
            # independently of T and preserves uninterpreted K > 6 fields for
            # every raw identity.
            raw_data = self.dataset.raw_data
            for volume_index in range(self.raw_T):
                result[volume_index, :, :] = np.nan
                result[volume_index, : self.raw_N, :] = raw_data[volume_index]

            for neuron_id in self.delete_all_ids:
                if 0 <= neuron_id < n_export:
                    result[:, neuron_id, :6] = np.nan

            for (volume_index, neuron_id), patch in sorted(
                self.observation_patches.items()
            ):
                if not 0 <= neuron_id < n_export:
                    continue
                if patch.state == DELETED:
                    result[volume_index, neuron_id, :6] = np.nan
                    continue
                assert patch.box is not None
                box = patch.box
                z, y, x = box.center_zyx
                depth, height, width = box.size_zyx
                result[volume_index, neuron_id, :6] = (
                    x,
                    y,
                    z * self.dataset.z_divisor,
                    width,
                    height,
                    depth * self.dataset.z_divisor,
                )

            result.flush()
            # Drop the mapping before replace, which is required on Windows.
            del result
            result = None
            with open(temporary, "rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except Exception:
            # If construction failed while the mapping was still live, drop
            # the local reference before trying to unlink its backing file.
            if result is not None:
                del result
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        return target


def _id_set(values: Any, name: str) -> set[int]:
    if not isinstance(values, list):
        raise SidecarError(f"{name} must be a list")
    result: set[int] = set()
    for value in values:
        try:
            value = _require_int(value, f"{name} ID")
        except TypeError as exc:
            raise SidecarError(f"invalid {name} ID") from exc
        if value < 0 or value in result:
            raise SidecarError(f"invalid or duplicate {name} ID")
        result.add(value)
    return result


def _canonical_json_value(value: Any) -> Any:
    """Normalize optional metadata and reject non-finite values."""
    if value is None or isinstance(value, str | bool | int | np.integer):
        if isinstance(value, np.integer):
            return int(value)
        return value
    if isinstance(value, float | np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("metadata must not contain NaN/Inf")
        return number
    if isinstance(value, np.ndarray):
        return _canonical_json_value(value.tolist())
    if isinstance(value, list | tuple):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


__all__ = [
    "ABSENT",
    "DEFAULT_SIZE_ZYX",
    "DELETED",
    "ExternalSidecarChangeError",
    "PRESENT",
    "RAW",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ObservationPatch",
    "ProofreadStore",
    "ProofreadStatus",
    "SidecarError",
]
