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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

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
            _validate_size(self.restore_size_zyx)

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
        self.observation_patches: dict[tuple[int, int], ObservationPatch] = {}
        self.delete_all_ids: set[int] = set()
        self.placement_size: dict[int, tuple[float, float, float]] = {}
        self.committed_added_ids: set[int] = set()
        self.provisional_added_ids: set[int] = set()
        self.retired_ids: set[int] = set()
        self._next_neuron_id = self.dataset.raw_N
        self._bound_sidecar_path: Path | None = None
        self._saved_snapshot = self._canonical_state()

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
    def neuron_ids(self) -> list[int]:
        """Current non-retired identities in deterministic order."""
        ids = set(range(self.raw_N))
        ids.update(self.committed_added_ids)
        ids.update(self.provisional_added_ids)
        ids.difference_update(self.retired_ids)
        return sorted(ids)

    @property
    def all_neuron_ids(self) -> list[int]:
        ids = set(range(self.raw_N))
        ids.update(self.committed_added_ids)
        ids.update(self.provisional_added_ids)
        ids.update(self.retired_ids)
        return sorted(ids)

    @property
    def identity_state(self) -> dict[int, str]:
        states = dict.fromkeys(range(self.raw_N), "raw")
        states.update(dict.fromkeys(self.committed_added_ids, "committed_added"))
        states.update(dict.fromkeys(self.provisional_added_ids, "provisional_added"))
        states.update(dict.fromkeys(self.retired_ids, "retired"))
        return states

    @property
    def observation_count(self) -> int:
        return sum(
            self.resolve(t, i) is not None
            for t in range(self.raw_T)
            for i in self.neuron_ids
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
        if not allow_retired and value in self.retired_ids:
            raise ValueError(f"neuron_id is retired: {value}")
        return value

    def _raw_box(self, volume_index: int, neuron_id: int) -> NeuronBox | None:
        if neuron_id >= self.raw_N:
            return None
        return self.dataset.get_box_at_volume_index(volume_index, neuron_id)

    # ------------------------------------------------------------------
    # Resolver and edit operations
    # ------------------------------------------------------------------
    def resolve(self, volume_index: int, neuron_id: int) -> NeuronBox | None:
        """Resolve one observation using canonical patch precedence."""
        volume_index = self._check_volume(volume_index)
        neuron_id = self._check_id(neuron_id)
        patch = self.observation_patches.get((volume_index, neuron_id))
        if patch is not None:
            if patch.state == PRESENT:
                return patch.box
            return None
        if neuron_id in self.delete_all_ids:
            return None
        return self._raw_box(volume_index, neuron_id)

    def effective_state(self, volume_index: int, neuron_id: int) -> str:
        """Return ``present``, ``deleted``, ``raw`` or ``absent``."""
        volume_index = self._check_volume(volume_index)
        neuron_id = self._check_id(neuron_id)
        patch = self.observation_patches.get((volume_index, neuron_id))
        if patch is not None:
            return patch.state
        if neuron_id in self.delete_all_ids:
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
            neuron_id not in self.delete_all_ids
            and raw_box is not None
            and raw_box.center_zyx == new_box.center_zyx
            and raw_box.size_zyx == new_box.size_zyx
        ):
            self.observation_patches.pop((volume_index, neuron_id), None)
        else:
            self.observation_patches[(volume_index, neuron_id)] = (
                ObservationPatch.present(new_box)
            )
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
        if neuron_id in self.delete_all_ids:
            # A PRESENT exception is removed; no redundant DELETED patch is
            # needed because the marker already expresses the deletion.
            self.observation_patches.pop((volume_index, neuron_id), None)
            return
        existing = self.observation_patches.get((volume_index, neuron_id))
        if existing is not None and existing.state == DELETED:
            return
        if existing is None and self._raw_box(volume_index, neuron_id) is None:
            # Deleting a naturally absent observation is a no-op.  This keeps
            # modified_observations aligned with actual proofreading effects.
            return
        restore_size = None
        if existing is not None and existing.state == PRESENT:
            assert existing.box is not None
            restore_size = existing.box.size_zyx
        else:
            raw = self._raw_box(volume_index, neuron_id)
            if raw is not None:
                restore_size = raw.size_zyx
            elif neuron_id in self.placement_size:
                restore_size = self.placement_size[neuron_id]
        self.observation_patches[(volume_index, neuron_id)] = (
            ObservationPatch.deleted(restore_size)
        )

    set_deleted = set_observation_deleted

    def delete_all_observations(self, neuron_id: int) -> None:
        """Logically remove an identity's observations at every volume."""
        neuron_id = self._check_id(neuron_id)
        if neuron_id not in self.placement_size:
            # Infer before clearing PRESENT patches or adding the marker;
            # afterwards the resolver would see every observation as absent.
            self.placement_size[neuron_id] = self.size_for_placement(neuron_id)
        for key in [
            key
            for key in self.observation_patches
            if key[1] == neuron_id
        ]:
            del self.observation_patches[key]
        self.delete_all_ids.add(neuron_id)

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
        self.provisional_added_ids.add(neuron_id)
        self.placement_size[neuron_id] = size
        self.set_observation_present(
            volume_index,
            neuron_id,
            center_zyx=center,
            size_zyx=size,
        )
        return neuron_id

    def retire_added_neuron(self, neuron_id: int) -> None:
        """Retire an added identity without renumbering other IDs."""
        neuron_id = self._check_id(neuron_id)
        if neuron_id < self.raw_N:
            raise ValueError("raw neuron IDs cannot be retired")
        self.provisional_added_ids.discard(neuron_id)
        self.committed_added_ids.discard(neuron_id)
        self.retired_ids.add(neuron_id)
        self.delete_all_ids.discard(neuron_id)
        self.placement_size.pop(neuron_id, None)
        for key in [key for key in self.observation_patches if key[1] == neuron_id]:
            del self.observation_patches[key]

    def size_for_placement(
        self, neuron_id: int, volume_index: int | None = None
    ) -> tuple[float, float, float]:
        """Infer a committed placement size using the documented priority."""
        neuron_id = self._check_id(neuron_id)
        if neuron_id in self.placement_size:
            return self.placement_size[neuron_id]
        if volume_index is not None:
            volume_index = self._check_volume(volume_index)
            patch = self.observation_patches.get((volume_index, neuron_id))
            if patch is not None and patch.restore_size_zyx is not None:
                return patch.restore_size_zyx
            indices = sorted(
                range(self.raw_T),
                key=lambda value: (abs(value - volume_index), value),
            )
        else:
            indices = list(range(self.raw_T))
        for candidate in indices:
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
        self.placement_size[neuron_id] = size
        # Snapshot the resolved boxes before mutating patches.  Missing
        # observations (including Delete-all volumes) are not created.
        boxes = [
            (volume_index, self.resolve(volume_index, neuron_id))
            for volume_index in range(self.raw_T)
        ]
        for volume_index, box in boxes:
            if box is None:
                continue
            self.set_observation_present(
                volume_index,
                neuron_id,
                center_zyx=box.center_zyx,
                size_zyx=size,
            )
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
            for volume_index in range(self.raw_T)
        ]
        if (
            all(box is None or box.size_zyx == size for box in boxes)
            and neuron_id in self.placement_size
            and self.placement_size[neuron_id] == size
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
        result = set(self.observation_patches)
        for neuron_id in self.delete_all_ids:
            for volume_index in range(self.raw_T):
                if self._raw_box(volume_index, neuron_id) is not None:
                    result.add((volume_index, neuron_id))
        return result

    @property
    def modified_ids(self) -> set[int]:
        return {neuron_id for _, neuron_id in self.modified_observations}

    def _changed_fields_for_patch(
        self,
        volume_index: int,
        neuron_id: int,
        patch: ObservationPatch,
        *,
        delete_all_ids: set[int] | None = None,
    ) -> tuple[str, ...]:
        """Derive v2's field list from raw geometry and a complete patch.

        The list is metadata only; resolution and NPY export continue to use
        the complete patch box.  Exact tuple equality is intentional here and
        matches the canonical redundant-patch normalization rules.
        """
        raw_box = self._raw_box(volume_index, neuron_id)
        markers = (
            self.delete_all_ids
            if delete_all_ids is None
            else delete_all_ids
        )
        if patch.state == DELETED:
            return ("presence",)
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
        patch = self.observation_patches.get((volume_index, neuron_id))
        if patch is not None:
            return self._changed_fields_for_patch(volume_index, neuron_id, patch)
        if (
            neuron_id in self.delete_all_ids
            and self._raw_box(volume_index, neuron_id) is not None
        ):
            return ("presence",)
        return ()

    # ``observation_change_fields`` is a convenient mapping for UI and
    # downstream consumers; the method above remains useful for point lookup.
    @property
    def observation_change_fields(
        self,
    ) -> dict[tuple[int, int], tuple[str, ...]]:
        result: dict[tuple[int, int], tuple[str, ...]] = {}
        for key, patch in self.observation_patches.items():
            result[key] = self._changed_fields_for_patch(*key, patch)
        for neuron_id in self.delete_all_ids:
            for volume_index in range(self.raw_T):
                key = (volume_index, neuron_id)
                if key in result or self._raw_box(volume_index, neuron_id) is None:
                    continue
                result[key] = ("presence",)
        return result

    @property
    def center_changed_observations(self) -> set[tuple[int, int]]:
        return {
            key
            for key, fields in self.observation_change_fields.items()
            if "center_zyx" in fields
        }

    @property
    def size_changed_observations(self) -> set[tuple[int, int]]:
        return {
            key
            for key, fields in self.observation_change_fields.items()
            if "size_zyx" in fields
        }

    @property
    def presence_changed_observations(self) -> set[tuple[int, int]]:
        return {
            key
            for key, fields in self.observation_change_fields.items()
            if "presence" in fields
        }

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
        patch = self.observation_patches.get((volume_index, neuron_id))
        has_center = "center_zyx" in fields
        has_size = "size_zyx" in fields
        # Presence plus geometry (e.g. a Delete-all local restoration) keeps
        # the meaningful geometry classification; presence alone maps to the
        # placed/deleted/added labels below.
        if "presence" in fields and not (has_center or has_size):
            # A delete-all marker without an explicit per-volume patch is a
            # presence-only deletion represented by the global operation.
            if patch is None and neuron_id in self.delete_all_ids:
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
            for (volume_index, candidate), _fields in (
                self.observation_change_fields.items()
            )
            if candidate == neuron_id
            for status in [self.classify_observation(volume_index, candidate)]
            if status is not None
        }

    def _canonical_state(self) -> dict[str, Any]:
        patches = [
            _patch_to_json(volume_index, neuron_id, patch)
            for (volume_index, neuron_id), patch in sorted(
                self.observation_patches.items()
            )
        ]
        return {
            "observation_patches": patches,
            "delete_all_ids": sorted(self.delete_all_ids),
            "placement_size": {
                str(neuron_id): [float(v) for v in size]
                for neuron_id, size in sorted(self.placement_size.items())
            },
            "committed_added_ids": sorted(self.committed_added_ids),
            "provisional_added_ids": sorted(self.provisional_added_ids),
            "retired_ids": sorted(self.retired_ids),
            "next_neuron_id": int(self._next_neuron_id),
        }

    @property
    def saved_snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._saved_snapshot)

    @property
    def dirty(self) -> bool:
        return self._canonical_state() != self._saved_snapshot

    def _restore_state(self, state: dict[str, Any]) -> None:
        self.observation_patches = {}
        for record in state["observation_patches"]:
            volume_index = int(record["volume_index"])
            neuron_id = int(record["neuron_id"])
            if record["state"] == PRESENT:
                box_data = record["box"]
                box = _box_from_parts(
                    neuron_id,
                    volume_index,
                    box_data["center_zyx"],
                    box_data["size_zyx"],
                )
                patch = ObservationPatch.present(box)
            else:
                restore = record.get("restore_size_zyx")
                patch = ObservationPatch.deleted(restore)
            self.observation_patches[(volume_index, neuron_id)] = patch
        self.delete_all_ids = {int(v) for v in state["delete_all_ids"]}
        self.placement_size = {
            int(neuron_id): tuple(float(v) for v in size)
            for neuron_id, size in state["placement_size"].items()
        }
        self.committed_added_ids = {
            int(v) for v in state["committed_added_ids"]
        }
        self.provisional_added_ids = {
            int(v) for v in state["provisional_added_ids"]
        }
        self.retired_ids = {int(v) for v in state["retired_ids"]}
        self._next_neuron_id = int(state["next_neuron_id"])

    def discard(self) -> None:
        """Restore the most recently saved/loaded canonical snapshot."""
        self._restore_state(copy.deepcopy(self._saved_snapshot))

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

    def _payload_for_save(self) -> dict[str, Any]:
        state = copy.deepcopy(self._canonical_state())
        # Provisional identities with at least one PRESENT patch become
        # committed on the successful save.  Empty provisional identities are
        # omitted from the persisted lineage.
        provisional = set(state["provisional_added_ids"])
        present_ids = {
            int(record["neuron_id"])
            for record in state["observation_patches"]
            if record["state"] == PRESENT
        }
        committed = set(state["committed_added_ids"])
        committed.update(provisional & present_ids)
        empty = provisional - present_ids
        # Saving establishes an identity lineage even if the provisional ID
        # has no remaining observation.  Reserve it as retired so a later
        # load cannot reuse the numeric ID.
        state["retired_ids"] = sorted(
            set(state["retired_ids"]) | empty
        )
        state["committed_added_ids"] = sorted(committed)
        state["provisional_added_ids"] = []
        if empty:
            state["observation_patches"] = [
                record
                for record in state["observation_patches"]
                if int(record["neuron_id"]) not in empty
            ]
            state["delete_all_ids"] = [
                value
                for value in state["delete_all_ids"]
                if int(value) not in empty
            ]
            state["placement_size"] = {
                key: value
                for key, value in state["placement_size"].items()
                if int(key) not in empty
            }
        state["next_neuron_id"] = max(
            [
                self.raw_N,
                *state["committed_added_ids"],
                *state["retired_ids"],
            ]
        ) + 1 if (
            state["committed_added_ids"] or state["retired_ids"]
        ) else max(self.raw_N, int(state["next_neuron_id"]))
        # v2 records carry an ordered field classification derived from the
        # raw geometry and the complete patch.  Keep the box itself as the
        # sole geometry authority; ``changed_fields`` is descriptive only.
        v2_patches: list[dict[str, Any]] = []
        for record in state["observation_patches"]:
            volume_index = int(record["volume_index"])
            neuron_id = int(record["neuron_id"])
            if record["state"] == PRESENT:
                box_data = record["box"]
                patch = ObservationPatch.present(
                    _box_from_parts(
                        neuron_id,
                        volume_index,
                        box_data["center_zyx"],
                        box_data["size_zyx"],
                    )
                )
            else:
                patch = ObservationPatch.deleted(record.get("restore_size_zyx"))
            fields = self._changed_fields_for_patch(volume_index, neuron_id, patch)
            if not fields:
                # Drop a redundant PRESENT patch if an external caller placed
                # a box exactly equal to raw geometry.
                continue
            v2_patches.append(
                _patch_to_json(
                    volume_index,
                    neuron_id,
                    patch,
                    changed_fields=fields,
                )
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "raw": {
                "shape": list(self.dataset.raw_shape),
                "dtype": self.dataset.raw_dtype.str,
                "z_divisor": float(self.dataset.z_divisor),
                "sha256": self._raw_fingerprint(),
            },
            "image_signature": copy.deepcopy(self.image_signature),
            "observation_patches": v2_patches,
            "delete_all_ids": state["delete_all_ids"],
            "placement_size": state["placement_size"],
            "added_neurons": {
                "committed": state["committed_added_ids"],
                "retired": state["retired_ids"],
            },
        }

    def _commit_saved_payload(self, payload: dict[str, Any]) -> None:
        state = {
            "observation_patches": payload["observation_patches"],
            "delete_all_ids": payload["delete_all_ids"],
            "placement_size": payload["placement_size"],
            "committed_added_ids": payload["added_neurons"]["committed"],
            "provisional_added_ids": [],
            "retired_ids": payload["added_neurons"]["retired"],
            "next_neuron_id": max(
                [
                    self.raw_N - 1,
                    *payload["added_neurons"]["committed"],
                    *payload["added_neurons"]["retired"],
                ],
                default=self.raw_N - 1,
            )
            + 1,
        }
        self._restore_state(state)
        self._saved_snapshot = self._canonical_state()

    def save(self, path: str | Path | None = None) -> Path:
        """Atomically save canonical edits to a JSON sidecar."""
        target = self._sidecar_path(path)
        payload = self._payload_for_save()
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        self._commit_saved_payload(payload)
        self._bound_sidecar_path = target
        return target

    save_as = save

    def _sidecar_path(self, path: str | Path | None) -> Path:
        if path is None:
            if self._bound_sidecar_path is not None:
                return self._bound_sidecar_path
            if self.dataset.path is None:
                raise ValueError(
                    "a sidecar path is required for an in-memory dataset"
                )
            target = self.dataset.path.with_suffix(".proofread.json")
        else:
            target = Path(path)
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
            with source.open("r", encoding="utf-8") as stream:
                payload = json.load(
                    stream,
                    parse_constant=_reject_json_constants,
                    object_pairs_hook=_reject_duplicate_keys,
                )
        except SidecarError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SidecarError(f"cannot load proof sidecar: {exc}") from exc
        state = self._validate_payload(payload)
        # No mutation has happened before this point.
        self._restore_state(state)
        self._saved_snapshot = self._canonical_state()
        self._bound_sidecar_path = source

    load_sidecar = load

    def _validate_payload(self, payload: Any) -> dict[str, Any]:
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
                patches[key] = patch

        # Retired IDs may remain reserved but cannot be active in markers.
        if delete_all_ids & retired:
            raise SidecarError("retired ID cannot have delete_all marker")
        max_id = max(
            [self.raw_N - 1, *committed, *retired], default=self.raw_N - 1
        )
        state = {
            "observation_patches": [
                _patch_to_json(volume_index, neuron_id, patch)
                for (volume_index, neuron_id), patch in sorted(patches.items())
            ],
            "delete_all_ids": sorted(delete_all_ids),
            "placement_size": {
                str(neuron_id): [float(v) for v in size]
                for neuron_id, size in sorted(placement_size.items())
            },
            "committed_added_ids": sorted(committed),
            "provisional_added_ids": [],
            "retired_ids": sorted(retired),
            "next_neuron_id": max_id + 1,
        }
        return state

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
    "PRESENT",
    "RAW",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ObservationPatch",
    "ProofreadStore",
    "SidecarError",
]
