import itertools

import pytest

from napari_worm_neuron_annotator._orientation import (
    OrientationState,
    resolve_orientation,
)


@pytest.mark.parametrize(
    ("rotation", "expected_order", "expected_orientation"),
    [
        (0, (0, 1, 2, 3), ("down", "right")),
        (90, (0, 1, 3, 2), ("down", "left")),
        (180, (0, 1, 2, 3), ("up", "left")),
        (270, (0, 1, 3, 2), ("up", "right")),
    ],
)
def test_clockwise_rotation_from_default_baseline(
    rotation, expected_order, expected_orientation
):
    result = resolve_orientation(
        (0, 1, 2, 3),
        ("down", "right"),
        OrientationState(rotation),
        y_axis=2,
        x_axis=3,
    )

    assert result == (expected_order, expected_orientation)


@pytest.mark.parametrize(
    ("rotation", "expected_orientation"),
    [
        (0, ("up", "left")),
        (90, ("up", "right")),
        (180, ("down", "right")),
        (270, ("down", "left")),
    ],
)
def test_rotation_composes_from_nondefault_baseline(
    rotation, expected_orientation
):
    order, orientation = resolve_orientation(
        (0, 1, 3, 2),
        ("up", "left"),
        OrientationState(rotation),
        y_axis=2,
        x_axis=3,
    )

    expected_order = (
        (0, 1, 2, 3) if rotation in (90, 270) else (0, 1, 3, 2)
    )
    assert order == expected_order
    assert orientation == expected_orientation


@pytest.mark.parametrize(
    ("rotation", "flip_horizontal", "flip_vertical"),
    itertools.product((0, 90, 180, 270), (False, True), (False, True)),
)
def test_all_rotation_and_flip_states_are_absolute(
    rotation, flip_horizontal, flip_vertical
):
    state = OrientationState(rotation, flip_horizontal, flip_vertical)

    first = resolve_orientation(
        (0, 1, 2),
        ("down", "right"),
        state,
        y_axis=1,
        x_axis=2,
    )
    second = resolve_orientation(
        (0, 1, 2),
        ("down", "right"),
        state,
        y_axis=1,
        x_axis=2,
    )

    assert first == second
    assert set(first[0]) == {0, 1, 2}
    assert first[1][0] in ("up", "down")
    assert first[1][1] in ("left", "right")


def test_screen_flips_apply_after_rotation():
    order, orientation = resolve_orientation(
        (0, 1, 2),
        ("down", "right"),
        OrientationState(90, flip_horizontal=True, flip_vertical=True),
        y_axis=1,
        x_axis=2,
    )

    assert order == (0, 2, 1)
    assert orientation == ("up", "right")


@pytest.mark.parametrize("rotation", (-90, 45, 360))
def test_invalid_rotation_is_rejected(rotation):
    with pytest.raises(ValueError, match="0, 90, 180, or 270"):
        OrientationState(rotation)


def test_invalid_baseline_is_rejected():
    state = OrientationState()

    with pytest.raises(ValueError, match="duplicate"):
        resolve_orientation(
            (0, 1, 1),
            ("down", "right"),
            state,
            y_axis=1,
            x_axis=2,
        )
    with pytest.raises(ValueError, match="vertical orientation"):
        resolve_orientation(
            (0, 1, 2),
            ("sideways", "right"),  # type: ignore[arg-type]
            state,
            y_axis=1,
            x_axis=2,
        )
