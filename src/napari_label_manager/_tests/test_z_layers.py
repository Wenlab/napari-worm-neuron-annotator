import numpy as np
import pytest

from napari_label_manager._z_layers import (
    ZLayerRange,
    build_z_layer_ranges,
    find_z_layer,
    parse_z_cuts,
    shifted_z_translation,
    slice_z_range,
    z_threshold_count_profile,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ()),
        ("   ", ()),
        ("[]", ()),
        ("4,10", (4, 10)),
        (" [ 4, 10 ] ", (4, 10)),
    ],
)
def test_parse_z_cuts(text, expected):
    assert parse_z_cuts(text) == expected


@pytest.mark.parametrize(
    "text",
    ["4.5", "four", "4,,10", ",4", "4,", "[4,10", "4,10]", "[[4,10]]"],
)
def test_parse_z_cuts_rejects_malformed_input(text):
    with pytest.raises(ValueError, match="Z cut"):
        parse_z_cuts(text)


def test_build_z_layer_ranges_spans_volume():
    ranges = build_z_layer_ranges(18, parse_z_cuts("4,10"))

    assert ranges == (
        ZLayerRange(index=0, start=0, stop=4),
        ZLayerRange(index=1, start=4, stop=10),
        ZLayerRange(index=2, start=10, stop=18),
    )


def test_build_z_layer_ranges_accepts_no_cuts():
    assert build_z_layer_ranges(18, ()) == (
        ZLayerRange(index=0, start=0, stop=18),
    )


@pytest.mark.parametrize(
    ("z_size", "cuts", "message"),
    [
        (0, (), "positive"),
        (18, (4, 4), "duplicates"),
        (18, (10, 4), "increasing"),
        (18, (0, 4), "inside"),
        (18, (4, 18), "inside"),
        (18, (-1, 4), "inside"),
        (18, (4, 19), "inside"),
        (18, (4.0, 10), "integer"),
    ],
)
def test_build_z_layer_ranges_rejects_invalid_cuts(z_size, cuts, message):
    with pytest.raises(ValueError, match=message):
        build_z_layer_ranges(z_size, cuts)


@pytest.mark.parametrize(
    ("shape", "z_range", "expected_shape"),
    [
        ((18, 5, 7), ZLayerRange(1, 4, 10), (6, 5, 7)),
        ((3, 18, 5, 7), ZLayerRange(1, 4, 10), (3, 6, 5, 7)),
    ],
)
def test_slice_z_range_returns_numpy_view(shape, z_range, expected_shape):
    source = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)

    result = slice_z_range(source, z_range)

    assert result.shape == expected_shape
    assert np.shares_memory(source, result)
    np.testing.assert_array_equal(
        result,
        source[(slice(None),) * (len(shape) - 3) + (slice(4, 10),)],
    )


def test_slice_z_range_returns_memory_map_view(tmp_path):
    path = tmp_path / "volume.dat"
    source = np.memmap(path, mode="w+", dtype=np.uint16, shape=(2, 18, 3, 4))
    source[:] = np.arange(source.size, dtype=np.uint16).reshape(source.shape)

    result = slice_z_range(source, ZLayerRange(2, 10, 18))

    assert isinstance(result, np.memmap)
    assert result.shape == (2, 8, 3, 4)
    assert np.shares_memory(source, result)


@pytest.mark.parametrize("shape", [(5, 6), (2, 3, 4, 5, 6)])
def test_slice_z_range_rejects_unsupported_dimensions(shape):
    with pytest.raises(ValueError, match="supports only"):
        slice_z_range(np.zeros(shape), ZLayerRange(0, 0, shape[-1]))


def test_slice_z_range_rejects_range_outside_data():
    with pytest.raises(ValueError, match="outside"):
        slice_z_range(np.zeros((5, 4, 3)), ZLayerRange(0, 0, 6))


@pytest.mark.parametrize(
    ("translate", "scale", "start", "expected"),
    [
        ((10, 20, 30), (5, 2, 0.5), 4, (30, 20, 30)),
        ((100, 10, 20, 30), (2, 5, 2, 0.5), 4, (100, 30, 20, 30)),
    ],
)
def test_shifted_z_translation_preserves_world_position(
    translate, scale, start, expected
):
    assert shifted_z_translation(translate, scale, start) == expected


@pytest.mark.parametrize(
    ("translate", "scale", "message"),
    [
        ((0, 0), (1, 1), "3D or 4D"),
        ((0, 0, 0), (1, 1), "same length"),
        ((0, 0, 0), (1, 0, 1), "positive"),
        ((0, 0, 0), (1, -1, 1), "positive"),
        ((0, 0, 0), (1, np.inf, 1), "finite"),
    ],
)
def test_shifted_z_translation_validates_transform(translate, scale, message):
    with pytest.raises(ValueError, match=message):
        shifted_z_translation(translate, scale, 1)


@pytest.mark.parametrize(
    ("center_z", "expected_index"),
    [
        (-0.01, None),
        (0, 0),
        (3.999, 0),
        (4, 1),
        (9.999, 1),
        (10, 2),
        (17.999, 2),
        (18, None),
        (np.nan, None),
        (np.inf, None),
    ],
)
def test_find_z_layer_uses_half_open_center_membership(
    center_z, expected_index
):
    ranges = build_z_layer_ranges(18, (4, 10))

    result = find_z_layer(center_z, ranges)

    assert (None if result is None else result.index) == expected_index


def test_z_threshold_count_profile_counts_pixels_strictly_above_threshold():
    data = np.asarray(
        [
            [[1, 2], [3, 4]],
            [[-4, -3], [-2, -1]],
            [[5, np.nan], [2, 1]],
        ],
        dtype=float,
    )

    np.testing.assert_array_equal(
        z_threshold_count_profile(data, threshold=2),
        [2, 0, 1],
    )


def test_z_threshold_count_profile_uses_only_requested_4d_time_point():
    data = np.zeros((2, 3, 2, 2), dtype=np.uint16)
    data[0, :, 0, 0] = [1, 2, 3]
    data[1, :, 1, 1] = [10, 20, 30]

    np.testing.assert_array_equal(
        z_threshold_count_profile(data, threshold=15, time_index=1),
        [0, 1, 1],
    )


def test_z_threshold_count_profile_does_not_count_nan():
    data = np.ones((2, 2, 2), dtype=float)
    data[1] = np.nan

    result = z_threshold_count_profile(data, threshold=0)

    np.testing.assert_array_equal(result, [4, 0])


@pytest.mark.parametrize(
    ("data", "threshold", "time_index", "message"),
    [
        (np.zeros((2, 2)), 170, 0, "only"),
        (np.zeros((1, 2, 2, 2)), 170, 1, "inside"),
        (np.zeros((2, 2, 2)), np.inf, 0, "finite"),
        (np.zeros((2, 2, 2), dtype=complex), 170, 0, "complex"),
    ],
)
def test_z_threshold_count_profile_rejects_unsupported_inputs(
    data, threshold, time_index, message
):
    with pytest.raises(ValueError, match=message):
        z_threshold_count_profile(
            data,
            threshold=threshold,
            time_index=time_index,
        )
