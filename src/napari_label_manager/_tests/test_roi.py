import numpy as np
import pytest

from napari_label_manager._roi import (
    NeuronBox,
    NeuronBoxDataset,
    add_time_axis,
    box_vectors_2d,
    box_vectors_3d,
    label_value_to_neuron_id,
    neuron_id_to_label_value,
)


def test_dataset_validates_and_maps_source_coordinates(tmp_path):
    data = np.full((6, 2, 8), np.nan, dtype=np.float32)
    data[3, 1, :6] = [20, 10, 25, 8, 6, 10]
    path = tmp_path / "neuron_pt_tuple.npy"
    np.save(path, data)

    dataset = NeuronBoxDataset.from_npy(
        path,
        z_divisor=5,
        volume_start=1,
        volume_stride=2,
    )
    box = dataset.get_box(viewer_t=1, neuron_id=1)

    assert dataset.time_count == 6
    assert dataset.neuron_count == 2
    assert box is not None
    assert box.source_t == 3
    assert box.center_zyx == (5, 10, 20)
    assert box.size_zyx == (2, 6, 8)
    assert dataset.valid_ids(1) == [1]


@pytest.mark.parametrize(
    "shape",
    [(2, 3), (2, 3, 5), (2, 3, 4, 5)],
)
def test_dataset_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        NeuronBoxDataset(np.zeros(shape, dtype=float))


def test_dataset_treats_invalid_observations_as_missing():
    data = np.zeros((1, 3, 8), dtype=float)
    data[0, 0, :6] = [1, 2, np.nan, 4, 5, 6]
    data[0, 1, :6] = [1, 2, 3, -1, 5, 6]
    data[0, 2, :6] = [1, 2, 3, 4, 5, 0]
    dataset = NeuronBoxDataset(data)

    assert dataset.valid_ids(0) == []
    assert dataset.get_box(10, 0) is None


def test_id_mapping_is_explicit_and_round_trips():
    assert neuron_id_to_label_value(0) == 1
    assert neuron_id_to_label_value(22) == 23
    assert label_value_to_neuron_id(1) == 0
    assert label_value_to_neuron_id(23) == 22

    with pytest.raises(ValueError):
        neuron_id_to_label_value(-1)
    with pytest.raises(ValueError):
        label_value_to_neuron_id(0)


def test_box_geometry_has_four_2d_and_twelve_3d_edges():
    box = NeuronBox(
        neuron_id=4,
        source_t=2,
        center_zyx=(5, 10, 20),
        size_zyx=(4, 6, 8),
    )

    vectors_3d = box_vectors_3d(box)
    vectors_2d = box_vectors_2d(box, z_index=5)

    assert vectors_3d.shape == (12, 2, 3)
    assert vectors_2d.shape == (4, 2, 3)
    np.testing.assert_allclose(
        vectors_3d[:, 0] + vectors_3d[:, 1],
        np.asarray(
            [
                [3, 7, 24],
                [3, 13, 24],
                [3, 13, 16],
                [3, 7, 16],
                [7, 7, 24],
                [7, 13, 24],
                [7, 13, 16],
                [7, 7, 16],
                [7, 7, 16],
                [7, 7, 24],
                [7, 13, 16],
                [7, 13, 24],
            ]
        ),
    )
    assert box_vectors_2d(box, z_index=7).shape == (0, 2, 3)


def test_add_time_axis_preserves_geometry():
    box = NeuronBox(
        neuron_id=0,
        source_t=0,
        center_zyx=(1, 2, 3),
        size_zyx=(2, 2, 2),
    )
    vectors = box_vectors_3d(box)

    promoted = add_time_axis(vectors, viewer_t=6)

    assert promoted.shape == (12, 2, 4)
    assert np.all(promoted[:, 0, 0] == 6)
    assert np.all(promoted[:, 1, 0] == 0)
    np.testing.assert_allclose(promoted[:, :, 1:], vectors)
