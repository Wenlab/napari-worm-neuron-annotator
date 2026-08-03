import numpy as np
import pytest

from napari_worm_neuron_annotator._colors import neuron_color


@pytest.mark.parametrize(
    ("neuron_id", "expected"),
    [
        (0, (0.95, 0.7174400000000001, 0.266, 1.0)),
        (1, (0.5178514898295687, 0.266, 0.95, 1.0)),
        (2, (0.266, 0.95, 0.31826297965913775, 1.0)),
        (42, (0.95, 0.5427225728418822, 0.266, 1.0)),
    ],
)
def test_neuron_color_uses_approved_golden_ratio_palette(
    neuron_id, expected
):
    assert neuron_color(neuron_id) == expected


def test_neuron_color_is_exactly_stable_for_the_same_id():
    first = neuron_color(1234)

    assert neuron_color(1234) == first
    assert neuron_color(np.int64(1234)) == first


@pytest.mark.parametrize("neuron_id", [-1, np.int64(-2)])
def test_neuron_color_rejects_negative_ids(neuron_id):
    with pytest.raises(ValueError, match="non-negative"):
        neuron_color(neuron_id)


@pytest.mark.parametrize(
    "neuron_id",
    [True, False, 1.0, "1", None, np.asarray(1)],
)
def test_neuron_color_rejects_non_integer_ids(neuron_id):
    with pytest.raises(TypeError, match="integer"):
        neuron_color(neuron_id)
