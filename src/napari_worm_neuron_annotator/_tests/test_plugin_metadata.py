from importlib import resources
from importlib.metadata import distribution

import yaml

import napari_worm_neuron_annotator


def test_renamed_distribution_preserves_npe2_identity():
    package = distribution("napari-worm-neuron-annotator")
    entry_points = {
        entry_point.name: entry_point.value
        for entry_point in package.entry_points
        if entry_point.group == "napari.manifest"
    }

    assert entry_points == {
        "napari-label-manager": "napari_worm_neuron_annotator:napari.yaml"
    }


def test_manifest_uses_new_visible_name_and_import_path():
    manifest_path = resources.files(napari_worm_neuron_annotator).joinpath(
        "napari.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest["name"] == "napari-label-manager"
    assert manifest["display_name"] == "Worm Neuron Annotator"
    assert manifest["contributions"]["commands"] == [
        {
            "id": "napari-label-manager.LabelManager",
            "python_name": (
                "napari_worm_neuron_annotator._widget:LabelManager"
            ),
            "title": "Worm Neuron Annotator",
        }
    ]
