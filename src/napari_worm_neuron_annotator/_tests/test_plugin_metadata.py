from importlib import resources
from importlib.metadata import distribution

import yaml

import napari_worm_neuron_annotator

DISTRIBUTION_NAME = "napari-worm-neuron-annotator"
MANIFEST_ENTRY_POINT = "napari_worm_neuron_annotator:napari.yaml"


def test_distribution_and_npe2_entry_point_names_match():
    package = distribution(DISTRIBUTION_NAME)
    package_name = package.metadata["Name"]
    entry_points = {
        entry_point.name: entry_point.value
        for entry_point in package.entry_points
        if entry_point.group == "napari.manifest"
    }

    assert package_name == DISTRIBUTION_NAME
    assert entry_points == {package_name: MANIFEST_ENTRY_POINT}


def test_manifest_identity_matches_distribution():
    manifest_path = resources.files(napari_worm_neuron_annotator).joinpath(
        "napari.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    widget_command_id = f"{DISTRIBUTION_NAME}.NeuronAnnotatorWidget"
    legacy_command_id = f"{DISTRIBUTION_NAME}.LabelManager"

    assert manifest["name"] == DISTRIBUTION_NAME
    assert manifest["display_name"] == "Worm Neuron Annotator"
    assert manifest["contributions"]["commands"] == [
        {
            "id": widget_command_id,
            "python_name": (
                "napari_worm_neuron_annotator._widget:NeuronAnnotatorWidget"
            ),
            "title": "Worm Neuron Annotator",
        },
        {
            "id": legacy_command_id,
            "python_name": (
                "napari_worm_neuron_annotator._widget:LabelManager"
            ),
            "title": "Worm Neuron Annotator",
        }
    ]
    assert manifest["contributions"]["widgets"] == [
        {
            "command": widget_command_id,
            "display_name": "Worm Neuron Annotator",
        }
    ]


def test_public_widget_export_keeps_label_manager_alias():
    assert (
        napari_worm_neuron_annotator.LabelManager
        is napari_worm_neuron_annotator.NeuronAnnotatorWidget
    )
