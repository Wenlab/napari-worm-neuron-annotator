"""Open the validated 20260304_w3_immobile dataset in the ROI navigator."""

from pathlib import Path

import napari
import numpy as np

from napari_worm_neuron_annotator import NeuronAnnotatorWidget

DATASET_NAME = "20260304_w3_immobile"
DATA_DIR = (
    Path(__file__).resolve().parents[1] / "data" / f"{DATASET_NAME}_npy"
)
IMAGE_PATH = DATA_DIR / "volumes.npy"
LABELS_PATH = DATA_DIR / "neuron_mask.npy"
ROI_PATH = DATA_DIR / "neuron_point_tuple.npy"

# Set this to True only when validating the optional Labels integration.
LOAD_OPTIONAL_LABELS = False
Z_DIVISOR = 5.0
LAYER_SCALE_TZYX = (1.0, 5.0, 1.0, 1.0)
IMAGE_CONTRAST_LIMITS = (89.0, 152.0)


def main() -> None:
    volumes = np.load(IMAGE_PATH, mmap_mode="r", allow_pickle=False)
    if volumes.ndim != 4:
        raise ValueError("Expected a (t,z,y,x) Image array")

    viewer = napari.Viewer(title=f"{DATASET_NAME} – Worm Neuron Annotator")
    image_layer = viewer.add_image(
        volumes,
        name=f"{DATASET_NAME} volumes",
        scale=LAYER_SCALE_TZYX,
        axis_labels=("t", "z", "y", "x"),
        contrast_limits=IMAGE_CONTRAST_LIMITS,
        colormap="gray",
    )
    labels_layer = None
    if LOAD_OPTIONAL_LABELS:
        labels = np.load(LABELS_PATH, mmap_mode="r", allow_pickle=False)
        if labels.shape != volumes.shape:
            raise ValueError("Expected Labels to match the Image shape")
        labels_layer = viewer.add_labels(
            labels,
            name=f"{DATASET_NAME} neuron mask",
            scale=LAYER_SCALE_TZYX,
            axis_labels=("t", "z", "y", "x"),
        )
    viewer.dims.current_step = (0, volumes.shape[1] // 2, 0, 0)

    widget = NeuronAnnotatorWidget(viewer)
    viewer.window.add_dock_widget(
        widget,
        name="Worm Neuron Annotator",
        area="right",
    )
    image_index = widget.image_combo.findData(image_layer)
    if image_index >= 0:
        widget.image_combo.setCurrentIndex(image_index)
    if labels_layer is not None:
        labels_index = widget.labels_combo.findData(labels_layer)
        if labels_index >= 0:
            widget.labels_combo.setCurrentIndex(labels_index)
    widget.z_divisor_spin.setValue(Z_DIVISOR)
    widget.load_roi_path(ROI_PATH)
    if widget.active_id is not None:
        widget.activate_id(widget.active_id, locate=True)

    viewer.window.show()
    napari.run()


if __name__ == "__main__":
    main()
