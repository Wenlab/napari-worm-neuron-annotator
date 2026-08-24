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
ROI_PATH = DATA_DIR / "neuron_point_tuple.npy"

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
    widget.z_divisor_spin.setValue(Z_DIVISOR)
    widget.load_roi_path(ROI_PATH)
    if widget.active_id is not None:
        widget.activate_id(widget.active_id, locate=True)

    viewer.window.show()
    napari.run()


if __name__ == "__main__":
    main()
