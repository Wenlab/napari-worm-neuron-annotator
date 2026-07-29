"""Open the validated 20260417_w2 dataset in the ROI navigator."""

from pathlib import Path

import napari
import numpy as np

from napari_label_manager import LabelManager

DATA_DIR = Path(
    r"H:\Process_temporary\WJH\zephir_modifypoints\data\20260417_w2_npy"
)
IMAGE_PATH = DATA_DIR / "volumes.npy"
LABELS_PATH = DATA_DIR / "neuron_mask.npy"
ROI_PATH = DATA_DIR / "neuron_point_tuple.npy"

Z_DIVISOR = 5.0
LAYER_SCALE_TZYX = (1.0, 5.0, 1.0, 1.0)
IMAGE_CONTRAST_LIMITS = (89.0, 152.0)


def main() -> None:
    volumes = np.load(IMAGE_PATH, mmap_mode="r", allow_pickle=False)
    labels = np.load(LABELS_PATH, mmap_mode="r", allow_pickle=False)
    if volumes.shape != labels.shape or volumes.ndim != 4:
        raise ValueError(
            "Expected matching (t,z,y,x) volumes and Labels arrays"
        )

    viewer = napari.Viewer(title="20260417_w2 – Neuron ROI Navigator")
    viewer.add_image(
        volumes,
        name="20260417_w2 volumes",
        scale=LAYER_SCALE_TZYX,
        axis_labels=("t", "z", "y", "x"),
        contrast_limits=IMAGE_CONTRAST_LIMITS,
        colormap="gray",
    )
    viewer.add_labels(
        labels,
        name="20260417_w2 neuron mask",
        scale=LAYER_SCALE_TZYX,
        axis_labels=("t", "z", "y", "x"),
    )
    viewer.dims.current_step = (0, labels.shape[1] // 2, 0, 0)

    widget = LabelManager(viewer)
    viewer.window.add_dock_widget(
        widget,
        name="Neuron ROI Navigator",
        area="right",
    )
    widget.z_divisor_spin.setValue(Z_DIVISOR)
    widget.load_roi_path(ROI_PATH)
    if widget.active_id is not None:
        widget.activate_id(widget.active_id, locate=True)

    viewer.window.show()
    napari.run()


if __name__ == "__main__":
    main()
