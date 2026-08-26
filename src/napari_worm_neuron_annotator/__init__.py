try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("napari-worm-neuron-annotator")
    except PackageNotFoundError:
        __version__ = "0+unknown"

from ._proofread import ObservationPatch, ProofreadStore, SidecarError
from ._widget import LabelManager, NeuronAnnotatorWidget

__all__ = (
    "NeuronAnnotatorWidget",
    "LabelManager",
    "ObservationPatch",
    "ProofreadStore",
    "SidecarError",
)
