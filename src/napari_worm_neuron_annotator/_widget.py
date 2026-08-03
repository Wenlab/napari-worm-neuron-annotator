"""Napari widget for ROI-driven neuron navigation and label visibility."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import napari
import numpy as np
from napari.layers import Image, Labels, Points, Vectors
from napari.utils import colormaps as cmap
from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QCloseEvent, QColor, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._colors import neuron_color
from ._roi import (
    NeuronBoxDataset,
    add_time_axis,
    box_label_point_2d,
    box_label_point_3d,
    box_vectors_2d,
    box_vectors_3d,
    neuron_id_to_label_value,
)
from ._z_layers import (
    ZLayerRange,
    build_z_layer_ranges,
    find_z_layer,
    parse_z_cuts,
    shifted_z_translation,
    slice_z_range,
    z_threshold_count_profile,
)
from ._z_profile import ZThresholdProfileWidget

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment

    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False


# Preserve the legacy metadata namespace so managed layers from earlier
# releases remain identifiable during cleanup.
ROLE_KEY = "napari_label_manager.role"
ROLE_SELECTED = "roi_boxes_selected"
LEGACY_ROLE_ALL = "roi_boxes_all"
ROLE_ACTIVE = "roi_box_active"
ROLE_BOX_LABELS = "roi_box_labels"
MANAGED_VECTOR_ROLES = (ROLE_SELECTED, LEGACY_ROLE_ALL, ROLE_ACTIVE)
MANAGED_ROI_ROLES = (*MANAGED_VECTOR_ROLES, ROLE_BOX_LABELS)
ROLE_Z_IMAGE = "z_layer_image"
ROLE_Z_LABELS = "z_layer_labels"
MANAGED_Z_ROLES = (ROLE_Z_IMAGE, ROLE_Z_LABELS)
MAX_EXACT_LABEL_PIXELS = 10_000_000
Z_SOURCE_GEOMETRY_EVENTS = (
    "scale",
    "translate",
    "rotate",
    "shear",
    "affine",
    "axis_labels",
    "units",
)


class NeuronAnnotatorWidget(QWidget):
    """Navigate read-only neuron boxes on an Image with optional Labels."""

    def __init__(self, napari_viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.current_image: Image | None = None
        self.current_labels: Labels | None = None
        self.roi_dataset: NeuronBoxDataset | None = None
        self.active_id: int | None = None
        self.checked_ids: set[int] = set()
        self._available_ids: list[int] = []
        self._selection_items: dict[int, QTreeWidgetItem] = {}
        self._base_colors: dict[int, tuple[float, float, float, float]] = {}
        self._original_colormap = None
        self._original_opacity: float | None = None
        self._box_label_color = "#ffffff"
        self._ui_sync = False
        self._closed = False
        self._keys_bound: list[str] = []
        self._z_ranges: tuple[ZLayerRange, ...] = ()
        self._z_source_image: Image | None = None
        self._z_image_layers: list[Image] = []
        self._z_labels_proxy: Labels | None = None
        self._z_active_index: int | None = None
        self._z_source_image_visible: bool | None = None
        self._z_source_labels_visible: bool | None = None
        self._z_source_labels_data = None
        self._z_cleanup = False
        self._z_session_token = f"{id(self):x}"
        self._z_profile_source: Image | None = None
        self._z_profile_time: int | None = None

        self._setup_ui()
        self._connect_viewer_events()
        self._bind_keys()
        self._refresh_image_layers()
        self._refresh_label_layers()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setMinimumWidth(320)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_content = QWidget()
        layout = QVBoxLayout(self.scroll_content)

        header = QLabel("Worm Neuron Annotator")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        self.image_layer_group = self._build_image_layer_group()
        layout.addWidget(self.image_layer_group)
        self.labels_layer_group = self._build_labels_layer_group()
        layout.addWidget(self.labels_layer_group)
        layout.addWidget(self._build_z_layer_group())
        layout.addWidget(self._build_roi_group())
        layout.addWidget(self._build_selection_group())
        layout.addWidget(self._build_annotation_group())
        layout.addWidget(self._build_status_group())
        layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_content)
        outer_layout.addWidget(self.scroll_area)
        self.setLayout(outer_layout)

    def _build_image_layer_group(self) -> QGroupBox:
        group = QGroupBox("Image Layer")
        group_layout = QVBoxLayout()
        self.image_combo = QComboBox()
        self.image_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.image_combo.setMinimumContentsLength(10)
        self.image_combo.currentIndexChanged.connect(
            self._on_image_changed
        )
        group_layout.addWidget(QLabel("Spatial source:"))
        group_layout.addWidget(self.image_combo)
        group.setLayout(group_layout)
        # Compatibility for callers that previously used the Z-only selector.
        self.z_image_combo = self.image_combo
        return group

    def _build_labels_layer_group(self) -> QGroupBox:
        group = QGroupBox("Labels Layer")
        group_layout = QVBoxLayout()
        self.labels_combo = QComboBox()
        self.labels_combo.currentIndexChanged.connect(
            self._on_labels_changed
        )
        group_layout.addWidget(QLabel("Optional binding:"))
        group_layout.addWidget(self.labels_combo)
        self.layer_combo = self.labels_combo
        self._add_labels_appearance_controls(group_layout)
        group.setLayout(group_layout)
        return group

    def _build_z_layer_group(self) -> QGroupBox:
        group = QGroupBox("Z Layers")
        group_layout = QVBoxLayout()

        cuts_layout = QHBoxLayout()
        cuts_layout.addWidget(QLabel("Cuts:"))
        self.z_cuts_input = QLineEdit()
        self.z_cuts_input.setPlaceholderText("4,10")
        self.z_cuts_input.setToolTip(
            "Z cuts define half-open ranges [start, stop); "
            "a boundary slice belongs to the following layer."
        )
        self.z_cuts_input.textChanged.connect(self._sync_z_profile_cuts)
        cuts_layout.addWidget(self.z_cuts_input, 1)
        self.split_z_btn = QPushButton("Split")
        self.split_z_btn.clicked.connect(self.split_z_layers)
        cuts_layout.addWidget(self.split_z_btn)
        group_layout.addLayout(cuts_layout)

        profile_header = QHBoxLayout()
        self.z_profile_label = QLabel("Pixels >")
        profile_header.addWidget(self.z_profile_label)
        self.z_profile_threshold_spin = QDoubleSpinBox()
        self.z_profile_threshold_spin.setRange(-1_000_000_000, 1_000_000_000)
        self.z_profile_threshold_spin.setDecimals(3)
        self.z_profile_threshold_spin.setValue(170)
        self.z_profile_threshold_spin.setKeyboardTracking(False)
        self.z_profile_threshold_spin.setToolTip(
            "Count pixels with intensity strictly greater than this threshold."
        )
        self.z_profile_threshold_spin.valueChanged.connect(
            self.refresh_z_profile
        )
        profile_header.addWidget(self.z_profile_threshold_spin)
        self.z_profile_state_label = QLabel("")
        profile_header.addWidget(self.z_profile_state_label)
        profile_header.addStretch(1)
        self.z_profile_refresh_btn = QPushButton("Refresh")
        self.z_profile_refresh_btn.clicked.connect(self.refresh_z_profile)
        profile_header.addWidget(self.z_profile_refresh_btn)
        group_layout.addLayout(profile_header)
        self.z_profile = ZThresholdProfileWidget()
        self.z_profile.cutToggled.connect(self._toggle_z_profile_cut)
        group_layout.addWidget(self.z_profile)

        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("Show:"))
        self.z_view_combo = QComboBox()
        self.z_view_combo.addItem("All", None)
        self.z_view_combo.setEnabled(False)
        self.z_view_combo.currentIndexChanged.connect(self._on_z_view_changed)
        view_layout.addWidget(self.z_view_combo, 1)
        self.clear_z_btn = QPushButton("Clear")
        self.clear_z_btn.setEnabled(False)
        self.clear_z_btn.clicked.connect(self.clear_z_layers)
        view_layout.addWidget(self.clear_z_btn)
        group_layout.addLayout(view_layout)

        description = QLabel("Split Image by z; sync optional Labels and boxes.")
        description.setToolTip(
            "Neuron boxes are assigned to one layer by center Z."
        )
        group_layout.addWidget(description)
        group.setLayout(group_layout)
        return group

    def _build_roi_group(self) -> QGroupBox:
        group = QGroupBox("Neuron ROI Source")
        group_layout = QVBoxLayout()

        path_layout = QHBoxLayout()
        self.roi_path_input = QLineEdit()
        self.roi_path_input.setReadOnly(True)
        self.roi_path_input.setPlaceholderText("Load neuron_pt_tuple.npy")
        self.load_roi_btn = QPushButton("Load NPY")
        self.load_roi_btn.setEnabled(False)
        self.load_roi_btn.clicked.connect(self.load_roi_npy)
        self.unload_roi_btn = QPushButton("Unload")
        self.unload_roi_btn.clicked.connect(self.unload_roi)
        self.unload_roi_btn.setEnabled(False)
        path_layout.addWidget(self.roi_path_input, 1)
        path_layout.addWidget(self.load_roi_btn)
        path_layout.addWidget(self.unload_roi_btn)
        group_layout.addLayout(path_layout)

        config_layout = QGridLayout()
        config_layout.addWidget(QLabel("Z divisor:"), 0, 0)
        self.z_divisor_spin = QDoubleSpinBox()
        self.z_divisor_spin.setDecimals(3)
        self.z_divisor_spin.setRange(0.001, 1_000_000)
        self.z_divisor_spin.setValue(5.0)
        config_layout.addWidget(self.z_divisor_spin, 0, 1)

        config_layout.addWidget(QLabel("Volume start:"), 1, 0)
        self.volume_start_spin = QSpinBox()
        self.volume_start_spin.setRange(0, 1_000_000)
        config_layout.addWidget(self.volume_start_spin, 1, 1)

        config_layout.addWidget(QLabel("Stride:"), 2, 0)
        self.volume_stride_spin = QSpinBox()
        self.volume_stride_spin.setRange(1, 1_000_000)
        self.volume_stride_spin.setValue(1)
        config_layout.addWidget(self.volume_stride_spin, 2, 1)
        config_layout.setColumnStretch(1, 1)
        group_layout.addLayout(config_layout)

        self.roi_info_label = QLabel("No ROI loaded")
        self.roi_info_label.setWordWrap(True)
        group_layout.addWidget(self.roi_info_label)
        group.setLayout(group_layout)
        return group

    def _build_selection_group(self) -> QGroupBox:
        group = QGroupBox("Neuron Selection")
        group_layout = QVBoxLayout()

        self.navigation_help_label = QLabel("Q/W: last/next")
        group_layout.addWidget(self.navigation_help_label)

        controls_layout = QHBoxLayout()
        self.check_all_btn = QPushButton("All")
        self.check_all_btn.clicked.connect(self.check_all)
        self.check_none_btn = QPushButton("None")
        self.check_none_btn.clicked.connect(self.check_none)
        controls_layout.addWidget(self.check_all_btn)
        controls_layout.addWidget(self.check_none_btn)
        controls_layout.addStretch(1)
        group_layout.addLayout(controls_layout)

        self.show_box_labels_checkbox = QCheckBox(
            "Show selected box labels"
        )
        self.show_box_labels_checkbox.toggled.connect(
            self._refresh_roi_layers
        )
        group_layout.addWidget(self.show_box_labels_checkbox)

        box_label_color_layout = QHBoxLayout()
        box_label_color_layout.addWidget(QLabel("Text color:"))
        self.box_label_color_btn = QPushButton()
        self.box_label_color_btn.setToolTip(
            "Choose the color of selected neuron box text."
        )
        self.box_label_color_btn.clicked.connect(
            self._choose_box_label_color
        )
        box_label_color_layout.addWidget(self.box_label_color_btn)
        box_label_color_layout.addStretch(1)
        group_layout.addLayout(box_label_color_layout)
        self._update_box_label_color_button()

        self.selection_tree = QTreeWidget()
        self.selection_tree.setColumnCount(2)
        self.selection_tree.setHeaderLabels(["", "Neuron"])
        self.selection_tree.setRootIsDecorated(False)
        self.selection_tree.setAlternatingRowColors(False)
        self.selection_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.selection_tree.setMinimumHeight(190)
        self.selection_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.selection_tree.header().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.selection_tree.itemChanged.connect(
            self._on_selection_item_changed
        )
        self.selection_tree.itemClicked.connect(
            self._on_selection_item_clicked
        )
        group_layout.addWidget(self.selection_tree)
        group.setLayout(group_layout)
        return group

    def _choose_box_label_color(self) -> None:
        color = QColorDialog.getColor(
            QColor(self._box_label_color),
            self,
            "Select box label text color",
        )
        if not color.isValid():
            return
        self._box_label_color = color.name()
        self._update_box_label_color_button()
        box_label_layer = self._managed_box_label_layer()
        if box_label_layer is not None:
            box_label_layer.text.color = self._box_label_color
            box_label_layer.refresh()

    def _update_box_label_color_button(self) -> None:
        color = QColor(self._box_label_color)
        foreground = "#000000" if color.lightness() > 127 else "#ffffff"
        self.box_label_color_btn.setText(color.name().upper())
        self.box_label_color_btn.setStyleSheet(
            f"background-color: {color.name()}; color: {foreground};"
        )

    def _add_labels_appearance_controls(
        self, group_layout: QVBoxLayout
    ) -> None:
        selected_layout = QHBoxLayout()
        selected_layout.addWidget(QLabel("Checked label opacity:"))
        self.selected_opacity_slider = QSlider(Qt.Horizontal)
        self.selected_opacity_slider.setRange(0, 100)
        self.selected_opacity_slider.setValue(50)
        self.selected_opacity_label = QLabel("0.50")
        self.selected_opacity_slider.valueChanged.connect(
            self._on_opacity_changed
        )
        selected_layout.addWidget(self.selected_opacity_slider)
        selected_layout.addWidget(self.selected_opacity_label)
        group_layout.addLayout(selected_layout)

        other_layout = QHBoxLayout()
        other_layout.addWidget(QLabel("Unchecked label opacity:"))
        self.other_opacity_slider = QSlider(Qt.Horizontal)
        self.other_opacity_slider.setRange(0, 100)
        self.other_opacity_slider.setValue(0)
        self.other_opacity_label = QLabel("0.00")
        self.other_opacity_slider.valueChanged.connect(
            self._on_opacity_changed
        )
        other_layout.addWidget(self.other_opacity_slider)
        other_layout.addWidget(self.other_opacity_label)
        group_layout.addLayout(other_layout)

        controls_layout = QHBoxLayout()
        self.hide_unchecked_checkbox = QCheckBox("Hide unchecked labels")
        self.hide_unchecked_checkbox.toggled.connect(self._on_opacity_changed)
        controls_layout.addWidget(self.hide_unchecked_checkbox)
        controls_layout.addStretch(1)
        group_layout.addLayout(controls_layout)

    def _build_annotation_group(self) -> QGroupBox:
        group = QGroupBox("Neuron Annotation")
        group_layout = QVBoxLayout()
        sheet_layout = QHBoxLayout()

        sheet_layout.addWidget(QLabel("Sheet:"))
        self.sheet_name_input = QLineEdit("Neuron Annotations")
        sheet_layout.addWidget(self.sheet_name_input, 1)
        group_layout.addLayout(sheet_layout)

        controls = QHBoxLayout()
        self.load_current_labels_btn = QPushButton("Current IDs")
        self.load_current_labels_btn.clicked.connect(
            self.load_current_ids_to_annotation
        )
        controls.addWidget(self.load_current_labels_btn)

        self.load_excel_btn = QPushButton("Load")
        self.load_excel_btn.clicked.connect(self.load_excel_to_annotation)
        self.load_excel_btn.setEnabled(EXCEL_AVAILABLE)
        controls.addWidget(self.load_excel_btn)
        self.save_annotation_btn = QPushButton("Save")
        self.save_annotation_btn.clicked.connect(self.save_annotation_to_excel)
        self.save_annotation_btn.setEnabled(EXCEL_AVAILABLE)
        controls.addWidget(self.save_annotation_btn)
        group_layout.addLayout(controls)

        self.annotation_table = QTableWidget(0, 3)
        self.annotation_table.setHorizontalHeaderLabels(
            ["digital", "biological", "annotation"]
        )
        self.annotation_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.annotation_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.annotation_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self.annotation_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.annotation_table.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self.annotation_table.itemSelectionChanged.connect(
            self._on_annotation_selection_changed
        )
        self.annotation_table.itemChanged.connect(
            self._on_annotation_item_changed
        )
        group_layout.addWidget(self.annotation_table)
        group.setLayout(group_layout)
        return group

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("Status")
        group_layout = QVBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green;")
        group_layout.addWidget(self.status_label)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(90)
        group_layout.addWidget(self.info_text)
        group.setLayout(group_layout)
        return group

    # ------------------------------------------------------------------
    # Viewer and layer lifecycle
    # ------------------------------------------------------------------
    def _connect_viewer_events(self) -> None:
        events = self.viewer.layers.events
        events.inserted.connect(self._refresh_label_layers)
        events.inserted.connect(self._refresh_image_layers)
        events.removed.connect(self._on_layer_removed)
        events.reordered.connect(self._refresh_label_layers)
        events.reordered.connect(self._refresh_image_layers)
        if hasattr(events, "renamed"):
            events.renamed.connect(self._refresh_label_layers)
            events.renamed.connect(self._refresh_image_layers)

        dims_events = self.viewer.dims.events
        dims_events.point.connect(self._on_dims_changed)
        dims_events.ndisplay.connect(self._on_dims_changed)
        dims_events.order.connect(self._on_dims_changed)

    def _disconnect_viewer_events(self) -> None:
        events = self.viewer.layers.events
        for emitter, callback in (
            (events.inserted, self._refresh_label_layers),
            (events.inserted, self._refresh_image_layers),
            (events.removed, self._on_layer_removed),
            (events.reordered, self._refresh_label_layers),
            (events.reordered, self._refresh_image_layers),
        ):
            with suppress(TypeError, ValueError):
                emitter.disconnect(callback)
        if hasattr(events, "renamed"):
            with suppress(TypeError, ValueError):
                events.renamed.disconnect(self._refresh_label_layers)
            with suppress(TypeError, ValueError):
                events.renamed.disconnect(self._refresh_image_layers)

        dims_events = self.viewer.dims.events
        for emitter in (
            dims_events.point,
            dims_events.ndisplay,
            dims_events.order,
        ):
            with suppress(TypeError, ValueError):
                emitter.disconnect(self._on_dims_changed)

    def _bind_keys(self) -> None:
        bindings = (("Q", self._previous_key), ("W", self._next_key))
        for key, callback in bindings:
            try:
                self.viewer.bind_key(key, callback)
                self._keys_bound.append(key)
            except ValueError:
                self.update_status(
                    f"Hotkey {key} is already assigned; button navigation remains available",
                    "orange",
                )

    def _unbind_keys(self) -> None:
        for key in self._keys_bound:
            with suppress(KeyError, ValueError):
                self.viewer.bind_key(key, None, overwrite=True)
        self._keys_bound.clear()

    def _refresh_label_layers(self, event=None) -> None:
        del event
        if self._closed:
            return
        current = self.current_labels
        layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, Labels)
            and layer.metadata.get(ROLE_KEY) != ROLE_Z_LABELS
        ]

        self.labels_combo.blockSignals(True)
        self.labels_combo.clear()
        self.labels_combo.addItem("None", None)
        for layer in layers:
            self.labels_combo.addItem(layer.name, layer)
        if current in layers:
            self.labels_combo.setCurrentIndex(layers.index(current) + 1)
        else:
            self.labels_combo.setCurrentIndex(0)
        self.labels_combo.setEnabled(bool(layers))
        self.labels_combo.blockSignals(False)

        if current is not None and current not in layers:
            self._detach_labels_layer(restore=False)
        self._set_labels_controls_enabled()

    def _refresh_image_layers(self, event=None) -> None:
        del event
        if self._closed:
            return
        current = self.current_image
        sources = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and not isinstance(layer, Labels)
            and layer.metadata.get(ROLE_KEY) != ROLE_Z_IMAGE
            and self._is_compatible_image_source(layer)
        ]

        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        if sources:
            for layer in sources:
                self.image_combo.addItem(layer.name, layer)
            target = current if current in sources else sources[0]
            self.image_combo.setCurrentIndex(sources.index(target))
            self.image_combo.setEnabled(self._z_source_image is None)
            self.split_z_btn.setEnabled(True)
        else:
            target = None
            self.image_combo.addItem("No compatible Image layers")
            self.image_combo.setEnabled(False)
            self.split_z_btn.setEnabled(False)
        self.image_combo.blockSignals(False)
        self.load_roi_btn.setEnabled(
            target is not None and self.roi_dataset is None
        )
        if target is not current:
            self._set_current_image(target)

    def _on_image_changed(self, index: int) -> None:
        del index
        source = self.image_combo.currentData()
        self._set_current_image(
            source
            if isinstance(source, Image) and not isinstance(source, Labels)
            else None
        )

    def _set_current_image(self, source: Image | None) -> None:
        if source is self.current_image:
            return
        self._disconnect_current_image_events()
        if self._z_ranges:
            self._clear_z_layers()
        self._remove_roi_layers()
        self.current_image = source
        self.load_roi_btn.setEnabled(
            source is not None and self.roi_dataset is None
        )
        if source is not None:
            # Clearing an active Z session refreshes this selector against the
            # previous source. Re-sync it after the new authority is installed.
            self._refresh_image_layers()
        if self.current_labels is not None:
            try:
                self._validate_labels_binding(source, self.current_labels)
            except ValueError:
                self._detach_labels_layer(restore=True)
                self._refresh_label_layers()

        if not isinstance(source, Image) or isinstance(source, Labels):
            self._z_profile_source = None
            self._z_profile_time = None
            self.z_profile.clear_profile()
            self.z_profile_state_label.clear()
            self.z_profile_refresh_btn.setEnabled(False)
            self._refresh_selection_item_styles()
            return
        self._validate_image_source(source)
        source.events.data.connect(self._on_current_image_changed)
        for event_name in Z_SOURCE_GEOMETRY_EVENTS:
            getattr(source.events, event_name).connect(
                self._on_current_image_changed
            )
        self.z_profile_refresh_btn.setEnabled(True)
        if source is not self._z_profile_source:
            self._z_profile_source = None
            self._z_profile_time = None
            self.z_profile.clear_profile()
            self.refresh_z_profile()
        if self.roi_dataset is not None:
            self._ensure_roi_layers()
            self._refresh_roi_layers()
        self._refresh_selection_item_styles()
        self._update_roi_info()

    def _disconnect_current_image_events(self) -> None:
        source = self.current_image
        if source is None:
            return
        with suppress(TypeError, ValueError):
            source.events.data.disconnect(self._on_current_image_changed)
        for event_name in Z_SOURCE_GEOMETRY_EVENTS:
            with suppress(TypeError, ValueError):
                getattr(source.events, event_name).disconnect(
                    self._on_current_image_changed
                )

    def _on_current_image_changed(self, event=None) -> None:
        del event
        source = self.current_image
        if source is None:
            return
        if self._z_ranges:
            self._clear_z_layers()
        if not self._is_compatible_image_source(source):
            self._set_current_image(None)
            self._refresh_image_layers()
            return
        if self.current_labels is not None:
            try:
                self._validate_labels_binding(source, self.current_labels)
            except ValueError as error:
                self._detach_labels_layer(restore=True)
                self._refresh_label_layers()
                self.update_status(
                    f"Labels binding cleared: {error}", "orange"
                )
        self._remove_roi_layers()
        self._ensure_roi_layers()
        self._refresh_roi_layers()

    # Compatibility with the old Z-specific callback name.
    def _on_z_image_changed(self, index: int) -> None:
        self._on_image_changed(index)

    def refresh_z_profile(self) -> None:
        """Count above-threshold pixels per Z for the current Image and time."""

        source = self.z_image_combo.currentData()
        if not isinstance(source, Image) or isinstance(source, Labels):
            return
        time_index = self._z_profile_time_index(source)
        threshold = float(self.z_profile_threshold_spin.value())
        self.z_profile_refresh_btn.setEnabled(False)
        self.z_profile_state_label.setText("computing…")
        try:
            values = z_threshold_count_profile(
                source.data,
                threshold=threshold,
                time_index=time_index,
            )
        except Exception as error:  # noqa: BLE001 - Qt callback boundary
            self._z_profile_source = None
            self._z_profile_time = None
            self.z_profile.clear_profile()
            self.z_profile_state_label.setText("unavailable")
            self.update_status(f"Could not compute Z profile: {error}", "red")
            return
        finally:
            self.z_profile_refresh_btn.setEnabled(True)

        self._z_profile_source = source
        self._z_profile_time = time_index
        self.z_profile.set_profile(values)
        self._sync_z_profile_cuts(self.z_cuts_input.text())
        time_text = f"t={time_index}" if source.ndim == 4 else ""
        self.z_profile_state_label.setText(time_text)

    def _z_profile_time_index(self, source: Image) -> int:
        if source.ndim != 4:
            return 0
        steps = self.viewer.dims.current_step
        value = int(steps[-4]) if len(steps) >= 4 else 0
        return min(max(value, 0), int(source.data.shape[0]) - 1)

    def _sync_z_profile_cuts(self, text: str) -> None:
        try:
            cuts = parse_z_cuts(text)
        except ValueError:
            cuts = ()
        self.z_profile.set_cuts(cuts)

    def _toggle_z_profile_cut(self, cut: int) -> None:
        try:
            cuts = set(parse_z_cuts(self.z_cuts_input.text()))
        except ValueError as error:
            self.update_status(
                f"Fix Z cuts before using the plot: {error}", "orange"
            )
            return
        if cut in cuts:
            cuts.remove(cut)
        else:
            cuts.add(cut)
        source = self.z_image_combo.currentData()
        if not isinstance(source, Image):
            return
        ordered = tuple(sorted(cuts))
        build_z_layer_ranges(int(source.data.shape[-3]), ordered)
        self.z_cuts_input.setText(",".join(str(value) for value in ordered))

    def split_z_layers(self) -> None:
        """Create runtime Image slices and synchronize Labels and boxes."""
        try:
            self._create_z_layers()
        except Exception as error:  # noqa: BLE001 - Qt callback boundary
            QMessageBox.critical(
                self,
                "Invalid Z layers",
                f"Could not split the selected layers:\n{error}",
            )
            self.update_status("Z-layer split failed", "red")

    def _create_z_layers(self) -> None:
        source = self.current_image
        if not isinstance(source, Image) or isinstance(source, Labels):
            raise RuntimeError("Select a 3D or 4D Image layer")
        self._validate_z_layer_sources(source, self.current_labels)

        cuts = parse_z_cuts(self.z_cuts_input.text())
        ranges = build_z_layer_ranges(int(source.data.shape[-3]), cuts)
        if not cuts:
            raise ValueError("Enter at least one Z cut")

        if self._z_ranges:
            self._clear_z_layers()

        self._z_source_image = source
        self._z_ranges = ranges
        self._z_source_image_visible = bool(source.visible)
        if self.current_labels is not None:
            self._z_source_labels_visible = bool(self.current_labels.visible)
            self._z_source_labels_data = self.current_labels.data

        try:
            source.events.data.connect(self._on_z_source_image_data_changed)
            for event_name in Z_SOURCE_GEOMETRY_EVENTS:
                getattr(source.events, event_name).connect(
                    self._on_z_source_geometry_changed
                )
                if self.current_labels is not None:
                    getattr(self.current_labels.events, event_name).connect(
                        self._on_z_source_geometry_changed
                    )
            source_index = list(self.viewer.layers).index(source)
            for z_range in ranges:
                derived = self.viewer.add_image(
                    slice_z_range(source.data, z_range),
                    name=f"{source.name} – Layer {z_range.index + 1}",
                    colormap=source.colormap,
                    contrast_limits=tuple(source.contrast_limits),
                    gamma=float(source.gamma),
                    opacity=float(source.opacity),
                    blending="additive",
                    rendering=source.rendering,
                    depiction=source.depiction,
                    attenuation=float(source.attenuation),
                    iso_threshold=float(source.iso_threshold),
                    projection_mode=source.projection_mode,
                    interpolation2d=source.interpolation2d,
                    interpolation3d=source.interpolation3d,
                    rgb=source.rgb,
                    scale=tuple(source.scale),
                    translate=shifted_z_translation(
                        tuple(source.translate),
                        tuple(source.scale),
                        z_range.start,
                    ),
                    axis_labels=tuple(source.axis_labels),
                    units=tuple(source.units),
                    metadata={
                        ROLE_KEY: ROLE_Z_IMAGE,
                        "z_layer_index": z_range.index,
                        "z_start": z_range.start,
                        "z_stop": z_range.stop,
                        "z_session": self._z_session_token,
                    },
                )
                self._z_image_layers.append(derived)
                current_index = list(self.viewer.layers).index(derived)
                target_index = source_index + len(self._z_image_layers)
                if current_index != target_index:
                    self.viewer.layers.move(current_index, target_index)

            if self.current_labels is not None:
                first_range = ranges[0]
                self._z_labels_proxy = self.viewer.add_labels(
                    slice_z_range(self.current_labels.data, first_range),
                    name=f"{self.current_labels.name} – Z view",
                    colormap=self.current_labels.colormap,
                    opacity=float(self.current_labels.opacity),
                    blending=self.current_labels.blending,
                    rendering=self.current_labels.rendering,
                    depiction=self.current_labels.depiction,
                    iso_gradient_mode=self.current_labels.iso_gradient_mode,
                    projection_mode=self.current_labels.projection_mode,
                    scale=tuple(self.current_labels.scale),
                    translate=shifted_z_translation(
                        tuple(self.current_labels.translate),
                        tuple(self.current_labels.scale),
                        first_range.start,
                    ),
                    axis_labels=tuple(self.current_labels.axis_labels),
                    units=tuple(self.current_labels.units),
                    visible=False,
                    metadata={
                        ROLE_KEY: ROLE_Z_LABELS,
                        "z_layer_index": first_range.index,
                        "z_start": first_range.start,
                        "z_stop": first_range.stop,
                        "z_session": self._z_session_token,
                    },
                )
                self._z_labels_proxy.editable = False
                self._z_labels_proxy.contour = self.current_labels.contour
            source.visible = False

            self.z_view_combo.blockSignals(True)
            self.z_view_combo.clear()
            self.z_view_combo.addItem("All", None)
            for z_range in ranges:
                self.z_view_combo.addItem(
                    f"Layer {z_range.index + 1} · "
                    f"z {z_range.start}–{z_range.stop - 1}",
                    z_range.index,
                )
            self.z_view_combo.setCurrentIndex(0)
            self.z_view_combo.blockSignals(False)
            self.z_view_combo.setEnabled(True)
            self.clear_z_btn.setEnabled(True)
            self.z_image_combo.setEnabled(False)
            self._z_active_index = None
            self._apply_z_view()
            self._apply_label_opacity()
        except Exception:
            self._clear_z_layers()
            raise

        self.update_status(
            f"Split {source.name} into {len(ranges)} Z layers",
            "green",
        )

    def _validate_z_layer_sources(
        self, source: Image, labels: Labels | None
    ) -> None:
        self._validate_image_source(source)
        if labels is not None:
            self._validate_labels_binding(source, labels)

    def _validate_image_source(self, source: Image) -> None:
        if source.rgb:
            raise ValueError("RGB Image layers are not supported")
        if source.multiscale:
            raise ValueError("Multiscale Image layers are not supported")
        if source.depiction != "volume":
            raise ValueError("Only volume depiction is supported")
        if len(source.experimental_clipping_planes):
            raise ValueError(
                "Experimental clipping planes are not supported"
            )
        if source.ndim not in (3, 4):
            raise ValueError("Image must use (z,y,x) or (t,z,y,x)")
        if not self._uses_axis_aligned_transform(source):
            raise ValueError("Image transform must be axis-aligned")
        data_module = type(source.data).__module__.split(".", 1)[0]
        if data_module == "zarr":
            raise ValueError(
                "Direct Zarr arrays are not supported; wrap the array "
                "as a Dask array before use"
            )

    def _validate_labels_binding(
        self, source: Image | None, labels: Labels
    ) -> None:
        if source is None:
            raise ValueError("Select an Image before binding Labels")
        if labels.multiscale:
            raise ValueError("Multiscale Labels layers are not supported")
        if labels.depiction != "volume":
            raise ValueError("Only volume depiction is supported")
        if len(labels.experimental_clipping_planes):
            raise ValueError("Experimental clipping planes are not supported")
        if labels.ndim not in (3, 4):
            raise ValueError("Labels must use (z,y,x) or (t,z,y,x)")
        if source.ndim != labels.ndim:
            raise ValueError("Image and Labels dimensions do not match")
        if tuple(source.data.shape) != tuple(labels.data.shape):
            raise ValueError("Image and Labels shapes do not match")
        if tuple(source.axis_labels) != tuple(labels.axis_labels):
            raise ValueError("Image and Labels axis labels do not match")
        if not np.array_equal(source.scale, labels.scale):
            raise ValueError("Image and Labels scale do not match")
        if not np.array_equal(source.translate, labels.translate):
            raise ValueError("Image and Labels translation do not match")
        if tuple(source.units) != tuple(labels.units):
            raise ValueError("Image and Labels units do not match")
        if not self._uses_axis_aligned_transform(labels):
            raise ValueError("Labels transform must be axis-aligned")
        data_module = type(labels.data).__module__.split(".", 1)[0]
        if data_module == "zarr":
            raise ValueError(
                "Direct Zarr arrays are not supported; wrap the array "
                "as a Dask array before use"
            )

    def _is_compatible_image_source(self, source: Image) -> bool:
        try:
            self._validate_image_source(source)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _uses_axis_aligned_transform(layer: Image | Labels) -> bool:
        ndim = layer.ndim
        return (
            np.allclose(layer.rotate, np.eye(ndim))
            and np.allclose(layer.shear, 0.0)
            and np.allclose(layer.affine.affine_matrix, np.eye(ndim + 1))
        )

    def _on_z_source_image_data_changed(self, event=None) -> None:
        del event
        if self._z_ranges:
            self._clear_z_layers()
            self.update_status(
                "Z layers cleared because the source Image changed",
                "orange",
            )

    def _on_z_source_geometry_changed(self, event=None) -> None:
        del event
        if self._z_ranges:
            self._clear_z_layers()
            self.update_status(
                "Z layers cleared because a source transform changed",
                "orange",
            )

    def _on_z_view_changed(self, index: int) -> None:
        del index
        if not self._z_ranges:
            return
        value = self.z_view_combo.currentData()
        self._z_active_index = None if value is None else int(value)
        self._apply_z_view()

    def _active_z_range(self) -> ZLayerRange | None:
        if self._z_active_index is None:
            return None
        if not 0 <= self._z_active_index < len(self._z_ranges):
            return None
        return self._z_ranges[self._z_active_index]

    def _apply_z_view(self) -> None:
        if not self._z_ranges:
            return
        active_range = self._active_z_range()
        for z_range, layer in zip(
            self._z_ranges, self._z_image_layers, strict=True
        ):
            layer.visible = (
                active_range is None or z_range.index == active_range.index
            )

        if self._z_source_image is not None:
            self._z_source_image.visible = False
        if active_range is None:
            if self.current_labels is not None:
                self.current_labels.visible = True
            if self._z_labels_proxy is not None:
                self._z_labels_proxy.visible = False
        elif self._z_labels_proxy is not None and self.current_labels is not None:
            self.current_labels.visible = False
            self._z_labels_proxy.data = slice_z_range(
                self.current_labels.data, active_range
            )
            self._z_labels_proxy.editable = False
            self._z_labels_proxy.translate = shifted_z_translation(
                tuple(self.current_labels.translate),
                tuple(self.current_labels.scale),
                active_range.start,
            )
            self._z_labels_proxy.metadata.update(
                {
                    "z_layer_index": active_range.index,
                    "z_start": active_range.start,
                    "z_stop": active_range.stop,
                }
            )
            self._z_labels_proxy.visible = True
        if active_range is not None:
            self._clamp_z_to_active_layer(active_range)

        self._refresh_selection_item_styles()
        self._refresh_roi_layers()
        self._update_info()

    def _clamp_z_to_active_layer(self, z_range: ZLayerRange) -> None:
        if self.viewer.dims.ndisplay != 2:
            return
        steps = list(self.viewer.dims.current_step)
        if len(steps) < 3:
            return
        z_axis = len(steps) - 3
        current_z = int(steps[z_axis])
        if not z_range.start <= current_z < z_range.stop:
            steps[z_axis] = (z_range.start + z_range.stop - 1) // 2
            self.viewer.dims.current_step = tuple(steps)

    def clear_z_layers(self) -> None:
        """Remove runtime Z layers and restore source visibility."""
        if not self._z_ranges:
            return
        self._clear_z_layers()
        self.update_status("Z layers cleared", "green")

    def _clear_z_layers(self) -> None:
        source_image = self._z_source_image
        source_labels = self.current_labels
        image_visible = self._z_source_image_visible
        labels_visible = self._z_source_labels_visible

        if source_image is not None:
            with suppress(TypeError, ValueError):
                source_image.events.data.disconnect(
                    self._on_z_source_image_data_changed
                )
            for event_name in Z_SOURCE_GEOMETRY_EVENTS:
                with suppress(TypeError, ValueError):
                    getattr(source_image.events, event_name).disconnect(
                        self._on_z_source_geometry_changed
                    )
        if source_labels is not None:
            for event_name in Z_SOURCE_GEOMETRY_EVENTS:
                with suppress(TypeError, ValueError):
                    getattr(source_labels.events, event_name).disconnect(
                        self._on_z_source_geometry_changed
                    )

        self._z_cleanup = True
        try:
            managed = [
                layer
                for layer in self.viewer.layers
                if layer in self._z_image_layers
                or layer is self._z_labels_proxy
                or (
                    layer.metadata.get(ROLE_KEY) in MANAGED_Z_ROLES
                    and layer.metadata.get("z_session")
                    == self._z_session_token
                )
            ]
            for layer in managed:
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
        finally:
            self._z_cleanup = False

        if (
            source_image is not None
            and source_image in self.viewer.layers
            and image_visible is not None
        ):
            source_image.visible = image_visible
        if (
            source_labels is not None
            and source_labels in self.viewer.layers
            and labels_visible is not None
        ):
            source_labels.visible = labels_visible

        self._z_ranges = ()
        self._z_source_image = None
        self._z_image_layers.clear()
        self._z_labels_proxy = None
        self._z_active_index = None
        self._z_source_image_visible = None
        self._z_source_labels_visible = None
        self._z_source_labels_data = None

        self.z_view_combo.blockSignals(True)
        self.z_view_combo.clear()
        self.z_view_combo.addItem("All", None)
        self.z_view_combo.blockSignals(False)
        self.z_view_combo.setEnabled(False)
        self.clear_z_btn.setEnabled(False)
        self._refresh_image_layers()
        self._refresh_label_layers()
        self._refresh_selection_item_styles()
        self._refresh_roi_layers()
        self._update_info()

    def _on_labels_changed(self, index: int) -> None:
        layer = self.labels_combo.itemData(index) if index >= 0 else None
        if layer is self.current_labels:
            return
        if not isinstance(layer, Labels):
            if layer is not None:
                self._set_labels_combo_layer(self.current_labels)
                return
        elif layer is not None:
            try:
                self._validate_labels_binding(self.current_image, layer)
            except ValueError as error:
                self._set_labels_combo_layer(self.current_labels)
                self.update_status(f"Labels binding rejected: {error}", "red")
                return

        if self._z_ranges:
            self._clear_z_layers()
        self._detach_labels_layer(restore=True)
        if layer is None:
            self._reset_labels_combo_to_none()
            self._set_labels_controls_enabled()
            self.update_status("Labels binding cleared", "blue")
            return
        self.current_labels = layer
        self._original_colormap = layer.colormap
        self._original_opacity = float(layer.opacity)
        self._base_colors.clear()
        layer.events.data.connect(self._on_label_data_changed)
        layer.events.labels_update.connect(self._on_label_data_changed)
        for event_name in Z_SOURCE_GEOMETRY_EVENTS:
            getattr(layer.events, event_name).connect(
                self._on_bound_labels_geometry_changed
            )
        self._cache_base_colors(self._available_ids)
        self._apply_label_opacity()
        self._set_labels_combo_layer(layer)
        self._set_labels_controls_enabled()
        self.update_status(f"Bound Labels: {layer.name}", "blue")

    def _reset_labels_combo_to_none(self) -> None:
        self._set_labels_combo_layer(None)

    def _set_labels_combo_layer(self, layer: Labels | None) -> None:
        index = 0
        if layer is not None:
            for candidate in range(1, self.labels_combo.count()):
                if self.labels_combo.itemData(candidate) is layer:
                    index = candidate
                    break
        self.labels_combo.blockSignals(True)
        self.labels_combo.setCurrentIndex(index)
        self.labels_combo.blockSignals(False)

    # Compatibility adapter for integrations using the previous callback.
    def _on_layer_changed(self, layer_name: str) -> None:
        index = self.labels_combo.findText(layer_name)
        self.labels_combo.setCurrentIndex(index if index >= 0 else 0)

    def _detach_labels_layer(self, *, restore: bool) -> None:
        if self.current_labels is None:
            return
        with suppress(TypeError, ValueError):
            self.current_labels.events.data.disconnect(
                self._on_label_data_changed
            )
        with suppress(TypeError, ValueError):
            self.current_labels.events.labels_update.disconnect(
                self._on_label_data_changed
            )
        for event_name in Z_SOURCE_GEOMETRY_EVENTS:
            with suppress(TypeError, ValueError):
                getattr(self.current_labels.events, event_name).disconnect(
                    self._on_bound_labels_geometry_changed
                )
        if restore:
            self._restore_layer_display()
        self.current_labels = None
        self._original_colormap = None
        self._original_opacity = None
        self._base_colors.clear()
        self._set_labels_controls_enabled()

    def _detach_current_layer(self, *, restore: bool) -> None:
        self._detach_labels_layer(restore=restore)

    def _on_layer_removed(self, event) -> None:
        if self._closed:
            return
        removed = getattr(event, "value", None)
        metadata = getattr(removed, "metadata", {})
        role = metadata.get(ROLE_KEY)
        if role in MANAGED_Z_ROLES:
            if (
                metadata.get("z_session") == self._z_session_token
                and not self._z_cleanup
            ):
                self._clear_z_layers()
            return
        if (
            isinstance(removed, Points | Vectors)
            and removed.metadata.get(ROLE_KEY) in MANAGED_ROI_ROLES
        ):
            return
        if removed is self._z_source_image or removed is self.current_labels:
            self._clear_z_layers()
        if removed is self.current_labels:
            self._detach_labels_layer(restore=False)
            self._reset_labels_combo_to_none()
        if removed is self.current_image:
            self._set_current_image(None)
        self._refresh_label_layers()
        self._refresh_image_layers()

    def _on_label_data_changed(self, event=None) -> None:
        del event
        if (
            self._z_ranges
            and self.current_labels is not None
            and self.current_labels.data is not self._z_source_labels_data
        ):
            self._clear_z_layers()
        if self.current_labels is not None:
            try:
                self._validate_labels_binding(
                    self.current_image, self.current_labels
                )
            except ValueError as error:
                self._detach_labels_layer(restore=True)
                self._reset_labels_combo_to_none()
                self.update_status(
                    f"Labels binding cleared: {error}", "orange"
                )
                return
        self._cache_base_colors(self._available_ids)
        self._apply_label_opacity()
        if self._z_labels_proxy is not None:
            self._z_labels_proxy.refresh()
        self._refresh_roi_layers()

    def _on_bound_labels_geometry_changed(self, event=None) -> None:
        del event
        if self.current_labels is None:
            return
        try:
            self._validate_labels_binding(
                self.current_image, self.current_labels
            )
        except ValueError as error:
            if self._z_ranges:
                self._clear_z_layers()
            self._detach_labels_layer(restore=True)
            self._reset_labels_combo_to_none()
            self.update_status(f"Labels binding cleared: {error}", "orange")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        """Restore managed display state and disconnect all external events."""
        if self._closed:
            return
        self._closed = True
        self._disconnect_viewer_events()
        self._unbind_keys()
        self._clear_z_layers()
        self._disconnect_current_image_events()
        self.current_image = None
        self._detach_current_layer(restore=True)
        self._remove_roi_layers()

    # ------------------------------------------------------------------
    # ROI identities, checkable selection, and navigation
    # ------------------------------------------------------------------
    def _refresh_available_ids(self, *, select_first: bool) -> None:
        ids = (
            self.roi_dataset.neuron_ids
            if self.roi_dataset is not None
            else []
        )
        self._available_ids = ids

        if self.active_id not in ids:
            self.active_id = None
        self.checked_ids.intersection_update(ids)

        if select_first and ids and self.active_id is None:
            valid = self._valid_navigation_ids()
            self.active_id = valid[0] if valid else ids[0]
            self.checked_ids = {self.active_id}

        self._cache_base_colors(ids)
        self._rebuild_selection_items()
        self._sync_annotation_ids()
        self._apply_label_opacity()
        self._set_labels_controls_enabled()

    def _background_value(self) -> int:
        colormap = (
            self._original_colormap
            if self._original_colormap is not None
            else getattr(self.current_labels, "colormap", None)
        )
        return int(getattr(colormap, "background_value", 0))

    def _label_value(self, display_id: int) -> int:
        return neuron_id_to_label_value(display_id)

    def _cache_base_colors(self, display_ids: list[int]) -> None:
        if self.current_labels is None or self._original_colormap is None:
            return
        label_values = {self._label_value(item_id) for item_id in display_ids}
        data = self.current_labels.data
        size = getattr(data, "size", None)
        if (
            isinstance(data, np.ndarray)
            and size is not None
            and int(size) <= MAX_EXACT_LABEL_PIXELS
        ):
            background = self._background_value()
            label_values.update(
                int(value)
                for value in np.unique(data)
                if int(value) != background
            )

        managed_colormap = self.current_labels.colormap
        if managed_colormap is not self._original_colormap:
            self.current_labels.colormap = self._original_colormap
        try:
            for label_value in label_values:
                if label_value in self._base_colors:
                    continue
                mapped = np.asarray(
                    self.current_labels.get_color(label_value), dtype=float
                ).reshape(-1)
                if mapped.size != 4 or not np.all(np.isfinite(mapped)):
                    raise ValueError(
                        "Labels colormap returned an invalid RGBA value "
                        f"for label {label_value}"
                    )
                self._base_colors[label_value] = tuple(mapped)
        finally:
            if managed_colormap is not self._original_colormap:
                self.current_labels.colormap = managed_colormap

    def _rebuild_selection_items(self) -> None:
        self._ui_sync = True
        try:
            self.selection_tree.clear()
            self._selection_items.clear()
            for display_id in self._available_ids:
                item = QTreeWidgetItem()
                item.setFlags(
                    item.flags()
                    | Qt.ItemIsUserCheckable
                    | Qt.ItemIsSelectable
                    | Qt.ItemIsEnabled
                )
                item.setData(0, Qt.UserRole, display_id)
                item.setCheckState(
                    0,
                    Qt.Checked
                    if display_id in self.checked_ids
                    else Qt.Unchecked,
                )
                self.selection_tree.addTopLevelItem(item)
                self._selection_items[display_id] = item
        finally:
            self._ui_sync = False
        self._refresh_selection_item_styles()

    def _refresh_selection_item_styles(self) -> None:
        names = self._annotation_names()
        valid_time_ids = set(self._valid_time_ids())
        self._ui_sync = True
        try:
            for display_id in self._available_ids:
                self._style_selection_item(
                    display_id,
                    valid_at_time=display_id in valid_time_ids,
                    in_active_layer=self._id_in_active_z_layer(display_id),
                    biological_name=names.get(display_id, ""),
                )
            if self.active_id is None:
                self.selection_tree.clearSelection()
                self.selection_tree.setCurrentItem(None)
            else:
                item = self._selection_items.get(self.active_id)
                if item is not None:
                    self.selection_tree.setCurrentItem(item)
                    self.selection_tree.scrollToItem(item)
        finally:
            self._ui_sync = False

    def _style_selection_item(
        self,
        display_id: int,
        *,
        valid_at_time: bool | None = None,
        in_active_layer: bool | None = None,
        biological_name: str = "",
    ) -> None:
        item = self._selection_items.get(display_id)
        if item is None:
            return
        if valid_at_time is None:
            valid_at_time = display_id in set(self._valid_time_ids())
        if in_active_layer is None:
            in_active_layer = self._id_in_active_z_layer(display_id)
        if not biological_name:
            biological_name = self._annotation_names().get(display_id, "")

        text = f"Neuron {display_id}"
        if biological_name:
            text += f" · {biological_name}"
        if self.roi_dataset is not None and not valid_at_time:
            text += " (missing)"
        item.setText(1, text)
        item.setCheckState(
            0,
            Qt.Checked if display_id in self.checked_ids else Qt.Unchecked,
        )
        font = item.font(1)
        font.setBold(display_id == self.active_id)
        item.setFont(1, font)
        if valid_at_time and in_active_layer:
            rgba = self._display_color(display_id)
            color = QColor.fromRgbF(*rgba[:3])
        else:
            color = QColor("#777777")
        item.setForeground(1, QBrush(color))

    def _display_color(self, display_id: int) -> np.ndarray:
        return np.asarray(neuron_color(display_id), dtype=float)

    def _on_selection_item_changed(
        self, item: QTreeWidgetItem, column: int
    ) -> None:
        if self._ui_sync or column != 0:
            return
        display_id = int(item.data(0, Qt.UserRole))
        checked = item.checkState(0) == Qt.Checked
        if checked:
            self.checked_ids.add(display_id)
            self.active_id = display_id
        else:
            self.checked_ids.discard(display_id)
            if self.active_id == display_id:
                self.active_id = None
        self._selection_changed(locate=checked)

    def _on_selection_item_clicked(
        self, item: QTreeWidgetItem, column: int
    ) -> None:
        if self._ui_sync or column == 0:
            return
        display_id = int(item.data(0, Qt.UserRole))
        self.activate_id(display_id, locate=True)

    def activate_id(self, display_id: int, *, locate: bool = True) -> None:
        """Check and activate one available identity."""
        if display_id not in self._available_ids:
            return
        self.checked_ids.add(display_id)
        self.active_id = display_id
        self._selection_changed(locate=locate)

    def check_all(self) -> None:
        """Check every available identity without changing an existing active."""
        self.checked_ids = set(self._available_ids)
        locate = False
        if self.active_id is None:
            valid = self._valid_navigation_ids()
            if valid:
                self.active_id = valid[0]
                locate = True
        self._selection_changed(locate=locate)

    def check_none(self) -> None:
        """Clear both the checked collection and active identity."""
        self.checked_ids.clear()
        self.active_id = None
        self._selection_changed(locate=False)

    def _selection_changed(self, *, locate: bool) -> None:
        self._refresh_selection_item_styles()
        self._apply_label_opacity()
        self._refresh_roi_layers()
        self._sync_annotation_to_active()
        if locate:
            self._locate_active_box()
        self._update_info()

    def _valid_navigation_ids(self) -> list[int]:
        return [
            display_id
            for display_id in self._valid_time_ids()
            if self._id_in_active_z_layer(display_id)
        ]

    def _valid_time_ids(self) -> list[int]:
        if self.roi_dataset is None:
            return list(self._available_ids)
        return self.roi_dataset.valid_ids(self._viewer_time())

    def _id_in_active_z_layer(self, display_id: int) -> bool:
        active_range = self._active_z_range()
        if active_range is None or self.roi_dataset is None:
            return True
        box = self.roi_dataset.get_box(self._viewer_time(), display_id)
        if box is None:
            return False
        owner = find_z_layer(box.center_zyx[0], self._z_ranges)
        return owner is not None and owner.index == active_range.index

    def navigate(self, step: int) -> None:
        valid_ids = self._valid_navigation_ids()
        if not valid_ids:
            self.update_status(
                "No valid neuron at the current volume", "orange"
            )
            return
        if self.active_id not in valid_ids:
            next_id = valid_ids[0] if step >= 0 else valid_ids[-1]
        else:
            index = valid_ids.index(self.active_id)
            next_id = valid_ids[(index + step) % len(valid_ids)]
        self.activate_id(next_id, locate=True)

    def _previous_key(self, viewer=None) -> None:
        del viewer
        self.navigate(-1)

    def _next_key(self, viewer=None) -> None:
        del viewer
        self.navigate(1)

    # ------------------------------------------------------------------
    # Labels appearance
    # ------------------------------------------------------------------
    def _on_opacity_changed(self, value=None) -> None:
        del value
        self.selected_opacity_label.setText(
            f"{self.selected_opacity_slider.value() / 100:.2f}"
        )
        self._set_labels_controls_enabled()
        other = (
            0.0
            if self.hide_unchecked_checkbox.isChecked()
            else self.other_opacity_slider.value() / 100
        )
        self.other_opacity_label.setText(f"{other:.2f}")
        self._apply_label_opacity()

    def _set_labels_controls_enabled(self) -> None:
        enabled = self.current_labels is not None and self.roi_dataset is not None
        self.selected_opacity_slider.setEnabled(enabled)
        self.hide_unchecked_checkbox.setEnabled(enabled)
        self.other_opacity_slider.setEnabled(
            enabled and not self.hide_unchecked_checkbox.isChecked()
        )

    def _apply_label_opacity(self) -> None:
        if (
            self.current_labels is None
            or self._original_colormap is None
            or not self._available_ids
        ):
            return

        selected_alpha = self.selected_opacity_slider.value() / 100
        other_alpha = (
            0.0
            if self.hide_unchecked_checkbox.isChecked()
            else self.other_opacity_slider.value() / 100
        )
        colors: dict[int | None, tuple[float, float, float, float]] = {
            None: (0.0, 0.0, 0.0, 0.0)
        }
        for label_value, base in self._base_colors.items():
            display_id = label_value - 1
            rgba = list(base)
            rgba[3] = (
                selected_alpha
                if display_id in self.checked_ids
                else other_alpha
            )
            colors[label_value] = tuple(rgba)

        direct = cmap.direct_colormap(colors)
        direct.background_value = self._background_value()
        direct.name = "neuron_roi_visibility"
        targets = [self.current_labels]
        if self._z_labels_proxy is not None:
            targets.append(self._z_labels_proxy)
        for layer in targets:
            layer.colormap = direct
            layer.opacity = 1.0

    def _restore_layer_display(self) -> None:
        if self.current_labels is None:
            return
        targets = [self.current_labels]
        if self._z_labels_proxy is not None:
            targets.append(self._z_labels_proxy)
        for layer in targets:
            if self._original_colormap is not None:
                layer.colormap = self._original_colormap
            if self._original_opacity is not None:
                layer.opacity = self._original_opacity

    # ------------------------------------------------------------------
    # ROI loading and geometry layers
    # ------------------------------------------------------------------
    def load_roi_npy(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load neuron_pt_tuple",
            "",
            "NumPy arrays (*.npy);;All files (*)",
        )
        if not path:
            return

        try:
            self.load_roi_path(path)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            QMessageBox.critical(
                self, "Invalid ROI file", f"Could not load ROI data:\n{error}"
            )
            self.update_status("ROI load failed", "red")

    def load_roi_path(self, path: str | Path) -> None:
        """Load and activate a read-only ROI NPY without opening a dialog."""
        if self.current_image is None:
            raise RuntimeError("Select an Image layer before loading ROI data")
        if self.current_image.ndim not in (3, 4):
            raise ValueError(
                "ROI overlays require a (z,y,x) or (t,z,y,x) Image layer"
            )

        dataset = NeuronBoxDataset.from_npy(
            path,
            z_divisor=self.z_divisor_spin.value(),
            volume_start=self.volume_start_spin.value(),
            volume_stride=self.volume_stride_spin.value(),
        )
        source_path = Path(path)

        self.roi_dataset = dataset
        self.roi_path_input.setText(str(source_path))
        self.unload_roi_btn.setEnabled(True)
        self._set_roi_config_enabled(False)
        self.active_id = None
        self.checked_ids.clear()
        self._refresh_available_ids(select_first=True)
        self._ensure_roi_layers()
        self._refresh_roi_layers()
        self._update_roi_info()
        self.update_status(f"Loaded ROI: {source_path.name}", "green")

    def unload_roi(self) -> None:
        self.roi_dataset = None
        self.roi_path_input.clear()
        self.unload_roi_btn.setEnabled(False)
        self._set_roi_config_enabled(True)
        self._remove_roi_layers()
        self.active_id = None
        self.checked_ids.clear()
        self._available_ids = []
        self.roi_info_label.setText("No ROI loaded")
        self._rebuild_selection_items()
        self._restore_layer_display()
        self._set_labels_controls_enabled()
        self._update_info()
        self.update_status("ROI unloaded", "green")

    def _set_roi_config_enabled(self, enabled: bool) -> None:
        self.z_divisor_spin.setEnabled(enabled)
        self.volume_start_spin.setEnabled(enabled)
        self.volume_stride_spin.setEnabled(enabled)
        self.load_roi_btn.setEnabled(enabled and self.current_image is not None)

    def _update_roi_info(self) -> None:
        if self.roi_dataset is None:
            return
        viewer_t = self._viewer_time()
        source_t = self.roi_dataset.source_time(viewer_t)
        valid = len(self.roi_dataset.valid_ids(viewer_t))
        self.roi_info_label.setText(
            f"T={self.roi_dataset.time_count}, "
            f"N={self.roi_dataset.neuron_count}; "
            f"viewer t={viewer_t} → source t={source_t}; "
            f"{valid} valid boxes"
        )

    def _viewer_time(self) -> int:
        if self.current_image is None or self.current_image.ndim == 3:
            return 0
        steps = self.viewer.dims.current_step
        return int(steps[-4]) if len(steps) >= 4 else 0

    def _viewer_z(self) -> int:
        if self.current_image is None:
            return 0
        steps = self.viewer.dims.current_step
        if self.current_image.ndim == 4 and len(steps) >= 4:
            return int(steps[-3])
        if len(steps) >= 3:
            return int(steps[-3])
        return 0

    def _shape_zyx(self) -> tuple[int, int, int]:
        if self.current_image is None:
            raise RuntimeError("No Image layer selected")
        shape = self.current_image.data.shape
        return tuple(int(value) for value in shape[-3:])

    def _view_axes_supported(self) -> bool:
        if self.current_image is None:
            return False
        ndim = int(self.viewer.dims.ndim)
        order = tuple(self.viewer.dims.order)
        if self.viewer.dims.ndisplay == 2:
            return order[-2:] == (ndim - 2, ndim - 1)
        return order[-3:] == (ndim - 3, ndim - 2, ndim - 1)

    def _on_dims_changed(self, event=None) -> None:
        del event
        if (
            self._z_profile_source is not None
            and self._z_profile_source.ndim == 4
            and self._z_profile_time is not None
            and self._z_profile_time_index(self._z_profile_source)
            != self._z_profile_time
        ):
            self.z_profile_state_label.setText(
                f"t={self._z_profile_time} · Refresh"
            )
        if (
            self.roi_dataset is not None
            and self.current_image is not None
            and self.current_image in self.viewer.layers
        ):
            self._update_roi_info()
            self._refresh_selection_item_styles()
            self._refresh_roi_layers()

    def _ensure_roi_layers(self) -> None:
        if (
            self.current_image is None
            or self.roi_dataset is None
            or self.current_image.ndim not in (3, 4)
        ):
            return
        ndim = self.current_image.ndim
        empty_vectors = np.empty((0, 2, ndim), dtype=float)
        empty_points = np.empty((0, ndim), dtype=float)
        axis_labels = tuple(self.current_image.axis_labels)
        units = tuple(self.current_image.units)

        legacy_layer = self._managed_vector_layer(LEGACY_ROLE_ALL)
        if legacy_layer is not None and legacy_layer in self.viewer.layers:
            self.viewer.layers.remove(legacy_layer)

        if self._managed_vector_layer(ROLE_SELECTED) is None:
            self.viewer.add_vectors(
                empty_vectors,
                ndim=ndim,
                name="Neuron boxes – selected",
                vector_style="line",
                edge_width=1.0,
                edge_color="white",
                opacity=0.25,
                blending="translucent",
                scale=tuple(self.current_image.scale),
                translate=tuple(self.current_image.translate),
                axis_labels=axis_labels,
                units=units,
                metadata={ROLE_KEY: ROLE_SELECTED},
            )
        if self._managed_vector_layer(ROLE_ACTIVE) is None:
            self.viewer.add_vectors(
                empty_vectors,
                ndim=ndim,
                name="Neuron box – active",
                vector_style="line",
                edge_width=3.0,
                edge_color="yellow",
                opacity=1.0,
                blending="translucent",
                scale=tuple(self.current_image.scale),
                translate=tuple(self.current_image.translate),
                axis_labels=axis_labels,
                units=units,
                metadata={ROLE_KEY: ROLE_ACTIVE},
            )
        if self._managed_box_label_layer() is None:
            box_label_layer = self.viewer.add_points(
                empty_points,
                ndim=ndim,
                name="Neuron labels – selected",
                size=1,
                face_color="transparent",
                border_color="transparent",
                border_width=0,
                opacity=1.0,
                blending="translucent",
                features={
                    "neuron_id": np.empty(0, dtype=int),
                    "display_text": np.empty(0, dtype=str),
                },
                text={
                    "string": "{display_text}",
                    "color": self._box_label_color,
                    "size": 12,
                    "anchor": "center",
                },
                visible=self.show_box_labels_checkbox.isChecked(),
                scale=tuple(self.current_image.scale),
                translate=tuple(self.current_image.translate),
                axis_labels=axis_labels,
                units=units,
                metadata={ROLE_KEY: ROLE_BOX_LABELS},
            )
            box_label_layer.editable = False

    def _managed_vector_layer(self, role: str) -> Vectors | None:
        for layer in self.viewer.layers:
            if (
                isinstance(layer, Vectors)
                and layer.metadata.get(ROLE_KEY) == role
            ):
                return layer
        return None

    def _managed_box_label_layer(self) -> Points | None:
        for layer in self.viewer.layers:
            if (
                isinstance(layer, Points)
                and layer.metadata.get(ROLE_KEY) == ROLE_BOX_LABELS
            ):
                return layer
        return None

    def _remove_roi_layers(self) -> None:
        managed = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, Points | Vectors)
            and layer.metadata.get(ROLE_KEY) in MANAGED_ROI_ROLES
        ]
        for layer in managed:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)

    def _refresh_roi_layers(self, event=None) -> None:
        del event
        if (
            self.roi_dataset is None
            or self.current_image is None
            or self.current_image.ndim not in (3, 4)
        ):
            return
        selected_layer = self._managed_vector_layer(ROLE_SELECTED)
        active_layer = self._managed_vector_layer(ROLE_ACTIVE)
        box_label_layer = self._managed_box_label_layer()
        if selected_layer is None or active_layer is None:
            return

        if not self._view_axes_supported():
            self._set_vector_data(selected_layer, [], [])
            self._set_vector_data(active_layer, [], [], active=True)
            if box_label_layer is not None:
                self._set_box_label_data(box_label_layer, [], [], [])
            self.update_status(
                "ROI overlays require spatial y/x axes in 2D or z/y/x axes in 3D",
                "orange",
            )
            return

        viewer_t = self._viewer_time()
        z_index = self._viewer_z()
        shape_zyx = self._shape_zyx()
        selected_vectors: list[np.ndarray] = []
        selected_neuron_ids: list[int] = []
        active_vectors: list[np.ndarray] = []
        active_ids: list[int] = []
        label_points: list[np.ndarray] = []
        label_neuron_ids: list[int] = []
        display_texts: list[str] = []
        biological_names = self._annotation_names()
        show_box_labels = (
            box_label_layer is not None
            and self.show_box_labels_checkbox.isChecked()
        )

        for neuron_id in self.roi_dataset.valid_ids(viewer_t):
            if (
                neuron_id not in self.checked_ids
                and neuron_id != self.active_id
            ):
                continue
            box = self.roi_dataset.get_box(viewer_t, neuron_id)
            if box is None:
                continue
            active_range = self._active_z_range()
            if active_range is not None:
                owner = find_z_layer(box.center_zyx[0], self._z_ranges)
                if owner is None or owner.index != active_range.index:
                    continue
            if self.viewer.dims.ndisplay == 2:
                geometry = box_vectors_2d(box, z_index, shape_zyx=shape_zyx)
                label_point = box_label_point_2d(
                    box, z_index, shape_zyx=shape_zyx
                )
            else:
                geometry = box_vectors_3d(box, shape_zyx=shape_zyx)
                label_point = box_label_point_3d(box, shape_zyx=shape_zyx)
            if self.current_image.ndim == 4:
                geometry = add_time_axis(geometry, viewer_t)
                if label_point is not None:
                    label_point = np.concatenate(
                        ([float(viewer_t)], label_point)
                    )
            if len(geometry):
                if neuron_id in self.checked_ids:
                    selected_vectors.append(geometry)
                    selected_neuron_ids.extend([neuron_id] * len(geometry))
                    if show_box_labels and label_point is not None:
                        label_points.append(
                            np.asarray(label_point, dtype=float)
                        )
                        label_neuron_ids.append(neuron_id)
                        biological = biological_names.get(neuron_id, "").strip()
                        display_texts.append(biological or str(neuron_id))
                if neuron_id == self.active_id:
                    active_vectors.append(geometry)
                    active_ids.extend([neuron_id] * len(geometry))

        self._set_vector_data(
            selected_layer, selected_vectors, selected_neuron_ids
        )
        self._set_vector_data(
            active_layer, active_vectors, active_ids, active=True
        )
        if box_label_layer is not None:
            self._set_box_label_data(
                box_label_layer,
                label_points,
                label_neuron_ids,
                display_texts,
            )

    def _set_vector_data(
        self,
        layer: Vectors,
        vector_blocks: list[np.ndarray],
        neuron_ids: list[int],
        *,
        active: bool = False,
    ) -> None:
        ndim = layer.ndim
        data = (
            np.concatenate(vector_blocks, axis=0)
            if vector_blocks
            else np.empty((0, 2, ndim), dtype=float)
        )
        layer.data = data
        layer.features = {"neuron_id": np.asarray(neuron_ids, dtype=int)}
        if active:
            layer.edge_color = "yellow"
        elif neuron_ids:
            layer.edge_color = np.asarray(
                [self._display_color(item_id) for item_id in neuron_ids]
            )
        else:
            layer.edge_color = "white"

    def _set_box_label_data(
        self,
        layer: Points,
        points: list[np.ndarray],
        neuron_ids: list[int],
        display_texts: list[str],
    ) -> None:
        ndim = layer.ndim
        layer.data = (
            np.asarray(points, dtype=float).reshape(-1, ndim)
            if points
            else np.empty((0, ndim), dtype=float)
        )
        layer.features = {
            "neuron_id": np.asarray(neuron_ids, dtype=int),
            "display_text": np.asarray(display_texts, dtype=str),
        }
        layer.editable = False
        layer.visible = self.show_box_labels_checkbox.isChecked()

    def _locate_active_box(self) -> None:
        if (
            self.roi_dataset is None
            or self.current_image is None
            or self.active_id is None
        ):
            return
        box = self.roi_dataset.get_box(self._viewer_time(), self.active_id)
        if box is None:
            self.update_status(
                f"Neuron {self.active_id} is missing at this volume", "orange"
            )
            return
        if not self._id_in_active_z_layer(self.active_id):
            self.update_status(
                f"Neuron {self.active_id} is outside the current Z layer",
                "orange",
            )
            return

        z, y, x = box.center_zyx
        steps = list(self.viewer.dims.current_step)
        if self.current_image.ndim == 4 and len(steps) >= 4:
            z_axis = len(steps) - 3
        else:
            z_axis = len(steps) - 3
        z_size = self._shape_zyx()[0]
        steps[z_axis] = int(np.clip(round(z), 0, z_size - 1))
        self.viewer.dims.current_step = tuple(steps)

        point = (
            (self._viewer_time(), z, y, x)
            if self.current_image.ndim == 4
            else (z, y, x)
        )
        world = self.current_image.data_to_world(point)
        self.viewer.camera.center = tuple(world[-3:])

    # ------------------------------------------------------------------
    # Annotation and Excel
    # ------------------------------------------------------------------
    def _annotation_names(self) -> dict[int, str]:
        names: dict[int, str] = {}
        for row in range(self.annotation_table.rowCount()):
            digital = self.annotation_table.item(row, 0)
            biological = self.annotation_table.item(row, 1)
            if digital is None:
                continue
            try:
                item_id = int(digital.text())
            except ValueError:
                continue
            names[item_id] = (
                biological.text() if biological is not None else ""
            )
        return names

    def _annotation_rows(
        self,
    ) -> dict[int, tuple[str, str]]:
        rows: dict[int, tuple[str, str]] = {}
        for row in range(self.annotation_table.rowCount()):
            digital = self.annotation_table.item(row, 0)
            if digital is None:
                continue
            try:
                item_id = int(digital.text())
            except ValueError:
                continue
            biological = self.annotation_table.item(row, 1)
            annotation = self.annotation_table.item(row, 2)
            rows[item_id] = (
                biological.text() if biological is not None else "",
                annotation.text() if annotation is not None else "",
            )
        return rows

    def _set_annotation_rows(
        self,
        rows: list[tuple[int, str, str]],
    ) -> None:
        self._ui_sync = True
        self.annotation_table.blockSignals(True)
        try:
            self.annotation_table.setRowCount(len(rows))
            for row, (item_id, biological, annotation) in enumerate(rows):
                digital_item = QTableWidgetItem(str(item_id))
                digital_item.setTextAlignment(Qt.AlignCenter)
                self.annotation_table.setItem(row, 0, digital_item)
                self.annotation_table.setItem(
                    row, 1, QTableWidgetItem(biological)
                )
                self.annotation_table.setItem(
                    row, 2, QTableWidgetItem(annotation)
                )
        finally:
            self.annotation_table.blockSignals(False)
            self._ui_sync = False
        self._refresh_selection_item_styles()
        self._refresh_roi_layers()

    def _sync_annotation_ids(self) -> int:
        existing = self._annotation_rows()
        rows = [
            (item_id, *existing.get(item_id, ("", "")))
            for item_id in self._available_ids
        ]
        self._set_annotation_rows(rows)
        return len(rows)

    def load_current_ids_to_annotation(self) -> None:
        count = self._sync_annotation_ids()
        self.update_status(
            f"Loaded {count} identities into annotation table", "green"
        )

    def _sync_annotation_to_active(self) -> None:
        if self.active_id is None:
            self._ui_sync = True
            try:
                self.annotation_table.clearSelection()
            finally:
                self._ui_sync = False
            return
        self._ui_sync = True
        try:
            for row in range(self.annotation_table.rowCount()):
                item = self.annotation_table.item(row, 0)
                if item is not None and item.text() == str(self.active_id):
                    self.annotation_table.selectRow(row)
                    self.annotation_table.scrollToItem(item)
                    break
        finally:
            self._ui_sync = False

    def _on_annotation_selection_changed(self) -> None:
        if self._ui_sync:
            return
        rows = self.annotation_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.annotation_table.item(rows[0].row(), 0)
        if item is None:
            return
        try:
            item_id = int(item.text())
        except ValueError:
            return
        if item_id not in self._available_ids:
            return
        self.activate_id(item_id, locate=True)

    def _on_annotation_item_changed(self, item: QTableWidgetItem) -> None:
        if self._ui_sync:
            return
        if item.column() in (0, 1):
            self._refresh_selection_item_styles()
            self._refresh_roi_layers()

    def save_annotation_to_excel(self) -> None:
        if not EXCEL_AVAILABLE:
            QMessageBox.critical(
                self, "Excel unavailable", "Install the 'excel' extra first."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save neuron annotations",
            "neuron_annotations.xlsx",
            "Excel workbooks (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        sheet_name = (
            self.sheet_name_input.text().strip() or "Neuron Annotations"
        )

        try:
            if Path(path).exists():
                workbook = load_workbook(path)
                if sheet_name in workbook.sheetnames:
                    reply = QMessageBox.question(
                        self,
                        "Sheet exists",
                        f"Overwrite sheet '{sheet_name}'?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if reply != QMessageBox.Yes:
                        workbook.close()
                        return
                    workbook.remove(workbook[sheet_name])
                sheet = workbook.create_sheet(sheet_name)
            else:
                workbook = Workbook()
                sheet = workbook.active
                sheet.title = sheet_name

            sheet.append(["digital", "biological", "annotation"])
            for cell in sheet[1]:
                cell.alignment = Alignment(
                    horizontal="center", vertical="center"
                )
            for item_id, (
                biological,
                annotation,
            ) in self._annotation_rows().items():
                sheet.append([item_id, biological, annotation])
            for column in sheet.columns:
                width = max(
                    (
                        len(str(cell.value))
                        for cell in column
                        if cell.value is not None
                    ),
                    default=0,
                )
                sheet.column_dimensions[column[0].column_letter].width = (
                    width + 2
                )
            workbook.save(path)
            workbook.close()
        except (OSError, PermissionError, ValueError) as error:
            QMessageBox.critical(
                self, "Save failed", f"Could not save workbook:\n{error}"
            )
            self.update_status("Annotation save failed", "red")
            return
        self.update_status(f"Saved annotations to {sheet_name}", "green")

    def load_excel_to_annotation(self) -> None:
        if not EXCEL_AVAILABLE:
            QMessageBox.critical(
                self, "Excel unavailable", "Install the 'excel' extra first."
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load neuron annotations",
            "",
            "Excel workbooks (*.xlsx)",
        )
        if not path:
            return
        requested = self.sheet_name_input.text().strip()

        try:
            workbook = load_workbook(path, read_only=True, data_only=True)
            if requested and requested in workbook.sheetnames:
                sheet = workbook[requested]
            else:
                sheet = workbook[workbook.sheetnames[0]]
                self.sheet_name_input.setText(sheet.title)

            rows: list[tuple[int, str, str]] = []
            for values in sheet.iter_rows(values_only=True):
                if not values or values[0] is None:
                    continue
                try:
                    numeric_id = float(values[0])
                except (TypeError, ValueError):
                    continue
                if (
                    not np.isfinite(numeric_id)
                    or not numeric_id.is_integer()
                    or numeric_id < 0
                ):
                    continue
                item_id = int(numeric_id)
                biological = (
                    ""
                    if len(values) < 2 or values[1] is None
                    else str(values[1])
                )
                annotation = (
                    ""
                    if len(values) < 3 or values[2] is None
                    else str(values[2])
                )
                rows.append((item_id, biological, annotation))
            workbook.close()
        except (OSError, PermissionError, ValueError) as error:
            QMessageBox.critical(
                self, "Load failed", f"Could not load workbook:\n{error}"
            )
            self.update_status("Annotation load failed", "red")
            return

        self._set_annotation_rows(rows)
        self.update_status(f"Loaded {len(rows)} annotations", "green")

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def _update_info(self) -> None:
        mode = "ROI" if self.roi_dataset is not None else "Idle"
        checked = sorted(self.checked_ids)
        text = (
            f"Mode: {mode}\n"
            f"Active: {self.active_id if self.active_id is not None else 'none'}\n"
            f"Checked: {checked[:12]}"
        )
        active_range = self._active_z_range()
        if self._z_ranges:
            z_view = (
                "All"
                if active_range is None
                else f"Layer {active_range.index + 1} "
                f"[{active_range.start},{active_range.stop})"
            )
            text += f"\nZ view: {z_view}"
        if len(checked) > 12:
            text += f" … (+{len(checked) - 12})"
        self.info_text.setText(text)

    def update_status(self, message: str, color: str = "black") -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color};")

    @property
    def current_layer(self) -> Labels | None:
        """Compatibility alias for the optional controlled Labels layer."""
        return self.current_labels


LabelManager = NeuronAnnotatorWidget

__all__ = (
    "EXCEL_AVAILABLE",
    "LabelManager",
    "NeuronAnnotatorWidget",
    "ROLE_ACTIVE",
    "ROLE_BOX_LABELS",
    "ROLE_KEY",
    "ROLE_SELECTED",
    "ROLE_Z_IMAGE",
    "ROLE_Z_LABELS",
)
