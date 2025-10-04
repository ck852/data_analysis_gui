"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Control Panel Widget

A self-contained widget for managing all analysis and plot settings in the PatchBatch Electrophysiology Data Analysis Tool.
Controls are always active, regardless of file loading state, and communicate user actions via Qt signals.
Provides themed UI elements for analysis ranges, dual range selection, stimulus period, axis configuration, and peak mode.
Validation is performed on input fields, with visual feedback for invalid states.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QGroupBox, QLabel, QPushButton, QCheckBox, QGridLayout
from PySide6.QtCore import Signal

# Import custom widgets that handle scrolling properly
from data_analysis_gui.widgets.custom_inputs import SelectAllSpinBox, NoScrollComboBox
from data_analysis_gui.config import DEFAULT_SETTINGS
from data_analysis_gui.core.params import AnalysisParameters

from data_analysis_gui.config.themes import (style_button, style_scroll_area, style_group_box, style_label, style_checkbox, apply_compact_layout, 
                                             style_spinbox_with_arrows, style_combo_simple, MODERN_COLORS, WIDGET_SIZES)


class ControlPanel(QWidget):
    """
    ControlPanel Widget

    Provides a themed panel for configuring analysis and plot settings.
    - Manages analysis ranges, dual range selection, and stimulus period.
    - Allows configuration of plot axes and peak calculation mode.
    - Emits signals for analysis requests, export actions, and range changes.
    - Performs validation and visual feedback for input fields.
    - Supports saving and restoring settings via dictionaries.

    Signals:
        analysis_requested: Emitted when the user requests to generate an analysis plot.
        export_requested: Emitted when the user requests to export analysis data.
        dual_range_toggled: Emitted when the dual range checkbox is toggled.
        range_values_changed: Emitted when any range spinbox value changes.

    Args:
        parent: Optional parent widget.
    """

    # Define signals for communication with main window
    analysis_requested = Signal()  # User wants to generate analysis plot
    export_requested = Signal()  # User wants to export data

    dual_range_toggled = Signal(bool)  # Dual range checkbox changed
    range_values_changed = Signal()  # Any range spinbox value changed

    def __init__(self, parent=None):
        super().__init__(parent)

        # Dictionary to track previous valid values
        self._previous_valid_values = {}

        # Track which fields have invalid state
        self._invalid_fields = set()

        # Store original style sheets for restoration
        self._original_styles = {}

        self._setup_ui()

        # Initialize tracking with starting values after UI setup
        self._previous_valid_values = {
            "start1": self.start_spin.value(),
            "end1": self.end_spin.value(),
            "start2": self.start_spin2.value(),
            "end2": self.end_spin2.value(),
        }

        # Store original styles for all spinboxes after themes are applied
        self._original_styles = {
            "start1": self.start_spin.styleSheet(),
            "end1": self.end_spin.styleSheet(),
            "start2": self.start_spin2.styleSheet(),
            "end2": self.end_spin2.styleSheet(),
        }

        self._connect_signals()

    def _setup_ui(self):
        """
        Set up the control panel UI with full theme integration.
        Creates all controls, layouts, and applies styling.
        """
        # Create scroll area for the controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(280)

        # Apply theme styling to scroll area
        style_scroll_area(scroll_area)

        # Main control widget inside scroll area
        control_widget = QWidget()
        scroll_area.setWidget(control_widget)

        # Layout for control widget with theme-based compact spacing
        layout = QVBoxLayout(control_widget)
        apply_compact_layout(control_widget, spacing=4, margin=6)

        # Add all control groups
        layout.addWidget(self._create_analysis_settings_group())
        layout.addWidget(self._create_plot_settings_group())

        # Export Plot Data button with theme styling
        self.export_plot_btn = QPushButton("Export Analysis Data")
        style_button(self.export_plot_btn, "secondary")
        self.export_plot_btn.clicked.connect(self.export_requested.emit)
        # Start enabled - will be validated immediately
        self.export_plot_btn.setEnabled(False)
        layout.addWidget(self.export_plot_btn)

        # Small spacing at bottom
        layout.addSpacing(5)

        # Set main layout for this widget
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

    def _create_analysis_settings_group(self):
        """
        Create the analysis settings group with themed controls.

        Returns:
            QGroupBox: The analysis settings group box.
        """
        analysis_group = QGroupBox("Analysis Settings")
        style_group_box(analysis_group)

        analysis_layout = QGridLayout(analysis_group)
        apply_compact_layout(analysis_group, spacing=4, margin=6)

        # Range 1 settings
        self._add_range1_settings(analysis_layout)

        # Dual range checkbox with theme styling
        self.dual_range_cb = QCheckBox("Use Dual Analysis")
        style_checkbox(self.dual_range_cb)
        self.dual_range_cb.stateChanged.connect(self._on_dual_range_changed)
        analysis_layout.addWidget(self.dual_range_cb, 2, 0, 1, 2)

        # Range 2 settings
        self._add_range2_settings(analysis_layout)

        return analysis_group

    def _add_range1_settings(self, layout):
        """
        Add Range 1 controls to the provided layout.

        Args:
            layout (QGridLayout): The layout to add controls to.
        """
        # Range 1 Start
        start_label = QLabel("Range 1 Start (ms):")
        style_label(start_label, "normal")
        layout.addWidget(start_label, 0, 0)

        self.start_spin = SelectAllSpinBox()
        self.start_spin.setRange(0, 100000)
        self.start_spin.setValue(DEFAULT_SETTINGS["range1_start"])
        self.start_spin.setSingleStep(0.05)
        self.start_spin.setDecimals(2)
        self.start_spin.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_spinbox_with_arrows(self.start_spin)
        layout.addWidget(self.start_spin, 0, 1)

        # Range 1 End
        end_label = QLabel("Range 1 End (ms):")
        style_label(end_label, "normal")
        layout.addWidget(end_label, 1, 0)

        self.end_spin = SelectAllSpinBox()
        self.end_spin.setRange(0, 100000)
        self.end_spin.setValue(DEFAULT_SETTINGS["range1_end"])
        self.end_spin.setSingleStep(0.05)
        self.end_spin.setDecimals(2)
        self.end_spin.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_spinbox_with_arrows(self.end_spin)
        layout.addWidget(self.end_spin, 1, 1)

    def _add_range2_settings(self, layout):
        """
        Add Range 2 controls to the provided layout.

        Args:
            layout (QGridLayout): The layout to add controls to.
        """
        # Range 2 Start
        start2_label = QLabel("Range 2 Start (ms):")
        style_label(start2_label, "normal")
        layout.addWidget(start2_label, 3, 0)

        self.start_spin2 = SelectAllSpinBox()
        self.start_spin2.setRange(0, 100000)
        self.start_spin2.setValue(DEFAULT_SETTINGS["range2_start"])
        self.start_spin2.setSingleStep(0.05)
        self.start_spin2.setDecimals(2)
        self.start_spin2.setEnabled(False)
        self.start_spin2.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_spinbox_with_arrows(self.start_spin2)
        layout.addWidget(self.start_spin2, 3, 1)

        # Range 2 End
        end2_label = QLabel("Range 2 End (ms):")
        style_label(end2_label, "normal")
        layout.addWidget(end2_label, 4, 0)

        self.end_spin2 = SelectAllSpinBox()
        self.end_spin2.setRange(0, 100000)
        self.end_spin2.setValue(DEFAULT_SETTINGS["range2_end"])
        self.end_spin2.setSingleStep(0.05)
        self.end_spin2.setDecimals(2)
        self.end_spin2.setEnabled(False)
        self.end_spin2.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_spinbox_with_arrows(self.end_spin2)
        layout.addWidget(self.end_spin2, 4, 1)

    def _create_plot_settings_group(self):
        """
        Create the plot settings group with themed controls.

        Returns:
            QGroupBox: The plot settings group box.
        """
        plot_group = QGroupBox("Plot Settings")
        style_group_box(plot_group)

        plot_layout = QGridLayout(plot_group)
        apply_compact_layout(plot_group, spacing=4, margin=6)

        # X-axis settings with NoScrollComboBox
        x_label = QLabel("X-Axis:")
        style_label(x_label, "normal")
        plot_layout.addWidget(x_label, 0, 0)

        self.x_measure_combo = NoScrollComboBox()
        self.x_measure_combo.addItems(["Time", "Peak", "Average"])
        self.x_measure_combo.setCurrentText("Average")
        self.x_measure_combo.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_combo_simple(self.x_measure_combo)
        plot_layout.addWidget(self.x_measure_combo, 0, 1)

        self.x_channel_combo = NoScrollComboBox()
        self.x_channel_combo.addItems(["Voltage", "Current"])
        self.x_channel_combo.setCurrentText("Voltage")
        self.x_channel_combo.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_combo_simple(self.x_channel_combo)
        plot_layout.addWidget(self.x_channel_combo, 0, 2)

        # Y-axis settings with NoScrollComboBox
        y_label = QLabel("Y-Axis:")
        style_label(y_label, "normal")
        plot_layout.addWidget(y_label, 1, 0)

        self.y_measure_combo = NoScrollComboBox()
        self.y_measure_combo.addItems(["Peak", "Average", "Time"])
        self.y_measure_combo.setCurrentText("Average")
        self.y_measure_combo.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_combo_simple(self.y_measure_combo)
        plot_layout.addWidget(self.y_measure_combo, 1, 1)

        self.y_channel_combo = NoScrollComboBox()
        self.y_channel_combo.addItems(["Voltage", "Current"])
        self.y_channel_combo.setCurrentText("Current")
        self.y_channel_combo.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_combo_simple(self.y_channel_combo)
        plot_layout.addWidget(self.y_channel_combo, 1, 2)

        # Update plot button with theme styling
        self.update_plot_btn = QPushButton("Generate Analysis Plot")
        style_button(self.update_plot_btn, "primary")
        self.update_plot_btn.clicked.connect(self.analysis_requested.emit)
        # Start disabled - will be validated immediately
        self.update_plot_btn.setEnabled(False)
        plot_layout.addWidget(self.update_plot_btn, 2, 0, 1, 3)

        # Peak Mode settings with NoScrollComboBox
        peak_label = QLabel("Peak Mode:")
        style_label(peak_label, "normal")
        plot_layout.addWidget(peak_label, 3, 0)

        self.peak_mode_combo = NoScrollComboBox()
        self.peak_mode_combo.addItems(["Absolute", "Positive", "Negative", "Peak-Peak"])
        self.peak_mode_combo.setCurrentText("Absolute")
        self.peak_mode_combo.setToolTip(
            "Peak calculation mode (applies when X or Y axis is set to Peak)"
        )
        self.peak_mode_combo.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_combo_simple(self.peak_mode_combo)
        plot_layout.addWidget(self.peak_mode_combo, 3, 1, 1, 2)

        # Connect signal to enable/disable peak mode based on axis selection
        self.x_measure_combo.currentTextChanged.connect(
            self._update_peak_mode_visibility
        )
        self.y_measure_combo.currentTextChanged.connect(
            self._update_peak_mode_visibility
        )

        return plot_group

    def _update_peak_mode_visibility(self):
        """
        Enable or disable the peak mode combo box based on axis selection.
        """
        is_peak_selected = (
            self.x_measure_combo.currentText() == "Peak"
            or self.y_measure_combo.currentText() == "Peak"
        )
        self.peak_mode_combo.setEnabled(is_peak_selected)

        # Update visual state when disabled
        if not is_peak_selected:
            # Re-apply combo box styling to ensure disabled state looks correct
            style_combo_simple(self.peak_mode_combo)

    def _connect_signals(self):
        """
        Connect internal widget signals and perform initial validation.
        """
        # Validate on any value change
        self.start_spin.valueChanged.connect(self._validate_and_update)
        self.end_spin.valueChanged.connect(self._validate_and_update)
        self.start_spin2.valueChanged.connect(self._validate_and_update)
        self.end_spin2.valueChanged.connect(self._validate_and_update)
        # Also re-validate when the dual range checkbox is toggled
        self.dual_range_cb.stateChanged.connect(self._validate_and_update)

        # Run initial validation to set button states correctly
        self._validate_and_update()

    def _validate_and_update(self):
        """
        Validate all range inputs, update UI feedback, and emit range change signals.
        Enables or disables action buttons based on validation.
        """
        # --- Validate Range 1 ---
        start1_val = self.start_spin.value()
        end1_val = self.end_spin.value()
        is_range1_valid = end1_val > start1_val

        if not is_range1_valid:
            self._mark_field_invalid("start1")
            self._mark_field_invalid("end1")
        else:
            self._clear_invalid_state("start1")
            self._clear_invalid_state("end1")

        # --- Validate Range 2 (if enabled) ---
        is_range2_valid = True
        if self.dual_range_cb.isChecked():
            start2_val = self.start_spin2.value()
            end2_val = self.end_spin2.value()
            is_range2_valid = end2_val > start2_val

            if not is_range2_valid:
                self._mark_field_invalid("start2")
                self._mark_field_invalid("end2")
            else:
                self._clear_invalid_state("start2")
                self._clear_invalid_state("end2")
        else:
            # If dual range is disabled, its fields can't be invalid
            self._clear_invalid_state("start2")
            self._clear_invalid_state("end2")

        # --- Update Button State ---
        # Buttons are enabled based on validation only
        is_all_valid = is_range1_valid and is_range2_valid
        self.update_plot_btn.setEnabled(is_all_valid)
        self.export_plot_btn.setEnabled(is_all_valid)

        # --- Sync Cursors ---
        self.range_values_changed.emit()

    def _mark_field_invalid(self, spinbox_key: str):
        """
        Mark a spinbox field as invalid, applying themed visual feedback.

        Args:
            spinbox_key (str): Key for the spinbox ('start1', 'end1', etc.).
        """
        spinbox_map = {
            "start1": self.start_spin,
            "end1": self.end_spin,
            "start2": self.start_spin2,
            "end2": self.end_spin2,
        }
        spinbox = spinbox_map.get(spinbox_key)
        if spinbox and spinbox_key not in self._invalid_fields:
            self._invalid_fields.add(spinbox_key)
            # Save the current style before applying invalid state
            if spinbox_key not in self._original_styles:
                self._original_styles[spinbox_key] = spinbox.styleSheet()

            # Apply invalid background color (hardcoded since it's not in refactored theme)
            current_style = spinbox.styleSheet()
            invalid_bg = "#ffcccc"  # Light red background for invalid state
            invalid_style = f"{current_style}\nQDoubleSpinBox {{ background-color: {invalid_bg}; border-color: {MODERN_COLORS['danger']}; }}"
            spinbox.setStyleSheet(invalid_style)

    def _clear_invalid_state(self, spinbox_key: str):
        """
        Clear the invalid state from a spinbox field, restoring original styling.

        Args:
            spinbox_key (str): Key for the spinbox ('start1', 'end1', etc.).
        """
        if spinbox_key in self._invalid_fields:
            self._invalid_fields.remove(spinbox_key)
            spinbox_map = {
                "start1": self.start_spin,
                "end1": self.end_spin,
                "start2": self.start_spin2,
                "end2": self.end_spin2,
            }
            spinbox = spinbox_map.get(spinbox_key)
            if spinbox:
                # Restore the original styled state
                style_spinbox_with_arrows(spinbox)

    def _on_dual_range_changed(self):
        """
        Handle changes to the dual range checkbox, enabling/disabling controls.
        """
        enabled = self.dual_range_cb.isChecked()
        self.start_spin2.setEnabled(enabled)
        self.end_spin2.setEnabled(enabled)

        # Re-apply styling to ensure disabled state looks correct
        style_spinbox_with_arrows(self.start_spin2)
        style_spinbox_with_arrows(self.end_spin2)

        self.dual_range_toggled.emit(enabled)
        # The validation is handled by the connected signal

    def get_parameters(self) -> AnalysisParameters:
        """
        Get current analysis parameters as an AnalysisParameters object.

        Returns:
            AnalysisParameters: Object containing all current control values.
        """
        from data_analysis_gui.core.params import AnalysisParameters, AxisConfig

        # Determine peak mode
        peak_mode = self.peak_mode_combo.currentText()

        # Create X-axis config
        x_measure = self.x_measure_combo.currentText()
        x_axis = AxisConfig(
            measure=x_measure,
            channel=self.x_channel_combo.currentText() if x_measure != "Time" else None,
            peak_type=peak_mode if x_measure == "Peak" else None,
        )

        # Create Y-axis config
        y_measure = self.y_measure_combo.currentText()
        y_axis = AxisConfig(
            measure=y_measure,
            channel=self.y_channel_combo.currentText() if y_measure != "Time" else None,
            peak_type=peak_mode if y_measure == "Peak" else None,
        )

        # Return clean parameters object (no current_units, no channels_swapped)
        return AnalysisParameters(
            range1_start=self.start_spin.value(),
            range1_end=self.end_spin.value(),
            use_dual_range=self.dual_range_cb.isChecked(),
            range2_start=(
                self.start_spin2.value() if self.dual_range_cb.isChecked() else None
            ),
            range2_end=(
                self.end_spin2.value() if self.dual_range_cb.isChecked() else None
            ),
            x_axis=x_axis,
            y_axis=y_axis,
            channel_config={},  # No longer needed from UI
        )

    # --- Public methods for data access and updates ---

    def get_range_values(self) -> dict:
        """
        Get current range values as a dictionary.

        Returns:
            dict: Dictionary of current range values.
        """
        return {
            "range1_start": self.start_spin.value(),
            "range1_end": self.end_spin.value(),
            "use_dual_range": self.dual_range_cb.isChecked(),
            "range2_start": (
                self.start_spin2.value() if self.dual_range_cb.isChecked() else None
            ),
            "range2_end": (
                self.end_spin2.value() if self.dual_range_cb.isChecked() else None
            ),
        }

    def get_range_spinboxes(self) -> dict:
        """
        Get references to range spinboxes for external use.

        Returns:
            dict: Dictionary mapping keys to spinbox widgets.
        """
        spinboxes = {"start1": self.start_spin, "end1": self.end_spin}
        if self.dual_range_cb.isChecked():
            spinboxes["start2"] = self.start_spin2
            spinboxes["end2"] = self.end_spin2
        return spinboxes

    def update_range_value(self, spinbox_key: str, value: float):
        """
        Update a specific range spinbox value.

        Args:
            spinbox_key (str): Key for the spinbox ('start1', 'end1', etc.).
            value (float): Value to set.
        """
        spinbox_map = {
            "start1": self.start_spin,
            "end1": self.end_spin,
            "start2": self.start_spin2,
            "end2": self.end_spin2,
        }
        if spinbox_key in spinbox_map:
            # setValue() triggers validation automatically
            spinbox_map[spinbox_key].setValue(value)

    def set_analysis_range(self, max_time: float):
        """
        Set the maximum value for analysis range spinboxes and clamp current values.

        Args:
            max_time (float): Maximum allowed time value.
        """
        self.start_spin.setRange(0, max_time)
        self.end_spin.setRange(0, max_time)
        self.start_spin2.setRange(0, max_time)
        self.end_spin2.setRange(0, max_time)

        # Clamp existing values to the new range
        if self.start_spin.value() > max_time:
            self.start_spin.setValue(max_time)
        if self.end_spin.value() > max_time:
            self.end_spin.setValue(max_time)
        if self.start_spin2.value() > max_time:
            self.start_spin2.setValue(max_time)
        if self.end_spin2.value() > max_time:
            self.end_spin2.setValue(max_time)

        # After clamping, sync the valid state
        self._previous_valid_values = {
            "start1": self.start_spin.value(),
            "end1": self.end_spin.value(),
            "start2": self.start_spin2.value(),
            "end2": self.end_spin2.value(),
        }

    def set_parameters_from_dict(self, params: dict):
        """
        Set analysis parameters from a dictionary, blocking signals during update.

        Args:
            params (dict): Dictionary of analysis parameter values.
        """
        # Store current signal state and disconnect
        signals_were_blocked = [
            self.start_spin.blockSignals(True),
            self.end_spin.blockSignals(True),
            self.start_spin2.blockSignals(True),
            self.end_spin2.blockSignals(True),
            self.dual_range_cb.blockSignals(True),
            # self.period_spin.blockSignals(True),
        ]

        try:
            # Set Range 1 values
            if "range1_start" in params:
                self.start_spin.setValue(params["range1_start"])
            if "range1_end" in params:
                self.end_spin.setValue(params["range1_end"])

            # Set dual range checkbox
            use_dual = params.get("use_dual_range", False)
            self.dual_range_cb.setChecked(use_dual)

            # Enable/disable Range 2 controls based on dual range
            self.start_spin2.setEnabled(use_dual)
            self.end_spin2.setEnabled(use_dual)

            # Set Range 2 values
            if use_dual:
                if "range2_start" in params and params["range2_start"] is not None:
                    self.start_spin2.setValue(params["range2_start"])
                if "range2_end" in params and params["range2_end"] is not None:
                    self.end_spin2.setValue(params["range2_end"])

        finally:
            # Restore signal states
            self.start_spin.blockSignals(signals_were_blocked[0])
            self.end_spin.blockSignals(signals_were_blocked[1])
            self.start_spin2.blockSignals(signals_were_blocked[2])
            self.end_spin2.blockSignals(signals_were_blocked[3])
            self.dual_range_cb.blockSignals(signals_were_blocked[4])
            #self.period_spin.blockSignals(signals_were_blocked[5])

            # Re-apply styling to ensure correct appearance
            style_spinbox_with_arrows(self.start_spin2)
            style_spinbox_with_arrows(self.end_spin2)

            # Now validate everything once and emit signals
            self._validate_and_update()
            if use_dual:
                self.dual_range_toggled.emit(use_dual)

    def set_plot_settings_from_dict(self, params: dict):
        """
        Set plot settings from a dictionary, blocking signals during update.

        Args:
            params (dict): Dictionary of plot setting values.
        """
        # Block signals during mass update
        signals_blocked = [
            self.x_measure_combo.blockSignals(True),
            self.x_channel_combo.blockSignals(True),
            self.y_measure_combo.blockSignals(True),
            self.y_channel_combo.blockSignals(True),
            self.peak_mode_combo.blockSignals(True),
        ]

        try:
            # Set X-axis settings
            if "x_measure" in params:
                index = self.x_measure_combo.findText(params["x_measure"])
                if index >= 0:
                    self.x_measure_combo.setCurrentIndex(index)

            if "x_channel" in params and params["x_channel"]:
                index = self.x_channel_combo.findText(params["x_channel"])
                if index >= 0:
                    self.x_channel_combo.setCurrentIndex(index)

            # Set Y-axis settings
            if "y_measure" in params:
                index = self.y_measure_combo.findText(params["y_measure"])
                if index >= 0:
                    self.y_measure_combo.setCurrentIndex(index)

            if "y_channel" in params and params["y_channel"]:
                index = self.y_channel_combo.findText(params["y_channel"])
                if index >= 0:
                    self.y_channel_combo.setCurrentIndex(index)

            # Set peak mode
            if "peak_mode" in params:
                index = self.peak_mode_combo.findText(params["peak_mode"])
                if index >= 0:
                    self.peak_mode_combo.setCurrentIndex(index)

        finally:
            # Restore signals
            self.x_measure_combo.blockSignals(signals_blocked[0])
            self.x_channel_combo.blockSignals(signals_blocked[1])
            self.y_measure_combo.blockSignals(signals_blocked[2])
            self.y_channel_combo.blockSignals(signals_blocked[3])
            self.peak_mode_combo.blockSignals(signals_blocked[4])

            # Update peak mode visibility based on current selections
            self._update_peak_mode_visibility()

    def get_all_settings_dict(self) -> dict:
        """
        Get all control panel settings as a dictionary for saving/restoring state.

        Returns:
            dict: Dictionary containing all current settings.
        """
        return {
            "analysis": {
                "range1_start": self.start_spin.value(),
                "range1_end": self.end_spin.value(),
                "use_dual_range": self.dual_range_cb.isChecked(),
                "range2_start": (
                    self.start_spin2.value() if self.dual_range_cb.isChecked() else None
                ),
                "range2_end": (
                    self.end_spin2.value() if self.dual_range_cb.isChecked() else None
                ),
            },
            "plot": {
                "x_measure": self.x_measure_combo.currentText(),
                "x_channel": self.x_channel_combo.currentText(),
                "y_measure": self.y_measure_combo.currentText(),
                "y_channel": self.y_channel_combo.currentText(),
                "peak_mode": self.peak_mode_combo.currentText(),
            },
        }
