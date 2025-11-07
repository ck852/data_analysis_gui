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
from PySide6.QtCore import Signal, Qt

# Import custom widgets that handle scrolling properly
from data_analysis_gui.widgets.custom_inputs import SelectAllSpinBox, NoScrollComboBox, NumericLineEdit
from data_analysis_gui.config import DEFAULT_SETTINGS
from data_analysis_gui.core.params import AnalysisParameters

from data_analysis_gui.config.themes import (style_button, style_scroll_area, style_group_box, style_label, style_checkbox, apply_compact_layout, 
                                             style_spinbox_with_arrows, style_combo_simple, MODERN_COLORS, WIDGET_SIZES)

# Import logging
from data_analysis_gui.config.logging import get_logger

from typing import Tuple, Optional

logger = get_logger(__name__)


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
        
        logger.debug("Initializing ControlPanel")

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
        
        logger.info("ControlPanel initialized with default settings")

    def _setup_ui(self):
        """
        Set up the control panel UI with full theme integration.
        Creates all controls, layouts, and applies styling.
        """
        logger.debug("Setting up ControlPanel UI")
        
        # Create scroll area for the controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(100)

        # Apply theme styling to scroll area
        style_scroll_area(scroll_area)

        # Main control widget inside scroll area
        control_widget = QWidget()
        scroll_area.setWidget(control_widget)

        # Layout for control widget with theme-based compact spacing
        layout = QVBoxLayout(control_widget)
        apply_compact_layout(control_widget, spacing=4, margin=4)

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
        apply_compact_layout(analysis_group, spacing=4, margin=4)

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

        self.start_spin = NumericLineEdit()
        self.start_spin.setRange(0, 100000)
        self.start_spin.setValue(DEFAULT_SETTINGS["range1_start"])
        self.start_spin.setSingleStep(0.05)
        self.start_spin.setDecimals(2)
        self.start_spin.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_spinbox_with_arrows(self.start_spin)
        layout.addWidget(self.start_spin, 0, 1)
        self.start_spin.setMaximumWidth(90)

        # Range 1 End
        end_label = QLabel("Range 1 End (ms):")
        style_label(end_label, "normal")
        layout.addWidget(end_label, 1, 0)

        self.end_spin = NumericLineEdit()
        self.end_spin.setRange(0, 100000)
        self.end_spin.setValue(DEFAULT_SETTINGS["range1_end"])
        self.end_spin.setSingleStep(0.05)
        self.end_spin.setDecimals(2)
        self.end_spin.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_spinbox_with_arrows(self.end_spin)
        layout.addWidget(self.end_spin, 1, 1)
        self.end_spin.setMaximumWidth(90)

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

        self.start_spin2 = NumericLineEdit()
        self.start_spin2.setRange(0, 100000)
        self.start_spin2.setValue(DEFAULT_SETTINGS["range2_start"])
        self.start_spin2.setSingleStep(0.05)
        self.start_spin2.setDecimals(2)
        self.start_spin2.setEnabled(False)
        self.start_spin2.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_spinbox_with_arrows(self.start_spin2)
        layout.addWidget(self.start_spin2, 3, 1)
        self.start_spin2.setMaximumWidth(90)

        # Range 2 End
        end2_label = QLabel("Range 2 End (ms):")
        style_label(end2_label, "normal")
        layout.addWidget(end2_label, 4, 0)

        self.end_spin2 = NumericLineEdit()
        self.end_spin2.setRange(0, 100000)
        self.end_spin2.setValue(DEFAULT_SETTINGS["range2_end"])
        self.end_spin2.setSingleStep(0.05)
        self.end_spin2.setDecimals(2)
        self.end_spin2.setEnabled(False)
        self.end_spin2.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_spinbox_with_arrows(self.end_spin2)
        layout.addWidget(self.end_spin2, 4, 1)
        self.end_spin2.setMaximumWidth(90)

    def _create_plot_settings_group(self):
        """
        Create the plot settings group with themed controls.

        Returns:
            QGroupBox: The plot settings group box.
        """
        plot_group = QGroupBox("Plot Settings")
        style_group_box(plot_group)

        plot_layout = QGridLayout(plot_group)
        apply_compact_layout(plot_group, spacing=4, margin=4)

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
        self.y_measure_combo.addItems(["Peak", "Average", "Time", "Conductance"])
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

        # Conductance settings group (initially hidden)
        self.conductance_group = self._create_conductance_settings_group()
        plot_layout.addWidget(self.conductance_group, 2, 0, 1, 3)

        # Update plot button with theme styling
        self.update_plot_btn = QPushButton("Generate Analysis Plot")
        style_button(self.update_plot_btn, "primary")
        self.update_plot_btn.clicked.connect(self.analysis_requested.emit)
        self.update_plot_btn.setEnabled(False)
        plot_layout.addWidget(self.update_plot_btn, 3, 0, 1, 3)

        # Peak Mode settings with NoScrollComboBox
        peak_label = QLabel("Peak Mode:")
        peak_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        style_label(peak_label, "normal")
        plot_layout.addWidget(peak_label, 4, 0)

        self.peak_mode_combo = NoScrollComboBox()
        self.peak_mode_combo.addItems(["Absolute", "Positive", "Negative", "Peak-Peak"])
        self.peak_mode_combo.setCurrentText("Absolute")
        self.peak_mode_combo.setToolTip(
            "Peak calculation mode (applies when X or Y axis is set to Peak, "
            "or when Conductance uses Peak measurements)"
        )
        self.peak_mode_combo.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_combo_simple(self.peak_mode_combo)
        plot_layout.addWidget(self.peak_mode_combo, 4, 1, 1, 2)

        # Connect signals to enable/disable peak mode and channel combos
        self.x_measure_combo.currentTextChanged.connect(
            self._update_axis_dependent_controls
        )
        self.y_measure_combo.currentTextChanged.connect(
            self._update_axis_dependent_controls
        )

        return plot_group


    def _create_conductance_settings_group(self):
        """
        Create the conductance configuration group box with themed controls.
        
        Returns:
            QGroupBox: Group box containing conductance settings.
        """
        conductance_group = QGroupBox("Conductance Settings")
        style_group_box(conductance_group)
        
        cond_layout = QGridLayout(conductance_group)
        apply_compact_layout(conductance_group, spacing=4, margin=4)
        
        # Current measure selection
        i_label = QLabel("Current:")
        style_label(i_label, "normal")
        cond_layout.addWidget(i_label, 0, 0)
        
        self.cond_i_measure_combo = NoScrollComboBox()
        self.cond_i_measure_combo.addItems(["Average", "Peak"])
        self.cond_i_measure_combo.setCurrentText("Average")
        self.cond_i_measure_combo.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_combo_simple(self.cond_i_measure_combo)
        cond_layout.addWidget(self.cond_i_measure_combo, 0, 1)
        
        # Voltage measure selection
        v_label = QLabel("Voltage:")
        style_label(v_label, "normal")
        cond_layout.addWidget(v_label, 1, 0)
        
        self.cond_v_measure_combo = NoScrollComboBox()
        self.cond_v_measure_combo.addItems(["Average", "Peak"])
        self.cond_v_measure_combo.setCurrentText("Average")
        self.cond_v_measure_combo.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_combo_simple(self.cond_v_measure_combo)
        cond_layout.addWidget(self.cond_v_measure_combo, 1, 1)
        
        # Reversal potential input
        vrev_label = QLabel("Vrev (mV):")
        style_label(vrev_label, "normal")
        cond_layout.addWidget(vrev_label, 2, 0)
        
        self.cond_vrev_input = NumericLineEdit()
        self.cond_vrev_input.setRange(-200, 200)
        self.cond_vrev_input.setValue(0)
        self.cond_vrev_input.setSingleStep(1.0)
        self.cond_vrev_input.setDecimals(1)
        self.cond_vrev_input.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_spinbox_with_arrows(self.cond_vrev_input)
        cond_layout.addWidget(self.cond_vrev_input, 2, 1)
        self.cond_vrev_input.setMaximumWidth(90)
        
        # Units selection
        units_label = QLabel("Units:")
        style_label(units_label, "normal")
        cond_layout.addWidget(units_label, 3, 0)
        
        self.cond_units_combo = NoScrollComboBox()
        self.cond_units_combo.addItems(["pS", "nS", "μS", "mS", "S"])
        self.cond_units_combo.setCurrentText("nS")
        self.cond_units_combo.setMinimumHeight(WIDGET_SIZES["input_height"])
        style_combo_simple(self.cond_units_combo)
        cond_layout.addWidget(self.cond_units_combo, 3, 1)
        
        # Initially hidden
        conductance_group.setVisible(False)
        
        # Connect measure combos to update peak mode visibility
        self.cond_i_measure_combo.currentTextChanged.connect(
            self._update_axis_dependent_controls
        )
        self.cond_v_measure_combo.currentTextChanged.connect(
            self._update_axis_dependent_controls
        )
        
        return conductance_group

    def _update_axis_dependent_controls(self):
        """
        Update visibility and enabled state of controls based on axis selections.
        
        Handles:
        - Peak mode combo (enabled when Peak is used anywhere)
        - X/Y channel combos (disabled for Time and Conductance)
        - Conductance settings group (shown when Y-axis is Conductance)
        - Dual range checkbox (disabled when Conductance selected)
        """
        x_measure = self.x_measure_combo.currentText()
        y_measure = self.y_measure_combo.currentText()
        
        # Update X-axis channel combo visibility (disabled for Time)
        x_is_time = x_measure == "Time"
        self.x_channel_combo.setEnabled(not x_is_time)
        if not x_is_time:
            style_combo_simple(self.x_channel_combo)
        
        # Update Y-axis channel combo visibility (disabled for Time and Conductance)
        y_needs_channel = y_measure not in ["Time", "Conductance"]
        self.y_channel_combo.setEnabled(y_needs_channel)
        if y_needs_channel:
            style_combo_simple(self.y_channel_combo)
        
        # Show/hide conductance settings group
        is_conductance = y_measure == "Conductance"
        self.conductance_group.setVisible(is_conductance)
        
        # If conductance selected and dual range is checked, uncheck it
        if is_conductance and self.dual_range_cb.isChecked():
            logger.warning("Conductance selected: disabling dual range (not supported)")
            self.dual_range_cb.setChecked(False)
        
        # Disable dual range checkbox when conductance is selected
        self.dual_range_cb.setEnabled(not is_conductance)
        if is_conductance:
            style_checkbox(self.dual_range_cb)
        
        # Determine if peak mode should be enabled
        # Peak mode is used when:
        # 1. X or Y axis is set to "Peak", OR
        # 2. Conductance is selected and either I or V measure is "Peak"
        is_peak_used = (x_measure == "Peak" or y_measure == "Peak")
        
        if is_conductance:
            i_measure = self.cond_i_measure_combo.currentText()
            v_measure = self.cond_v_measure_combo.currentText()
            is_peak_used = is_peak_used or (i_measure == "Peak" or v_measure == "Peak")
        
        self.peak_mode_combo.setEnabled(is_peak_used)
        
        # Re-apply styling to ensure disabled state looks correct
        if not is_peak_used:
            style_combo_simple(self.peak_mode_combo)

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
        logger.debug("Connecting ControlPanel signals")
        
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
                logger.warning(f"Range 1 validation failed: start={start1_val}, end={end1_val}")
                self._mark_field_invalid("start1")
                self._mark_field_invalid("end1")
            else:
                if "start1" in self._invalid_fields or "end1" in self._invalid_fields:
                    logger.debug(f"Range 1 validation passed: start={start1_val}, end={end1_val}")
                self._clear_invalid_state("start1")
                self._clear_invalid_state("end1")

            # --- Validate Range 2 (if enabled) ---
            is_range2_valid = True
            if self.dual_range_cb.isChecked():
                start2_val = self.start_spin2.value()
                end2_val = self.end_spin2.value()
                is_range2_valid = end2_val > start2_val

                if not is_range2_valid:
                    logger.warning(f"Range 2 validation failed: start={start2_val}, end={end2_val}")
                    self._mark_field_invalid("start2")
                    self._mark_field_invalid("end2")
                else:
                    if "start2" in self._invalid_fields or "end2" in self._invalid_fields:
                        logger.debug(f"Range 2 validation passed: start={start2_val}, end={end2_val}")
                    self._clear_invalid_state("start2")
                    self._clear_invalid_state("end2")
            else:
                # If dual range is disabled, its fields can't be invalid
                self._clear_invalid_state("start2")
                self._clear_invalid_state("end2")

            # --- Update Button State ---
            is_all_valid = is_range1_valid and is_range2_valid
            previous_button_state = self.update_plot_btn.isEnabled()
            self.update_plot_btn.setEnabled(is_all_valid)
            self.export_plot_btn.setEnabled(is_all_valid)
            
            if previous_button_state != is_all_valid:
                logger.debug(f"Analysis buttons {'enabled' if is_all_valid else 'disabled'} (validation: {is_all_valid})")

            # --- Sync Cursors ---
            logger.info(f"DIAGNOSTIC: About to emit range_values_changed signal")
            self.range_values_changed.emit()
            logger.info(f"DIAGNOSTIC: Emitted range_values_changed signal")

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
            logger.debug(f"Marking field {spinbox_key} as invalid")
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
            logger.debug(f"Clearing invalid state from field {spinbox_key}")
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
        logger.info(f"Dual range {'enabled' if enabled else 'disabled'}")
        
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
        from data_analysis_gui.core.params import AnalysisParameters, AxisConfig, ConductanceConfig

        # Determine peak mode
        peak_mode = self.peak_mode_combo.currentText()

        # Create X-axis config
        x_measure = self.x_measure_combo.currentText()
        x_axis = AxisConfig(
            measure=x_measure,
            channel=self.x_channel_combo.currentText() if x_measure not in ["Time"] else None,
            peak_type=peak_mode if x_measure == "Peak" else None,
        )

        # Create Y-axis config
        y_measure = self.y_measure_combo.currentText()
        y_axis = AxisConfig(
            measure=y_measure,
            channel=self.y_channel_combo.currentText() if y_measure not in ["Time", "Conductance"] else None,
            peak_type=peak_mode if y_measure == "Peak" else None,
        )

        # Create conductance config if Y-axis is Conductance
        conductance_config = None
        if y_measure == "Conductance":
            conductance_config = ConductanceConfig(
                i_measure=self.cond_i_measure_combo.currentText(),
                v_measure=self.cond_v_measure_combo.currentText(),
                vrev=self.cond_vrev_input.value(),
                units=self.cond_units_combo.currentText(),
            )

        params = AnalysisParameters(
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
            channel_config={},
            conductance_config=conductance_config,
        )
        
        logger.debug(f"Generated parameters: R1=[{params.range1_start}, {params.range1_end}], "
                    f"dual={params.use_dual_range}, X={x_measure}/{x_axis.channel}, "
                    f"Y={y_measure}/{y_axis.channel}, conductance={conductance_config is not None}")
        
        return params

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

    def validate_ranges(self) -> Tuple[bool, Optional[str]]:
        """
        Validate all active analysis ranges.
        
        This is a public method for external callers (MainWindow, dialogs) to
        check if ranges are valid before launching analysis operations.
        
        Returns:
            Tuple[bool, Optional[str]]: 
                - First element: True if all ranges are valid, False otherwise
                - Second element: Error message if invalid, None if valid
        
        Example:
            >>> is_valid, error_msg = control_panel.validate_ranges()
            >>> if not is_valid:
            ...     QMessageBox.warning(self, "Invalid Range", error_msg)
            ...     return
        """
        vals = self.get_range_values()
        
        # Validate Range 1
        if vals["range1_end"] <= vals["range1_start"]:
            logger.warning(f"Range validation failed: Range 1 end ({vals['range1_end']}) <= start ({vals['range1_start']})")
            return False, "Range 1: End time must be greater than start time."
        
        # Validate Range 2 if dual range is enabled
        if vals["use_dual_range"]:
            range2_start = vals.get("range2_start")
            range2_end = vals.get("range2_end")
            
            if range2_start is None or range2_end is None:
                logger.warning("Range validation failed: Range 2 values are missing")
                return False, "Range 2: Values are missing."
            
            if range2_end <= range2_start:
                logger.warning(f"Range validation failed: Range 2 end ({range2_end}) <= start ({range2_start})")
                return False, "Range 2: End time must be greater than start time."
        
        logger.debug("Range validation passed")
        return True, None

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
            logger.debug(f"Updating {spinbox_key} to {value}")
            # setValue() triggers validation automatically
            spinbox_map[spinbox_key].setValue(value)

    def set_analysis_range(self, max_time: float):
        """
        Set the maximum value for analysis range spinboxes and clamp current values.

        Args:
            max_time (float): Maximum allowed time value.
        """
        logger.info(f"Setting analysis range maximum to {max_time} ms")
        
        self.start_spin.setRange(0, max_time)
        self.end_spin.setRange(0, max_time)
        self.start_spin2.setRange(0, max_time)
        self.end_spin2.setRange(0, max_time)

        # Clamp existing values to the new range
        clamped_values = []
        if self.start_spin.value() > max_time:
            self.start_spin.setValue(max_time)
            clamped_values.append("start1")
        if self.end_spin.value() > max_time:
            self.end_spin.setValue(max_time)
            clamped_values.append("end1")
        if self.start_spin2.value() > max_time:
            self.start_spin2.setValue(max_time)
            clamped_values.append("start2")
        if self.end_spin2.value() > max_time:
            self.end_spin2.setValue(max_time)
            clamped_values.append("end2")
        
        if clamped_values:
            logger.debug(f"Clamped values to max_time: {', '.join(clamped_values)}")

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
        logger.debug(f"Setting parameters from dict: {params}")
        
        # Store current signal state and disconnect
        signals_were_blocked = [
            self.start_spin.blockSignals(True),
            self.end_spin.blockSignals(True),
            self.start_spin2.blockSignals(True),
            self.end_spin2.blockSignals(True),
            self.dual_range_cb.blockSignals(True),
        ]

        try:
            # Set Range 1 values
            if "range1_start" in params:
                self.start_spin.setValue(params["range1_start"])
            if "range1_end" in params:
                self.end_spin.setValue(params["range1_end"])

            # Set dual range checkbox
            use_dual = params.get("use_dual_range", False)
            
            # EDGE CASE HANDLING: If loading settings with conductance + dual range,
            # force dual_range to False to maintain valid state
            # (This shouldn't happen normally, but defensive programming)
            if hasattr(self, 'y_measure_combo'):
                y_measure = self.y_measure_combo.currentText()
                if y_measure == "Conductance" and use_dual:
                    logger.warning(
                        "Loaded settings had Conductance with dual_range=True. "
                        "Forcing dual_range to False."
                    )
                    use_dual = False
            
            self.dual_range_cb.setChecked(use_dual)

            # Enable/disable Range 2 controls based on dual range
            self.start_spin2.setEnabled(use_dual)
            self.end_spin2.setEnabled(use_dual)

            # Always restore Range 2 values if they exist
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

            # Re-apply styling to ensure correct appearance
            style_spinbox_with_arrows(self.start_spin2)
            style_spinbox_with_arrows(self.end_spin2)

            # Now validate everything once and emit signals
            self._validate_and_update()
            if use_dual:
                self.dual_range_toggled.emit(use_dual)
        
        logger.info("Parameters applied from dict")

    def set_plot_settings_from_dict(self, params: dict):
        """
        Set plot settings from a dictionary, blocking signals during update.

        Args:
            params (dict): Dictionary of plot setting values.
        """
        logger.debug(f"Setting plot settings from dict: {params}")
        
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
            
            # Set conductance settings if present
            if "conductance" in params:
                cond = params["conductance"]
                
                # Block signals for conductance controls
                self.cond_i_measure_combo.blockSignals(True)
                self.cond_v_measure_combo.blockSignals(True)
                self.cond_vrev_input.blockSignals(True)
                self.cond_units_combo.blockSignals(True)
                
                try:
                    if "i_measure" in cond:
                        index = self.cond_i_measure_combo.findText(cond["i_measure"])
                        if index >= 0:
                            self.cond_i_measure_combo.setCurrentIndex(index)
                    
                    if "v_measure" in cond:
                        index = self.cond_v_measure_combo.findText(cond["v_measure"])
                        if index >= 0:
                            self.cond_v_measure_combo.setCurrentIndex(index)
                    
                    if "vrev" in cond:
                        self.cond_vrev_input.setValue(cond["vrev"])
                    
                    if "units" in cond:
                        index = self.cond_units_combo.findText(cond["units"])
                        if index >= 0:
                            self.cond_units_combo.setCurrentIndex(index)
                finally:
                    # Unblock conductance control signals
                    self.cond_i_measure_combo.blockSignals(False)
                    self.cond_v_measure_combo.blockSignals(False)
                    self.cond_vrev_input.blockSignals(False)
                    self.cond_units_combo.blockSignals(False)

        finally:
            # Restore signals
            self.x_measure_combo.blockSignals(signals_blocked[0])
            self.x_channel_combo.blockSignals(signals_blocked[1])
            self.y_measure_combo.blockSignals(signals_blocked[2])
            self.y_channel_combo.blockSignals(signals_blocked[3])
            self.peak_mode_combo.blockSignals(signals_blocked[4])

            # Update control visibility and states
            self._update_axis_dependent_controls()
        
        logger.info("Plot settings applied from dict")

    def get_all_settings_dict(self) -> dict:
        """
        Get all control panel settings as a dictionary for saving/restoring state.

        Returns:
            dict: Dictionary containing all current settings.
        """
        settings = {
            "analysis": {
                "range1_start": self.start_spin.value(),
                "range1_end": self.end_spin.value(),
                "use_dual_range": self.dual_range_cb.isChecked(),
                "range2_start": self.start_spin2.value(),
                "range2_end": self.end_spin2.value(),
            },
            "plot": {
                "x_measure": self.x_measure_combo.currentText(),
                "x_channel": self.x_channel_combo.currentText(),
                "y_measure": self.y_measure_combo.currentText(),
                "y_channel": self.y_channel_combo.currentText(),
                "peak_mode": self.peak_mode_combo.currentText(),
            },
        }
        
        # Add conductance settings if present
        if self.y_measure_combo.currentText() == "Conductance":
            settings["plot"]["conductance"] = {
                "i_measure": self.cond_i_measure_combo.currentText(),
                "v_measure": self.cond_v_measure_combo.currentText(),
                "vrev": self.cond_vrev_input.value(),
                "units": self.cond_units_combo.currentText(),
            }
        
        logger.debug(f"Retrieved all settings: {settings}")
        return settings

    def update_range_value_silent(self, spinbox_key: str, value: float):
        """Update spinbox without emitting signals (prevents feedback loop)."""
        spinbox_map = {
            "start1": self.start_spin,
            "end1": self.end_spin,
            "start2": self.start_spin2,
            "end2": self.end_spin2,
        }
        if spinbox_key in spinbox_map:
            spinbox = spinbox_map[spinbox_key]
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)