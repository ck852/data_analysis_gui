"""
Control Panel Widget - Extracted from main window for better modularity.
Handles all control settings and communicates via signals.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QScrollArea, QGroupBox,
                             QLabel, QPushButton, QCheckBox, QComboBox,
                             QGridLayout)
from PyQt5.QtCore import pyqtSignal

from data_analysis_gui.widgets import SelectAllSpinBox
from data_analysis_gui.config import DEFAULT_SETTINGS


class ControlPanel(QWidget):
    """
    Self-contained control panel widget that manages all analysis settings.
    Emits signals to communicate user actions to the main window.
    """
    
    # Define signals for communication with main window
    analysis_requested = pyqtSignal()  # User wants to generate analysis plot
    export_requested = pyqtSignal()  # User wants to export data
    swap_channels_requested = pyqtSignal()  # User wants to swap channels
    center_cursor_requested = pyqtSignal()  # User wants to center cursor
    dual_range_toggled = pyqtSignal(bool)  # Dual range checkbox changed
    range_values_changed = pyqtSignal()  # Any range spinbox value changed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Track channel swap state
        self._is_swapped = False
        self._pending_swap_state = False
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the control panel UI"""
        # Create scroll area for the controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(400)
        
        # Main control widget inside scroll area
        control_widget = QWidget()
        scroll_area.setWidget(control_widget)
        
        # Layout for control widget
        layout = QVBoxLayout(control_widget)
        
        # Add all control groups
        layout.addWidget(self._create_file_info_group())
        layout.addWidget(self._create_analysis_settings_group())
        layout.addWidget(self._create_plot_settings_group())
        
        # Export Plot Data button
        self.export_plot_btn = QPushButton("Export Plot Data")
        self.export_plot_btn.clicked.connect(self.export_requested.emit)
        self.export_plot_btn.setEnabled(False)
        layout.addWidget(self.export_plot_btn)
        
        layout.addStretch()
        
        # Set main layout for this widget
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
    
    def _create_file_info_group(self):
        """Create the file information group"""
        file_group = QGroupBox("File Information")
        file_layout = QVBoxLayout(file_group)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        self.sweep_count_label = QLabel("Sweeps: 0")
        file_layout.addWidget(self.sweep_count_label)
        
        return file_group
    
    def _create_analysis_settings_group(self):
        """Create the analysis settings group"""
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QGridLayout(analysis_group)
        
        # Range 1 settings
        self._add_range1_settings(analysis_layout)
        
        # Dual range checkbox
        self.dual_range_cb = QCheckBox("Use Dual Analysis")
        self.dual_range_cb.stateChanged.connect(self._on_dual_range_changed)
        analysis_layout.addWidget(self.dual_range_cb, 2, 0, 1, 2)
        
        # Range 2 settings
        self._add_range2_settings(analysis_layout)
        
        # Stimulus period
        analysis_layout.addWidget(QLabel("Stimulus Period (ms):"), 5, 0)
        self.period_spin = SelectAllSpinBox()
        self.period_spin.setRange(1, 100000)
        self.period_spin.setValue(DEFAULT_SETTINGS['stimulus_period'])
        self.period_spin.setSingleStep(100)
        analysis_layout.addWidget(self.period_spin, 5, 1)
        
        # Swap Channels button
        self.swap_channels_btn = QPushButton("Swap Channels")
        self.swap_channels_btn.setToolTip("Swap voltage and current channel assignments")
        self.swap_channels_btn.clicked.connect(self.swap_channels_requested.emit)
        self.swap_channels_btn.setEnabled(False) # Initially disabled
        analysis_layout.addWidget(self.swap_channels_btn, 6, 0, 1, 2)
        
        # Center Nearest Cursor button
        center_cursor_btn = QPushButton("Center Nearest Cursor")
        center_cursor_btn.setToolTip("Moves the nearest cursor to the center of the view")
        center_cursor_btn.clicked.connect(self.center_cursor_requested.emit)
        analysis_layout.addWidget(center_cursor_btn, 7, 0, 1, 2)
        
        return analysis_group
    
    def _add_range1_settings(self, layout):
        """Add Range 1 settings to layout"""
        layout.addWidget(QLabel("Range 1 Start (ms):"), 0, 0)
        self.start_spin = SelectAllSpinBox()
        self.start_spin.setRange(0, 100000)
        self.start_spin.setValue(DEFAULT_SETTINGS['range1_start'])
        self.start_spin.setSingleStep(0.05)
        self.start_spin.setDecimals(2)
        layout.addWidget(self.start_spin, 0, 1)
        
        layout.addWidget(QLabel("Range 1 End (ms):"), 1, 0)
        self.end_spin = SelectAllSpinBox()
        self.end_spin.setRange(0, 100000)
        self.end_spin.setValue(DEFAULT_SETTINGS['range1_end'])
        self.end_spin.setSingleStep(0.05)
        self.end_spin.setDecimals(2)
        layout.addWidget(self.end_spin, 1, 1)
    
    def _add_range2_settings(self, layout):
        """Add Range 2 settings to layout"""
        layout.addWidget(QLabel("Range 2 Start (ms):"), 3, 0)
        self.start_spin2 = SelectAllSpinBox()
        self.start_spin2.setRange(0, 100000)
        self.start_spin2.setValue(DEFAULT_SETTINGS['range2_start'])
        self.start_spin2.setSingleStep(0.05)
        self.start_spin2.setDecimals(2)
        self.start_spin2.setEnabled(False)
        layout.addWidget(self.start_spin2, 3, 1)
        
        layout.addWidget(QLabel("Range 2 End (ms):"), 4, 0)
        self.end_spin2 = SelectAllSpinBox()
        self.end_spin2.setRange(0, 100000)
        self.end_spin2.setValue(DEFAULT_SETTINGS['range2_end'])
        self.end_spin2.setSingleStep(0.05)
        self.end_spin2.setDecimals(2)
        self.end_spin2.setEnabled(False)
        layout.addWidget(self.end_spin2, 4, 1)
    
    def _create_plot_settings_group(self):
        """Create the plot settings group"""
        plot_group = QGroupBox("Plot Settings")
        plot_layout = QGridLayout(plot_group)
        
        # X-axis settings
        plot_layout.addWidget(QLabel("X-Axis:"), 0, 0)
        self.x_measure_combo = QComboBox()
        self.x_measure_combo.addItems(["Time", "Peak", "Average"])
        self.x_measure_combo.setCurrentText("Average")
        plot_layout.addWidget(self.x_measure_combo, 0, 1)
        
        self.x_channel_combo = QComboBox()
        self.x_channel_combo.addItems(["Voltage", "Current"])
        self.x_channel_combo.setCurrentText("Voltage")
        plot_layout.addWidget(self.x_channel_combo, 0, 2)
        
        # Y-axis settings
        plot_layout.addWidget(QLabel("Y-Axis:"), 1, 0)
        self.y_measure_combo = QComboBox()
        self.y_measure_combo.addItems(["Peak", "Average", "Time"])
        self.y_measure_combo.setCurrentText("Average")
        plot_layout.addWidget(self.y_measure_combo, 1, 1)
        
        self.y_channel_combo = QComboBox()
        self.y_channel_combo.addItems(["Voltage", "Current"])
        self.y_channel_combo.setCurrentText("Current")
        plot_layout.addWidget(self.y_channel_combo, 1, 2)
        
        # Update plot button
        self.update_plot_btn = QPushButton("Generate Analysis Plot")
        self.update_plot_btn.clicked.connect(self.analysis_requested.emit)
        self.update_plot_btn.setEnabled(False)
        plot_layout.addWidget(self.update_plot_btn, 2, 0, 1, 3)
        
        plot_layout.addWidget(QLabel("Peak Mode:"), 3, 0)
        self.peak_mode_combo = QComboBox()
        self.peak_mode_combo.addItems(["Absolute", "Positive", "Negative", "Peak-Peak"])
        self.peak_mode_combo.setCurrentText("Absolute")
        self.peak_mode_combo.setToolTip("Peak calculation mode (applies when X or Y axis is set to Peak)")
        plot_layout.addWidget(self.peak_mode_combo, 3, 1, 1, 2)
        
        # Connect signal to enable/disable peak mode based on axis selection
        self.x_measure_combo.currentTextChanged.connect(self._update_peak_mode_visibility)
        self.y_measure_combo.currentTextChanged.connect(self._update_peak_mode_visibility)

        return plot_group
    
    def _update_peak_mode_visibility(self):
        """Enable/disable peak mode combo based on whether Peak is selected"""
        is_peak_selected = (self.x_measure_combo.currentText() == "Peak" or 
                        self.y_measure_combo.currentText() == "Peak")
        self.peak_mode_combo.setEnabled(is_peak_selected)
    
    def _connect_signals(self):
        """Connect internal widget signals"""
        # Connect range spinbox changes
        self.start_spin.valueChanged.connect(self.range_values_changed.emit)
        self.end_spin.valueChanged.connect(self.range_values_changed.emit)
        self.start_spin2.valueChanged.connect(self.range_values_changed.emit)
        self.end_spin2.valueChanged.connect(self.range_values_changed.emit)
    
    def _on_dual_range_changed(self):
        """Handle dual range checkbox state change"""
        enabled = self.dual_range_cb.isChecked()
        self.start_spin2.setEnabled(enabled)
        self.end_spin2.setEnabled(enabled)
        self.dual_range_toggled.emit(enabled)
    
    # --- Public methods for data access and updates ---
    
    def collect_parameters(self) -> dict:
        """
        Collect all parameters from the control panel.
        Returns a dictionary with all settings.
        """
        peak_mode = self.peak_mode_combo.currentText()
        
        return {
            'range1_start': self.start_spin.value(),
            'range1_end': self.end_spin.value(),
            'use_dual_range': self.dual_range_cb.isChecked(),
            'range2_start': self.start_spin2.value(),
            'range2_end': self.end_spin2.value(),
            'stimulus_period': self.period_spin.value(),
            'x_measure': self.x_measure_combo.currentText(),
            'x_channel': self.x_channel_combo.currentText() if self.x_measure_combo.currentText() != "Time" else None,
            'x_peak_type': peak_mode if self.x_measure_combo.currentText() == "Peak" else None,
            'y_measure': self.y_measure_combo.currentText(),
            'y_channel': self.y_channel_combo.currentText() if self.y_measure_combo.currentText() != "Time" else None,
            'y_peak_type': peak_mode if self.y_measure_combo.currentText() == "Peak" else None,
            'channels_swapped': self._is_swapped,  # Add this line
        }
    
    def update_file_info(self, file_name: str, sweep_count: int):
        """Update file information labels"""
        self.file_label.setText(f"File: {file_name}")
        self.sweep_count_label.setText(f"Sweeps: {sweep_count}")
    
    def set_controls_enabled(self, enabled: bool):
        """Enable or disable analysis controls"""
        self.update_plot_btn.setEnabled(enabled)
        self.export_plot_btn.setEnabled(enabled)
        self.swap_channels_btn.setEnabled(enabled)  # Add this line
        
        # If enabling controls and we have a pending swap state, maintain button appearance
        if enabled and self._pending_swap_state:
            # The pending state visual should already be set, just update internal state
            self._is_swapped = self._pending_swap_state
            self._pending_swap_state = False
    
    def update_swap_button_state(self, is_swapped: bool):
        """Update the swap channels button appearance"""
        self._is_swapped = is_swapped  # Add this line
        if is_swapped:
            self.swap_channels_btn.setStyleSheet("QPushButton { background-color: #ffcc99; }")
            self.swap_channels_btn.setText("Channels Swapped ⇄")
        else:
            self.swap_channels_btn.setStyleSheet("")
            self.swap_channels_btn.setText("Swap Channels")
    
    def get_range_values(self) -> dict:
        """Get current range values"""
        return {
            'range1_start': self.start_spin.value(),
            'range1_end': self.end_spin.value(),
            'use_dual_range': self.dual_range_cb.isChecked(),
            'range2_start': self.start_spin2.value() if self.dual_range_cb.isChecked() else None,
            'range2_end': self.end_spin2.value() if self.dual_range_cb.isChecked() else None
        }
    
    def get_range_spinboxes(self) -> dict:
        """Get references to range spinboxes for plot manager"""
        spinboxes = {
            'start1': self.start_spin,
            'end1': self.end_spin
        }
        if self.dual_range_cb.isChecked():
            spinboxes['start2'] = self.start_spin2
            spinboxes['end2'] = self.end_spin2
        return spinboxes
    
    def update_range_value(self, spinbox_key: str, value: float):
        """Update a specific range spinbox value"""
        spinbox_map = {
            'start1': self.start_spin,
            'end1': self.end_spin,
            'start2': self.start_spin2,
            'end2': self.end_spin2
        }
        if spinbox_key in spinbox_map:
            spinbox_map[spinbox_key].setValue(value)

    def set_analysis_range(self, max_time: float):
        """Sets the maximum value for the analysis range spinboxes and clamps current values."""
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


    def apply_parameters(self, params: dict) -> None:
        """
        Apply parameters to control panel widgets.
        Only sets values for keys that exist in params, ignores unknown keys.
        
        Args:
            params: Dictionary of parameters to apply
        """
        # Range 1 settings (always available)
        if 'range1_start' in params:
            self.start_spin.setValue(params['range1_start'])
        if 'range1_end' in params:
            self.end_spin.setValue(params['range1_end'])
        
        # Dual range checkbox
        if 'use_dual_range' in params:
            self.dual_range_cb.setChecked(params['use_dual_range'])
        
        # Range 2 settings (only if dual range is enabled)
        if self.dual_range_cb.isChecked():
            if 'range2_start' in params:
                self.start_spin2.setValue(params['range2_start'])
            if 'range2_end' in params:
                self.end_spin2.setValue(params['range2_end'])
        
        # Stimulus period
        if 'stimulus_period' in params:
            self.period_spin.setValue(params['stimulus_period'])
        
        # Plot settings - X axis
        if 'x_measure' in params:
            idx = self.x_measure_combo.findText(params['x_measure'])
            if idx >= 0:
                self.x_measure_combo.setCurrentIndex(idx)
        if 'x_channel' in params and params['x_channel']:
            idx = self.x_channel_combo.findText(params['x_channel'])
            if idx >= 0:
                self.x_channel_combo.setCurrentIndex(idx)
        
        # Plot settings - Y axis
        if 'y_measure' in params:
            idx = self.y_measure_combo.findText(params['y_measure'])
            if idx >= 0:
                self.y_measure_combo.setCurrentIndex(idx)
        if 'y_channel' in params and params['y_channel']:
            idx = self.y_channel_combo.findText(params['y_channel'])
            if idx >= 0:
                self.y_channel_combo.setCurrentIndex(idx)
        
        # Peak mode (if peak is selected)
        if 'x_peak_type' in params and params['x_peak_type']:
            idx = self.peak_mode_combo.findText(params['x_peak_type'])
            if idx >= 0:
                self.peak_mode_combo.setCurrentIndex(idx)
        elif 'y_peak_type' in params and params['y_peak_type']:
            idx = self.peak_mode_combo.findText(params['y_peak_type'])
            if idx >= 0:
                self.peak_mode_combo.setCurrentIndex(idx)
        
        # Channel swap state - store as pending if no file loaded yet
        if 'channels_swapped' in params:
            if self.swap_channels_btn.isEnabled():
                # File already loaded, apply immediately
                self._is_swapped = params.get('channels_swapped', False)
            else:
                # No file loaded yet, store as pending
                self._pending_swap_state = params.get('channels_swapped', False)
                # Update button appearance to show pending state
                if self._pending_swap_state:
                    self.swap_channels_btn.setStyleSheet("QPushButton { background-color: #ffcc99; }")
                    self.swap_channels_btn.setText("Channels Swapped ⇄")
                else:
                    self.swap_channels_btn.setStyleSheet("")
                    self.swap_channels_btn.setText("Swap Channels")

    def get_pending_swap_state(self) -> bool:
        """
        Get the pending swap state that should be applied when a file is loaded.
        
        Returns:
            True if channels should be swapped after file load
        """
        return self._pending_swap_state

    def clear_pending_swap_state(self):
        """Clear the pending swap state after it's been applied."""
        self._pending_swap_state = False