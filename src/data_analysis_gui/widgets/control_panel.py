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
        
        return plot_group
    
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
        return {
            'range1_start': self.start_spin.value(),
            'range1_end': self.end_spin.value(),
            'use_dual_range': self.dual_range_cb.isChecked(),
            'range2_start': self.start_spin2.value(),
            'range2_end': self.end_spin2.value(),
            'stimulus_period': self.period_spin.value(),
            'x_measure': self.x_measure_combo.currentText(),
            'x_channel': self.x_channel_combo.currentText() if self.x_measure_combo.currentText() != "Time" else None,
            'y_measure': self.y_measure_combo.currentText(),
            'y_channel': self.y_channel_combo.currentText() if self.y_measure_combo.currentText() != "Time" else None,
        }
    
    def update_file_info(self, file_name: str, sweep_count: int):
        """Update file information labels"""
        self.file_label.setText(f"File: {file_name}")
        self.sweep_count_label.setText(f"Sweeps: {sweep_count}")
    
    def set_controls_enabled(self, enabled: bool):
        """Enable or disable analysis controls"""
        self.update_plot_btn.setEnabled(enabled)
        self.export_plot_btn.setEnabled(enabled)
    
    def update_swap_button_state(self, is_swapped: bool):
        """Update the swap channels button appearance"""
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