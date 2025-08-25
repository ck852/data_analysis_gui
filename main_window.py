import sys
import os
import numpy as np
import scipy.io
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QCheckBox, QFileDialog,
                             QMessageBox, QGroupBox, QLabel, QSplitter,
                             QScrollArea, QGridLayout, QProgressBar,
                             QStatusBar, QToolBar, QMenuBar, QMenu,
                             QAction, QActionGroup, QInputDialog, QApplication)
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Internal imports
from config import THEMES, get_theme_stylesheet, DEFAULT_SETTINGS
from dialogs import (ConcentrationResponseDialog, BatchResultDialog, 
                     AnalysisPlotDialog, CurrentDensityIVDialog)
from widgets import SelectAllSpinBox, NoScrollComboBox
from utils import (load_mat_file, export_to_csv, process_sweep_data,
                   apply_analysis_mode, calculate_average_voltage,
                   extract_file_number, format_voltage_label)

class ChannelConfiguration:
    """Manages the mapping between physical channels in the data and logical channel types"""
    
    def __init__(self):
        # Define what each physical channel represents
        self.channel_definitions = {
            0: "Voltage",  # Default: Channel 0 is Voltage
            1: "Current"   # Default: Channel 1 is Current
        }
        
    def swap_channels(self):
        """Swap the channel definitions"""
        self.channel_definitions = {
            0: self.channel_definitions[1],
            1: self.channel_definitions[0]
        }
    
    def get_channel_for_type(self, channel_type):
        """Get the physical channel number for a given type (Voltage or Current)
        
        Args:
            channel_type: "Voltage" or "Current"
            
        Returns:
            Physical channel number (0 or 1)
        """
        for channel_num, ch_type in self.channel_definitions.items():
            if ch_type == channel_type:
                return channel_num
        raise ValueError(f"Channel type '{channel_type}' not found")
    
    def get_type_for_channel(self, channel_num):
        """Get the type for a given physical channel number
        
        Args:
            channel_num: Physical channel number (0 or 1)
            
        Returns:
            Channel type string ("Voltage" or "Current")
        """
        return self.channel_definitions.get(channel_num, f"Channel {channel_num}")
    
    def get_available_types(self):
        """Get list of available channel types in order of their physical channels"""
        return [self.channel_definitions[i] for i in sorted(self.channel_definitions.keys())]
    
    def get_channel_label(self, channel_type, include_unit=True):
        """Get a formatted label for a channel type
        
        Args:
            channel_type: "Voltage" or "Current"
            include_unit: Whether to include the unit in the label
            
        Returns:
            Formatted label string
        """
        if include_unit:
            unit = "(mV)" if channel_type == "Voltage" else "(pA)"
            return f"{channel_type} {unit}"
        return channel_type
    
    def get_status_string(self):
        """Get a string describing the current channel configuration"""
        return f"Ch0={self.channel_definitions[0]}, Ch1={self.channel_definitions[1]}"
    
    def is_swapped(self):
        """Check if channels are in non-default configuration"""
        return self.channel_definitions[0] != "Voltage"

class SweepDataProcessor:
    """Handles all data processing and business logic operations"""
    
    @staticmethod
    def process_single_sweep(t, y, t_start, t_end, channel=0):
        """Process a single sweep for the given time range and channel"""
        return process_sweep_data(t, y, t_start, t_end, channel)
    
    @staticmethod
    def calculate_peak_values(data):
        """Calculate peak values from data array"""
        if data.size > 0:
            return np.max(np.abs(data))
        return np.nan
    
    @staticmethod
    def calculate_average_values(data):
        """Calculate average values from data array"""
        if data.size > 0:
            return np.mean(data)
        return np.nan
    
    @staticmethod
    def process_sweep_ranges(sweeps, range_params, dual_range=False, channel_config=None):
        """Process all sweeps for the given range parameters
        
        Args:
            sweeps: Dictionary of sweep data
            range_params: Dict with keys 't_start', 't_end', and optionally 't_start2', 't_end2'
            dual_range: Whether to process dual ranges
            
        Returns:
            Dictionary containing processed data for all sweeps
        """

        # Use default configuration if none provided
        if channel_config is None:
            channel_config = ChannelConfiguration()
        
        voltage_channel = channel_config.get_channel_for_type("Voltage")
        current_channel = channel_config.get_channel_for_type("Current")

        result = {
            "sweep_indices": [], "time_values": [],
            "peak_current": [], "peak_voltage": [],
            "average_current": [], "average_voltage": [],
            "peak_current2": [], "peak_voltage2": [],
            "average_current2": [], "average_voltage2": [],
            "avg_voltages_r1": [], "avg_voltages_r2": [],
        }
        
        period_sec = range_params.get('period_ms', 1000) / 1000.0
        
        for i, index in enumerate(sorted(sweeps.keys(), key=lambda x: int(x))):
            t, y = sweeps[index]
            
            # Process first range
            voltage_data = process_sweep_data(t, y, range_params['t_start'], 
                                            range_params['t_end'], channel=voltage_channel)
            current_data = process_sweep_data(t, y, range_params['t_start'], 
                                            range_params['t_end'], channel=current_channel)
            
            if current_data.size > 0 and voltage_data.size > 0:
                result["sweep_indices"].append(int(index))
                result["time_values"].append(i * period_sec)
                
                # Calculate values for Range 1
                result["peak_current"].append(SweepDataProcessor.calculate_peak_values(current_data))
                result["peak_voltage"].append(SweepDataProcessor.calculate_peak_values(voltage_data))
                result["average_current"].append(SweepDataProcessor.calculate_average_values(current_data))
                result["average_voltage"].append(SweepDataProcessor.calculate_average_values(voltage_data))
                result["avg_voltages_r1"].append(SweepDataProcessor.calculate_average_values(voltage_data))
                
                # Process Range 2 if enabled
                if dual_range and 't_start2' in range_params:
                    voltage_data2 = process_sweep_data(t, y, range_params['t_start2'], 
                                                      range_params['t_end2'], channel=voltage_channel)
                    current_data2 = process_sweep_data(t, y, range_params['t_start2'], 
                                                      range_params['t_end2'], channel=current_channel)
                    
                    if current_data2.size > 0 and voltage_data2.size > 0:
                        result["peak_current2"].append(SweepDataProcessor.calculate_peak_values(current_data2))
                        result["peak_voltage2"].append(SweepDataProcessor.calculate_peak_values(voltage_data2))
                        result["average_current2"].append(SweepDataProcessor.calculate_average_values(current_data2))
                        result["average_voltage2"].append(SweepDataProcessor.calculate_average_values(voltage_data2))
                        result["avg_voltages_r2"].append(SweepDataProcessor.calculate_average_values(voltage_data2))
                    else:
                        result["peak_current2"].append(np.nan)
                        result["peak_voltage2"].append(np.nan)
                        result["average_current2"].append(np.nan)
                        result["average_voltage2"].append(np.nan)
                        result["avg_voltages_r2"].append(np.nan)
        
        return result
    
    @staticmethod
    def extract_axis_data(plot_data, measure, channel=None):
        """Extract data for a specific axis configuration
        
        Args:
            plot_data: Dictionary containing processed sweep data
            measure: Type of measurement ('Time', 'Peak', 'Average')
            channel: Channel type ('Current' or 'Voltage'), None for Time
            
        Returns:
            Tuple of (data_array, label_string)
        """
        if measure == "Time":
            return plot_data["time_values"], "Time (s)"
        
        measure_key = "peak" if measure == "Peak" else "average"
        channel_key = "current" if channel == "Current" else "voltage"
        unit = "(pA)" if channel == "Current" else "(mV)"
        
        data = plot_data[f"{measure_key}_{channel_key}"]
        label = f"{measure} {channel} {unit}"
        
        return data, label
    
    @staticmethod
    def prepare_export_data(plot_data, axis_config, use_dual_range=False, channel_config=None):
        """Prepare data for CSV export with proper channel labels
        
        Args:
            plot_data: Processed sweep data
            axis_config: Dict with x/y measure and channel settings
            use_dual_range: Whether dual range is enabled
            channel_config: ChannelConfiguration object
            
        Returns:
            Tuple of (data_array, header_string)
        """
        if channel_config is None:
            channel_config = ChannelConfiguration()
        
        x_data, x_label = SweepDataProcessor.extract_axis_data(
            plot_data, axis_config['x_measure'], axis_config.get('x_channel')
        )
        
        y_data, y_label = SweepDataProcessor.extract_axis_data(
            plot_data, axis_config['y_measure'], axis_config.get('y_channel')
        )
        
        # Build proper labels using channel configuration
        if axis_config['x_measure'] != "Time":
            x_channel_type = axis_config['x_channel']
            x_label = f"{axis_config['x_measure']} {channel_config.get_channel_label(x_channel_type)}"
        
        if axis_config['y_measure'] != "Time":
            y_channel_type = axis_config['y_channel']
            y_label = f"{axis_config['y_measure']} {channel_config.get_channel_label(y_channel_type)}"
        
        if use_dual_range:
            # ... existing dual range logic but with proper labels ...
            measure_key = "peak" if axis_config['y_measure'] == "Peak" else "average"
            channel_key = "current" if axis_config['y_channel'] == "Current" else "voltage"
            y_data2 = plot_data[f"{measure_key}_{channel_key}2"]
            
            # Create descriptive headers with channel configuration
            y_label_r1 = y_label
            y_label_r2 = y_label
            
            if plot_data["avg_voltages_r1"]:
                mean_v1 = np.nanmean(plot_data["avg_voltages_r1"])
                y_label_r1 = f"{y_label} ({format_voltage_label(mean_v1)}mV)"
            
            if plot_data["avg_voltages_r2"]:
                mean_v2 = np.nanmean(plot_data["avg_voltages_r2"])
                y_label_r2 = f"{y_label} ({format_voltage_label(mean_v2)}mV)"
            
            header = f"{x_label},{y_label_r1},{y_label_r2}"
            output_data = np.column_stack((x_data, y_data, y_data2))
        else:
            header = f"{x_label},{y_label}"
            output_data = np.column_stack((x_data, y_data))
        
        return output_data, header


class BatchAnalyzer:
    """Handles batch analysis operations"""
    
    def __init__(self, parent):
        self.parent = parent
        self.processor = SweepDataProcessor()
    
    def get_output_folder(self, file_paths):
        """Prompt user for output folder and create it if necessary
        
        Returns:
            str: Path to output folder or None if cancelled
        """
        base_dir = os.path.dirname(file_paths[0])
        
        # Calculate unique default folder name
        default_folder_name = "MAT_analysis"
        temp_path = os.path.join(base_dir, default_folder_name)
        counter = 1
        while os.path.exists(temp_path):
            default_folder_name = f"MAT_analysis_{counter}"
            temp_path = os.path.join(base_dir, default_folder_name)
            counter += 1
        
        # Prompt user for folder name
        folder_name, ok = QInputDialog.getText(
            self.parent, "Name Output Folder",
            "Enter a name for the new results folder:",
            text=default_folder_name
        )
        
        if not ok or not folder_name:
            return None
        
        destination_folder = os.path.join(base_dir, folder_name)
        
        # Check if folder exists
        if os.path.exists(destination_folder):
            reply = QMessageBox.question(
                self.parent, 'Folder Exists',
                f"The folder '{folder_name}' already exists.\n\n"
                "Do you want to save files into this existing folder?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return None
        else:
            os.makedirs(destination_folder)
        
        return destination_folder
    
    def get_analysis_parameters(self):
        """Get current analysis parameters from UI
        
        Returns:
            Dictionary containing all analysis parameters
        """
        params = {
            't_start': self.parent.start_spin.value(),
            't_end': self.parent.end_spin.value(),
            'period_ms': self.parent.period_spin.value(),
            'use_dual_range': self.parent.use_dual_range,
            'x_measure': self.parent.x_measure_combo.currentText(),
            'y_measure': self.parent.y_measure_combo.currentText(),
            'x_channel': self.parent.x_channel_combo.currentText(),
            'y_channel': self.parent.y_channel_combo.currentText(),
        }
        
        if params['use_dual_range']:
            params['t_start2'] = self.parent.start_spin2.value()
            params['t_end2'] = self.parent.end_spin2.value()
        
        return params
    
    def create_axis_labels(self, params):
        """Create axis labels based on parameters
        
        Returns:
            Tuple of (x_label, y_label)
        """
        # X-axis label
        if params['x_measure'] == "Time":
            x_label = "Time (s)"
        else:
            unit = "(pA)" if params['x_channel'] == "Current" else "(mV)"
            x_label = f"{params['x_measure']} {params['x_channel']} {unit}"
        
        # Y-axis label
        if params['y_measure'] == "Time":
            y_label = "Time (s)"
        else:
            unit = "(pA)" if params['y_channel'] == "Current" else "(mV)"
            y_label = f"{params['y_measure']} {params['y_channel']} {unit}"
        
        return x_label, y_label
    
    def process_single_file(self, file_path, params):
        """Process a single MAT file"""
        base_name = os.path.basename(file_path).split('.mat')[0]
        if '[' in base_name:
            base_name = base_name.split('[')[0]
        
        sweeps = load_mat_file(file_path)
        
        # Process sweeps using the processor with channel configuration
        range_params = {
            't_start': params['t_start'],
            't_end': params['t_end'],
            'period_ms': params['period_ms']
        }
        
        if params['use_dual_range']:
            range_params['t_start2'] = params['t_start2']
            range_params['t_end2'] = params['t_end2']
        
        processed_data = self.processor.process_sweep_ranges(
            sweeps, range_params, params['use_dual_range'],
            channel_config=self.parent.channel_config  # Pass parent's configuration
        )
        
        # Extract axis data
        axis_config = {
            'x_measure': params['x_measure'],
            'y_measure': params['y_measure'],
            'x_channel': params['x_channel'],
            'y_channel': params['y_channel']
        }
        
        x_data, _ = self.processor.extract_axis_data(
            processed_data, params['x_measure'], params.get('x_channel')
        )
        y_data, _ = self.processor.extract_axis_data(
            processed_data, params['y_measure'], params.get('y_channel')
        )
        
        result = {
            'base_name': base_name,
            'x_data': x_data,
            'y_data': y_data,
            'processed_data': processed_data
        }
        
        if params['use_dual_range']:
            measure_key = "peak" if params['y_measure'] == "Peak" else "average"
            channel_key = "current" if params['y_channel'] == "Current" else "voltage"
            result['y_data2'] = processed_data[f"{measure_key}_{channel_key}2"]
        
        return result
    
    def export_file_data(self, file_data, params, destination_folder, x_label, y_label):
        """Export processed data for a single file to CSV"""
        export_path = os.path.join(destination_folder, f"{file_data['base_name']}.csv")
        
        axis_config = {
            'x_measure': params['x_measure'],
            'y_measure': params['y_measure'],
            'x_channel': params['x_channel'],
            'y_channel': params['y_channel']
        }
        
        output_data, header = self.processor.prepare_export_data(
            file_data['processed_data'], axis_config, params['use_dual_range']
        )
        
        export_to_csv(export_path, output_data, header, '%.5f')
    
    def prepare_iv_data(self, batch_data, params):
        """Prepare data for IV analysis if applicable"""
        iv_data = {}
        iv_file_mapping = {}
        
        # Check if we should prepare for IV
        prepare_for_iv = (params['x_measure'] == "Average" and
                         params['x_channel'] == "Voltage" and
                         params['y_measure'] == "Average" and
                         params['y_channel'] == "Current")
        
        if not prepare_for_iv:
            return iv_data, iv_file_mapping
        
        for idx, (base_name, data) in enumerate(batch_data.items()):
            for x_val, y_val in zip(data['x_values'], data['y_values']):
                rounded_voltage = round(x_val, 1)
                if rounded_voltage not in iv_data:
                    iv_data[rounded_voltage] = []
                iv_data[rounded_voltage].append(y_val)
            
            recording_id = f"Recording {idx + 1}"
            iv_file_mapping[recording_id] = base_name
        
        return iv_data, iv_file_mapping


class ModernMatSweepAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_theme_name = "Light"  # Set default theme
        self.channel_config = ChannelConfiguration()

        self.data_processor = SweepDataProcessor()
        self.batch_analyzer = BatchAnalyzer(self)

        self.sweeps = {}
        self.plot_data = {}
        self.loaded_file_path = None
        self.hold_timer = QTimer()
        self.hold_timer.timeout.connect(self.continue_hold)
        self.hold_direction = None
        self.dragging_line = None
        self.range_lines = []
        self.line_spinbox_map = {}
        self.conc_analysis_dialog = None  # Attribute to hold the tool window

        # Analysis settings
        self.analysis_mode = "Average"
        self.use_dual_range = False
        self.x_measure = "Average"
        self.y_measure = "Average"
        self.x_channel = "Voltage"
        self.y_channel = "Current"

        # Batch analysis data
        self.batch_plot_lines = {}
        self.batch_checkboxes = {}
        self.iv_file_mapping = {}

        self.init_ui()
        self.setStyleSheet(get_theme_stylesheet(THEMES[self.current_theme_name]))

    def init_ui(self):
        """Initialize the main UI components"""
        self.setWindowTitle("MAT File Sweep Analyzer - Modern Edition")
        self.setGeometry(100, 100, 1400, 900)

        # Create UI components
        self._create_menu_bar()
        self._create_toolbar()
        self._create_main_layout()
        self._create_status_bar()

    def _update_channel_combos(self):
        """Update all channel combo boxes with current configuration"""
        # Store current selections
        toolbar_selection = self.channel_combo.currentText() if hasattr(self, 'channel_combo') else None
        x_selection = self.x_channel_combo.currentText() if hasattr(self, 'x_channel_combo') else None
        y_selection = self.y_channel_combo.currentText() if hasattr(self, 'y_channel_combo') else None
        
        # Get available types from channel configuration
        available_types = self.channel_config.get_available_types()
        
        # Update toolbar channel combo
        if hasattr(self, 'channel_combo'):
            # Block signals to prevent update_plot() from being called with empty selection
            self.channel_combo.blockSignals(True)
            self.channel_combo.clear()
            self.channel_combo.addItems(available_types)
            
            # Try to restore selection (it might be swapped now)
            if toolbar_selection in available_types:
                self.channel_combo.setCurrentText(toolbar_selection)
            
            # Re-enable signals
            self.channel_combo.blockSignals(False)
        
        # Update plot settings combos
        if hasattr(self, 'x_channel_combo'):
            self.x_channel_combo.blockSignals(True)
            self.x_channel_combo.clear()
            self.x_channel_combo.addItems(available_types)
            
            if x_selection in available_types:
                self.x_channel_combo.setCurrentText(x_selection)
            
            self.x_channel_combo.blockSignals(False)
        
        if hasattr(self, 'y_channel_combo'):
            self.y_channel_combo.blockSignals(True)
            self.y_channel_combo.clear()
            self.y_channel_combo.addItems(available_types)
            
            if y_selection in available_types:
                self.y_channel_combo.setCurrentText(y_selection)
            
            self.y_channel_combo.blockSignals(False)

    def _create_main_layout(self):
        """Create the main layout with splitter"""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Create splitter for resizable panels
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_widget_layout = QVBoxLayout(main_widget)
        main_widget_layout.addWidget(main_splitter)

        # Left panel for controls
        left_panel = self._create_control_panel()
        main_splitter.addWidget(left_panel)

        # Right panel for plot
        right_panel = self._create_plot_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setStretchFactor(0, 0)  # Control panel fixed width
        main_splitter.setStretchFactor(1, 1)  # Plot panel expandable
        main_splitter.setSizes([400, 1000])

    def _create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _create_menu_bar(self):
        """Create the menu bar with all menus"""
        menubar = self.menuBar()
        
        self._create_file_menu(menubar)
        self._create_tools_menu(menubar)
        self._create_themes_menu(menubar)

    def _create_file_menu(self, menubar):
        """Create the File menu"""
        file_menu = menubar.addMenu('File')

        load_action = QAction('Load MAT File', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_mat_file)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        batch_action = QAction('Batch Analysis', self)
        batch_action.setShortcut('Ctrl+B')
        batch_action.triggered.connect(self.batch_analyze)
        file_menu.addAction(batch_action)

        file_menu.addSeparator()

        export_action = QAction('Export Plot Data', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_plot_data)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _create_tools_menu(self, menubar):
        """Create the Tools menu"""
        tools_menu = menubar.addMenu('Tools')
        conc_analysis_action = QAction('Concentration Response Analysis', self)
        conc_analysis_action.triggered.connect(self.open_conc_analysis)
        tools_menu.addAction(conc_analysis_action)

    def _create_themes_menu(self, menubar):
        """Create the Themes menu"""
        theme_menu = menubar.addMenu('Themes')
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)

        for theme_name in THEMES.keys():
            action = QAction(theme_name, self, checkable=True)
            action.triggered.connect(lambda checked, name=theme_name: self.set_theme(name))
            theme_menu.addAction(action)
            theme_group.addAction(action)
            if theme_name == self.current_theme_name:
                action.setChecked(True)

    def _create_toolbar(self):
        """Create the toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # Load file button
        load_btn = QPushButton("Load MAT File")
        load_btn.clicked.connect(self.load_mat_file)
        toolbar.addWidget(load_btn)

        toolbar.addSeparator()

        # Navigation buttons
        self._add_navigation_controls(toolbar)

        toolbar.addSeparator()

        # Channel selection
        self._add_channel_selection(toolbar)

        toolbar.addSeparator()

        # Batch analysis button
        self.batch_btn = QPushButton("Batch Analysis")
        self.batch_btn.clicked.connect(self.batch_analyze)
        toolbar.addWidget(self.batch_btn)

    def _add_navigation_controls(self, toolbar):
        """Add navigation controls to toolbar"""
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setMaximumWidth(40)
        self.prev_btn.pressed.connect(lambda: self.start_hold(self.prev_sweep))
        self.prev_btn.released.connect(self.stop_hold)
        toolbar.addWidget(self.prev_btn)

        self.sweep_combo = QComboBox()
        self.sweep_combo.setMinimumWidth(120)
        self.sweep_combo.currentTextChanged.connect(self.update_plot)
        toolbar.addWidget(self.sweep_combo)

        self.next_btn = QPushButton("▶")
        self.next_btn.setMaximumWidth(40)
        self.next_btn.pressed.connect(lambda: self.start_hold(self.next_sweep))
        self.next_btn.released.connect(self.stop_hold)
        toolbar.addWidget(self.next_btn)

    def _add_channel_selection(self, toolbar):
        """Add channel selection to toolbar"""
        toolbar.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(self.channel_config.get_available_types())
        self.channel_combo.currentTextChanged.connect(self.update_plot)
        toolbar.addWidget(self.channel_combo)

    def _create_control_panel(self):
        """Create the control panel with all settings groups"""
        # Create scrollable control panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(400)

        control_widget = QWidget()
        scroll_area.setWidget(control_widget)

        layout = QVBoxLayout(control_widget)

        # Add control groups
        layout.addWidget(self._create_file_info_group())
        layout.addWidget(self._create_analysis_settings_group())
        layout.addWidget(self._create_plot_settings_group())
        
        # Export Plot Data button
        self.export_plot_btn = QPushButton("Export Plot Data")
        self.export_plot_btn.clicked.connect(self.export_plot_data)
        self.export_plot_btn.setEnabled(False)
        layout.addWidget(self.export_plot_btn)

        # Add stretch
        layout.addStretch()

        return scroll_area

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
        self.dual_range_cb.stateChanged.connect(self.toggle_dual_range)
        analysis_layout.addWidget(self.dual_range_cb, 2, 0, 1, 2)

        # Range 2 settings
        self._add_range2_settings(analysis_layout)
        
        # Stimulus period
        analysis_layout.addWidget(QLabel("Stimulus Period (ms):"), 5, 0)
        self.period_spin = SelectAllSpinBox()
        self.period_spin.setRange(1, 10000)
        self.period_spin.setValue(DEFAULT_SETTINGS['stimulus_period'])
        self.period_spin.setSingleStep(100)
        analysis_layout.addWidget(self.period_spin, 5, 1)

        # Add Swap Channels button
        self.swap_channels_btn = QPushButton("Swap Channels")
        self.swap_channels_btn.setToolTip("Swap voltage and current channel assignments")
        self.swap_channels_btn.clicked.connect(self.swap_channels)
        analysis_layout.addWidget(self.swap_channels_btn, 6, 0, 1, 2)

        # Center Nearest Cursor button
        center_cursor_btn = QPushButton("Center Nearest Cursor")
        center_cursor_btn.setToolTip("Moves the nearest cursor to the center of the view")
        center_cursor_btn.clicked.connect(self.center_nearest_cursor)
        analysis_layout.addWidget(center_cursor_btn, 7, 0, 1, 2)

        return analysis_group

    # def swap_channels(self):
    #     """Swap the voltage and current channel assignments"""
    #     # Swap the channel indices
    #     self.voltage_channel, self.current_channel = self.current_channel, self.voltage_channel
    #     self.channels_swapped = not self.channels_swapped
        
    #     # Update button appearance to indicate state
    #     if self.channels_swapped:
    #         self.swap_channels_btn.setStyleSheet("QPushButton { background-color: #ffcc99; }")
    #         self.swap_channels_btn.setText("Channels Swapped ⇄")
    #     else:
    #         self.swap_channels_btn.setStyleSheet("")  # Reset to default style
    #         self.swap_channels_btn.setText("Swap Channels")
        
    #     # Update status bar
    #     self.status_bar.showMessage(
    #         f"Channels {'swapped' if self.channels_swapped else 'reset'}: "
    #         f"Voltage=Ch{self.voltage_channel}, Current=Ch{self.current_channel}"
    #     )
        
    #     # Refresh the current plot if data is loaded
    #     if self.sweeps:
    #         self.update_plot()
    #         # If analysis data exists, reprocess it
    #         if hasattr(self, 'plot_data') and self.plot_data:
    #             self.process_all_sweeps()

    def swap_channels(self):
        """Swap the voltage and current channel assignments"""
        # Validate that we have data with at least 2 channels
        if self.sweeps:
            first_key = list(self.sweeps.keys())[0]
            _, y = self.sweeps[first_key]
            if y.shape[1] < 2:
                QMessageBox.warning(self, "Cannot Swap", 
                                "Data has fewer than 2 channels")
                return
        
        # Perform the swap by calling the channel_config's swap method
        self.channel_config.swap_channels()
        
        # Update button appearance
        if self.channel_config.is_swapped():
            self.swap_channels_btn.setStyleSheet("QPushButton { background-color: #ffcc99; }")
            self.swap_channels_btn.setText("Channels Swapped ⇄")
        else:
            self.swap_channels_btn.setStyleSheet("")
            self.swap_channels_btn.setText("Swap Channels")
        
        # Update combo boxes to reflect new channel assignments
        self._update_channel_combos()
        
        # Update status bar
        self.status_bar.showMessage(
            f"Channel configuration: {self.channel_config.get_status_string()}"
        )
        
        # Refresh the current plot if data is loaded
        if self.sweeps:
            self.update_plot()
            if hasattr(self, 'plot_data') and self.plot_data:
                self.process_all_sweeps()

    def _add_range1_settings(self, layout):
        """Add Range 1 settings to layout"""
        layout.addWidget(QLabel("Range 1 Start (ms):"), 0, 0)
        self.start_spin = SelectAllSpinBox()
        self.start_spin.setRange(0, 10000)
        self.start_spin.setValue(DEFAULT_SETTINGS['range1_start'])
        self.start_spin.setSingleStep(0.05)
        self.start_spin.setDecimals(2)
        self.start_spin.valueChanged.connect(self.update_lines_from_entries)
        layout.addWidget(self.start_spin, 0, 1)

        layout.addWidget(QLabel("Range 1 End (ms):"), 1, 0)
        self.end_spin = SelectAllSpinBox()
        self.end_spin.setRange(0, 10000)
        self.end_spin.setValue(DEFAULT_SETTINGS['range1_end'])
        self.end_spin.setSingleStep(0.05)
        self.end_spin.setDecimals(2)
        self.end_spin.valueChanged.connect(self.update_lines_from_entries)
        layout.addWidget(self.end_spin, 1, 1)

    def _add_range2_settings(self, layout):
        """Add Range 2 settings to layout"""
        layout.addWidget(QLabel("Range 2 Start (ms):"), 3, 0)
        self.start_spin2 = SelectAllSpinBox()
        self.start_spin2.setRange(0, 10000)
        self.start_spin2.setValue(DEFAULT_SETTINGS['range2_start'])
        self.start_spin2.setSingleStep(0.05)
        self.start_spin2.setDecimals(2)
        self.start_spin2.setEnabled(False)
        self.start_spin2.valueChanged.connect(self.update_lines_from_entries)
        layout.addWidget(self.start_spin2, 3, 1)

        layout.addWidget(QLabel("Range 2 End (ms):"), 4, 0)
        self.end_spin2 = SelectAllSpinBox()
        self.end_spin2.setRange(0, 10000)
        self.end_spin2.setValue(DEFAULT_SETTINGS['range2_end'])
        self.end_spin2.setSingleStep(0.05)
        self.end_spin2.setDecimals(2)
        self.end_spin2.setEnabled(False)
        self.end_spin2.valueChanged.connect(self.update_lines_from_entries)
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
        self.x_channel_combo.addItems(self.channel_config.get_available_types())
        self.x_channel_combo.setCurrentText("Voltage")
        plot_layout.addWidget(self.x_channel_combo, 0, 2)

        # Y-axis settings
        plot_layout.addWidget(QLabel("Y-Axis:"), 1, 0)
        self.y_measure_combo = QComboBox()
        self.y_measure_combo.addItems(["Peak", "Average", "Time"])
        self.y_measure_combo.setCurrentText("Average")
        plot_layout.addWidget(self.y_measure_combo, 1, 1)

        self.y_channel_combo = QComboBox()
        self.y_channel_combo.addItems(self.channel_config.get_available_types())
        self.y_channel_combo.setCurrentText("Current")
        plot_layout.addWidget(self.y_channel_combo, 1, 2)

        # Update plot button
        self.update_plot_btn = QPushButton("Generate Analysis Plot")
        self.update_plot_btn.clicked.connect(self.update_plot_with_axis_selection)
        self.update_plot_btn.setEnabled(False)
        plot_layout.addWidget(self.update_plot_btn, 2, 0, 1, 3)

        return plot_group

    def _create_plot_panel(self):
        """Create the plot panel with matplotlib canvas"""
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=DEFAULT_SETTINGS['plot_figsize'])
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, plot_widget)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # Connect events
        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # Initialize range lines
        self.range_lines = [
            self.ax.axvline(150, color='green', linestyle='-', picker=5),
            self.ax.axvline(500, color='green', linestyle='-', picker=5)
        ]

        return plot_widget

    # ========== Theme and Tools Methods ==========
    
    def open_conc_analysis(self):
        """Opens the Concentration-Response Analysis tool window."""
        self.conc_analysis_dialog = ConcentrationResponseDialog(self)
        self.conc_analysis_dialog.show()

    def set_theme(self, name):
        """Applies the selected color theme to the application."""
        if name in THEMES:
            self.current_theme_name = name
            new_stylesheet = get_theme_stylesheet(THEMES[name])
            self.setStyleSheet(new_stylesheet)

    # ========== File Loading Methods ==========
    
    def load_mat_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load MAT File", "", "MAT files (*.mat)"
        )

        if not file_path:
            return

        try:
            self.sweeps = load_mat_file(file_path)
            self.loaded_file_path = file_path
            self.sweep_combo.clear()
            sweep_names = []

            for index in sorted(self.sweeps.keys(), key=lambda x: int(x)):
                sweep_names.append(f"Sweep {index}")

            self.sweep_combo.addItems(sweep_names)
            if sweep_names:
                self.sweep_combo.setCurrentIndex(0)
                self.update_plot()
                self.process_all_sweeps()

            # Update file info
            self.file_label.setText(f"File: {os.path.basename(file_path)}")
            self.sweep_count_label.setText(f"Sweeps: {len(sweep_names)}")
            self.status_bar.showMessage(f"Loaded {len(sweep_names)} sweeps from {os.path.basename(file_path)}")

            # Enable UI elements
            self.batch_btn.setEnabled(True)
            self.update_plot_btn.setEnabled(True)
            self.export_plot_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    # ========== Navigation Methods ==========
    
    def start_hold(self, direction_func):
        direction_func()
        self.hold_direction = direction_func
        self.hold_timer.start(150)

    def stop_hold(self):
        self.hold_timer.stop()
        self.hold_direction = None

    def continue_hold(self):
        if self.hold_direction:
            self.hold_direction()

    def next_sweep(self):
        current_index = self.sweep_combo.currentIndex()
        if current_index < self.sweep_combo.count() - 1:
            self.sweep_combo.setCurrentIndex(current_index + 1)

    def prev_sweep(self):
        current_index = self.sweep_combo.currentIndex()
        if current_index > 0:
            self.sweep_combo.setCurrentIndex(current_index - 1)

    # ========== Range Control Methods ==========
    
    def toggle_dual_range(self):
        enabled = self.dual_range_cb.isChecked()
        self.use_dual_range = enabled

        self.start_spin2.setEnabled(enabled)
        self.end_spin2.setEnabled(enabled)

        if enabled:
            # Add second range lines
            if len(self.range_lines) == 2:
                self.range_lines.append(
                    self.ax.axvline(self.start_spin2.value(), color='red', linestyle='-', picker=5)
                )
                self.range_lines.append(
                    self.ax.axvline(self.end_spin2.value(), color='red', linestyle='-', picker=5)
                )
                self.canvas.draw()
        else:
            # Remove second range lines
            if len(self.range_lines) == 4:
                self.range_lines[2].remove()
                self.range_lines[3].remove()
                self.range_lines = self.range_lines[:2]
                self.canvas.draw()

    def update_channel_visibility(self):
        """Update channel combobox visibility based on measurement type"""
        x_measure = self.x_measure_combo.currentText()
        y_measure = self.y_measure_combo.currentText()

        self.x_channel_combo.setEnabled(x_measure in ["Peak", "Average"])
        self.y_channel_combo.setEnabled(y_measure in ["Peak", "Average"])

        # Update measurements
        self.x_measure = x_measure
        self.y_measure = y_measure
        self.x_channel = self.x_channel_combo.currentText()
        self.y_channel = self.y_channel_combo.currentText()

    # ========== Data Processing Methods ==========
    
    def process_all_sweeps(self):
        """Process all sweeps to prepare data for different plotting modes"""
        if not self.sweeps:
            return

        try:
            range_params = {
                't_start': self.start_spin.value(),
                't_end': self.end_spin.value(),
                'period_ms': self.period_spin.value()
            }
            
            if self.use_dual_range:
                range_params['t_start2'] = self.start_spin2.value()
                range_params['t_end2'] = self.end_spin2.value()
            
            # Pass channel configuration to processor
            self.plot_data = self.data_processor.process_sweep_ranges(
                self.sweeps, range_params, self.use_dual_range,
                channel_config=self.channel_config
            )

        except Exception as e:
            self.status_bar.showMessage(f"Error processing sweeps: {e}")

    def update_plot_with_axis_selection(self):
        """Generate and display the analysis plot in a new window"""
        if not self.sweeps:
            QMessageBox.warning(self, "No Data", "Please load a MAT file first.")
            return

        self.update_channel_visibility()
        self.process_all_sweeps()

        # Get axis configuration
        axis_config = {
            'x_measure': self.x_measure_combo.currentText(),
            'y_measure': self.y_measure_combo.currentText(),
            'x_channel': self.x_channel_combo.currentText(),
            'y_channel': self.y_channel_combo.currentText()
        }

        # Extract data using processor
        x_data, x_label = self.data_processor.extract_axis_data(
            self.plot_data, axis_config['x_measure'], axis_config.get('x_channel')
        )
        y_data, y_label = self.data_processor.extract_axis_data(
            self.plot_data, axis_config['y_measure'], axis_config.get('y_channel')
        )

        # Get second range data if needed
        y_data2 = []
        if self.use_dual_range and axis_config['y_measure'] != "Time":
            measure_key = "peak" if axis_config['y_measure'] == "Peak" else "average"
            channel_key = "current" if axis_config['y_channel'] == "Current" else "voltage"
            y_data2 = self.plot_data[f"{measure_key}_{channel_key}2"]

        # Create descriptive labels for dual range
        y_label_r1 = y_label
        y_label_r2 = y_label
        if self.use_dual_range:
            if self.plot_data["avg_voltages_r1"]:
                mean_v1 = np.nanmean(self.plot_data["avg_voltages_r1"])
                y_label_r1 = f"{y_label} ({format_voltage_label(mean_v1)}mV)"
            if self.plot_data["avg_voltages_r2"]:
                mean_v2 = np.nanmean(self.plot_data["avg_voltages_r2"])
                y_label_r2 = f"{y_label} ({format_voltage_label(mean_v2)}mV)"

        plot_title = f"{y_label} vs {x_label}"

        plot_data_dict = {
            'x_data': x_data, 'y_data': y_data, 'y_data2': y_data2,
            'sweep_indices': self.plot_data["sweep_indices"],
            'use_dual_range': self.use_dual_range,
            'y_label_r1': y_label_r1, 'y_label_r2': y_label_r2
        }

        dialog = AnalysisPlotDialog(self, plot_data_dict, x_label, y_label, plot_title)
        dialog.exec()

    # ========== Plot Update Methods ==========
    
    def update_plot(self):
        """Update the plot to show current sweep data"""
        if not self.sweeps:
            return

        selection = self.sweep_combo.currentText()
        if not selection:
            return

        index = selection.split()[-1]
        t, y = self.sweeps[index]

        # Get the physical channel for the selected type
        selected_type = self.channel_combo.currentText()
        channel = self.channel_config.get_channel_for_type(selected_type)

        self.ax.clear()

        # Plot the data
        self.ax.plot(t, y[:, channel], linewidth=2)

        # Set labels and title
        self.ax.set_title(f"Sweep {index} - {selected_type}")
        self.ax.set_xlabel("Time (ms)")

        # Use proper units based on channel type
        unit = "mV" if selected_type == "Voltage" else "pA"
        self.ax.set_ylabel(f"{selected_type} ({unit})")
        
        self.ax.grid(True, alpha=0.3)

        # Force autoscaling on the data
        self.ax.relim()
        self.ax.autoscale_view(tight=True)

        # Get the data limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Add 5% padding to y-axis
        y_range = ylim[1] - ylim[0]
        y_padding = y_range * 0.05

        # Set up range lines
        self._update_range_lines()

        # Restore the y-limits with padding
        self.ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)

        # Set x-limits to data range with small padding
        x_range = xlim[1] - xlim[0]
        x_padding = x_range * 0.02
        self.ax.set_xlim(xlim[0] - x_padding, xlim[1] + x_padding)

        self.figure.tight_layout()
        self.canvas.draw()

        self._update_line_spinbox_map()

    def _update_range_lines(self):
        """Update range lines on the plot"""
        start = self.start_spin.value()
        end = self.end_spin.value()

        self.range_lines = [
            self.ax.axvline(start, color='green', linestyle='-', picker=5, linewidth=2),
            self.ax.axvline(end, color='green', linestyle='-', picker=5, linewidth=2)
        ]

        # Add second range lines if enabled
        if self.use_dual_range:
            start2 = self.start_spin2.value()
            end2 = self.end_spin2.value()

            self.range_lines.append(
                self.ax.axvline(start2, color='red', linestyle='-', picker=5, linewidth=2)
            )
            self.range_lines.append(
                self.ax.axvline(end2, color='red', linestyle='-', picker=5, linewidth=2)
            )

    def _update_line_spinbox_map(self):
        """Update the mapping between lines and spinboxes"""
        self.line_spinbox_map = {
            self.range_lines[0]: self.start_spin,
            self.range_lines[1]: self.end_spin
        }
        if self.use_dual_range and len(self.range_lines) == 4:
            self.line_spinbox_map[self.range_lines[2]] = self.start_spin2
            self.line_spinbox_map[self.range_lines[3]] = self.end_spin2

    def center_nearest_cursor(self):
        """Finds the horizontal center of the plot view and moves the nearest cursor line to it."""
        if not self.range_lines or not self.ax.has_data():
            return

        # Get the center of the current x-axis view
        x_min, x_max = self.ax.get_xlim()
        center_x = (x_min + x_max) / 2

        # Find the cursor line nearest to the center
        nearest_line = None
        min_distance = float('inf')

        for line in self.range_lines:
            line_pos_x = line.get_xdata()[0]
            distance = abs(line_pos_x - center_x)
            if distance < min_distance:
                min_distance = distance
                nearest_line = line

        # Move the nearest line and update its corresponding spinbox
        if nearest_line:
            nearest_line.set_xdata([center_x, center_x])
            if nearest_line in self.line_spinbox_map:
                spinbox_to_update = self.line_spinbox_map[nearest_line]
                spinbox_to_update.setValue(center_x)

            self.canvas.draw()

    def update_lines_from_entries(self):
        """Update range lines based on spinbox values"""
        if not self.range_lines:
            return

        start = self.start_spin.value()
        end = self.end_spin.value()

        self.range_lines[0].set_xdata([start, start])
        self.range_lines[1].set_xdata([end, end])

        if self.use_dual_range and len(self.range_lines) == 4:
            start2 = self.start_spin2.value()
            end2 = self.end_spin2.value()

            self.range_lines[2].set_xdata([start2, start2])
            self.range_lines[3].set_xdata([end2, end2])

        self.canvas.draw()

    # ========== Mouse Interaction Methods ==========
    
    def on_pick(self, event):
        if event.artist in self.range_lines:
            self.dragging_line = event.artist

    def on_drag(self, event):
        if self.dragging_line and event.xdata is not None:
            x = event.xdata
            self.dragging_line.set_xdata([x, x])

            # Update corresponding spinbox
            if self.dragging_line == self.range_lines[0]:
                self.start_spin.setValue(x)
            elif self.dragging_line == self.range_lines[1]:
                self.end_spin.setValue(x)
            elif self.use_dual_range and len(self.range_lines) > 2:
                if self.dragging_line == self.range_lines[2]:
                    self.start_spin2.setValue(x)
                elif self.dragging_line == self.range_lines[3]:
                    self.end_spin2.setValue(x)

            self.canvas.draw()

    def on_release(self, event):
        self.dragging_line = None

    # ========== Export Methods ==========
    
    def export_plot_data(self):
        """Export current plot X and Y data to a CSV file, supporting dual analysis."""
        if not self.loaded_file_path:
            QMessageBox.information(self, "Export Error", "No data to export. Please load a file first.")
            return

        try:
            # Ensure data is current
            self.process_all_sweeps()

            # Get axis configuration
            axis_config = {
                'x_measure': self.x_measure_combo.currentText(),
                'y_measure': self.y_measure_combo.currentText(),
                'x_channel': self.x_channel_combo.currentText(),
                'y_channel': self.y_channel_combo.currentText()
            }

            # Prepare export data with channel configuration
            output_data, header = self.data_processor.prepare_export_data(
                self.plot_data, axis_config, self.use_dual_range,
                channel_config=self.channel_config  # Pass the configuration
            )

            # Get save path
            base_name = os.path.basename(self.loaded_file_path).split('.mat')[0]
            if '[' in base_name:
                base_name = base_name.split('[')[0]

            save_path, _ = QFileDialog.getSaveFileName(
                self, "Export Plot Data",
                os.path.join(os.path.dirname(self.loaded_file_path), f"{base_name}_analyzed.csv"),
                "CSV files (*.csv)"
            )

            if save_path:
                export_to_csv(save_path, output_data, header, '%.6f')
                QMessageBox.information(self, "Export Successful",
                                      f"Data exported to:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")

    # ========== Batch Analysis Methods ==========
    
    def batch_analyze(self):
        """Analyze multiple files and plot results for each"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select MAT Files for Batch Analysis", "", "MAT files (*.mat)"
        )

        if not file_paths:
            return

        # Get output folder
        destination_folder = self.batch_analyzer.get_output_folder(file_paths)
        if not destination_folder:
            return

        # Sort files numerically
        file_paths = sorted(file_paths, key=extract_file_number)

        # Get analysis parameters
        params = self.batch_analyzer.get_analysis_parameters()
        x_label, y_label = self.batch_analyzer.create_axis_labels(params)

        # Create progress dialog
        progress = self._create_progress_dialog(len(file_paths))

        try:
            # Create batch figure
            batch_fig, batch_ax = self._create_batch_figure(x_label, y_label)
            
            # Process files and collect data
            batch_data = self._process_batch_files(
                file_paths, params, destination_folder, 
                batch_ax, progress, x_label, y_label
            )
            
            progress.setValue(len(file_paths))

            # Finalize and show results
            self._finalize_batch_results(
                batch_fig, batch_ax, batch_data, params, 
                x_label, y_label, destination_folder
            )

        except Exception as e:
            QMessageBox.critical(self, "Batch Analysis Error", f"Error: {str(e)}")
        finally:
            progress.close()

    def _create_progress_dialog(self, max_value):
        """Create and configure progress dialog"""
        progress = QProgressBar()
        progress.setMaximum(max_value)
        progress.setWindowTitle("Batch Analysis Progress")
        progress.show()
        return progress

    def _create_batch_figure(self, x_label, y_label):
        """Create figure for batch plotting"""
        batch_fig = plt.figure(figsize=(10, 6))
        batch_ax = batch_fig.add_subplot(111)
        batch_ax.set_xlabel(x_label)
        batch_ax.set_ylabel(y_label)
        batch_ax.set_title(f"{y_label} vs {x_label}")
        batch_ax.grid(True, alpha=0.3)
        return batch_fig, batch_ax

    def _process_batch_files(self, file_paths, params, destination_folder, 
                           batch_ax, progress, x_label, y_label):
        """Process all files in batch"""
        batch_data = {}
        
        for file_idx, file_path in enumerate(file_paths):
            progress.setValue(file_idx)
            QApplication.processEvents()

            try:
                # Process single file
                file_data = self.batch_analyzer.process_single_file(file_path, params)
                
                # Store batch data
                batch_data[file_data['base_name']] = {
                    'x_values': file_data['x_data'],
                    'y_values': file_data['y_data']
                }
                
                if params['use_dual_range'] and 'y_data2' in file_data:
                    batch_data[file_data['base_name']]['y_values2'] = file_data['y_data2']
                
                # Plot data
                self._plot_batch_file_data(
                    batch_ax, file_data, params['use_dual_range']
                )
                
                # Export CSV
                self.batch_analyzer.export_file_data(
                    file_data, params, destination_folder, x_label, y_label
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", 
                    f"Error processing {os.path.basename(file_path)}: {str(e)}"
                )
                continue
        
        return batch_data

    def _plot_batch_file_data(self, batch_ax, file_data, use_dual_range):
        """Plot data for a single file in batch"""
        base_name = file_data['base_name']
        x_data = file_data['x_data']
        y_data = file_data['y_data']
        
        if len(x_data) > 0 and len(y_data) > 0:
            batch_ax.plot(x_data, y_data, 'o-', label=f"{base_name} (Range 1)")
            
            if use_dual_range and 'y_data2' in file_data and len(file_data['y_data2']) > 0:
                batch_ax.plot(x_data, file_data['y_data2'], 's--', 
                            label=f"{base_name} (Range 2)")

    def _finalize_batch_results(self, batch_fig, batch_ax, batch_data, 
                               params, x_label, y_label, destination_folder):
        """Finalize batch analysis and show results"""
        if batch_ax.get_legend_handles_labels()[0]:
            batch_ax.legend()
            batch_fig.tight_layout()
            
            # Prepare IV data if applicable
            iv_data, iv_file_mapping = self.batch_analyzer.prepare_iv_data(
                batch_data, params
            )
            
            # Store mapping
            self.iv_file_mapping = iv_file_mapping
            
            # Show batch results dialog
            batch_dialog = BatchResultDialog(
                self, batch_data, batch_fig, iv_data, iv_file_mapping, 
                x_label, y_label, destination_folder=destination_folder
            )
            batch_dialog.exec()
        else:
            QMessageBox.information(
                self, "Batch Analysis", 
                "No valid data found in the selected files."
            )