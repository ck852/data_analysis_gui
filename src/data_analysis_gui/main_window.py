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
from data_analysis_gui.plot_manager import PlotManager

from data_analysis_gui.config import THEMES, get_theme_stylesheet, DEFAULT_SETTINGS
from data_analysis_gui.dialogs import (ConcentrationResponseDialog, BatchResultDialog, 
                     AnalysisPlotDialog, CurrentDensityIVDialog)
from data_analysis_gui.widgets import SelectAllSpinBox, NoScrollComboBox
from data_analysis_gui.utils import (load_mat_file, export_to_csv, process_sweep_data,
                   apply_analysis_mode, calculate_average_voltage,
                   extract_file_number, format_voltage_label)
from data_analysis_gui.utils.data_processing import (
    process_sweep_data, format_voltage_label, calculate_average
)
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.dataset import ElectrophysiologyDataset, DatasetLoader

class SweepDataProcessor:
    """Handles all data processing and business logic operations"""
    
    @staticmethod
    def process_single_sweep(t, y, t_start, t_end, channel_id=0):
        """Process a single sweep for the given time range and channel
        
        Args:
            t: Time array
            y: Data array
            t_start: Start time in milliseconds
            t_end: End time in milliseconds
            channel_id: The channel index to process
            
        Returns:
            Processed data array for the specified channel and time range
        """
        return process_sweep_data(t, y, t_start, t_end, channel_id)
    
    @staticmethod
    def calculate_peak_values(data):
        """Calculate peak values from data array
        
        Args:
            data: Data array
            
        Returns:
            Maximum absolute value or NaN if empty
        """
        if data.size > 0:
            return np.max(np.abs(data))
        return np.nan
    
    @staticmethod
    def calculate_average_values(data):
        """Calculate average values from data array
        
        Args:
            data: Data array
            
        Returns:
            Mean value or NaN if empty
        """
        if data.size > 0:
            return np.mean(data)
        return np.nan
    
    @staticmethod
    def process_sweep_ranges(sweeps, range_params, dual_range=False, channel_definitions=None):
        """Process all sweeps for the given range parameters
        
        Args:
            sweeps: Dictionary of sweep data
            range_params: Dict with keys 't_start', 't_end', and optionally 't_start2', 't_end2'
            dual_range: Whether to process dual ranges
            channel_definitions: ChannelDefinitions object for channel mapping
            
        Returns:
            Dictionary containing processed data for all sweeps
        """
        # Use default channel definitions if none provided
        if channel_definitions is None:
            channel_definitions = ChannelDefinitions()
        
        # Get the actual channel IDs from the definitions
        voltage_channel_id = channel_definitions.get_voltage_channel()
        current_channel_id = channel_definitions.get_current_channel()
        
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
            
            # Process first range using explicit channel IDs
            voltage_data = process_sweep_data(
                t, y, range_params['t_start'], 
                range_params['t_end'], 
                channel_id=voltage_channel_id
            )
            current_data = process_sweep_data(
                t, y, range_params['t_start'], 
                range_params['t_end'], 
                channel_id=current_channel_id
            )
            
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
                    voltage_data2 = process_sweep_data(
                        t, y, range_params['t_start2'], 
                        range_params['t_end2'], 
                        channel_id=voltage_channel_id
                    )
                    current_data2 = process_sweep_data(
                        t, y, range_params['t_start2'], 
                        range_params['t_end2'], 
                        channel_id=current_channel_id
                    )
                    
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
    def extract_axis_data(plot_data, measure, channel_type=None, channel_definitions=None):
        """Extract data for a specific axis configuration
        
        Args:
            plot_data: Dictionary containing processed sweep data
            measure: Type of measurement ('Time', 'Peak', 'Average')
            channel_type: Data type ('Current' or 'Voltage'), None for Time
            channel_definitions: ChannelDefinitions object for proper labeling
            
        Returns:
            Tuple of (data_array, label_string)
        """
        if measure == "Time":
            return plot_data["time_values"], "Time (s)"
        
        # Use default channel definitions if none provided
        if channel_definitions is None:
            channel_definitions = ChannelDefinitions()
        
        # Determine the data key based on measure and channel type
        measure_key = "peak" if measure == "Peak" else "average"
        channel_key = "current" if channel_type == "Current" else "voltage"
        
        # Get data from plot_data
        data = plot_data[f"{measure_key}_{channel_key}"]
        
        # Create label using channel definitions for proper units
        if channel_type == "Current":
            label = f"{measure} {channel_definitions.get_channel_label(
                channel_definitions.get_current_channel(), include_units=True
            )}"
        else:  # Voltage
            label = f"{measure} {channel_definitions.get_channel_label(
                channel_definitions.get_voltage_channel(), include_units=True
            )}"
        
        return data, label
    
    @staticmethod
    def prepare_export_data(plot_data, axis_config, use_dual_range=False, channel_definitions=None):
        """Prepare data for CSV export with proper channel labels
        
        Args:
            plot_data: Processed sweep data
            axis_config: Dict with x/y measure and channel settings
            use_dual_range: Whether dual range is enabled
            channel_definitions: ChannelDefinitions object for proper labeling
            
        Returns:
            Tuple of (data_array, header_string)
        """
        # Use default channel definitions if none provided
        if channel_definitions is None:
            channel_definitions = ChannelDefinitions()
        
        # Extract X-axis data
        x_data, x_label = SweepDataProcessor.extract_axis_data(
            plot_data, 
            axis_config['x_measure'], 
            axis_config.get('x_channel'),
            channel_definitions
        )
        
        # Extract Y-axis data
        y_data, y_label = SweepDataProcessor.extract_axis_data(
            plot_data, 
            axis_config['y_measure'], 
            axis_config.get('y_channel'),
            channel_definitions
        )
        
        # Handle dual range if enabled
        if use_dual_range:
            # Get data for second range
            measure_key = "peak" if axis_config['y_measure'] == "Peak" else "average"
            channel_key = "current" if axis_config['y_channel'] == "Current" else "voltage"
            y_data2 = plot_data[f"{measure_key}_{channel_key}2"]
            
            # Create descriptive headers with proper voltage labels
            y_label_r1 = y_label
            y_label_r2 = y_label
            
            # Add voltage information to labels if available
            if plot_data.get("avg_voltages_r1"):
                mean_v1 = np.nanmean(plot_data["avg_voltages_r1"])
                y_label_r1 = f"{y_label} ({format_voltage_label(mean_v1)}mV)"
            
            if plot_data.get("avg_voltages_r2"):
                mean_v2 = np.nanmean(plot_data["avg_voltages_r2"])
                y_label_r2 = f"{y_label} ({format_voltage_label(mean_v2)}mV)"
            
            # Create header and data array
            header = f"{x_label},{y_label_r1},{y_label_r2}"
            output_data = np.column_stack((x_data, y_data, y_data2))
        else:
            # Single range export
            header = f"{x_label},{y_label}"
            output_data = np.column_stack((x_data, y_data))
        
        return output_data, header
    
    @staticmethod
    def get_channel_data_for_type(t, y, t_start, t_end, data_type, channel_definitions):
        """Helper method to get data for a specific type (voltage or current)
        
        Args:
            t: Time array
            y: Data array
            t_start: Start time in milliseconds
            t_end: End time in milliseconds
            data_type: 'voltage' or 'current'
            channel_definitions: ChannelDefinitions object
            
        Returns:
            Processed data array for the specified data type
        """
        channel_id = channel_definitions.get_channel_for_type(data_type)
        return process_sweep_data(t, y, t_start, t_end, channel_id)

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
        
        # Load file using DatasetLoader
        dataset = DatasetLoader.load(file_path, self.parent.channel_definitions)
        
        # Convert dataset to legacy format for processor
        sweeps_dict = {}
        for sweep_idx in dataset.sweeps():
            time_ms, data_matrix = dataset.get_sweep(sweep_idx)
            sweeps_dict[sweep_idx] = (time_ms, data_matrix)
        
        # Process sweeps using the processor with channel definitions
        range_params = {
            't_start': params['t_start'],
            't_end': params['t_end'],
            'period_ms': params['period_ms']
        }
        
        if params['use_dual_range']:
            range_params['t_start2'] = params['t_start2']
            range_params['t_end2'] = params['t_end2']
        
        # Pass parent's channel definitions
        processed_data = self.processor.process_sweep_ranges(
            sweeps_dict, 
            range_params, 
            params['use_dual_range'],
            channel_definitions=self.parent.channel_definitions
        )
        
        # Extract axis data with channel definitions
        x_data, _ = self.processor.extract_axis_data(
            processed_data, 
            params['x_measure'], 
            params.get('x_channel'),
            self.parent.channel_definitions
        )
        y_data, _ = self.processor.extract_axis_data(
            processed_data, 
            params['y_measure'], 
            params.get('y_channel'),
            self.parent.channel_definitions
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
        
        # Pass channel definitions for proper labeling
        output_data, header = self.processor.prepare_export_data(
            file_data['processed_data'], 
            axis_config, 
            params['use_dual_range'],
            self.parent.channel_definitions
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
        
        # Initialize channel definitions for managing voltage/current channel assignments
        from data_analysis_gui.core.channel_definitions import ChannelDefinitions
        self.channel_definitions = ChannelDefinitions()
        self.channel_config = self.channel_definitions 

        self.current_dataset = None

        self.data_processor = SweepDataProcessor()
        self.batch_analyzer = BatchAnalyzer(self)

        self.plot_manager = PlotManager(self, figure_size=DEFAULT_SETTINGS['plot_figsize'])
        self.plot_manager.set_drag_callback(self.on_line_dragged)

        self.plot_data = {}
        self.loaded_file_path = None
        self.hold_timer = QTimer()
        self.hold_timer.timeout.connect(self.continue_hold)
        self.hold_direction = None
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

    def on_line_dragged(self, line, x_value):
        """Callback to update spinboxes when range lines are dragged"""
        if line == self.plot_manager.range_lines[0]:
            self.start_spin.setValue(x_value)
        elif line == self.plot_manager.range_lines[1]:
            self.end_spin.setValue(x_value)
        elif self.use_dual_range and len(self.plot_manager.range_lines) > 2:
            if line == self.plot_manager.range_lines[2]:
                self.start_spin2.setValue(x_value)
            elif line == self.plot_manager.range_lines[3]:
                self.end_spin2.setValue(x_value)

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
        
        # Get channel labels (these will always be "Voltage" and "Current")
        # The order doesn't change in the UI, just the underlying channel mapping
        available_types = ["Voltage", "Current"]
        
        # Update toolbar channel combo
        if hasattr(self, 'channel_combo'):
            self.channel_combo.blockSignals(True)
            self.channel_combo.clear()
            self.channel_combo.addItems(available_types)
            
            # Restore selection
            if toolbar_selection in available_types:
                self.channel_combo.setCurrentText(toolbar_selection)
            
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

        # Prefer method if present; otherwise fall back to an attribute or sane default
        types = (self.channel_config.get_available_types()
                if hasattr(self.channel_config, "get_available_types")
                else getattr(self.channel_config, "available_types", ["Voltage", "Current"]))

        self.channel_combo.addItems(types)
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
        self.period_spin.setRange(1, 100000)
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

    def swap_channels(self):
        """Swap the voltage and current channel assignments"""
        # Validate that we have data with at least 2 channels
        if self.current_dataset is not None and not self.current_dataset.is_empty():
            # Check channel count directly from dataset
            if self.current_dataset.channel_count() < 2:
                QMessageBox.warning(self, "Cannot Swap", 
                                    "Data has fewer than 2 channels")
                return
        
        # Perform the swap using the channel definitions
        self.channel_definitions.swap_channels()
        
        # Update button appearance based on swap state
        if self.channel_definitions.is_swapped():
            self.swap_channels_btn.setStyleSheet("QPushButton { background-color: #ffcc99; }")
            self.swap_channels_btn.setText("Channels Swapped ⇄")
        else:
            self.swap_channels_btn.setStyleSheet("")
            self.swap_channels_btn.setText("Swap Channels")
        
        # Update combo boxes to reflect new channel assignments
        self._update_channel_combos()
        
        # Update status bar with current configuration
        config = self.channel_definitions.get_configuration()
        self.status_bar.showMessage(
            f"Channel configuration: Voltage=Ch{config['voltage']}, Current=Ch{config['current']}"
        )
        
        # Refresh the current plot if data is loaded
        if self.current_dataset is not None and not self.current_dataset.is_empty():
            self.update_plot()
            if hasattr(self, 'plot_data') and self.plot_data:
                self.process_all_sweeps()

    def _add_range1_settings(self, layout):
        """Add Range 1 settings to layout"""
        layout.addWidget(QLabel("Range 1 Start (ms):"), 0, 0)
        self.start_spin = SelectAllSpinBox()
        self.start_spin.setRange(0, 100000)
        self.start_spin.setValue(DEFAULT_SETTINGS['range1_start'])
        self.start_spin.setSingleStep(0.05)
        self.start_spin.setDecimals(2)
        self.start_spin.valueChanged.connect(self.update_lines_from_entries)
        layout.addWidget(self.start_spin, 0, 1)

        layout.addWidget(QLabel("Range 1 End (ms):"), 1, 0)
        self.end_spin = SelectAllSpinBox()
        self.end_spin.setRange(0, 100000)
        self.end_spin.setValue(DEFAULT_SETTINGS['range1_end'])
        self.end_spin.setSingleStep(0.05)
        self.end_spin.setDecimals(2)
        self.end_spin.valueChanged.connect(self.update_lines_from_entries)
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
        self.start_spin2.valueChanged.connect(self.update_lines_from_entries)
        layout.addWidget(self.start_spin2, 3, 1)

        layout.addWidget(QLabel("Range 2 End (ms):"), 4, 0)
        self.end_spin2 = SelectAllSpinBox()
        self.end_spin2.setRange(0, 100000)
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
        return self.plot_manager.get_plot_widget()

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
            # Use DatasetLoader to load the file
            self.current_dataset = DatasetLoader.load(file_path, self.channel_definitions)
            self.loaded_file_path = file_path
            self.sweep_combo.clear()
            sweep_names = []

            # Iterate through sweeps using dataset methods
            for index in sorted(self.current_dataset.sweeps(), key=lambda x: int(x)):
                sweep_names.append(f"Sweep {index}")

            self.sweep_combo.addItems(sweep_names)
            if sweep_names:
                self.sweep_combo.setCurrentIndex(0)
                self.update_plot()
                self.process_all_sweeps()

            # Update file info
            self.file_label.setText(f"File: {os.path.basename(file_path)}")
            self.sweep_count_label.setText(f"Sweeps: {self.current_dataset.sweep_count()}")
            self.status_bar.showMessage(
                f"Loaded {self.current_dataset.sweep_count()} sweeps from {os.path.basename(file_path)}"
            )

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

        # Let PlotManager handle the visual updates
        self.plot_manager.toggle_dual_range(
            enabled, 
            self.start_spin2.value(), 
            self.end_spin2.value()
        )

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
        if self.current_dataset is None or self.current_dataset.is_empty():
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
            
            # Convert dataset to legacy format for processor
            # TODO: Eventually refactor processor to use dataset directly
            sweeps_dict = {}
            for sweep_idx in self.current_dataset.sweeps():
                time_ms, data_matrix = self.current_dataset.get_sweep(sweep_idx)
                sweeps_dict[sweep_idx] = (time_ms, data_matrix)
            
            # Pass to processor with channel definitions
            self.plot_data = self.data_processor.process_sweep_ranges(
                sweeps_dict, 
                range_params, 
                self.use_dual_range,
                channel_definitions=self.channel_definitions
            )

        except Exception as e:
            self.status_bar.showMessage(f"Error processing sweeps: {e}")

    def update_plot_with_axis_selection(self):
        """Generate and display the analysis plot in a new window"""
        if self.current_dataset is None or self.current_dataset.is_empty():
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
        if self.current_dataset is None or self.current_dataset.is_empty():
            return

        selection = self.sweep_combo.currentText()
        if not selection:
            return

        index = selection.split()[-1]
        
        # Get the selected channel type from the UI
        selected_type = self.channel_combo.currentText()
        
        # Get the physical channel ID based on the selected type
        if selected_type == "Voltage":
            channel_id = self.channel_definitions.get_voltage_channel()
        else:  # Current
            channel_id = self.channel_definitions.get_current_channel()
        
        # Get channel data from dataset
        time_ms, channel_data = self.current_dataset.get_channel_vector(index, channel_id)
        
        if time_ms is None or channel_data is None:
            return
        
        # Create a compatible data matrix for plot_manager
        # Reshape channel_data to be 2D for compatibility
        data_matrix = np.zeros((len(time_ms), max(2, self.current_dataset.channel_count())))
        data_matrix[:, channel_id] = channel_data

        # Update the plot using the channel ID and proper labeling
        self.plot_manager.update_sweep_plot(
            time_ms, data_matrix, channel_id, index, selected_type, self.channel_definitions
        )
        
        # Update range lines
        self.plot_manager.update_range_lines(
            self.start_spin.value(),
            self.end_spin.value(),
            self.use_dual_range,
            self.start_spin2.value() if self.use_dual_range else None,
            self.end_spin2.value() if self.use_dual_range else None
        )
        
        # Update spinbox mapping
        spinboxes = {
            'start1': self.start_spin,
            'end1': self.end_spin
        }
        if self.use_dual_range:
            spinboxes['start2'] = self.start_spin2
            spinboxes['end2'] = self.end_spin2
        
        self.plot_manager.update_line_spinbox_map(spinboxes)

    def center_nearest_cursor(self):
        """Finds the horizontal center of the plot view and moves the nearest cursor line to it."""
        line_moved, new_position = self.plot_manager.center_nearest_cursor()
        
        if line_moved and new_position is not None:
            # Update the corresponding spinbox
            if line_moved in self.plot_manager.line_spinbox_map:
                spinbox_to_update = self.plot_manager.line_spinbox_map[line_moved]
                spinbox_to_update.setValue(new_position)

    def update_lines_from_entries(self):
        """Update range lines based on spinbox values"""
        self.plot_manager.update_lines_from_values(
            self.start_spin.value(),
            self.end_spin.value(),
            self.use_dual_range,
            self.start_spin2.value() if self.use_dual_range else None,
            self.end_spin2.value() if self.use_dual_range else None
        )

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
        return self.plot_manager.create_batch_figure(x_label, y_label)

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
        
        self.plot_manager.plot_batch_data(
            batch_ax,
            file_data['x_data'],
            file_data['y_data'],
            f"{base_name} (Range 1)",
            'o-',
            file_data.get('y_data2') if use_dual_range else None,
            f"{base_name} (Range 2)" if use_dual_range else None,
            's--'
        )

    def _finalize_batch_results(self, batch_fig, batch_ax, batch_data, 
                            params, x_label, y_label, destination_folder):
        """Finalize batch analysis and show results"""
        self.plot_manager.finalize_batch_plot(batch_fig, batch_ax)
        
        if batch_ax.get_legend_handles_labels()[0]:
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