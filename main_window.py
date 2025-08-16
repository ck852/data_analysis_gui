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


class ModernMatSweepAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_theme_name = "Light"  # Set default theme

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
        self.setWindowTitle("MAT File Sweep Analyzer - Modern Edition")
        self.setGeometry(100, 100, 1400, 900)

        # Create menu bar
        self.create_menu_bar()

        # Create toolbar
        self.create_toolbar()

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Create splitter for resizable panels
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_widget_layout = QVBoxLayout(main_widget)
        main_widget_layout.addWidget(main_splitter)

        # Left panel for controls
        left_panel = self.create_control_panel()
        main_splitter.addWidget(left_panel)

        # Right panel for plot
        right_panel = self.create_plot_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setStretchFactor(0, 0)  # Control panel fixed width
        main_splitter.setStretchFactor(1, 1)  # Plot panel expandable
        main_splitter.setSizes([400, 1000])

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File menu
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

        # Tools Menu
        tools_menu = menubar.addMenu('Tools')
        conc_analysis_action = QAction('Concentration Response Analysis', self)
        conc_analysis_action.triggered.connect(self.open_conc_analysis)
        tools_menu.addAction(conc_analysis_action)

        # Themes menu
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

    def create_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # Load file button
        load_btn = QPushButton("Load MAT File")
        load_btn.clicked.connect(self.load_mat_file)
        toolbar.addWidget(load_btn)

        toolbar.addSeparator()

        # Navigation buttons
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

        toolbar.addSeparator()

        # Channel selection
        toolbar.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Current", "Voltage"])
        self.channel_combo.currentTextChanged.connect(self.update_plot)
        toolbar.addWidget(self.channel_combo)

        toolbar.addSeparator()

        # Batch analysis button
        self.batch_btn = QPushButton("Batch Analysis")
        self.batch_btn.clicked.connect(self.batch_analyze)
        toolbar.addWidget(self.batch_btn)

    def create_control_panel(self):
        # Create scrollable control panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(400)

        control_widget = QWidget()
        scroll_area.setWidget(control_widget)

        layout = QVBoxLayout(control_widget)

        # File info group
        file_group = QGroupBox("File Information")
        file_layout = QVBoxLayout(file_group)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        self.sweep_count_label = QLabel("Sweeps: 0")
        file_layout.addWidget(self.sweep_count_label)

        layout.addWidget(file_group)

        # Analysis settings group
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QGridLayout(analysis_group)

        # Range 1 settings
        analysis_layout.addWidget(QLabel("Range 1 Start (ms):"), 0, 0)
        self.start_spin = SelectAllSpinBox()
        self.start_spin.setRange(0, 10000)
        self.start_spin.setValue(DEFAULT_SETTINGS['range1_start'])
        self.start_spin.setSingleStep(0.05)
        self.start_spin.setDecimals(2)
        self.start_spin.valueChanged.connect(self.update_lines_from_entries)
        analysis_layout.addWidget(self.start_spin, 0, 1)

        analysis_layout.addWidget(QLabel("Range 1 End (ms):"), 1, 0)
        self.end_spin = SelectAllSpinBox()
        self.end_spin.setRange(0, 10000)
        self.end_spin.setValue(DEFAULT_SETTINGS['range1_end'])
        self.end_spin.setSingleStep(0.05)
        self.end_spin.setDecimals(2)
        self.end_spin.valueChanged.connect(self.update_lines_from_entries)
        analysis_layout.addWidget(self.end_spin, 1, 1)

        # Dual range checkbox
        self.dual_range_cb = QCheckBox("Use Dual Analysis")
        self.dual_range_cb.stateChanged.connect(self.toggle_dual_range)
        analysis_layout.addWidget(self.dual_range_cb, 2, 0, 1, 2)

        # Range 2 settings
        analysis_layout.addWidget(QLabel("Range 2 Start (ms):"), 3, 0)
        self.start_spin2 = SelectAllSpinBox()
        self.start_spin2.setRange(0, 10000)
        self.start_spin2.setValue(DEFAULT_SETTINGS['range2_start'])
        self.start_spin2.setSingleStep(0.05)
        self.start_spin2.setDecimals(2)
        self.start_spin2.setEnabled(False)
        self.start_spin2.valueChanged.connect(self.update_lines_from_entries)
        analysis_layout.addWidget(self.start_spin2, 3, 1)

        analysis_layout.addWidget(QLabel("Range 2 End (ms):"), 4, 0)
        self.end_spin2 = SelectAllSpinBox()
        self.end_spin2.setRange(0, 10000)
        self.end_spin2.setValue(DEFAULT_SETTINGS['range2_end'])
        self.end_spin2.setSingleStep(0.05)
        self.end_spin2.setDecimals(2)
        self.end_spin2.setEnabled(False)
        self.end_spin2.valueChanged.connect(self.update_lines_from_entries)
        analysis_layout.addWidget(self.end_spin2, 4, 1)

        # Stimulus period
        analysis_layout.addWidget(QLabel("Stimulus Period (ms):"), 5, 0)
        self.period_spin = SelectAllSpinBox()
        self.period_spin.setRange(1, 10000)
        self.period_spin.setValue(DEFAULT_SETTINGS['stimulus_period'])
        self.period_spin.setSingleStep(100)
        analysis_layout.addWidget(self.period_spin, 5, 1)

        # Center Nearest Cursor button
        center_cursor_btn = QPushButton("Center Nearest Cursor")
        center_cursor_btn.setToolTip("Moves the nearest cursor to the center of the view")
        center_cursor_btn.clicked.connect(self.center_nearest_cursor)
        analysis_layout.addWidget(center_cursor_btn, 6, 0, 1, 2)

        layout.addWidget(analysis_group)

        # Plot settings group
        plot_group = QGroupBox("Plot Settings")
        plot_layout = QGridLayout(plot_group)

        # X-axis settings
        plot_layout.addWidget(QLabel("X-Axis:"), 0, 0)
        self.x_measure_combo = QComboBox()
        self.x_measure_combo.addItems(["Time", "Peak", "Average"])
        self.x_measure_combo.setCurrentText("Average")
        plot_layout.addWidget(self.x_measure_combo, 0, 1)

        self.x_channel_combo = QComboBox()
        self.x_channel_combo.addItems(["Current", "Voltage"])
        self.x_channel_combo.setCurrentText("Voltage")
        plot_layout.addWidget(self.x_channel_combo, 0, 2)

        # Y-axis settings
        plot_layout.addWidget(QLabel("Y-Axis:"), 1, 0)
        self.y_measure_combo = QComboBox()
        self.y_measure_combo.addItems(["Peak", "Average", "Time"])
        self.y_measure_combo.setCurrentText("Average")
        plot_layout.addWidget(self.y_measure_combo, 1, 1)

        self.y_channel_combo = QComboBox()
        self.y_channel_combo.addItems(["Current", "Voltage"])
        self.y_channel_combo.setCurrentText("Current")
        plot_layout.addWidget(self.y_channel_combo, 1, 2)

        # Update plot button
        self.update_plot_btn = QPushButton("Generate Analysis Plot")
        self.update_plot_btn.clicked.connect(self.update_plot_with_axis_selection)
        self.update_plot_btn.setEnabled(False)
        plot_layout.addWidget(self.update_plot_btn, 2, 0, 1, 3)

        layout.addWidget(plot_group)

        # Export Plot Data button
        self.export_plot_btn = QPushButton("Export Plot Data")
        self.export_plot_btn.clicked.connect(self.export_plot_data)
        self.export_plot_btn.setEnabled(False)
        layout.addWidget(self.export_plot_btn)

        # Add stretch
        layout.addStretch()

        return scroll_area

    def create_plot_panel(self):
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

    def update_plot_with_axis_selection(self):
        """Generate and display the analysis plot in a new window"""
        if not self.sweeps:
            QMessageBox.warning(self, "No Data", "Please load a MAT file first.")
            return

        self.update_channel_visibility()
        self.process_all_sweeps()

        x_measure = self.x_measure_combo.currentText()
        y_measure = self.y_measure_combo.currentText()
        x_channel = self.x_channel_combo.currentText()
        y_channel = self.y_channel_combo.currentText()

        # Get X-axis data and label
        if x_measure == "Time":
            x_data = self.plot_data["time_values"]
            x_label = "Time (s)"
        else:
            measure_key = "peak" if x_measure == "Peak" else "average"
            channel_key = "current" if x_channel == "Current" else "voltage"
            x_data = self.plot_data[f"{measure_key}_{channel_key}"]
            x_label = f"{x_measure} {x_channel}"

        # Get Y-axis data and label
        y_data2 = []
        if y_measure == "Time":
            y_data = self.plot_data["time_values"]
            y_label = "Time (s)"
        else:
            measure_key = "peak" if y_measure == "Peak" else "average"
            channel_key = "current" if y_channel == "Current" else "voltage"
            y_data = self.plot_data[f"{measure_key}_{channel_key}"]
            y_label = f"{y_measure} {y_channel}"
            if self.use_dual_range:
                y_data2 = self.plot_data[f"{measure_key}_{channel_key}2"]

        # Create descriptive labels for export
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

    def process_all_sweeps(self):
        """Process all sweeps to prepare data for different plotting modes"""
        if not self.sweeps:
            return

        t_start = self.start_spin.value()
        t_end = self.end_spin.value()

        # Clear existing processed data
        self.plot_data = {
            "sweep_indices": [], "time_values": [],
            "peak_current": [], "peak_voltage": [],
            "average_current": [], "average_voltage": [],
            "peak_current2": [], "peak_voltage2": [],
            "average_current2": [], "average_voltage2": [],
            "avg_voltages_r1": [], "avg_voltages_r2": [],
        }

        # Process each sweep
        try:
            period_ms = self.period_spin.value()
            period_sec = period_ms / 1000.0

            for i, index in enumerate(sorted(self.sweeps.keys(), key=lambda x: int(x))):
                t, y = self.sweeps[index]
                
                # Process first range
                voltage_data = process_sweep_data(t, y, t_start, t_end, channel=0)
                current_data = process_sweep_data(t, y, t_start, t_end, channel=1)

                if current_data.size > 0 and voltage_data.size > 0:
                    self.plot_data["sweep_indices"].append(int(index))
                    self.plot_data["time_values"].append(i * period_sec)

                    # Calculate values for Range 1
                    self.plot_data["peak_current"].append(np.max(np.abs(current_data)))
                    self.plot_data["peak_voltage"].append(np.max(np.abs(voltage_data)))
                    self.plot_data["average_current"].append(np.mean(current_data))
                    self.plot_data["average_voltage"].append(np.mean(voltage_data))
                    self.plot_data["avg_voltages_r1"].append(np.mean(voltage_data))

                    # Process Range 2 if enabled
                    if self.use_dual_range:
                        t_start2 = self.start_spin2.value()
                        t_end2 = self.end_spin2.value()
                        voltage_data2 = process_sweep_data(t, y, t_start2, t_end2, channel=0)
                        current_data2 = process_sweep_data(t, y, t_start2, t_end2, channel=1)

                        if current_data2.size > 0 and voltage_data2.size > 0:
                            self.plot_data["peak_current2"].append(np.max(np.abs(current_data2)))
                            self.plot_data["peak_voltage2"].append(np.max(np.abs(voltage_data2)))
                            self.plot_data["average_current2"].append(np.mean(current_data2))
                            self.plot_data["average_voltage2"].append(np.mean(voltage_data2))
                            self.plot_data["avg_voltages_r2"].append(np.mean(voltage_data2))
                        else:
                            self.plot_data["peak_current2"].append(np.nan)
                            self.plot_data["peak_voltage2"].append(np.nan)
                            self.plot_data["average_current2"].append(np.nan)
                            self.plot_data["average_voltage2"].append(np.nan)
                            self.plot_data["avg_voltages_r2"].append(np.nan)

        except Exception as e:
            self.status_bar.showMessage(f"Error processing sweeps: {e}")

    def update_plot(self):
        """Update the plot to show current sweep data"""
        if not self.sweeps:
            return

        selection = self.sweep_combo.currentText()
        if not selection:
            return

        index = selection.split()[-1]
        t, y = self.sweeps[index]

        # Current is channel 1, Voltage is channel 0
        channel = 1 if self.channel_combo.currentText() == "Current" else 0

        self.ax.clear()

        # Plot the data
        self.ax.plot(t, y[:, channel], linewidth=2)

        # Set labels and title
        self.ax.set_title(f"Sweep {index} - {self.channel_combo.currentText()}")
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Amplitude")
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

        # Restore the y-limits with padding
        self.ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)

        # Set x-limits to data range with small padding
        x_range = xlim[1] - xlim[0]
        x_padding = x_range * 0.02
        self.ax.set_xlim(xlim[0] - x_padding, xlim[1] + x_padding)

        self.figure.tight_layout()
        self.canvas.draw()

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

    def export_plot_data(self):
        """Export current plot X and Y data to a CSV file, supporting dual analysis."""
        if not self.loaded_file_path:
            QMessageBox.information(self, "Export Error", "No data to export. Please load a file first.")
            return

        try:
            # Ensure data is current
            self.process_all_sweeps()

            # Get the current axis selections
            x_measure = self.x_measure_combo.currentText()
            y_measure = self.y_measure_combo.currentText()
            x_channel = self.x_channel_combo.currentText()
            y_channel = self.y_channel_combo.currentText()

            # Get X-axis data and label
            if x_measure == "Time":
                x_data = self.plot_data["time_values"]
                x_label = "Time (s)"
            else:
                measure_key = "peak" if x_measure == "Peak" else "average"
                channel_key = "current" if x_channel == "Current" else "voltage"
                unit = "(pA)" if x_channel == "Current" else "(mV)"
                x_data = self.plot_data[f"{measure_key}_{channel_key}"]
                x_label = f"{x_measure} {x_channel} {unit}"

            # Get Y-axis data and labels
            y_data2 = []
            if y_measure == "Time":
                y_data = self.plot_data["time_values"]
                y_label = "Time (s)"
            else:
                measure_key = "peak" if y_measure == "Peak" else "average"
                channel_key = "current" if y_channel == "Current" else "voltage"
                unit = "(pA)" if y_channel == "Current" else "(mV)"
                y_data = self.plot_data[f"{measure_key}_{channel_key}"]
                y_label = f"{y_measure} {y_channel} {unit}"
                if self.use_dual_range:
                    y_data2 = self.plot_data[f"{measure_key}_{channel_key}2"]

            # Create descriptive headers
            header = f"{x_label},{y_label}"
            if self.use_dual_range:
                y_label_r1 = y_label
                y_label_r2 = y_label
                if self.plot_data["avg_voltages_r1"]:
                    mean_v1 = np.nanmean(self.plot_data["avg_voltages_r1"])
                    y_label_r1 = f"{y_label} ({format_voltage_label(mean_v1)}mV)"

                if self.plot_data["avg_voltages_r2"]:
                    mean_v2 = np.nanmean(self.plot_data["avg_voltages_r2"])
                    y_label_r2 = f"{y_label} ({format_voltage_label(mean_v2)}mV)"
                header = f"{x_label},{y_label_r1},{y_label_r2}"

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
                # Stack data columns
                output_data = np.column_stack((x_data, y_data, y_data2)) if self.use_dual_range else np.column_stack((x_data, y_data))

                # Save to CSV
                export_to_csv(save_path, output_data, header, '%.6f')

                QMessageBox.information(self, "Export Successful",
                                    f"Data exported to:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")

    def batch_analyze(self):
        """Analyze multiple files and plot results for each"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select MAT Files for Batch Analysis", "", "MAT files (*.mat)"
        )

        if not file_paths:
            return

        # Prompt for output folder name
        try:
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
            folder_name, ok = QInputDialog.getText(self, "Name Output Folder",
                                                   "Enter a name for the new results folder:",
                                                   text=default_folder_name)

            if not ok or not folder_name:
                return

            destination_folder = os.path.join(base_dir, folder_name)

            # Check if folder exists
            if os.path.exists(destination_folder):
                reply = QMessageBox.question(self, 'Folder Exists',
                                             f"The folder '{folder_name}' already exists.\n\n"
                                             "Do you want to save files into this existing folder?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    return
            else:
                os.makedirs(destination_folder)

        except Exception as e:
            QMessageBox.critical(self, "Folder Creation Error", f"An error occurred: {e}")
            return

        # Sort files numerically
        file_paths = sorted(file_paths, key=extract_file_number)

        # Create progress dialog
        progress = QProgressBar()
        progress.setMaximum(len(file_paths))
        progress.setWindowTitle("Batch Analysis Progress")
        progress.show()

        try:
            # Get analysis parameters
            t_start = self.start_spin.value()
            t_end = self.end_spin.value()

            if self.use_dual_range:
                t_start2 = self.start_spin2.value()
                t_end2 = self.end_spin2.value()

            period_ms = self.period_spin.value()
            period_sec = period_ms / 1000.0

            # Get axis selections
            x_measure = self.x_measure_combo.currentText()
            y_measure = self.y_measure_combo.currentText()
            x_channel = self.x_channel_combo.currentText()
            y_channel = self.y_channel_combo.currentText()

            # Set labels
            if x_measure == "Time":
                x_label = "Time (s)"
            elif x_measure == "Peak":
                unit = "(pA)" if x_channel == "Current" else "(mV)"
                x_label = f"Peak {x_channel} {unit}"
            elif x_measure == "Average":
                unit = "(pA)" if x_channel == "Current" else "(mV)"
                x_label = f"Average {x_channel} {unit}"
            else:
                x_label = "Sweep Number"

            if y_measure == "Time":
                y_label = "Time (s)"
            elif y_measure == "Peak":
                unit = "(pA)" if y_channel == "Current" else "(mV)"
                y_label = f"Peak {y_channel} {unit}"
            elif y_measure == "Average":
                unit = "(pA)" if y_channel == "Current" else "(mV)"
                y_label = f"Average {y_channel} {unit}"
            else:
                y_label = "Sweep Number"

            # Create batch figure
            batch_fig = plt.figure(figsize=(10, 6))
            batch_ax = batch_fig.add_subplot(111)
            batch_ax.set_xlabel(x_label)
            batch_ax.set_ylabel(y_label)
            batch_ax.set_title(f"{y_label} vs {x_label}")
            batch_ax.grid(True, alpha=0.3)

            # Check if we should prepare for IV
            prepare_for_iv = (x_measure == "Average" and
                            x_channel == "Voltage" and
                            y_measure == "Average" and
                            y_channel == "Current")

            iv_data = {}
            batch_data = {}
            iv_file_mapping = {}

            # Process each file
            for file_idx, file_path in enumerate(file_paths):
                progress.setValue(file_idx)
                QApplication.processEvents()

                try:
                    # Extract base name
                    base_name = os.path.basename(file_path).split('.mat')[0]
                    if '[' in base_name:
                        base_name = base_name.split('[')[0]

                    # Load MAT file
                    sweeps = load_mat_file(file_path)

                    # Process sweeps
                    x_data = []
                    y_data = []
                    y_data2 = []
                    avg_voltages_r1 = []
                    avg_voltages_r2 = []

                    for i, index in enumerate(sorted(sweeps.keys(), key=lambda x: int(x))):
                        t, y = sweeps[index]

                        # Process first range
                        voltage_trace1 = process_sweep_data(t, y, t_start, t_end, channel=0)
                        current_trace1 = process_sweep_data(t, y, t_start, t_end, channel=1)

                        # Process second range if enabled
                        if self.use_dual_range:
                            voltage_trace2 = process_sweep_data(t, y, t_start2, t_end2, channel=0)
                            current_trace2 = process_sweep_data(t, y, t_start2, t_end2, channel=1)

                        # Check if data is valid
                        if voltage_trace1.size > 0 and current_trace1.size > 0:
                            # Calculate values
                            peak_current = np.max(np.abs(current_trace1))
                            peak_voltage = np.max(np.abs(voltage_trace1))
                            average_current = np.mean(current_trace1)
                            average_voltage = np.mean(voltage_trace1)
                            avg_voltages_r1.append(average_voltage)

                            # Get X data
                            if x_measure == "Time":
                                x_val = i * period_sec
                            elif x_measure == "Peak":
                                x_val = peak_current if x_channel == "Current" else peak_voltage
                            elif x_measure == "Average":
                                x_val = average_current if x_channel == "Current" else average_voltage
                            else:
                                x_val = int(index)

                            # Get Y data
                            if y_measure == "Time":
                                y_val = i * period_sec
                            elif y_measure == "Peak":
                                y_val = peak_current if y_channel == "Current" else peak_voltage
                            elif y_measure == "Average":
                                y_val = average_current if y_channel == "Current" else average_voltage
                            else:
                                y_val = int(index)

                            x_data.append(x_val)
                            y_data.append(y_val)

                            # Second range data
                            if self.use_dual_range and voltage_trace2.size > 0 and current_trace2.size > 0:
                                peak_current2 = np.max(np.abs(current_trace2))
                                peak_voltage2 = np.max(np.abs(voltage_trace2))
                                average_current2 = np.mean(current_trace2)
                                average_voltage2 = np.mean(voltage_trace2)
                                avg_voltages_r2.append(average_voltage2)

                                if y_measure == "Peak":
                                    y_val2 = peak_current2 if y_channel == "Current" else peak_voltage2
                                elif y_measure == "Average":
                                    y_val2 = average_current2 if y_channel == "Current" else average_voltage2
                                else:
                                    y_val2 = y_val

                                y_data2.append(y_val2)

                            # Collect IV data
                            if prepare_for_iv:
                                rounded_voltage = round(average_voltage, 1)
                                if rounded_voltage not in iv_data:
                                    iv_data[rounded_voltage] = []
                                iv_data[rounded_voltage].append(average_current)

                    # Store batch data
                    batch_data[base_name] = {'x_values': x_data, 'y_values': y_data}
                    if self.use_dual_range:
                        batch_data[base_name]['y_values2'] = y_data2

                    # Plot data
                    if len(x_data) > 0 and len(y_data) > 0:
                        batch_ax.plot(x_data, y_data, 'o-', label=f"{base_name} (Range 1)")

                        if self.use_dual_range and len(y_data2) > 0:
                            batch_ax.plot(x_data, y_data2, 's--', label=f"{base_name} (Range 2)")

                    # Export CSV
                    export_path = os.path.join(destination_folder, f"{base_name}.csv")

                    if self.use_dual_range:
                        header_r1_suffix = "(Range 1)"
                        header_r2_suffix = "(Range 2)"

                        if avg_voltages_r1:
                            mean_v1 = np.mean(avg_voltages_r1)
                            header_r1_suffix = f"({format_voltage_label(mean_v1)}mV)"

                        if avg_voltages_r2:
                            mean_v2 = np.mean(avg_voltages_r2)
                            header_r2_suffix = f"({format_voltage_label(mean_v2)}mV)"

                        output_data = np.column_stack((x_data, y_data, y_data2))
                        header = f"{x_label},{y_label} {header_r1_suffix},{y_label} {header_r2_suffix}"
                    else:
                        output_data = np.column_stack((x_data, y_data))
                        header = f"{x_label},{y_label}"

                    export_to_csv(export_path, output_data, header, '%.5f')

                    # Store file mapping for IV
                    recording_id = f"Recording {file_idx+1}"
                    iv_file_mapping[recording_id] = base_name

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error processing {os.path.basename(file_path)}: {str(e)}")
                    continue

            progress.setValue(len(file_paths))

            # Finalize plot
            if batch_ax.get_legend_handles_labels()[0]:
                batch_ax.legend()
                batch_fig.tight_layout()

                # Store mapping
                self.iv_file_mapping = iv_file_mapping

                # Show batch results dialog
                batch_dialog = BatchResultDialog(self, batch_data, batch_fig, iv_data, iv_file_mapping, x_label, y_label, )
                batch_dialog.exec()
            else:
                QMessageBox.information(self, "Batch Analysis", "No valid data found in the selected files.")

        except Exception as e:
            QMessageBox.critical(self, "Batch Analysis Error", f"Error: {str(e)}")
        finally:
            progress.close()