"""
Main Window - Fixed to use shared services from controller
"""

import os
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QMessageBox, QSplitter, QAction, QToolBar, 
                             QStatusBar, QLabel, QPushButton, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QKeySequence

# Core imports
from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.models import FileInfo
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.config.logging import get_logger

# Widget imports
from data_analysis_gui.widgets.control_panel import ControlPanel
from data_analysis_gui.plot_manager import PlotManager

# Dialog imports
from data_analysis_gui.dialogs.analysis_plot_dialog import AnalysisPlotDialog
from data_analysis_gui.dialogs.batch_dialog import BatchAnalysisDialog

# Service imports
from data_analysis_gui.gui_services import FileDialogService

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window using shared services from controller.
    No duplicate service instantiation.
    """
    
    # Application events
    file_loaded = pyqtSignal(str)
    analysis_completed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Initialize controller (which creates all services)
        self.controller = ApplicationController()
        
        # Get shared services from controller
        services = self.controller.get_services()
        self.channel_definitions = services['channel_definitions']
        self.data_manager = services['data_manager']
        self.analysis_manager = services['analysis_manager']
        self.batch_processor = services['batch_processor']
        
        # GUI services
        self.file_dialog_service = FileDialogService()
        
        # Controller callbacks
        self.controller.on_file_loaded = self._on_file_loaded
        self.controller.on_error = lambda msg: QMessageBox.critical(self, "Error", msg)
        self.controller.on_status_update = lambda msg: self.status_bar.showMessage(msg, 5000)
        
        # State
        self.current_file_path: Optional[str] = None
        self.analysis_dialog: Optional[AnalysisPlotDialog] = None
        
        # Navigation timer
        self.hold_timer = QTimer()
        self.hold_timer.timeout.connect(self._continue_navigation)
        self.navigation_direction = None
        
        # Build UI
        self._init_ui()
        
        # Configure window
        self.setWindowTitle("Electrophysiology Data Analysis")
        self.setGeometry(100, 100, 1400, 800)

    def _init_ui(self):
        """Initialize the complete UI"""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        QHBoxLayout(central).addWidget(splitter)
        
        # Control panel (left)
        self.control_panel = ControlPanel()
        splitter.addWidget(self.control_panel)
        
        # Plot manager (right)
        self.plot_manager = PlotManager()
        splitter.addWidget(self.plot_manager.get_plot_widget())
        
        splitter.setSizes([400, 1000])
        
        # Menus and toolbar
        self._create_menus()
        self._create_toolbar()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Connect signals
        self._connect_signals()

    def _create_menus(self):
        """Create application menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        self.open_action = QAction("&Open...", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.triggered.connect(self._open_file)
        file_menu.addAction(self.open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        
        self.swap_action = QAction("&Swap Channels", self)
        self.swap_action.setShortcut("Ctrl+Shift+S")
        self.swap_action.triggered.connect(self._swap_channels)
        self.swap_action.setEnabled(False)
        analysis_menu.addAction(self.swap_action)

        self.batch_action = QAction("&Batch Analyze...", self)
        self.batch_action.setShortcut("Ctrl+B")
        self.batch_action.triggered.connect(self._batch_analyze)
        self.batch_action.setEnabled(True)
        analysis_menu.addAction(self.batch_action)

    def _create_toolbar(self):
        """Create toolbar with navigation"""
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # File operations
        toolbar.addAction("Open", self._open_file)
        toolbar.addSeparator()
        
        # Navigation
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setMaximumWidth(40)
        self.prev_btn.setEnabled(False)
        self.prev_btn.pressed.connect(lambda: self._start_navigation(self._prev_sweep))
        self.prev_btn.released.connect(self._stop_navigation)
        toolbar.addWidget(self.prev_btn)
        
        self.sweep_combo = QComboBox()
        self.sweep_combo.setMinimumWidth(120)
        self.sweep_combo.setEnabled(False)
        self.sweep_combo.currentTextChanged.connect(self._on_sweep_changed)
        toolbar.addWidget(self.sweep_combo)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.setMaximumWidth(40)
        self.next_btn.setEnabled(False)
        self.next_btn.pressed.connect(lambda: self._start_navigation(self._next_sweep))
        self.next_btn.released.connect(self._stop_navigation)
        toolbar.addWidget(self.next_btn)
        
        toolbar.addSeparator()
        
        # Channel selection
        toolbar.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Voltage", "Current"])
        self.channel_combo.setEnabled(False)
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        toolbar.addWidget(self.channel_combo)
        
        toolbar.addSeparator()

            # File Information Labels
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("QLabel { margin-left: 5px; }") # Optional styling
        toolbar.addWidget(self.file_label)
        
        self.sweep_count_label = QLabel("") # Start empty
        self.sweep_count_label.setStyleSheet("QLabel { margin-left: 5px; }") # Optional styling
        toolbar.addWidget(self.sweep_count_label)

        toolbar.addSeparator()

        # Swap Channels Button
        self.swap_channels_btn = QPushButton("Swap Channels")
        self.swap_channels_btn.setToolTip("Swap voltage and current channel assignments (Ctrl+Shift+S)")
        self.swap_channels_btn.setEnabled(False)
        self.swap_channels_btn.clicked.connect(self._swap_channels)
        toolbar.addWidget(self.swap_channels_btn)
        
        # Center Cursor Button
        center_cursor_btn = QPushButton("Center Nearest Cursor")
        center_cursor_btn.setToolTip("Moves the nearest cursor to the center of the view")
        # Direct connection to the function that was previously called by the signal
        center_cursor_btn.clicked.connect(
            lambda: self._sync_cursor_to_control(*self.plot_manager.center_nearest_cursor())
        )
        center_cursor_btn.setEnabled(False) # Will be enabled on file load
        self.center_cursor_btn = center_cursor_btn # Store reference if needed
        toolbar.addWidget(self.center_cursor_btn)

        toolbar.addSeparator()

    def _connect_signals(self):
        """Connect all signals"""
        # Control panel
        self.control_panel.analysis_requested.connect(self._generate_analysis)
        self.control_panel.export_requested.connect(self._export_data)
        self.control_panel.dual_range_toggled.connect(self._toggle_dual_range)
        self.control_panel.range_values_changed.connect(self._sync_cursors_to_plot)
        
        # Plot manager
        self.plot_manager.line_state_changed.connect(self._on_cursor_moved)

    def _open_file(self):
        """Open file using controller"""
        file_types = (
            "Data files (*.mat *.abf);;"
            "MAT files (*.mat);;"
            "ABF files (*.abf);;"
            "All files (*.*)"
        )
        
        default_dir = str(Path(self.current_file_path).parent) if self.current_file_path else None
        
        file_path = self.file_dialog_service.get_import_path(
            self, "Open Data File", default_dir, file_types
        )
        
        if file_path:
            # Use controller to load file
            result = self.controller.load_file(file_path)
            
            if result.success:
                self.current_file_path = file_path
                self.file_loaded.emit(file_path)
            # Error handling is done by controller callbacks

    def _on_file_loaded(self, file_info: FileInfo):
        """Handle successful file load"""
        # Update control panel
        self.file_label.setText(f"File: {file_info.name}")
        self.sweep_count_label.setText(f"Sweeps: {file_info.sweep_count}")
        self.control_panel.set_controls_enabled(True)
        
        if file_info.max_sweep_time:
            self.control_panel.set_analysis_range(file_info.max_sweep_time)
        
        # Enable UI elements
        self.swap_action.setEnabled(True)
        self.swap_channels_btn.setEnabled(True)
        self.center_cursor_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.sweep_combo.setEnabled(True)
        self.channel_combo.setEnabled(True)
        
        # Populate sweeps
        self.sweep_combo.clear()
        self.sweep_combo.addItems(file_info.sweep_names)
        
        # Handle any pending swap state
        if getattr(self.control_panel, '_is_swapped', False):
            self._swap_channels()
        
        # Show first sweep
        if file_info.sweep_names:
            self.sweep_combo.setCurrentIndex(0)

    def _on_sweep_changed(self):
        """Update plot when sweep selection changes"""
        self._update_plot()

    def _on_channel_changed(self):
        """Update plot when channel selection changes"""
        self._update_plot()

    def _update_plot(self):
        """Update the sweep plot using controller"""
        if not self.controller.has_data():
            return
            
        sweep = self.sweep_combo.currentText()
        if not sweep:
            return
        
        channel_type = self.channel_combo.currentText()
        
        # Get plot data from controller
        result = self.controller.get_sweep_plot_data(sweep, channel_type)
        
        if result.success:
            plot_data = result.data
            self.plot_manager.update_sweep_plot(
                t=plot_data.time_ms,
                y=plot_data.data_matrix,
                channel=plot_data.channel_id,
                sweep_index=int(sweep) if sweep.isdigit() else 0,
                channel_type=channel_type
            )
            self._sync_cursors_to_plot()
        else:
            logger.debug(f"Could not load sweep {sweep}: {result.error_message}")

    def _generate_analysis(self):
        """Generate analysis plot using controller"""
        if not self.controller.has_data():
            QMessageBox.warning(self, "No Data", "Please load a data file first.")
            return
        
        # Get parameters from control panel
        params = self.control_panel.get_parameters()
        
        # Perform analysis using controller
        result = self.controller.perform_analysis(params)
        
        if not result.success:
            QMessageBox.critical(self, "Analysis Failed", 
                               f"Analysis failed:\n{result.error_message}")
            return
        
        analysis_result = result.data
        
        if not analysis_result or not analysis_result.x_data.size:
            QMessageBox.warning(self, "No Results", 
                              "No data available for selected parameters.")
            return
        
        # Prepare plot data
        plot_data = {
            'x_data': analysis_result.x_data,
            'y_data': analysis_result.y_data,
            'sweep_indices': analysis_result.sweep_indices,
            'use_dual_range': analysis_result.use_dual_range
        }
        
        if analysis_result.use_dual_range and hasattr(analysis_result, 'y_data2'):
            plot_data['y_data2'] = analysis_result.y_data2
            plot_data['y_label_r1'] = getattr(analysis_result, 'y_label_r1', analysis_result.y_label)
            plot_data['y_label_r2'] = getattr(analysis_result, 'y_label_r2', analysis_result.y_label)
        
        # Show dialog
        if self.analysis_dialog:
            self.analysis_dialog.close()
        
        title = f"Analysis - {Path(self.current_file_path).stem}" if self.current_file_path else "Analysis"
        
        # Pass controller for export functionality
        self.analysis_dialog = AnalysisPlotDialog(
            self, plot_data, analysis_result.x_label, analysis_result.y_label,
            title, self.controller, params
        )
        self.analysis_dialog.show()
        self.analysis_completed.emit()

    def _export_data(self):
        """Export data using controller"""
        if not self.controller.has_data():
            QMessageBox.warning(self, "No Data", "Please load a data file first.")
            return
        
        # Get parameters
        params = self.control_panel.get_parameters()
        
        # Get suggested filename
        suggested = self.controller.get_suggested_export_filename(params)
        
        file_path = self.file_dialog_service.get_export_path(
            self, suggested, file_types="CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        # Export using controller
        result = self.controller.export_analysis_data(params, file_path)
        
        if result.success:
            QMessageBox.information(self, "Success", 
                                  f"Exported {result.records_exported} records to {Path(file_path).name}")
        else:
            QMessageBox.critical(self, "Export Failed", result.error_message)

    def _swap_channels(self):
        """Swap channels using controller"""
        result = self.controller.swap_channels()
        
        if result['success']:
            # Update UI
            self.update_swap_button_state(result['is_swapped'])
            
            # Update current plot
            self._update_plot()
            
            # Switch displayed channel
            current = self.channel_combo.currentText()
            self.channel_combo.setCurrentText("Current" if current == "Voltage" else "Voltage")
            
            self.status_bar.showMessage(
                f"Channels {'swapped' if result['is_swapped'] else 'restored'}", 3000
            )
        else:
            QMessageBox.warning(self, "Cannot Swap", result['reason'])

    def _batch_analyze(self):
        """Open batch analysis dialog"""
        # Get current parameters
        params = self.control_panel.get_parameters()
        
        # Open batch dialog with shared batch processor
        dialog = BatchAnalysisDialog(self, self.batch_processor, params)
        dialog.show()

    def _toggle_dual_range(self, enabled):
        """Toggle dual range cursors"""
        if enabled:
            vals = self.control_panel.get_range_values()
            self.plot_manager.toggle_dual_range(
                True, vals.get('range2_start', 600), vals.get('range2_end', 900)
            )
        else:
            self.plot_manager.toggle_dual_range(False, 0, 0)

    def _sync_cursors_to_plot(self):
        """Sync cursor positions from control panel to plot"""
        vals = self.control_panel.get_range_values()
        self.plot_manager.update_range_lines(
            vals['range1_start'], vals['range1_end'],
            vals['use_dual_range'],
            vals.get('range2_start'), vals.get('range2_end')
        )

    def _sync_cursor_to_control(self, line_id, position):
        """Sync cursor position from plot to control panel"""
        if line_id and position is not None:
            mapping = {
                'range1_start': 'start1',
                'range1_end': 'end1',
                'range2_start': 'start2',
                'range2_end': 'end2'
            }
            if line_id in mapping:
                self.control_panel.update_range_value(mapping[line_id], position)

    def _on_cursor_moved(self, action, line_id, position):
        """Handle cursor movement from plot"""
        if action == 'dragged':
            self._sync_cursor_to_control(line_id, position)

    # Navigation methods
    def _start_navigation(self, direction):
        """Start continuous navigation"""
        direction()
        self.navigation_direction = direction
        self.hold_timer.start(150)

    def _stop_navigation(self):
        """Stop continuous navigation"""
        self.hold_timer.stop()
        self.navigation_direction = None

    def _continue_navigation(self):
        """Continue navigation while held"""
        if self.navigation_direction:
            self.navigation_direction()

    def _next_sweep(self):
        """Go to next sweep"""
        idx = self.sweep_combo.currentIndex()
        if idx < self.sweep_combo.count() - 1:
            self.sweep_combo.setCurrentIndex(idx + 1)

    def _prev_sweep(self):
        """Go to previous sweep"""
        idx = self.sweep_combo.currentIndex()
        if idx > 0:
            self.sweep_combo.setCurrentIndex(idx - 1)

    def closeEvent(self, event):
        """Clean shutdown"""
        if self.analysis_dialog:
            self.analysis_dialog.close()
        event.accept()