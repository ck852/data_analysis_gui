"""
Refactored Main Window - Simplified by delegating control panel to separate widget.
All business operations are delegated to the ApplicationController.
"""

import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QCheckBox, QFileDialog,
                             QMessageBox, QGroupBox, QLabel, QSplitter,
                             QScrollArea, QGridLayout, QProgressBar,
                             QStatusBar, QToolBar, QMenuBar, QMenu,
                             QAction, QActionGroup, QInputDialog, QApplication)
from PyQt5.QtCore import Qt, QTimer

from data_analysis_gui.core.session_settings import save_last_session, load_last_session

import base64
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# GUI-only imports
from data_analysis_gui.plot_manager import PlotManager
from data_analysis_gui.config import THEMES, get_theme_stylesheet, DEFAULT_SETTINGS
from data_analysis_gui.dialogs import (BatchResultDialog, 
                     AnalysisPlotDialog, CurrentDensityIVDialog)
from data_analysis_gui.widgets import SelectAllSpinBox, NoScrollComboBox
from data_analysis_gui.widgets.control_panel import ControlPanel  # Import the new widget

# Import the controller
from data_analysis_gui.core.app_controller import ApplicationController, FileInfo, PlotData


class ModernMatSweepAnalyzer(QMainWindow):
    """
    Simplified main window implementation using the extracted ControlPanel widget.
    This class focuses on high-level application coordination.
    """
    
    def __init__(self, controller: ApplicationController = None):
        super().__init__()
        
        # Create the controller and PASS IN the callback method
        self.controller = controller or ApplicationController(
            get_save_path_callback=self._show_save_file_dialog
        )
        
        # Set up controller callbacks
        self.controller.on_file_loaded = self._handle_file_loaded
        self.controller.on_error = self._show_error
        self.controller.on_status_update = self._update_status
        
        # GUI state
        self.current_theme_name = "Light"
        
        # Plot manager for visualization
        self.plot_manager = PlotManager(self, figure_size=DEFAULT_SETTINGS['plot_figsize'])
        self.plot_manager.set_drag_callback(self._on_line_dragged)
        
        # Dialog references
        self.conc_analysis_dialog = None
        
        # Timer for navigation
        self.hold_timer = QTimer()
        self.hold_timer.timeout.connect(self._continue_hold)
        self.hold_direction = None
        
        # Initialize UI
        self._init_ui()
        self.setStyleSheet(get_theme_stylesheet(THEMES[self.current_theme_name]))

        # Load last session settings
        last_session = load_last_session()
        if last_session:
            self.control_panel.apply_parameters(last_session)
            # TODO(cursor): Refresh plot/cursor positions if needed after parameter restore

    def _show_save_file_dialog(self, suggested_path: str) -> str:
        """This is the callback function. It handles showing the GUI dialog."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Data", suggested_path, "CSV files (*.csv)"
        )
        return file_path  # Returns the path or an empty string if cancelled

    def _export_data(self):
        """Delegates the entire export process to the controller."""
        if not self.controller.has_data():
            QMessageBox.information(self, "Export Error", "No data to export.")
            return
            
        params = self._collect_parameters()
        
        # This one line now handles everything!
        self.controller.trigger_export_dialog(params)
    
    # ============ UI Initialization ============
    
    def _init_ui(self):
        """Initialize the main UI components"""
        self.setWindowTitle("PatchBatch v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        self._create_menu_bar()
        self._create_toolbar()
        self._create_main_layout()
        self._create_status_bar()
    
    def _create_menu_bar(self):
        """Create the menu bar with all menus"""
        menubar = self.menuBar()
        
        self._create_file_menu(menubar)
        self._create_themes_menu(menubar)
    
    def _create_file_menu(self, menubar):
        """Create the File menu"""
        file_menu = menubar.addMenu('File')
        
        load_action = QAction('Load File', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self._load_file)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        batch_action = QAction('Batch Analysis', self)
        batch_action.setShortcut('Ctrl+B')
        batch_action.triggered.connect(self._batch_analyze)
        file_menu.addAction(batch_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('Export Plot Data', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self._export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
    
    def _create_themes_menu(self, menubar):
        """Create the Themes menu"""
        theme_menu = menubar.addMenu('Themes')
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        
        for theme_name in THEMES.keys():
            action = QAction(theme_name, self, checkable=True)
            action.triggered.connect(lambda checked, name=theme_name: self._set_theme(name))
            theme_menu.addAction(action)
            theme_group.addAction(action)
            if theme_name == self.current_theme_name:
                action.setChecked(True)
    
    def _create_toolbar(self):
        """Create the toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Load file button
        load_btn = QPushButton("Load File")
        load_btn.clicked.connect(self._load_file)
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
        self.batch_btn.clicked.connect(self._batch_analyze)
        toolbar.addWidget(self.batch_btn)
    
    def _add_navigation_controls(self, toolbar):
        """Add navigation controls to toolbar"""
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setMaximumWidth(40)
        self.prev_btn.pressed.connect(lambda: self._start_hold(self._prev_sweep))
        self.prev_btn.released.connect(self._stop_hold)
        toolbar.addWidget(self.prev_btn)
        
        self.sweep_combo = QComboBox()
        self.sweep_combo.setMinimumWidth(120)
        self.sweep_combo.currentTextChanged.connect(self._update_plot)
        toolbar.addWidget(self.sweep_combo)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.setMaximumWidth(40)
        self.next_btn.pressed.connect(lambda: self._start_hold(self._next_sweep))
        self.next_btn.released.connect(self._stop_hold)
        toolbar.addWidget(self.next_btn)
    
    def _add_channel_selection(self, toolbar):
        """Add channel selection to toolbar"""
        toolbar.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Voltage", "Current"])
        self.channel_combo.currentTextChanged.connect(self._update_plot)
        toolbar.addWidget(self.channel_combo)
    
    def _create_main_layout(self):
        """Create the main layout with splitter - SIMPLIFIED VERSION"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_widget_layout = QVBoxLayout(main_widget)
        main_widget_layout.addWidget(main_splitter)
        
        # Left panel - Now using the extracted ControlPanel widget
        self.control_panel = ControlPanel()
        self._connect_control_panel_signals()
        main_splitter.addWidget(self.control_panel)
        
        # Right panel for plot
        right_panel = self._create_plot_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([400, 1000])
    
    def _connect_control_panel_signals(self):
        """Connect signals from the control panel to handlers"""
        self.control_panel.analysis_requested.connect(self._generate_analysis_plot)
        self.control_panel.export_requested.connect(self._export_data)
        self.control_panel.swap_channels_requested.connect(self._swap_channels)
        self.control_panel.center_cursor_requested.connect(self._center_nearest_cursor)
        self.control_panel.dual_range_toggled.connect(self._toggle_dual_range)
        self.control_panel.range_values_changed.connect(self._update_lines_from_entries)
    
    def _create_plot_panel(self):
        """Create the plot panel with matplotlib canvas"""
        return self.plot_manager.get_plot_widget()
    
    def _create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    # ============ GUI Event Handlers (delegate to controller) ============
    
    def _load_file(self):
        """Handle file loading"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Data File", "", 
            "Data files (*.mat *.abf);;MAT files (*.mat);;ABF files (*.abf);;All files (*.*)"
        )
        
        if file_path:
            self.controller.load_file(file_path)
    
    def _handle_file_loaded(self, file_info: FileInfo):
        """Handle successful file load (callback from controller)"""
        # Update control panel file info
        self.control_panel.update_file_info(file_info.name, file_info.sweep_count)

        # Set the analysis range based on the longest sweep
        if file_info.max_sweep_time is not None:
            self.control_panel.set_analysis_range(file_info.max_sweep_time)
        
        # Populate sweep combo
        self.sweep_combo.clear()
        self.sweep_combo.addItems(file_info.sweep_names)
        
        # Enable controls
        self.batch_btn.setEnabled(True)
        self.control_panel.set_controls_enabled(True)
        
        # Check if we need to apply swap state
        needs_swap = False
        if hasattr(self.control_panel, '_pending_swap_state') and self.control_panel._pending_swap_state:
            needs_swap = True
            self.control_panel._is_swapped = True
            self.control_panel._pending_swap_state = False
        elif hasattr(self.control_panel, '_is_swapped') and self.control_panel._is_swapped:
            needs_swap = True
        
        # Apply swap if needed
        if needs_swap:
            # Force swap after file load
            result = self.controller.swap_channels()
            if result['success']:
                self.control_panel.update_swap_button_state(True)
                config = result['configuration']
                self.status_bar.showMessage(
                    f"Loaded {file_info.sweep_count} sweeps. Channels: Voltage=Ch{config['voltage']}, Current=Ch{config['current']}"
                )
        else:
            self.status_bar.showMessage(f"Loaded {file_info.sweep_count} sweeps from {file_info.name}")
        
        # Update plot - this will now show the swapped data
        if file_info.sweep_names:
            self.sweep_combo.setCurrentIndex(0)
            self._update_plot()
    
    def _update_plot(self):
        """Update the plot display"""
        selection = self.sweep_combo.currentText()
        if not selection:
            return
        
        channel_type = self.channel_combo.currentText()
        plot_data = self.controller.get_sweep_plot_data(selection, channel_type)
        
        if plot_data:
            # Get channel config through controller method
            channel_config = self.controller.get_channel_configuration()
            
            # Pass to plot manager for visualization
            self.plot_manager.update_sweep_plot(
                t=plot_data.time_ms,
                y=plot_data.data_matrix,
                channel=plot_data.channel_id,
                sweep_index=plot_data.sweep_index,
                channel_type=plot_data.channel_type,
                channel_config=channel_config
            )
            
            # Update range lines using values from control panel
            range_values = self.control_panel.get_range_values()
            self.plot_manager.update_range_lines(
                range_values['range1_start'],
                range_values['range1_end'],
                range_values['use_dual_range'],
                range_values['range2_start'],
                range_values['range2_end']
            )
            
            # Update spinbox mapping for dragging
            self.plot_manager.update_line_spinbox_map(
                self.control_panel.get_range_spinboxes()
            )
    
    def _generate_analysis_plot(self):
        """Generate and display analysis plot"""
        params = self._collect_parameters()
        result = self.controller.perform_analysis(params)
        
        if result:
            plot_data_dict = {
                'x_data': result.x_data,
                'y_data': result.y_data,
                'y_data2': result.y_data2,
                'sweep_indices': result.sweep_indices,
                'use_dual_range': result.use_dual_range,
                'y_label_r1': result.y_label,
                'y_label_r2': result.y_label
            }
            
            dialog = AnalysisPlotDialog(
                self, plot_data_dict, result.x_label, result.y_label,
                f"{result.y_label} vs {result.x_label}",
                controller=self.controller,
                params=params
            )
            dialog.exec()
        else:
            QMessageBox.warning(self, "No Data", "Please load a MAT file first.")
    
    def _export_data(self):
        """Export analysis data"""
        if not self.controller.has_data():
            QMessageBox.information(self, "Export Error", "No data to export.")
            return
        
        # Use the controller's suggested filename
        suggested = self.controller.get_suggested_export_filename()
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot Data", suggested, "CSV files (*.csv)"
        )
        
        if file_path:
            params = self._collect_parameters()
            success = self.controller.export_analysis_data_to_file(params, file_path)
            # Success/error messages handled by controller callbacks
    
    def _batch_analyze(self):
        """Perform batch analysis"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files for Batch Analysis", "", 
            "Data files (*.mat *.abf);;MAT files (*.mat);;ABF files (*.abf);;All files (*.*)"
        )
        
        if not file_paths:
            return
        
        # Get the shortest max sweep time from all selected files
        min_max_sweep_time = self.controller.get_min_max_sweep_time_for_files(file_paths)
        params = self._collect_parameters()

        if params['range1_end'] > min_max_sweep_time or \
           (params['use_dual_range'] and params['range2_end'] > min_max_sweep_time):
            QMessageBox.warning(self, "Invalid Range for Batch Analysis",
                                f"The analysis range is outside the valid time range for at least one of the selected files. "
                                f"Please adjust the analysis range to be within 0 - {min_max_sweep_time:.2f} ms for all files.")
            return
        
        destination_folder = self._get_batch_output_folder(file_paths)
        if not destination_folder:
            return
        
        params = self._collect_parameters()
        
        # Create progress dialog
        progress = self._create_progress_dialog(len(file_paths))
        
        def update_progress(current, total):
            progress.setValue(current)
            QApplication.processEvents()
        
        try:
            # Get data from controller
            result = self.controller.perform_batch_analysis(
                file_paths,
                params,
                destination_folder,
                progress_callback=update_progress
            )
            
            if result.success:
                # Create the plot using PlotService
                from data_analysis_gui.services.plot_service import PlotService
                plot_service = PlotService()
                
                plot_data = plot_service.create_batch_plot(
                    result.batch_result,
                    params,
                    result.x_label,
                    result.y_label
                )
                
                # Create the plot using PlotService
                from data_analysis_gui.services.plot_service import PlotService
                plot_service = PlotService()

                figure, plot_count = plot_service.build_batch_figure(
                    result.batch_result,
                    params,
                    result.x_label,
                    result.y_label
                )
                
                # Show batch results dialog
                batch_dialog = BatchResultDialog(
                    self, 
                    result.batch_result, 
                    figure,
                    result.iv_data, 
                    result.iv_file_mapping,
                    result.x_label, 
                    result.y_label, 
                    destination_folder=destination_folder,
                    iv_data_range2=result.iv_data_range2,
                    use_dual_range=result.use_dual_range  
                )
                batch_dialog.exec()
                
                # Update status
                self.status_bar.showMessage(
                    f"Batch complete. Processed {result.successful_count} files, "
                    f"{result.failed_count} failed."
                )
            else:
                QMessageBox.warning(self, "Batch Analysis Failed", 
                                "No files could be processed successfully.")
        
        except Exception as e:
            QMessageBox.critical(self, "Batch Analysis Error", str(e))
        finally:
            progress.close()
    
    def _swap_channels(self):
        """Handle channel swapping"""
        result = self.controller.swap_channels()
        
        if not result['success']:
            if 'reason' in result:
                QMessageBox.warning(self, "Cannot Swap", result['reason'])
            return
        
        # Update control panel button appearance
        self.control_panel.update_swap_button_state(result['is_swapped'])
        
        # Update status bar
        config = result['configuration']
        self.status_bar.showMessage(
            f"Channel configuration: Voltage=Ch{config['voltage']}, Current=Ch{config['current']}"
        )
        
        # Switch displayed channel
        current_type = self.channel_combo.currentText()
        new_type = "Current" if current_type == "Voltage" else "Voltage"
        self.channel_combo.setCurrentText(new_type)
    
    # ============ Helper Methods ============
    
    def _collect_parameters(self):
        """Collect parameters from control panel"""
        gui_state = self.control_panel.collect_parameters()
        return self.controller.create_parameters_from_dict(gui_state)
    
    def _get_batch_output_folder(self, file_paths):
        """Prompt user for output folder"""
        base_dir = os.path.dirname(file_paths[0])
        
        # Calculate unique default folder name
        default_folder_name = "PB_analysis"
        temp_path = os.path.join(base_dir, default_folder_name)
        counter = 1
        while os.path.exists(temp_path):
            default_folder_name = f"PB_analysis_{counter}"
            temp_path = os.path.join(base_dir, default_folder_name)
            counter += 1
        
        # Prompt user for folder name
        folder_name, ok = QInputDialog.getText(
            self, "Name Output Folder",
            "Enter a name for the new results folder:",
            text=default_folder_name
        )
        
        if not ok or not folder_name:
            return None
        
        destination_folder = os.path.join(base_dir, folder_name)
        
        # Check if folder exists
        if os.path.exists(destination_folder):
            reply = QMessageBox.question(
                self, 'Folder Exists',
                f"The folder '{folder_name}' already exists.\n\n"
                "Do you want to save files into this existing folder?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return None
        else:
            os.makedirs(destination_folder)
        
        return destination_folder
    
    def _create_progress_dialog(self, max_value):
        """Create and configure progress dialog"""
        progress = QProgressBar()
        progress.setMaximum(max_value)
        progress.setWindowTitle("Batch Analysis Progress")
        progress.show()
        return progress
    
    # ============ Pure GUI Methods ============
    
    def _toggle_dual_range(self, enabled):
        """Handle dual range toggle from control panel"""
        range_values = self.control_panel.get_range_values()
        self.plot_manager.toggle_dual_range(
            enabled, 
            range_values['range2_start'], 
            range_values['range2_end']
        )
    
    def _on_line_dragged(self, line, x_value):
        """Callback when range lines are dragged"""
        # Determine which spinbox to update based on the line
        spinboxes = self.control_panel.get_range_spinboxes()
        
        if line == self.plot_manager.range_lines[0]:
            self.control_panel.update_range_value('start1', x_value)
        elif line == self.plot_manager.range_lines[1]:
            self.control_panel.update_range_value('end1', x_value)
        elif self.control_panel.dual_range_cb.isChecked() and len(self.plot_manager.range_lines) > 2:
            if line == self.plot_manager.range_lines[2]:
                self.control_panel.update_range_value('start2', x_value)
            elif line == self.plot_manager.range_lines[3]:
                self.control_panel.update_range_value('end2', x_value)
    
    def _update_lines_from_entries(self):
        """Update range lines based on control panel values"""
        range_values = self.control_panel.get_range_values()
        self.plot_manager.update_lines_from_values(
            range_values['range1_start'],
            range_values['range1_end'],
            range_values['use_dual_range'],
            range_values['range2_start'],
            range_values['range2_end']
        )
    
    def _center_nearest_cursor(self):
        """Center the nearest cursor line"""
        line_moved, new_position = self.plot_manager.center_nearest_cursor()
        
        if line_moved and new_position is not None:
            # Update the corresponding spinbox
            if line_moved in self.plot_manager.line_spinbox_map:
                spinbox = self.plot_manager.line_spinbox_map[line_moved]
                # Find which spinbox and update through control panel
                spinboxes = self.control_panel.get_range_spinboxes()
                for key, sb in spinboxes.items():
                    if sb == spinbox:
                        self.control_panel.update_range_value(key, new_position)
                        break
    
    def _start_hold(self, direction_func):
        """Start continuous navigation"""
        direction_func()
        self.hold_direction = direction_func
        self.hold_timer.start(150)
    
    def _stop_hold(self):
        """Stop continuous navigation"""
        self.hold_timer.stop()
        self.hold_direction = None
    
    def _continue_hold(self):
        """Continue navigation while held"""
        if self.hold_direction:
            self.hold_direction()
    
    def _next_sweep(self):
        """Navigate to next sweep"""
        current_index = self.sweep_combo.currentIndex()
        if current_index < self.sweep_combo.count() - 1:
            self.sweep_combo.setCurrentIndex(current_index + 1)
    
    def _prev_sweep(self):
        """Navigate to previous sweep"""
        current_index = self.sweep_combo.currentIndex()
        if current_index > 0:
            self.sweep_combo.setCurrentIndex(current_index - 1)
    
    def _set_theme(self, name):
        """Apply a color theme"""
        if name in THEMES:
            self.current_theme_name = name
            new_stylesheet = get_theme_stylesheet(THEMES[name])
            self.setStyleSheet(new_stylesheet)

    # =========== Save/Load Session Settings ===========
    def closeEvent(self, event):
        """Save session settings before closing."""
        try:
            # Collect and save current settings
            params = self.control_panel.collect_parameters()
            save_last_session(params)
        except Exception as e:
            # Don't prevent app closure if save fails
            print(f"Failed to save session on exit: {e}")
        
        # Continue with normal close
        event.accept()
    
    # ============ Callbacks for Controller ============
    
    def _show_error(self, message: str):
        """Show error message from controller"""
        QMessageBox.critical(self, "Error", message)
    
    def _update_status(self, message: str):
        """Update status bar with message from controller"""
        self.status_bar.showMessage(message)


# ============ Main Entry Point ============

def main():
    """Main entry point for the application"""
    import sys
    
    app = QApplication(sys.argv)
    
    # Create controller and view
    controller = ApplicationController()
    window = ModernMatSweepAnalyzer(controller)
    
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()