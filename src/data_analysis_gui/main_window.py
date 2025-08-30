"""
Refactored Main Window - Pure GUI implementation with no business logic.
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

import base64
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# GUI-only imports
from data_analysis_gui.plot_manager import PlotManager
from data_analysis_gui.config import THEMES, get_theme_stylesheet, DEFAULT_SETTINGS
from data_analysis_gui.dialogs import (ConcentrationResponseDialog, BatchResultDialog, 
                     AnalysisPlotDialog, CurrentDensityIVDialog)
from data_analysis_gui.widgets import SelectAllSpinBox, NoScrollComboBox

# Import the controller
from data_analysis_gui.core.app_controller import ApplicationController, FileInfo, PlotData


class ModernMatSweepAnalyzer(QMainWindow):
    """
    Pure GUI implementation. This class only handles:
    - Widget creation and layout
    - User input collection
    - Display of results
    - Event handling that delegates to the controller
    """
    
    def __init__(self, controller: ApplicationController = None):
        super().__init__()
        
        # Use provided controller or create one
        self.controller = controller or ApplicationController()
        
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
    
    # ============ UI Initialization ============
    
    def _init_ui(self):
        """Initialize the main UI components"""
        self.setWindowTitle("MAT File Sweep Analyzer - Modern Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        self._create_menu_bar()
        self._create_toolbar()
        self._create_main_layout()
        self._create_status_bar()
    
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
    
    def _create_tools_menu(self, menubar):
        """Create the Tools menu"""
        tools_menu = menubar.addMenu('Tools')
        conc_analysis_action = QAction('Concentration Response Analysis', self)
        conc_analysis_action.triggered.connect(self._open_conc_analysis)
        tools_menu.addAction(conc_analysis_action)
    
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
        load_btn = QPushButton("Load MAT File")
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
        """Create the main layout with splitter"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
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
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([400, 1000])
    
    def _create_control_panel(self):
        """Create the control panel with all settings groups"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(400)
        
        control_widget = QWidget()
        scroll_area.setWidget(control_widget)
        
        layout = QVBoxLayout(control_widget)
        
        layout.addWidget(self._create_file_info_group())
        layout.addWidget(self._create_analysis_settings_group())
        layout.addWidget(self._create_plot_settings_group())
        
        # Export Plot Data button
        self.export_plot_btn = QPushButton("Export Plot Data")
        self.export_plot_btn.clicked.connect(self._export_data)
        self.export_plot_btn.setEnabled(False)
        layout.addWidget(self.export_plot_btn)
        
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
        self.dual_range_cb.stateChanged.connect(self._toggle_dual_range)
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
        self.swap_channels_btn.clicked.connect(self._swap_channels)
        analysis_layout.addWidget(self.swap_channels_btn, 6, 0, 1, 2)
        
        # Center Nearest Cursor button
        center_cursor_btn = QPushButton("Center Nearest Cursor")
        center_cursor_btn.setToolTip("Moves the nearest cursor to the center of the view")
        center_cursor_btn.clicked.connect(self._center_nearest_cursor)
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
        self.start_spin.valueChanged.connect(self._update_lines_from_entries)
        layout.addWidget(self.start_spin, 0, 1)
        
        layout.addWidget(QLabel("Range 1 End (ms):"), 1, 0)
        self.end_spin = SelectAllSpinBox()
        self.end_spin.setRange(0, 100000)
        self.end_spin.setValue(DEFAULT_SETTINGS['range1_end'])
        self.end_spin.setSingleStep(0.05)
        self.end_spin.setDecimals(2)
        self.end_spin.valueChanged.connect(self._update_lines_from_entries)
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
        self.start_spin2.valueChanged.connect(self._update_lines_from_entries)
        layout.addWidget(self.start_spin2, 3, 1)
        
        layout.addWidget(QLabel("Range 2 End (ms):"), 4, 0)
        self.end_spin2 = SelectAllSpinBox()
        self.end_spin2.setRange(0, 100000)
        self.end_spin2.setValue(DEFAULT_SETTINGS['range2_end'])
        self.end_spin2.setSingleStep(0.05)
        self.end_spin2.setDecimals(2)
        self.end_spin2.setEnabled(False)
        self.end_spin2.valueChanged.connect(self._update_lines_from_entries)
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
        self.update_plot_btn.clicked.connect(self._generate_analysis_plot)
        self.update_plot_btn.setEnabled(False)
        plot_layout.addWidget(self.update_plot_btn, 2, 0, 1, 3)
        
        return plot_group
    
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
            self, "Load MAT File", "", "MAT files (*.mat)"
        )
        
        if file_path:
            self.controller.load_file(file_path)
    
    def _handle_file_loaded(self, file_info: FileInfo):
        """Handle successful file load (callback from controller)"""
        # Update file info display
        self.file_label.setText(f"File: {file_info.name}")
        self.sweep_count_label.setText(f"Sweeps: {file_info.sweep_count}")
        
        # Populate sweep combo
        self.sweep_combo.clear()
        self.sweep_combo.addItems(file_info.sweep_names)
        if file_info.sweep_names:
            self.sweep_combo.setCurrentIndex(0)
            self._update_plot()
        
        # Enable controls
        self.batch_btn.setEnabled(True)
        self.update_plot_btn.setEnabled(True)
        self.export_plot_btn.setEnabled(True)
        
        self.status_bar.showMessage(f"Loaded {file_info.sweep_count} sweeps from {file_info.name}")
    
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
                channel_config=channel_config  # Use the returned config
            )
            
            # Update range lines
            self.plot_manager.update_range_lines(
                self.start_spin.value(),
                self.end_spin.value(),
                self.dual_range_cb.isChecked(),
                self.start_spin2.value() if self.dual_range_cb.isChecked() else None,
                self.end_spin2.value() if self.dual_range_cb.isChecked() else None
            )
            
            # Update spinbox mapping
            spinboxes = {
                'start1': self.start_spin,
                'end1': self.end_spin
            }
            if self.dual_range_cb.isChecked():
                spinboxes['start2'] = self.start_spin2
                spinboxes['end2'] = self.end_spin2
            
            self.plot_manager.update_line_spinbox_map(spinboxes)
    
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
                controller=self.controller,  # Pass controller
                params=params  # Pass parameters
            )
            dialog.exec()
        else:
            QMessageBox.warning(self, "No Data", "Please load a MAT file first.")
    
   
    def _get_export_file_path(self, dialog_title="Export Plot Data"):
        """Common method to get export file path with smart defaults"""
        if self.controller.loaded_file_path:
            base_name = os.path.basename(self.controller.loaded_file_path).split('.mat')[0]
            if '[' in base_name:
                base_name = base_name.split('[')[0]
            suggested_filename = f"{base_name}_analyzed.csv"
        else:
            suggested_filename = "analyzed.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            dialog_title,
            suggested_filename,
            "CSV files (*.csv)"
        )
        return file_path

    def _export_data(self):
        """Export analysis data"""
        if not self.controller.has_data():
            QMessageBox.information(self, "Export Error", "No data to export.")
            return
        
        # Use the new method for suggested filename
        suggested = self.controller.get_suggested_export_filename()
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot Data", suggested, "CSV files (*.csv)"
        )
        
        if file_path:
            params = self._collect_parameters()
            success = self.controller.export_analysis_data_to_file(params, file_path)
            # Success/error messages handled by controller callbacks
    
    def _batch_analyze(self):
        """Perform batch analysis - using the new architecture"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select MAT Files for Batch Analysis", "", "MAT files (*.mat)"
        )
        
        if not file_paths:
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
            # Get data from controller (no plotting)
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
                
                # Reconstruct figure for the dialog
                figure = self._deserialize_figure(
                    plot_data['figure_data'],
                    plot_data['figure_size']
                )
                
                # Show batch results dialog
                batch_dialog = BatchResultDialog(
                    self, 
                    result.batch_data, 
                    figure,
                    result.iv_data, 
                    result.iv_file_mapping,
                    result.x_label, 
                    result.y_label, 
                    destination_folder=destination_folder
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
    
    def _deserialize_figure(self, figure_data: str, figure_size: tuple) -> Figure:
        """Reconstruct a matplotlib figure from serialized data"""
        # Decode the base64 PNG data
        img_data = base64.b64decode(figure_data)
        
        # Create a new figure with the original size
        fig = Figure(figsize=figure_size)
        
        # Create a canvas for the figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        canvas = FigureCanvasQTAgg(fig)
        
        # Load the image into the figure
        from PIL import Image
        import numpy as np
        
        img = Image.open(BytesIO(img_data))
        ax = fig.add_subplot(111)
        ax.imshow(np.array(img))
        ax.axis('off')  # Hide axes since we're showing a rendered image
        
        return fig
    
    def _swap_channels(self):
        """Handle channel swapping"""
        result = self.controller.swap_channels()
        
        if not result['success']:
            if 'reason' in result:
                QMessageBox.warning(self, "Cannot Swap", result['reason'])
            return
        
        # Update button appearance
        if result['is_swapped']:
            self.swap_channels_btn.setStyleSheet("QPushButton { background-color: #ffcc99; }")
            self.swap_channels_btn.setText("Channels Swapped ⇄")
        else:
            self.swap_channels_btn.setStyleSheet("")
            self.swap_channels_btn.setText("Swap Channels")
        
        # Update status bar
        config = result['configuration']
        self.status_bar.showMessage(
            f"Channel configuration: Voltage=Ch{config['voltage']}, Current=Ch{config['current']}"
        )
        
        # Switch displayed channel
        current_type = self.channel_combo.currentText()
        new_type = "Current" if current_type == "Voltage" else "Voltage"
        self.channel_combo.setCurrentText(new_type)
    
    # ============ Helper Methods (GUI-only logic) ============
    
    def _collect_parameters(self):
        """Collect parameters from GUI widgets as a simple dictionary"""
        gui_state = {
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
        return self.controller.create_parameters_from_dict(gui_state)
    
    def _get_batch_output_folder(self, file_paths):
        """Prompt user for output folder"""
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
    
    # ============ Pure GUI Methods (no business logic) ============
    
    def _toggle_dual_range(self):
        """Toggle dual range UI elements"""
        enabled = self.dual_range_cb.isChecked()
        
        self.start_spin2.setEnabled(enabled)
        self.end_spin2.setEnabled(enabled)
        
        # Let PlotManager handle the visual updates
        self.plot_manager.toggle_dual_range(
            enabled, 
            self.start_spin2.value(), 
            self.end_spin2.value()
        )
    
    def _on_line_dragged(self, line, x_value):
        """Callback when range lines are dragged"""
        if line == self.plot_manager.range_lines[0]:
            self.start_spin.setValue(x_value)
        elif line == self.plot_manager.range_lines[1]:
            self.end_spin.setValue(x_value)
        elif self.dual_range_cb.isChecked() and len(self.plot_manager.range_lines) > 2:
            if line == self.plot_manager.range_lines[2]:
                self.start_spin2.setValue(x_value)
            elif line == self.plot_manager.range_lines[3]:
                self.end_spin2.setValue(x_value)
    
    def _update_lines_from_entries(self):
        """Update range lines based on spinbox values"""
        self.plot_manager.update_lines_from_values(
            self.start_spin.value(),
            self.end_spin.value(),
            self.dual_range_cb.isChecked(),
            self.start_spin2.value() if self.dual_range_cb.isChecked() else None,
            self.end_spin2.value() if self.dual_range_cb.isChecked() else None
        )
    
    def _center_nearest_cursor(self):
        """Center the nearest cursor line"""
        line_moved, new_position = self.plot_manager.center_nearest_cursor()
        
        if line_moved and new_position is not None:
            # Update the corresponding spinbox
            if line_moved in self.plot_manager.line_spinbox_map:
                spinbox_to_update = self.plot_manager.line_spinbox_map[line_moved]
                spinbox_to_update.setValue(new_position)
    
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
    
    def _open_conc_analysis(self):
        """Open concentration analysis dialog"""
        self.conc_analysis_dialog = ConcentrationResponseDialog(self)
        self.conc_analysis_dialog.show()
    
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