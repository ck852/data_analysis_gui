"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Main application window for the PatchBatch Electrophysiology Data Analysis Tool.

This module implements the primary user interface for analyzing patch-clamp
electrophysiology data. It serves as the central coordinator between:
- User interactions (file loading, parameter adjustment, analysis requests)
- Data processing backend (controller, analysis engine)
- Visualization components (plot manager, analysis dialogs)

The MainWindow class handles:
1. File I/O operations for ABF and WCP format data files
2. Real-time sweep visualization with adjustable analysis ranges
3. Parameter configuration for various analysis modes (IV curves, peaks, time series)
4. Batch processing coordination for multiple file analysis
5. Session state persistence to remember user preferences

Key design principles:
- Controls remain active at all times (no complex enable/disable logic)
- Settings auto-save on change to preserve user workflow
- Separation of concerns: UI logic here, analysis logic in controllers
- Signal/slot pattern for loose coupling between components
"""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
    QMessageBox, QSplitter, QToolBar, QStatusBar, QLabel,
    QComboBox, QDialog
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QKeySequence, QAction

# Import refactored theme functions
from data_analysis_gui.config.themes import (apply_modern_theme, create_styled_button, style_combo_box,
                                            style_label
                                )

from data_analysis_gui.core.session_settings import (extract_settings_from_main_window, apply_settings_to_main_window,
                                                    revalidate_ranges_for_file, load_session_settings, save_session_settings,
)

from data_analysis_gui.config.plot_style import add_zero_axis_lines

# Core imports
from data_analysis_gui.core.app_controller import ApplicationController
from data_analysis_gui.core.models import FileInfo
from data_analysis_gui.config.logging import get_logger
from data_analysis_gui.core.plot_formatter import PlotFormatter

# Widget imports
from data_analysis_gui.widgets.control_panel import ControlPanel
from data_analysis_gui.plot_manager import PlotManager

# Dialog imports
from data_analysis_gui.dialogs.analysis_plot_dialog import AnalysisPlotDialog
from data_analysis_gui.dialogs.batch_dialog import BatchAnalysisDialog
from data_analysis_gui.dialogs.bg_subtraction_dialog import BackgroundSubtractionDialog
from data_analysis_gui.dialogs.ramp_iv_dialog import RampIVDialog
from data_analysis_gui.dialogs import ConcentrationResponseDialog

# Service imports
from data_analysis_gui.gui_services import FileDialogService

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window for PatchBatch Electrophysiology Data Analysis Tool.

    This class manages the core GUI, including file operations, sweep navigation,
    analysis parameter configuration, plotting, and batch processing. Controls remain
    active at all times, streamlining user interaction and minimizing state complexity.

    Key Features:
        - Modern, compact themed UI
        - Always-active controls for efficient workflow
        - Persistent session settings
        - Real-time plot updates and analysis visualization
        - Batch analysis and export support
    """

    # Application events
    file_loaded = Signal(str)
    analysis_completed = Signal()

    def __init__(self):
        super().__init__()

        # Initialize controller (which creates all services)
        self.controller = ApplicationController()

        # Initialize plot formatter for consistent plot labeling
        self.plot_formatter = PlotFormatter()

        # Get shared services from controller
        services = self.controller.get_services()
        self.data_manager = services["data_manager"]
        self.analysis_manager = services["analysis_manager"]
        self.batch_processor = services["batch_processor"]

        # GUI services
        self.file_dialog_service = FileDialogService()

        # Controller callbacks
        self.controller.on_file_loaded = self._on_file_loaded
        self.controller.on_error = lambda msg: QMessageBox.critical(self, "Error", msg)
        self.controller.on_status_update = lambda msg: self.status_bar.showMessage(
            msg, 5000
        )

        # State
        self.current_file_path: Optional[str] = None
        self.analysis_dialog: Optional[AnalysisPlotDialog] = None

        # Navigation timer
        self.hold_timer = QTimer()
        self.hold_timer.timeout.connect(self._continue_navigation)
        self.navigation_direction = None

        # Initialize default values for settings that may be loaded
        self.last_channel_view = "Voltage"
        self.last_directory = None

        # self.current_units = "pA"  # Default current units

        # Build UI
        self._init_ui()

        # Load and apply settings immediately (shows user's last configuration)
        saved_settings = load_session_settings()
        if saved_settings:
            apply_settings_to_main_window(self, saved_settings)

        # Apply modern theme to the main window (handles everything including toolbars and menus)
        apply_modern_theme(self)

        # Configure window
        self.setWindowTitle("PatchBatch BETA")

    def _init_ui(self):
        """Initialize the complete UI with full theme integration"""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout with NO spacing or margins for seamless appearance
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        main_layout.addWidget(splitter)

        # Control panel (left)
        self.control_panel = ControlPanel()
        splitter.addWidget(self.control_panel)

        # Plot manager (right)
        self.plot_manager = PlotManager()
        splitter.addWidget(self.plot_manager.get_plot_widget())

        # Set the plot manager to expand and leave the control panel at its minimum size
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # splitter.setMinimumWidth(600)

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
        """
        Create and configure application menus.

        Includes File menu (Open, Exit) and Analysis menu (Batch Analyze).
        Menu styling is handled by the modern theme system.
        """
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        self.open_action = QAction("&Open...", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self._open_file)
        file_menu.addAction(self.open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")

        self.batch_action = QAction("&Batch Analyze...", self)
        self.batch_action.setShortcut("Ctrl+B")
        self.batch_action.triggered.connect(self._batch_analyze)
        self.batch_action.setEnabled(True)
        analysis_menu.addAction(self.batch_action)

        # Background Subtraction
        self.bg_subtract_action = QAction("&Background Subtraction...", self)
        self.bg_subtract_action.triggered.connect(self._background_subtraction)
        analysis_menu.addAction(self.bg_subtract_action)

        # Ramp IV Analysis
        self.ramp_iv_action = QAction("&Ramp IV Analysis...", self)
        self.ramp_iv_action.triggered.connect(self._ramp_iv_analysis)
        analysis_menu.addAction(self.ramp_iv_action)

        # Concentration Response Analysis
        conc_resp_action = analysis_menu.addAction("Concentration Response...")
        conc_resp_action.triggered.connect(self._open_concentration_response)

        # Sweep Extractor
        sweep_extract_action = analysis_menu.addAction("Extract Sweeps...")
        sweep_extract_action.triggered.connect(self._sweep_extraction)

    def _open_concentration_response(self):
        """Open the concentration-response analysis dialog."""
        dialog = ConcentrationResponseDialog(self)
        dialog.exec()

    def _background_subtraction(self):
        if not self.controller.has_data():
            QMessageBox.warning(self, "No Data", "Please load a data file first.")
            return
        
        sweep = self.sweep_combo.currentText()
        if not sweep:
            return
        
        dialog = BackgroundSubtractionDialog(
            dataset=self.controller.current_dataset,
            sweep_index=sweep,
            parent=self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Refresh plot and clear caches
            self._update_plot()
            if hasattr(self.analysis_manager, "clear_caches"):
                self.analysis_manager.clear_caches()
            self.status_bar.showMessage("Background subtraction applied", 3000)

    def _ramp_iv_analysis(self):
        """Open the ramp IV analysis dialog."""
        if not self.controller.has_data():
            QMessageBox.warning(self, "No Data", "Please load a data file first.")
            return
        
        # Get current analysis range from control panel
        params = self.control_panel.get_parameters()
        
        # Get current units from loaded file metadata
        dataset = self.controller.current_dataset
        channel_config = dataset.metadata.get("channel_config")
        if not channel_config:
            logger.warning("No channel configuration found - using default units")
            current_units = "pA"
        else:
            current_units = channel_config.get("current_units", "pA")
        
        # Create dialog with Range 1 parameters
        dialog = RampIVDialog(
            dataset=dataset,
            start_ms=params.range1_start,
            end_ms=params.range1_end,
            current_units=current_units,
            parent=self
        )
        
        # Use the special show method that gets voltage targets first
        # This will show voltage input dialog, then main dialog if user doesn't cancel
        dialog.show_with_voltage_input()

    def _create_toolbar(self):
        """
        Construct the main toolbar with themed controls.

        Toolbar provides file operations, sweep navigation, channel selection,
        current units, file information, cursor centering, and channel toggling.
        All controls are styled and sized for a compact, modern appearance.
        """
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # File operations
        open_action = toolbar.addAction("Open", self._open_file)
        toolbar.addSeparator()

        # Navigation buttons - start disabled until file is loaded
        self.prev_btn = create_styled_button("◀", "secondary")
        self.prev_btn.setMaximumWidth(10)
        self.prev_btn.setEnabled(False)
        self.prev_btn.pressed.connect(lambda: self._start_navigation(self._prev_sweep))
        self.prev_btn.released.connect(self._stop_navigation)
        toolbar.addWidget(self.prev_btn)

        # Sweep combo - disabled until file is loaded
        self.sweep_combo = QComboBox()
        self.sweep_combo.setMinimumWidth(80)
        self.sweep_combo.setEnabled(True)
        self.sweep_combo.currentTextChanged.connect(self._on_sweep_changed)
        style_combo_box(self.sweep_combo)
        toolbar.addWidget(self.sweep_combo)

        self.next_btn = create_styled_button("▶", "secondary")
        self.next_btn.setMaximumWidth(10)
        self.next_btn.setEnabled(False)
        self.next_btn.pressed.connect(lambda: self._start_navigation(self._next_sweep))
        self.next_btn.released.connect(self._stop_navigation)
        toolbar.addWidget(self.next_btn)

        toolbar.addSeparator()

        channel_label = QLabel("Channel:")
        style_label(channel_label, "normal")
        toolbar.addWidget(channel_label)

        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Voltage", "Current"])
        self.channel_combo.setMaximumWidth(80)
        self.channel_combo.setEnabled(True)
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        style_combo_box(self.channel_combo)
        toolbar.addWidget(self.channel_combo)

        toolbar.addSeparator()

        # File Information Labels with theme styling
        self.file_label = QLabel("No file loaded")
        style_label(self.file_label, "muted")
        toolbar.addWidget(self.file_label)

        toolbar.addSeparator()

        self.sweep_count_label = QLabel("")
        self.sweep_count_label.setMaximumWidth(80)
        style_label(self.sweep_count_label, "muted")
        toolbar.addWidget(self.sweep_count_label)

        toolbar.addSeparator()

        # Center Cursor Button - always enabled
        self.center_cursor_btn = create_styled_button(
            "Center Nearest Cursor", "secondary"
        )
        self.center_cursor_btn.setToolTip(
            "Moves the nearest cursor to the center of the view"
        )
        self.center_cursor_btn.clicked.connect(
            lambda: self._sync_cursor_to_control(
                *self.plot_manager.center_nearest_cursor()
            )
        )
        # Always enabled - cursor centering works regardless of file state
        self.center_cursor_btn.setEnabled(True)
        toolbar.addWidget(self.center_cursor_btn)

        toolbar.addSeparator()

        # Connect toolbar controls to auto-save
        self.channel_combo.currentTextChanged.connect(self._auto_save_settings)

    def _connect_signals(self):
        """
        Connect UI signals to their respective logic handlers.

        Ensures that user actions and control changes trigger appropriate updates,
        including auto-saving settings and synchronizing plot/cursor states.
        """
        # Control panel
        self.control_panel.analysis_requested.connect(self._generate_analysis)
        self.control_panel.export_requested.connect(self._export_data)
        self.control_panel.dual_range_toggled.connect(self._toggle_dual_range)
        self.control_panel.range_values_changed.connect(self._sync_cursors_to_plot)

        # Auto-save settings when they change
        self.control_panel.dual_range_toggled.connect(self._auto_save_settings)
        self.control_panel.range_values_changed.connect(self._auto_save_settings)

        # Connect to plot setting combo boxes for auto-save
        self.control_panel.x_measure_combo.currentTextChanged.connect(
            self._auto_save_settings
        )
        self.control_panel.x_channel_combo.currentTextChanged.connect(
            self._auto_save_settings
        )
        self.control_panel.y_measure_combo.currentTextChanged.connect(
            self._auto_save_settings
        )
        self.control_panel.y_channel_combo.currentTextChanged.connect(
            self._auto_save_settings
        )
        self.control_panel.peak_mode_combo.currentTextChanged.connect(
            self._auto_save_settings
        )

        # Plot manager
        self.plot_manager.line_state_changed.connect(self._on_cursor_moved)

    def _open_file(self):
        """
        Prompt the user to select a data file and load it via the controller.

        Opens a file dialog for supported formats, loads the selected file,
        and updates the UI. Emits file_loaded signal on success.
        """
        file_types = (
            "WCP files (*.wcp);;"
            "ABF files (*.abf);;"
            "Data files (*.wcp *.abf);;"
            "All files (*.*)"
        )

        # Use ONLY the service's stored directory - no fallback logic
        # The service handles fallbacks internally if needed
        file_path = self.file_dialog_service.get_import_path(
            parent=self,
            title="Open Data File",
            default_directory=None,  # Let service use its memory
            file_types=file_types,
            dialog_type="import_data",
        )

        if file_path:
            # Use controller to load file
            result = self.controller.load_file(file_path)

            if result.success:
                self.current_file_path = file_path
                self.file_loaded.emit(file_path)
                
                # Auto-save settings to persist the directory choice
                self._auto_save_settings()
            # Error handling is done by controller callbacks

    def _on_file_loaded(self, file_info: FileInfo):
        """
        Respond to successful file load and update UI components.

        Updates file labels, sweep count, revalidates ranges,
        and populates the sweep selection combo box.
        """
        # Update file labels with proper theme styling
        self.file_label.setText(f"File: {file_info.name}")
        style_label(self.file_label, "normal")  # Switch from muted to normal

        self.sweep_count_label.setText(f"Sweeps: {file_info.sweep_count}")
        style_label(self.sweep_count_label, "normal")  # Switch from muted to normal

        # Revalidate ranges with file's max sweep time if available
        if file_info.max_sweep_time:
            revalidate_ranges_for_file(self, file_info.max_sweep_time)

        # Apply saved channel view preference
        if hasattr(self, "last_channel_view"):
            self.channel_combo.setCurrentText(self.last_channel_view)

        # Enable navigation controls (these still depend on having a file loaded)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.sweep_combo.setEnabled(True)
        self.channel_combo.setEnabled(True)

        # Populate sweeps
        self.sweep_combo.clear()
        self.sweep_combo.addItems(file_info.sweep_names)

        # Show first sweep
        if file_info.sweep_names:
            self.sweep_combo.setCurrentIndex(0)

    def _on_sweep_changed(self):
        """
        Update the plot when the sweep selection changes.
        """
        self._update_plot()

    def _on_channel_changed(self):
        """
        Update the plot when the channel selection changes.
        """
        self._update_plot()

    def _update_plot(self):
        """
        Refresh the sweep plot using controller data and centralized formatting.
        """
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

            # Get current units from loaded file metadata
            dataset = self.controller.current_dataset
            channel_config = dataset.metadata.get("channel_config")
            if not channel_config:
                logger.warning("No channel configuration found - using default units")
                current_units = "pA"
                channel_config = {"current_units": "pA"}
            else:
                current_units = channel_config.get("current_units", "pA")

            # Use centralized formatter for consistent labels
            sweep_info = {
                "sweep_index": int(sweep) if sweep.isdigit() else 0,
                "channel_type": channel_type,
                "current_units": current_units,
            }
            plot_labels = self.plot_formatter.get_plot_titles_and_labels(
                "sweep", sweep_info=sweep_info
            )

            # Update plot with formatted labels AND channel_config for cursor text
            self.plot_manager.update_sweep_plot(
                t=plot_data.time_ms,
                y=plot_data.data_matrix,
                channel=plot_data.channel_id,
                sweep_index=sweep_info["sweep_index"],
                channel_type=channel_type,
                title=plot_labels["title"],
                x_label=plot_labels["x_label"],
                y_label=plot_labels["y_label"],
                channel_config=channel_config,  # Pass config for cursor text units
            )

            # Add prominent gridlines at x=0 and y=0
            add_zero_axis_lines(self.plot_manager.ax, alpha=0.4, linewidth=0.8)
            self.plot_manager.redraw()

            self._sync_cursors_to_plot()
        else:
            logger.debug(f"Could not load sweep {sweep}: {result.error_message}")

    def _generate_analysis(self):
            """
            Generate and display an analysis plot using the controller.

            Validates data availability, retrieves analysis parameters, performs analysis,
            and displays results in a dedicated dialog. Handles errors and empty results gracefully.
            """
            if not self.controller.has_data():
                QMessageBox.warning(self, "No Data", "Please load a data file first.")
                return

            params = self.control_panel.get_parameters()

            # Get current units from loaded file metadata
            dataset = self.controller.current_dataset
            channel_config = dataset.metadata.get("channel_config")
            if not channel_config:
                logger.warning("No channel configuration found - using default units")
                current_units = "pA"
            else:
                current_units = channel_config.get("current_units", "pA")

            # Add current units from metadata to parameters
            params = params.with_updates(
                channel_config={
                    **params.channel_config,
                    "current_units": current_units,
                }
            )

            result = self.controller.perform_analysis(params)

            if not result.success:
                QMessageBox.critical(
                    self, "Analysis Failed", f"Analysis failed:\n{result.error_message}"
                )
                return

            analysis_result = result.data

            if not analysis_result or not analysis_result.x_data.size:
                QMessageBox.warning(
                    self, "No Results", "No data available for selected parameters."
                )
                return

            plot_data = {
                "x_data": analysis_result.x_data,
                "y_data": analysis_result.y_data,
                "sweep_indices": analysis_result.sweep_indices,
                "use_dual_range": analysis_result.use_dual_range,
            }

            if analysis_result.use_dual_range and hasattr(analysis_result, "y_data2"):
                plot_data["y_data2"] = analysis_result.y_data2
                plot_data["y_label_r1"] = getattr(
                    analysis_result, "y_label_r1", analysis_result.y_label
                )
                plot_data["y_label_r2"] = getattr(
                    analysis_result, "y_label_r2", analysis_result.y_label
                )

            if self.analysis_dialog:
                self.analysis_dialog.close()

            # Pass the parameters with units and file path to the dialog
            self.analysis_dialog = AnalysisPlotDialog(
                self, plot_data, params, self.current_file_path, self.controller
            )
            self.analysis_dialog.show()
            self.analysis_completed.emit()

    def _sweep_extraction(self):
        """Open the sweep extraction dialog."""
        if not self.controller.has_data():
            QMessageBox.warning(self, "No Data", "Please load a data file first.")
            return
        
        # Get current dataset and file path
        dataset = self.controller.current_dataset
        
        # Import the dialog (lazy import to avoid circular dependencies)
        from data_analysis_gui.dialogs.extract_sweeps_dialog import SweepExtractorDialog
        
        # Create and show dialog
        dialog = SweepExtractorDialog(self, dataset, self.current_file_path)
        dialog.exec()

    def _export_data(self):
        """
        Export analysis data using the controller.

        Presents a file dialog for export location, performs export, and displays success or error messages.
        """
        if not self.controller.has_data():
            QMessageBox.warning(self, "No Data", "Please load a data file first.")
            return

        # Get parameters
        params = self.control_panel.get_parameters()

        # Get suggested filename
        suggested = self.controller.get_suggested_export_filename(params)

        file_path = self.file_dialog_service.get_export_path(
            parent=self,
            suggested_name=suggested,
            default_directory=None,  # Let service use its memory
            file_types="CSV files (*.csv);;All files (*.*)",
            dialog_type="export_analysis",
        )

        if not file_path:
            return

        # Export using controller
        result = self.controller.export_analysis_data(params, file_path)

        if result.success:
            QMessageBox.information(
                self,
                "Success",
                f"Exported {result.records_exported} records to {Path(file_path).name}",
            )
            # Auto-save settings to persist the directory choice
            self._auto_save_settings()
        else:
            QMessageBox.critical(self, "Export Failed", result.error_message)

    def _batch_analyze(self):
        """
        Open the batch analysis dialog.

        Passes current analysis parameters and batch processor to the dialog for batch processing.
        """
        # Get current parameters
        params = self.control_panel.get_parameters()

        # Open batch dialog with shared batch processor
        dialog = BatchAnalysisDialog(self, self.batch_processor, params)
        dialog.show()

    def _toggle_dual_range(self, enabled):
        """
        Toggle dual range cursors on the plot.

        Args:
            enabled (bool): True to enable dual range, False to disable.
        """
        if enabled:
            vals = self.control_panel.get_range_values()
            self.plot_manager.toggle_dual_range(
                True, vals.get("range2_start", 600), vals.get("range2_end", 900)
            )
        else:
            self.plot_manager.toggle_dual_range(False, 0, 0)

    def _sync_cursors_to_plot(self):
        """
        Synchronize cursor positions from the control panel to the plot manager.
        """
        vals = self.control_panel.get_range_values()
        self.plot_manager.update_range_lines(
            vals["range1_start"],
            vals["range1_end"],
            vals["use_dual_range"],
            vals.get("range2_start"),
            vals.get("range2_end"),
        )

    def _auto_save_settings(self):
        """
        Automatically save current user settings whenever they change.

        Ensures user preferences are preserved across sessions. Silent on failure.
        """
        try:
            settings = extract_settings_from_main_window(self)
            save_session_settings(settings)
            logger.debug("Auto-saved settings")
        except Exception as e:
            logger.warning(f"Failed to auto-save settings: {e}")
            # Don't show error to user for auto-save failures

    def _sync_cursor_to_control(self, line_id, position):
        """
        Synchronize cursor position from the plot to the control panel.

        Args:
            line_id (str): Identifier for the cursor line.
            position (float): New position value for the cursor.
        """
        if line_id and position is not None:
            mapping = {
                "range1_start": "start1",
                "range1_end": "end1",
                "range2_start": "start2",
                "range2_end": "end2",
            }
            if line_id in mapping:
                self.control_panel.update_range_value(mapping[line_id], position)

    def _on_cursor_moved(self, action, line_id, position):
        """
        Handle cursor movement events from the plot manager.

        Args:
            action (str): Type of cursor action (e.g., "dragged").
            line_id (str): Identifier for the cursor line.
            position (float): New position value for the cursor.
        """
        if action == "dragged":
            self._sync_cursor_to_control(line_id, position)

    # Navigation methods
    def _start_navigation(self, direction):
        """
        Start continuous sweep navigation in the specified direction.

        Args:
            direction (callable): Function to invoke for navigation.
        """
        direction()
        self.navigation_direction = direction
        self.hold_timer.start(150)

    def _stop_navigation(self):
        """
        Stop continuous sweep navigation.
        """
        self.hold_timer.stop()
        self.navigation_direction = None

    def _continue_navigation(self):
        """
        Continue sweep navigation while navigation button is held.
        """
        if self.navigation_direction:
            self.navigation_direction()

    def _next_sweep(self):
        """
        Navigate to the next sweep in the sweep combo box.
        """
        idx = self.sweep_combo.currentIndex()
        if idx < self.sweep_combo.count() - 1:
            self.sweep_combo.setCurrentIndex(idx + 1)

    def _prev_sweep(self):
        """
        Navigate to the previous sweep in the sweep combo box.
        """
        idx = self.sweep_combo.currentIndex()
        if idx > 0:
            self.sweep_combo.setCurrentIndex(idx - 1)
