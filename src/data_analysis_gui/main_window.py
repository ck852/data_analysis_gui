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
from typing import Optional, Set
import re

from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
    QMessageBox, QSplitter, QToolBar, QStatusBar, QLabel, QCheckBox,
    QComboBox, QDialog, QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QKeySequence, QAction

# Import refactored theme functions
from data_analysis_gui.config.themes import (apply_modern_theme, create_styled_button, style_combo_box,
                                            style_label, style_checkbox
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
from data_analysis_gui.widgets.sweep_navigation_panel import SweepNavigationPanel
from data_analysis_gui.plot_manager import PlotManager

# Dialog imports
from data_analysis_gui.dialogs.analysis_plot_dialog import AnalysisPlotDialog
from data_analysis_gui.dialogs.batch_dialog import BatchAnalysisDialog
from data_analysis_gui.dialogs.bg_subtraction_dialog import BackgroundSubtractionDialog
from data_analysis_gui.dialogs.ramp_iv_dialog import RampIVDialog
from data_analysis_gui.dialogs import ConcentrationResponseDialog

# Service imports
from data_analysis_gui.gui_services import FileDialogService
from data_analysis_gui.gui_services.main_range_coordinator import MainRangeCoordinator 

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

    # Feature flags for experimental/in-development features
    ENABLE_BG_BATCH = False  # Batch Analysis with Background Subtraction
    ENABLE_CONC_RESPONSE = True  # Concentration Response Analysis

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

        self.rejected_sweeps: Set[int] = set()  # Track rejected sweep indices

        # Splitter auto-save timer
        self.splitter_save_timer = QTimer()
        self.splitter_save_timer.setSingleShot(True)
        self.splitter_save_timer.setInterval(500)  # 500ms delay after dragging stops
        self.splitter_save_timer.timeout.connect(self._auto_save_settings)

        # Initialize default values for settings that may be loaded
        self.last_channel_view = "Voltage"
        self.last_directory = None

        # Build UI (this calls _connect_signals internally)
        self._init_ui()
        
        # Initialize range coordinator AFTER UI is built
        # (needs control_panel and plot_manager to exist)
        self.range_coordinator = MainRangeCoordinator(
            self.control_panel, 
            self.plot_manager
        )
        
        # Connect coordinator signals (must be after coordinator creation)
        self._connect_coordinator_signals()

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

        # Main splitter - store as instance variable for settings persistence
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        main_layout.addWidget(self.splitter)

        # Control panel (left)
        self.control_panel = ControlPanel()
        self.splitter.addWidget(self.control_panel)

        # Plot manager (right)
        self.plot_manager = PlotManager(file_dialog_service=self.file_dialog_service)
        self.splitter.addWidget(self.plot_manager.get_plot_widget())

        # Set the plot manager to expand and leave the control panel at its minimum size
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
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

        # Batch Analysis
        self.batch_action = QAction("&Batch Analyze...", self)
        self.batch_action.setShortcut("Ctrl+B")
        self.batch_action.triggered.connect(self._batch_analyze)
        self.batch_action.setEnabled(True)
        analysis_menu.addAction(self.batch_action)

        # Batch Analysis with Background Subtraction
        self.batch_bg_action = QAction("Batch Analyze with BG Subtraction...", self)
        self.batch_bg_action.triggered.connect(self._batch_analyze_with_bg_subtraction)
        self.batch_bg_action.setEnabled(True)
        self.batch_bg_action.setVisible(self.ENABLE_BG_BATCH) # Experimental feature flag
        analysis_menu.addAction(self.batch_bg_action)

        analysis_menu.addSeparator()

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
        conc_resp_action.setVisible(self.ENABLE_CONC_RESPONSE) # Experimental feature flag

        # Sweep Extractor
        sweep_extract_action = analysis_menu.addAction("Extract Sweeps...")
        sweep_extract_action.triggered.connect(self._sweep_extraction)

        # About button (no submenu)
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about_dialog)
        menubar.addAction(about_action)

    def _open_concentration_response(self):
        """Open the concentration-response analysis dialog."""
        dialog = ConcentrationResponseDialog(self)
        dialog.exec()

    def _background_subtraction(self):
        if not self.controller.has_data():
            self._show_no_data_warning()
            return
        
        sweep = self.sweep_nav_panel.get_current_sweep()
        if not sweep:
            return
        
        dialog = BackgroundSubtractionDialog(
            dataset=self.controller.current_dataset,
            sweep_index=sweep,
            parent=self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Refresh plot
            self._update_plot()
            self.status_bar.showMessage("Background subtraction applied", 3000)

    def _ramp_iv_analysis(self):
        """Open the ramp IV analysis dialog."""
        if not self.controller.has_data():
            self._show_no_data_warning()
            return
        
        # Validate analysis range using ControlPanel's public method
        is_valid, error_msg = self.control_panel.validate_ranges()
        if not is_valid:
            QMessageBox.warning(self, "Invalid Analysis Range", error_msg)
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

    def _show_no_data_warning(self):
        """Display a warning that no data file is loaded."""
        QMessageBox.warning(self, "No Data", "Please load a data file first.")

    def _create_toolbar(self):
        """
        Construct the main toolbar with themed controls.

        Toolbar provides file operations, sweep navigation, channel selection,
        current units, file information, cursor centering, sweep rejection,
        and channel toggling. All controls are styled and sized for a compact,
        modern appearance.
        """
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # File operations
        open_action = toolbar.addAction("Open", self._open_file)
        toolbar.addSeparator()

        # Channel selection
        channel_label = QLabel("Channel:")
        style_label(channel_label, "normal")
        toolbar.addWidget(channel_label)

        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Voltage", "Current"])
        self.channel_combo.setEnabled(True)
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        style_combo_box(self.channel_combo)
        self.channel_combo.setMinimumWidth(100)
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

        # Center Cursor Button
        self.center_cursor_btn = create_styled_button(
            "Center Nearest Cursor", "secondary"
        )
        self.center_cursor_btn.setToolTip(
            "Moves the nearest cursor to the center of the view"
        )
        self.center_cursor_btn.clicked.connect(self._center_nearest_cursor)
        self.center_cursor_btn.setEnabled(True)
        toolbar.addWidget(self.center_cursor_btn)

        toolbar.addSeparator()

        # Sweep Navigation Panel (replaces individual controls)
        self.sweep_nav_panel = SweepNavigationPanel()
        toolbar.addWidget(self.sweep_nav_panel)

        toolbar.addSeparator()

        # Reject Sweep Checkbox
        self.reject_sweep_cb = QCheckBox("Reject Sweep")
        self.reject_sweep_cb.setToolTip(
            "Exclude this sweep from analysis (Generate Analysis Plot and Export Analysis Data)"
        )
        self.reject_sweep_cb.setEnabled(False)  # Disabled until file loaded
        self.reject_sweep_cb.stateChanged.connect(self._on_reject_sweep_toggled)
        style_checkbox(self.reject_sweep_cb)
        toolbar.addWidget(self.reject_sweep_cb)

        # Connect toolbar controls to auto-save
        self.channel_combo.currentTextChanged.connect(self._auto_save_settings)

    def _connect_signals(self):
        """
        Connect UI signals to their respective logic handlers.
        
        NOTE: This is called from _init_ui(), so it cannot reference
        self.range_coordinator (which doesn't exist yet).
        Coordinator signals are connected separately in _connect_coordinator_signals().
        """
        # Connect sweep navigation panel
        self.sweep_nav_panel.sweep_changed.connect(self._on_sweep_changed)

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

        # DIAGNOSTIC: Add logging wrappers to see if signals fire
        logger.info(f"DIAGNOSTIC: Range 2 spinboxes exist? start={hasattr(self.control_panel, 'start_spin2')}, end={hasattr(self.control_panel, 'end_spin2')}")
        logger.info(f"DIAGNOSTIC: Range 2 spinbox types: start={type(self.control_panel.start_spin2).__name__}, end={type(self.control_panel.end_spin2).__name__}")
        
        def debug_range2_start_changed(value):
            logger.info(f"DIAGNOSTIC: Range 2 START changed to {value}")
            self._auto_save_settings()
        
        def debug_range2_end_changed(value):
            logger.info(f"DIAGNOSTIC: Range 2 END changed to {value}")
            self._auto_save_settings()
        
        # Connect Range 2 spinboxes directly with debug wrappers
        self.control_panel.start_spin2.valueChanged.connect(debug_range2_start_changed)
        self.control_panel.end_spin2.valueChanged.connect(debug_range2_end_changed)
        
        logger.info("DIAGNOSTIC: Connected Range 2 spinboxes to debug handlers")

        # Splitter - debounced auto-save when user adjusts position
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

    def _on_splitter_moved(self):
        """
        Handle splitter movement with debouncing.
        
        Restarts the timer on each movement, so settings only save
        after the user has stopped dragging for 500ms.
        """
        self.splitter_save_timer.start()

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
                
                # Update batch_import to match current import_data location
                # (so batch dialogs start near your current work)
                dirs = self.file_dialog_service.get_last_directories()
                dirs['batch_import'] = dirs.get('import_data')
                self.file_dialog_service.set_last_directories(dirs)
                
                self.file_loaded.emit(file_path)
                
                # Auto-save settings to persist the directory choice
                self._auto_save_settings()
            # Error handling is done by controller callbacks

    def _batch_analyze_with_bg_subtraction(self):
        """
        Open batch analysis with background subtraction workflow.
        
        First opens the background subtraction dialog in batch mode to get
        the background range, then proceeds to batch analysis with that range.
        """
        # Show experimental feature warning
        QMessageBox.warning(
            self,
            "Experimental Feature",
            "Warning: This feature has not been tested.\nFor experimental use only."
        )
        
        # Check if file is loaded
        if not self.controller.has_data():
            self._show_no_data_warning()
            return
        
        # Validate analysis range using ControlPanel's public method
        is_valid, error_msg = self.control_panel.validate_ranges()
        if not is_valid:
            QMessageBox.warning(self, "Invalid Analysis Range", error_msg)
            return
        
        # Get current sweep for preview
        sweep = self.sweep_nav_panel.get_current_sweep()
        if not sweep:
            QMessageBox.warning(self, "No Sweep", "No sweep selected.")
            return
        
        # Open background subtraction dialog in batch mode
        bg_dialog = BackgroundSubtractionDialog(
            dataset=self.controller.current_dataset,
            sweep_index=sweep,
            parent=self,
            batch_mode=True  # Special mode for batch workflow
        )
        
        if bg_dialog.exec() == QDialog.DialogCode.Accepted:
            # Get the background range that was defined
            bg_range = bg_dialog.get_background_range()
            
            # Get current analysis parameters
            params = self.control_panel.get_parameters()
            
            # Open batch dialog with background subtraction enabled
            dialog = BatchAnalysisDialog(
                parent=self,
                batch_service=self.batch_processor,
                params=params,
                bg_subtraction_range=bg_range  # Pass the BG range
            )
            dialog.show()

    def _center_nearest_cursor(self):
        """
        Handle center cursor button click.
        
        Delegates to PlotManager, then uses coordinator to sync the result.
        """
        line_id, position = self.plot_manager.center_nearest_cursor()
        if line_id and position:
            # Coordinator will handle syncing to spinbox via its signal connection
            # No explicit sync needed here
            pass

    def _connect_coordinator_signals(self):
        """
        Connect range coordinator signals.
        
        Called from __init__ after range_coordinator is created.
        Separated from _connect_signals() because the coordinator
        doesn't exist when _connect_signals() is called.
        """
        # Range coordinator handles analysis/export requests
        self.range_coordinator.analysis_requested.connect(self._generate_analysis)
        self.range_coordinator.export_requested.connect(self._export_data)
        
        # NEW: Auto-save when cursor operations complete
        self.range_coordinator.settings_changed.connect(self._auto_save_settings)

        # Connect toolbar's plot_saved signal to auto-save settings
        self.plot_manager.toolbar.plot_saved.connect(self._auto_save_settings)

        self.plot_manager.welcome_clicked.connect(self._open_file)

    def _on_file_loaded(self, file_info: FileInfo):
        """
        Respond to successful file load and update UI components.

        Updates file labels, sweep count, revalidates ranges,
        populates the sweep selection combo box, and clears rejected sweeps.
        """
        # Clear welcome message on first file load
        self.plot_manager.clear_welcome_state()
        
        # Clear rejected sweeps for new file
        self.rejected_sweeps.clear()
        
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

        # Enable channel selection
        self.channel_combo.setEnabled(True)
        self.reject_sweep_cb.setEnabled(True)  # Enable reject checkbox

        # Set max time bound for X-axis zoom limiting
        if file_info.max_sweep_time:
            self.plot_manager.set_max_time_bound(file_info.max_sweep_time)

        # Reset plot manager for new file - clears view state so first sweep autoscales
        self.plot_manager.reset_for_new_file()

        # Initialize sweep navigation panel
        self.sweep_nav_panel.set_sweep_list(file_info.sweep_names)
        
        # Set sweep timing data from dataset metadata
        dataset = self.controller.current_dataset
        sweep_times = dataset.metadata.get("sweep_times", {})
        self.sweep_nav_panel.set_sweep_times(sweep_times)
        
        # Enable the navigation panel
        self.sweep_nav_panel.set_enabled(True)

        # Show first sweep
        if file_info.sweep_names:
            self.sweep_nav_panel.set_current_sweep(file_info.sweep_names[0])

    def _on_reject_sweep_toggled(self, state):
        """
        Handle the reject sweep checkbox toggle.
        
        Updates the rejected_sweeps set based on the current sweep selection.
        
        Args:
            state: Qt check state (Qt.CheckState.Checked or Qt.CheckState.Unchecked)
        """
        sweep = self.sweep_nav_panel.get_current_sweep()
        if not sweep:
            return
        
        try:
            sweep_idx = int(sweep)
        except ValueError:
            logger.warning(f"Could not parse sweep index: {sweep}")
            return
        
        if state == Qt.CheckState.Checked.value:
            self.rejected_sweeps.add(sweep_idx)
            logger.debug(f"Rejected sweep {sweep_idx}")
        else:
            self.rejected_sweeps.discard(sweep_idx)
            logger.debug(f"Un-rejected sweep {sweep_idx}")

    def _on_sweep_changed(self, sweep_index: str):
        """
        Update the plot when the sweep selection changes.
        Also updates the reject sweep checkbox state.
        
        Args:
            sweep_index: The new sweep index as a string (e.g., "0", "1", "2")
        """
        # Update reject checkbox to match current sweep's rejection state
        if sweep_index:
            try:
                sweep_idx = int(sweep_index)
                # Block signals to prevent triggering toggle handler
                self.reject_sweep_cb.blockSignals(True)
                self.reject_sweep_cb.setChecked(sweep_idx in self.rejected_sweeps)
                self.reject_sweep_cb.blockSignals(False)
            except ValueError:
                pass
        
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

        sweep = self.sweep_nav_panel.get_current_sweep()
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

            # Sync cursors and spinboxes (coordinator handles this now)
            self.range_coordinator.sync_cursors_to_spinboxes()
        else:
            logger.debug(f"Could not load sweep {sweep}: {result.error_message}")

    def _generate_analysis(self):
        """
        Generate and display an analysis plot using the controller.

        Validates data availability, retrieves analysis parameters, performs analysis
        excluding rejected sweeps, and displays results in a dedicated dialog.
        Handles errors and empty results gracefully.
        """
        if not self.controller.has_data():
            self._show_no_data_warning()
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

        # Pass rejected sweeps to controller
        result = self.controller.perform_analysis(params, rejected_sweeps=self.rejected_sweeps)

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

        # Pass AnalysisManager and dataset explicitly instead of controller
        self.analysis_dialog = AnalysisPlotDialog(
            parent=self,
            plot_data=plot_data,
            params=params,
            file_path=self.current_file_path,
            analysis_manager=self.analysis_manager,
            dataset=self.controller.current_dataset,
        )
        self.analysis_dialog.show()
        self.analysis_completed.emit()

    def _sweep_extraction(self):
        """Open the sweep extraction dialog."""
        if not self.controller.has_data():
            self._show_no_data_warning()
            return
        
        # Validate analysis range using ControlPanel's public method
        is_valid, error_msg = self.control_panel.validate_ranges()
        if not is_valid:
            QMessageBox.warning(self, "Invalid Analysis Range", error_msg)
            return
        
        # Get current dataset and file path
        dataset = self.controller.current_dataset
        
        # Get Range 1 values from control panel
        range_values = self.control_panel.get_range_values()
        default_start = range_values.get('range1_start', 0.0)
        default_end = range_values.get('range1_end', dataset.get_max_sweep_time())
        
        # Import the dialog (lazy import to avoid circular dependencies)
        from data_analysis_gui.dialogs.extract_sweeps_dialog import SweepExtractorDialog
        
        # Create and show dialog with default time range from Range 1
        dialog = SweepExtractorDialog(self, dataset, self.current_file_path, 
                                    default_start=default_start, 
                                    default_end=default_end)
        dialog.exec()

    def _export_data(self):
        """
        Export analysis data using the controller, excluding rejected sweeps.

        Presents a file dialog for export location, performs export, and displays
        success or error messages.
        """
        if not self.controller.has_data():
            self._show_no_data_warning()
            return

        # Get parameters
        params = self.control_panel.get_parameters()

        # Get suggested filename
        suggested = self.controller.get_suggested_export_filename(params)
        
        # Use current file's directory as fallback for first export
        fallback = str(Path(self.current_file_path).parent) if self.current_file_path else None

        file_path = self.file_dialog_service.get_export_path(
            parent=self,
            suggested_name=suggested,
            default_directory=fallback,
            file_types="CSV files (*.csv);;All files (*.*)",
            dialog_type="export",
        )

        if not file_path:
            return

        # Export using controller, passing rejected sweeps
        result = self.controller.export_analysis_data(
            params, file_path, rejected_sweeps=self.rejected_sweeps
        )

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
        # Check if file is loaded
        if not self.controller.has_data():
            self._show_no_data_warning()
            return
        
        # Validate analysis range using ControlPanel's public method
        is_valid, error_msg = self.control_panel.validate_ranges()
        if not is_valid:
            QMessageBox.warning(self, "Invalid Analysis Range", error_msg)
            return
        
        # Get current parameters
        params = self.control_panel.get_parameters()

        # Open batch dialog with shared batch processor
        dialog = BatchAnalysisDialog(self, self.batch_processor, params)
        dialog.show()

    def _auto_save_settings(self):
        """
        Automatically save current user settings whenever they change.
        """
        try:
            settings = extract_settings_from_main_window(self)
            
            # DIAGNOSTIC: Log what we're about to save
            if 'analysis' in settings:
                logger.info(f"AUTO-SAVE triggered - Range 2 values: "
                        f"start={settings['analysis'].get('range2_start')}, "
                        f"end={settings['analysis'].get('range2_end')}, "
                        f"dual={settings['analysis'].get('use_dual_range')}")
            
            save_session_settings(settings)
            logger.debug("Auto-saved settings")
        except Exception as e:
            logger.warning(f"Failed to auto-save settings: {e}")

    def _show_about_dialog(self):
        """
        Display the About dialog with application information.
        """
        version = QApplication.applicationVersion()
        
        about_text = f"""<h3>PatchBatch v{version}</h3>
    <p>© 2025 Ralph C Kissell<br>
    Licensed under MIT License</p>

    <p><b>This software uses:</b><br>
    • PySide6 (Qt for Python) © The Qt Company Ltd.<br>
    &nbsp;&nbsp;Licensed under GNU LGPLv3<br>
    &nbsp;&nbsp;<a href="https://www.qt.io/qt-for-python">https://www.qt.io/qt-for-python</a></p>

    """
        
        QMessageBox.about(self, "About PatchBatch", about_text)
