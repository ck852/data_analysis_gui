"""
Ramp IV Analysis Dialog for PatchBatch Electrophysiology Data Analysis Tool

This module provides dialog interfaces for voltage ramp IV curve analysis.
It includes a voltage input dialog and the main ramp IV analysis dialog
with sweep selection, plotting, and CSV export capabilities.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import json
from typing import Optional, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QWidget, QMessageBox, QFormLayout, QGroupBox, QSplitter, QApplication, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from data_analysis_gui.config.themes import apply_modern_theme, create_styled_button, style_group_box
from data_analysis_gui.config.plot_style import (
    apply_plot_style, style_axis, get_line_styles, COLORS, COLOR_CYCLE, add_zero_axis_lines
)
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.services.ramp_iv_service import RampIVService, RampIVResult
from data_analysis_gui.services.data_manager import DataManager
from data_analysis_gui.gui_services import FileDialogService, ClipboardService
from data_analysis_gui.widgets.sweep_select_list import SweepSelectionWidget

from data_analysis_gui.services.ramp_iv_batch_processor import RampIVBatchProcessor
from data_analysis_gui.dialogs.ramp_iv_batch_sweep_dialog import RampIVBatchSweepDialog
from data_analysis_gui.dialogs.batch_results_window import BatchResultsWindow
from data_analysis_gui.core.models import FileAnalysisResult

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


def _get_ramp_iv_settings_file():
    """Get the ramp IV settings file path."""
    from data_analysis_gui.core.session_settings import get_settings_dir
    return get_settings_dir() / "ramp_iv_settings.json"

def _load_ramp_iv_voltages():
    """Load saved ramp IV voltages."""
    try:
        settings_file = _get_ramp_iv_settings_file()
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                data = json.load(f)
                return data.get('voltages', "-80, -60, -40, -20, 0, 20, 40, 60, 80")
    except:
        pass
    return "-80, -60, -40, -20, 0, 20, 40"

def _save_ramp_iv_voltages(voltage_text):
    """Save ramp IV voltages."""
    try:
        settings_file = _get_ramp_iv_settings_file()
        with open(settings_file, 'w') as f:
            json.dump({'voltages': voltage_text}, f)
        return True
    except:
        return False

class VoltageInputDialog(QDialog):
    """
    Simple dialog for entering target voltages as comma-separated values.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Enter Target Voltages")
        self.setModal(True)
        self.setFixedSize(450, 250)
        
        self._init_ui()
        apply_modern_theme(self)
        
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Instructions
        instructions = QLabel(
            "Enter target voltages separated by commas.\n"
            "These are the voltages at which current will be extracted.\n\n"
            "Example: -80, -60, -40, -20, 0, 20, 40, 60, 80"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Input field
        self.voltage_input = QTextEdit()
        self.voltage_input.setMaximumHeight(80)
        # Load saved voltages or use defaults
        self.voltage_input.setPlainText(_load_ramp_iv_voltages())
        layout.addWidget(self.voltage_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = create_styled_button("Cancel", "secondary")
        self.ok_button = create_styled_button("OK", "primary")
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        layout.addLayout(button_layout)
        
        # Connect signals
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button.clicked.connect(self._validate_and_accept)
        
        # Select all text initially
        self.voltage_input.selectAll()
        self.voltage_input.setFocus()
        
    def _validate_and_accept(self):
        """Validate voltage input and accept if valid."""
        voltage_text = self.voltage_input.toPlainText().strip()
        
        if not voltage_text:
            QMessageBox.warning(self, "Input Error", "Please enter voltage values.")
            return
            
        is_valid, voltages, error_msg = RampIVService.validate_voltage_targets(voltage_text)
        
        if not is_valid:
            QMessageBox.warning(self, "Invalid Input", error_msg)
            return
            
        self.voltages = voltages
        _save_ramp_iv_voltages(voltage_text)
        
        self.accept()
        
    def get_voltages(self) -> List[float]:
        """Get the validated voltage list."""
        return getattr(self, 'voltages', [])

class RampIVBatchWorker(QThread):
    """
    Worker thread for performing batch ramp IV analysis.

    Emits:
        progress (int, int, str): Progress update (completed, total, filename).
        file_complete (FileAnalysisResult): Signal when a file is processed.
        finished (BatchAnalysisResult): Signal when batch is complete.
        error (str): Signal on error.
    """

    progress = Signal(int, int, str)
    file_complete = Signal(object)  # FileAnalysisResult
    finished = Signal(object)  # BatchAnalysisResult
    error = Signal(str)

    def __init__(
        self,
        file_paths,
        voltage_targets,
        start_ms,
        end_ms,
        current_units,
        sweep_mode,
        selected_sweeps,
    ):
        super().__init__()
        self.file_paths = file_paths
        self.voltage_targets = voltage_targets
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.current_units = current_units
        self.sweep_mode = sweep_mode
        self.selected_sweeps = selected_sweeps

    def run(self):
        """Run batch ramp IV analysis in a separate thread."""
        try:
            processor = RampIVBatchProcessor()

            # Set up progress callbacks
            processor.on_progress = lambda c, t, n: self.progress.emit(c, t, n)
            processor.on_file_complete = lambda r: self.file_complete.emit(r)

            # Process files
            result = processor.process_files(
                file_paths=self.file_paths,
                voltage_targets=self.voltage_targets,
                start_ms=self.start_ms,
                end_ms=self.end_ms,
                current_units=self.current_units,
                sweep_selection_mode=self.sweep_mode,
                selected_sweeps=self.selected_sweeps,
            )

            # Ensure result has selection state initialized
            if not hasattr(result, "selected_files") or result.selected_files is None:
                from dataclasses import replace

                result = replace(
                    result,
                    selected_files={r.base_name for r in result.successful_results},
                )

            self.finished.emit(result)

        except Exception as e:
            logger.error(f"Batch ramp IV analysis failed: {e}", exc_info=True)
            self.error.emit(str(e))


class RampIVDialog(QDialog):
    """
    Main dialog for ramp IV analysis with plot display and export capabilities.
    
    Follows the pattern of bg_subtraction_dialog.py for structure and theming,
    with plotting capabilities similar to analysis_plot_dialog.py.
    """
    
    def __init__(self, dataset: ElectrophysiologyDataset, 
                start_ms: float, end_ms: float,
                current_units: str = "pA",
                parent=None):
        super().__init__(parent)
        
        self.dataset = dataset
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.current_units = current_units
        
        # Services - follow existing pattern
        self.data_manager = DataManager()

        if hasattr(parent, 'file_dialog_service'):
            self.file_dialog_service = parent.file_dialog_service
        else:
            # Fallback to new instance if parent doesn't have one
            self.file_dialog_service = FileDialogService()
        
        # State
        self.voltage_targets = []
        self.current_result: Optional[RampIVResult] = None
        self.analysis_completed = False  # NEW: Track if analysis has been run
        self.batch_worker = None  # NEW: Track batch worker thread
        
        self.setWindowTitle("Ramp IV Analysis")
        self.setModal(True)
        
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.resize(int(avail.width() * 0.9), int(avail.height() * 0.9))
        fg = self.frameGeometry()
        fg.moveCenter(avail.center())
        self.move(fg.topLeft())
        
        # Apply global plot style for consistency
        apply_plot_style()
        
        # Get sweep names sorted numerically like other dialogs
        self.sweep_names = sorted(dataset.sweeps(), 
                                key=lambda x: int(x) if x.isdigit() else 0)
            
        self._init_ui()
        self._connect_signals()
        
        # Apply theme last, following existing pattern
        apply_modern_theme(self)
        
    def show_with_voltage_input(self):
        """Show the dialog, first getting voltage targets."""
        if self._get_voltage_targets():
            # Update the voltage label now that we have the targets
            voltages_str = ", ".join([f"{v:+.0f}" for v in self.voltage_targets])
            self.voltage_label.setText(voltages_str)
            self.show()
        else:
            # User cancelled voltage input, don't show main dialog
            self.close()

    def _get_voltage_targets(self) -> bool:
        """Get voltage targets from user input dialog."""
        dialog = VoltageInputDialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.voltage_targets = dialog.get_voltages()
            logger.info(f"Ramp IV voltage targets set: {self.voltage_targets}")
            return True
        
        return False
        
    def _init_ui(self):
        """Initialize the user interface following bg_subtraction_dialog pattern."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Create splitter for controls and plot
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Controls
        controls_panel = self._create_controls_panel()
        splitter.addWidget(controls_panel)
        
        # Right panel: Plot
        plot_panel = self._create_plot_panel()
        splitter.addWidget(plot_panel)
        
        # Set splitter sizes (30% controls, 70% plot)
        splitter.setSizes([300, 700])
        layout.addWidget(splitter)
        
        # Bottom buttons
        self._create_bottom_buttons(layout)
        
    def _create_controls_panel(self) -> QWidget:
        """Create the left control panel with analysis settings and sweep selection."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 10, 0)
        
        # Analysis info group - show current settings
        info_group = QGroupBox("Analysis Settings")
        style_group_box(info_group)
        info_layout = QFormLayout(info_group)
        
        # Display current settings (read-only)
        info_layout.addRow("Analysis Range (ms):", QLabel(f"{self.start_ms:.1f} - {self.end_ms:.1f}"))
        
        # Display voltage targets (create label but will update after voltage input)
        self.voltage_label = QLabel("")
        self.voltage_label.setWordWrap(True)
        info_layout.addRow("Target Voltages (mV):", self.voltage_label)
        
        layout.addWidget(info_group)
        
        # Sweep selection group
        sweep_group = QGroupBox(f"Sweep Selection ({len(self.sweep_names)} available)")
        style_group_box(sweep_group)
        sweep_layout = QVBoxLayout(sweep_group)
        
        self.sweep_selection = SweepSelectionWidget(self.sweep_names)
        sweep_layout.addWidget(self.sweep_selection)
        
        # Selection control buttons
        selection_buttons = QHBoxLayout()
        select_all_btn = create_styled_button("Select All", "secondary")
        select_none_btn = create_styled_button("Select None", "secondary")
        
        select_all_btn.clicked.connect(lambda: self.sweep_selection.select_all(True))
        select_none_btn.clicked.connect(lambda: self.sweep_selection.select_all(False))
        
        selection_buttons.addWidget(select_all_btn)
        selection_buttons.addWidget(select_none_btn)
        selection_buttons.addStretch()
        sweep_layout.addLayout(selection_buttons)
        
        layout.addWidget(sweep_group)
        
        # Analysis button - primary action
        self.generate_plot_btn = create_styled_button("Generate Analysis Plot", "primary")
        self.generate_plot_btn.setMinimumHeight(35)
        layout.addWidget(self.generate_plot_btn)
        
        # Status label for feedback
        self.status_label = QLabel("Ready to analyze")
        self.status_label.setStyleSheet("color: #6C757D; font-style: italic;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        return panel
        
    def _create_plot_panel(self) -> QWidget:
        """Create the plot panel following matplotlib integration patterns."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure with consistent theming
        self.figure = Figure(figsize=(8, 6), facecolor=COLORS["light"])
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Add toolbar for plot interaction
        toolbar = NavigationToolbar(self.canvas, panel)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        
        # Initial empty plot
        self._setup_empty_plot()
        
        return panel
        
    def _create_bottom_buttons(self, layout):
        """Create bottom action buttons with batch analysis support."""
        button_layout = QHBoxLayout()
        
        # Export button (initially disabled until analysis is done)
        self.export_btn = create_styled_button("Export Summary CSV...", "accent")
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)

        # Copy button (initially disabled until analysis is done)
        self.copy_btn = create_styled_button("Copy Data", "secondary")
        self.copy_btn.setEnabled(False)
        button_layout.addWidget(self.copy_btn)
        
        # NEW: Batch analyze button (disabled until single analysis is done)
        self.batch_analyze_btn = create_styled_button("Batch Analyze...", "primary")
        self.batch_analyze_btn.setEnabled(False)
        button_layout.addWidget(self.batch_analyze_btn)
        
        button_layout.addStretch()
        
        # Close button
        close_btn = create_styled_button("Close", "secondary")
        button_layout.addWidget(close_btn)
        close_btn.clicked.connect(self.close)
        
        layout.addLayout(button_layout)
        
    def _connect_signals(self):
        """Connect UI signals to their handlers."""
        self.generate_plot_btn.clicked.connect(self._generate_analysis_plot)
        self.export_btn.clicked.connect(self._export_summary_csv)
        self.copy_btn.clicked.connect(self._copy_data_to_clipboard)
        self.batch_analyze_btn.clicked.connect(self._batch_analyze)
        
    def _setup_empty_plot(self):
        """Set up initial empty plot with consistent styling."""
        self.ax.clear()
        
        # Use centralized plot styling
        style_axis(
            self.ax,
            title="Ramp IV Analysis",
            xlabel="Voltage (mV)",
            ylabel=f"Current ({self.current_units})",
            remove_top_right=True
        )
        
        # Add placeholder text
        self.ax.text(0.5, 0.5, "Click 'Generate Analysis Plot' to begin analysis",
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=12, alpha=0.6, style='italic')
                    
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw_idle()
        
    def _generate_analysis_plot(self):
        """Generate the ramp IV analysis plot using the service."""
        # Handle new tuple return from get_selected_sweeps
        selected_sweeps, invalid_sweeps = self.sweep_selection.get_selected_sweeps()
        
        # Show warning if invalid sweeps were requested
        if invalid_sweeps:
            QMessageBox.warning(
                self, "Invalid Sweeps",
                f"Sweep(s) {', '.join(invalid_sweeps)} not found in file.\n"
                f"Proceeding with valid sweeps only."
            )
        
        if not selected_sweeps:
            QMessageBox.warning(
                self, "No Sweeps Selected", 
                "Please select at least one sweep from the table or enter a valid range."
            )
            return
            
        # Update status
        self.status_label.setText("Analyzing...")
        self.generate_plot_btn.setEnabled(False)
        
        try:
            # Perform analysis using the ramp IV service
            self.current_result = RampIVService.extract_ramp_iv_data(
                dataset=self.dataset,
                selected_sweeps=selected_sweeps,
                target_voltages=self.voltage_targets,
                start_ms=self.start_ms,
                end_ms=self.end_ms,
                current_units=self.current_units
            )
            
            if not self.current_result.success:
                QMessageBox.critical(self, "Analysis Failed", 
                                f"Analysis failed: {self.current_result.error_message}")
                self.status_label.setText("Analysis failed")
                return
                
            # Update plot with results
            self._update_plot()
            
            # Enable export and batch analyze buttons
            self.export_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
            self.analysis_completed = True  # NEW: Mark analysis as completed
            self.batch_analyze_btn.setEnabled(True)  # NEW: Enable batch button
            
            # Update status
            processed = len(self.current_result.processed_sweeps)
            failed = len(self.current_result.failed_sweeps)
            self.status_label.setText(f"Analysis complete: {processed} sweeps processed, {failed} failed")
            
            logger.info(f"Ramp IV analysis completed: {processed} sweeps processed")
            
        except Exception as e:
            logger.error(f"Error during ramp IV analysis: {e}")
            QMessageBox.critical(self, "Analysis Error", 
                            f"An error occurred during analysis:\n{str(e)}")
            self.status_label.setText("Analysis error")
        finally:
            self.generate_plot_btn.setEnabled(True)
                               
    def _update_plot(self):
        """Update the plot with analysis results using centralized styling."""
        if not self.current_result or not self.current_result.success:
            return
            
        self.ax.clear()
        
        # Get plot data arrays from service
        plot_data = RampIVService.get_plot_data_arrays(self.current_result)
        
        if not plot_data:
            self.ax.text(0.5, 0.5, "No valid data points found",
                        ha='center', va='center', transform=self.ax.transAxes,
                        fontsize=12, alpha=0.6, color='red')
        else:
            # Plot each sweep using consistent color cycling
            colors = COLOR_CYCLE
            line_styles = get_line_styles()
            
            for i, (sweep_idx, (voltages, currents)) in enumerate(plot_data.items()):
                color = colors[i % len(colors)]
                
                self.ax.plot(voltages, currents, 'o-', 
                        label=f"Sweep {sweep_idx}",
                        color=color, markersize=6, linewidth=2, 
                        alpha=0.8, markeredgewidth=0)
                        
            # Add legend if multiple sweeps, following existing style
            if len(plot_data) > 1:
                self.ax.legend(
                    loc='best', 
                    frameon=True,
                    fancybox=False,
                    shadow=False,
                    framealpha=0.95,
                    edgecolor="#D0D0D0",
                    facecolor="white",
                    fontsize=9
                )
        
        # Add prominent gridlines at x=0 and y=0
        add_zero_axis_lines(self.ax)
                
        # Apply consistent axis styling
        style_axis(
            self.ax,
            title="Ramp IV Analysis",
            xlabel="Voltage (mV)",
            ylabel=f"Current ({self.current_units})",
            remove_top_right=True
        )
        
        # Add some padding for better visualization
        self.ax.margins(x=0.02, y=0.05)
        
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw()
        
    def _export_summary_csv(self):
        """Export the analysis results to CSV using existing data manager."""
        if not self.current_result or not self.current_result.success:
            QMessageBox.warning(self, "No Data", "No analysis results to export.")
            return
            
        # Get export file path using file dialog service
        file_path = self.file_dialog_service.get_export_path(
            parent=self,
            suggested_name="ramp_iv_summary.csv",
            file_types="CSV files (*.csv)",
            dialog_type="export"
        )
        
        if not file_path:
            return
            
        try:
            # Prepare export table using service
            export_table = RampIVService.prepare_export_table(self.current_result)
            
            # Export using existing data manager
            result = self.data_manager.export_to_csv(export_table, file_path)
            
            if result.success:
                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {result.records_exported} records to CSV file.\n"
                    f"File: {file_path.split('/')[-1]}"
                )
                logger.info(f"Ramp IV data exported to {file_path}")
                if hasattr(self.parent(), '_auto_save_settings'):
                    try:
                        self.parent()._auto_save_settings()
                    except Exception:
                        pass
            else:
                QMessageBox.critical(self, "Export Failed", 
                                   f"Export failed: {result.error_message}")
                                   
        except Exception as e:
            logger.error(f"Error during ramp IV export: {e}")
            QMessageBox.critical(self, "Export Error", 
                               f"An error occurred during export:\n{str(e)}")
            
    def _copy_data_to_clipboard(self):
        """
        Copy the analysis results to clipboard as tab-separated values.
        
        Allows users to paste data directly into Excel, Prism, or other applications
        without needing to save a CSV file first.
        """
        if not self.current_result or not self.current_result.success:
            QMessageBox.warning(self, "No Data", "No analysis results to copy.")
            return
            
        try:
            # Prepare export table using service (same as CSV export)
            export_table = RampIVService.prepare_export_table(self.current_result)
            
            # Copy to clipboard
            success = ClipboardService.copy_data_to_clipboard(export_table)
            
            if success:
                logger.info("Ramp IV data copied to clipboard")
                                    
        except Exception as e:
            logger.error(f"Error during ramp IV clipboard copy: {e}")
            QMessageBox.critical(
                self, "Copy Error", 
                f"An error occurred while copying:\n{str(e)}"
            )

    def _batch_analyze(self):
        """Launch batch analysis with current ramp IV parameters."""
        if not self.analysis_completed:
            QMessageBox.warning(
                self, "Analysis Required",
                "Please run a single-file analysis first to verify your parameters."
            )
            return
        
        # Get current sweep selection
        selected_sweeps, _ = self.sweep_selection.get_selected_sweeps()
        
        # Show sweep selection dialog
        sweep_dialog = RampIVBatchSweepDialog(self, selected_sweeps)
        if sweep_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        sweep_mode, sweep_list = sweep_dialog.get_selection()
        
        # Get files to process
        file_types = (
            "WCP files (*.wcp);;"
            "ABF files (*.abf);;"
            "Data files (*.wcp *.abf);;"
            "All files (*.*)"
        )
        
        file_paths = self.file_dialog_service.get_import_paths(
            self,
            "Select Files for Batch Ramp IV Analysis",
            file_types=file_types,
            dialog_type="batch_import"
        )
        
        if not file_paths:
            return
        
        # Create progress dialog
        self.batch_progress_dialog = QDialog(self)
        self.batch_progress_dialog.setWindowTitle("Batch Ramp IV Analysis")
        self.batch_progress_dialog.setModal(True)
        self.batch_progress_dialog.setFixedSize(500, 150)
        
        progress_layout = QVBoxLayout(self.batch_progress_dialog)
        
        self.batch_progress_label = QLabel("Starting batch analysis...")
        progress_layout.addWidget(self.batch_progress_label)
        
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setMaximum(len(file_paths))
        self.batch_progress_bar.setValue(0)
        progress_layout.addWidget(self.batch_progress_bar)
        
        self.batch_status_label = QLabel("Preparing...")
        progress_layout.addWidget(self.batch_status_label)
        
        button_layout = QHBoxLayout()
        self.batch_cancel_btn = create_styled_button("Cancel", "secondary")
        self.batch_cancel_btn.clicked.connect(self._cancel_batch_analysis)
        button_layout.addStretch()
        button_layout.addWidget(self.batch_cancel_btn)
        progress_layout.addLayout(button_layout)
        
        # Create and start worker thread
        self.batch_worker = RampIVBatchWorker(
            file_paths=file_paths,
            voltage_targets=self.voltage_targets,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            current_units=self.current_units,
            sweep_mode=sweep_mode,
            selected_sweeps=sweep_list,
        )
        
        # Connect worker signals
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.file_complete.connect(self._on_batch_file_complete)
        self.batch_worker.finished.connect(self._on_batch_complete)
        self.batch_worker.error.connect(self._on_batch_error)
        
        # Start worker and show dialog
        self.batch_worker.start()
        self.batch_progress_dialog.show()
        
        logger.info(f"Started batch ramp IV analysis of {len(file_paths)} files")


    def _cancel_batch_analysis(self):
        """Cancel the running batch analysis."""
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.quit()
            self.batch_worker.wait()
            self.batch_status_label.setText("Analysis cancelled")
            logger.info("Batch ramp IV analysis cancelled by user")


    def _on_batch_progress(self, completed: int, total: int, current_file: str):
        """Handle progress updates from batch worker."""
        self.batch_progress_bar.setValue(completed)
        self.batch_progress_label.setText(f"Processing file {completed} of {total}")
        self.batch_status_label.setText(f"Current: {current_file}")


    def _on_batch_file_complete(self, result: FileAnalysisResult):
        """Handle completion of individual file analysis."""
        status = "✓" if result.success else "✗"
        logger.debug(f"{status} Completed: {result.base_name}")


    def _on_batch_complete(self, batch_result):
        """Handle completion of batch analysis and open results window."""
        self.batch_progress_dialog.close()
        
        success_count = len(batch_result.successful_results)
        fail_count = len(batch_result.failed_results)
        
        logger.info(
            f"Batch ramp IV analysis complete: {success_count} succeeded, "
            f"{fail_count} failed"
        )
        
        if success_count == 0:
            QMessageBox.warning(
                self, "No Results",
                "Batch analysis completed but no files were successfully processed."
            )
            return
        
        # Open results window
        try:
            # Get data_service from parent if available
            data_service = None
            batch_service = None
            
            if hasattr(self.parent(), 'controller'):
                data_service = self.parent().controller.data_service
                # Create batch service for exports
                from data_analysis_gui.services.batch_processor import BatchProcessor
                batch_service = BatchProcessor()
            else:
                # Fallback to DataManager
                data_service = self.data_manager
                from data_analysis_gui.services.batch_processor import BatchProcessor
                batch_service = BatchProcessor()
            
            results_window = BatchResultsWindow(
                self,
                batch_result,
                batch_service,
                data_service,
            )
            results_window.show()
            
        except Exception as e:
            logger.error(f"Failed to show batch results: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Error",
                f"Failed to display results:\n{str(e)}"
            )


    def _on_batch_error(self, error_msg: str):
        """Handle errors from batch worker."""
        self.batch_progress_dialog.close()
        QMessageBox.critical(
            self, "Batch Analysis Error",
            f"Batch analysis failed:\n{error_msg}"
        )
        logger.error(f"Batch ramp IV analysis error: {error_msg}")