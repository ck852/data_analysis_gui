"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Batch analysis dialog with threaded processing and progress tracking.

This module provides a comprehensive dialog interface for performing batch analysis
on multiple electrophysiology data files. It includes file selection management,
threaded analysis processing with real-time progress updates, and integrated
results viewing capabilities.

Classes:
    - BatchAnalysisWorker: QThread subclass for background batch processing
    - BatchAnalysisDialog: Main dialog with file management and progress tracking

Features:
    - Interactive file list management (add, remove, clear operations)
    - Background threaded analysis with progress bar and status updates
    - Real-time processing feedback and error handling
    - Integrated results viewing and export capabilities
    - Modern themed UI with responsive button states
    - Cancellable analysis operations with proper cleanup
"""

from pathlib import Path
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QProgressBar, QLabel, QMessageBox, QAbstractItemView,
                                QGroupBox)
from PySide6.QtCore import QThread, Signal

from data_analysis_gui.gui_services import FileDialogService
from data_analysis_gui.core.models import FileAnalysisResult, BatchAnalysisResult
from data_analysis_gui.dialogs.batch_results_window import BatchResultsWindow
from data_analysis_gui.config.logging import get_logger

# Updated imports for refactored themes.py
from data_analysis_gui.config.themes import (apply_modern_theme, apply_compact_layout, style_list_widget, style_progress_bar, style_group_box,
                                            style_label, get_file_count_color, create_styled_button)

logger = get_logger(__name__)


class BatchAnalysisWorker(QThread):
    """
    Worker thread for performing batch analysis.

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

    def __init__(self, batch_service, file_paths, params):
        super().__init__()
        self.batch_service = batch_service
        self.file_paths = file_paths
        self.params = params

    def run(self):
        """
        Run batch analysis in a separate thread.

        Sets up progress and completion callbacks, processes files, and emits results.
        """
        try:
            # Set up progress callbacks
            self.batch_service.on_progress = lambda c, t, n: self.progress.emit(c, t, n)
            self.batch_service.on_file_complete = lambda r: self.file_complete.emit(r)

            # Simple sequential processing
            result = self.batch_service.process_files(self.file_paths, self.params)

            # Ensure result has selection state initialized
            if not hasattr(result, "selected_files") or result.selected_files is None:
                from dataclasses import replace

                result = replace(
                    result,
                    selected_files={r.base_name for r in result.successful_results},
                )

            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}", exc_info=True)
            self.error.emit(str(e))


class BatchAnalysisDialog(QDialog):
    """
    Dialog for batch analysis with file selection, progress tracking, and result viewing.

    Provides:
        - File list management (add, remove, clear).
        - Progress bar and status updates.
        - Start/cancel analysis.
        - View results after completion.
    """

    def __init__(self, parent, batch_service, params):
        """
        Initialize the BatchAnalysisDialog.

        Args:
            parent: Parent widget.
            batch_service: Service for batch analysis.
            params: Analysis parameters.
        """
        super().__init__(parent)
        self.batch_service = batch_service
        self.params = params
        self.file_paths = []
        self.worker = None
        self.batch_result = None

        # Share the file dialog service from parent instead of creating new instance
        # This ensures directory memory is shared across the application
        if hasattr(parent, 'file_dialog_service'):
            self.file_dialog_service = parent.file_dialog_service
        else:
            # Fallback to new instance if parent doesn't have one
            self.file_dialog_service = FileDialogService()

        # Set window title before applying theme
        self.setWindowTitle("Batch Analysis")

        self.init_ui()

        # Apply theme and layout helpers from refactored themes.py
        apply_modern_theme(self)
        apply_compact_layout(self)

    def init_ui(self):
        """
        Initialize the user interface, including file list, progress bar, and buttons.
        """
        layout = QVBoxLayout(self)

        # File List Section
        file_group = QGroupBox("Files to Analyze")
        style_group_box(file_group)
        file_group_layout = QVBoxLayout(file_group)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        style_list_widget(self.file_list)
        file_group_layout.addWidget(self.file_list)

        self.file_count_label = QLabel("0 files selected")
        style_label(
            self.file_count_label, "muted"
        )  # Updated to use style_label with type
        file_group_layout.addWidget(self.file_count_label)

        layout.addWidget(file_group)

        # File Management Buttons
        file_button_layout = QHBoxLayout()
        self.add_files_btn = create_styled_button("Add Files...", "secondary", self)
        self.remove_selected_btn = create_styled_button(
            "Remove Selected", "secondary", self
        )
        self.clear_all_btn = create_styled_button("Clear All", "secondary", self)

        file_button_layout.addWidget(self.add_files_btn)
        file_button_layout.addWidget(self.remove_selected_btn)
        file_button_layout.addWidget(self.clear_all_btn)
        file_button_layout.addStretch()
        layout.addLayout(file_button_layout)

        # Progress Section
        progress_group = QGroupBox("Progress")
        style_group_box(progress_group)
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        style_progress_bar(self.progress_bar)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        style_label(self.status_label, "muted")  # Updated to use style_label with type
        progress_layout.addWidget(self.status_label)

        layout.addWidget(progress_group)

        # Action Buttons
        button_layout = QHBoxLayout()

        self.analyze_btn = create_styled_button("Start Analysis", "primary", self)
        self.cancel_btn = create_styled_button("Cancel", "secondary", self)
        self.cancel_btn.setEnabled(False)
        self.view_results_btn = create_styled_button("View Results", "accent", self)
        self.view_results_btn.setEnabled(False)
        self.close_btn = create_styled_button("Close", "secondary", self)

        button_layout.addWidget(self.analyze_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.view_results_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        # Connect signals
        self.add_files_btn.clicked.connect(self.add_files)
        self.remove_selected_btn.clicked.connect(self.remove_selected)
        self.clear_all_btn.clicked.connect(self.clear_files)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        self.view_results_btn.clicked.connect(self.view_results)
        self.close_btn.clicked.connect(self.close)

        # Update button states
        self.update_button_states()

    def add_files(self):
        """
        Add files to the batch list via file dialog.
        """
        file_types = (
            "WCP files (*.wcp);;"
            "ABF files (*.abf);;"
            "Data files (*.wcp *.abf);;"
            "All files (*.*)"
        )

        # Use service's directory memory - no manual default_dir logic
        file_paths = self.file_dialog_service.get_import_paths(
            self, 
            "Select Files for Batch Analysis", 
            default_directory=None,  # Let service use its memory
            file_types=file_types,
            dialog_type="import_batch"  # Separate memory for batch imports
        )

        if file_paths:
            for file_path in file_paths:
                if file_path not in self.file_paths:
                    self.file_paths.append(file_path)
                    self.file_list.addItem(Path(file_path).name)

            self.update_file_count()
            self.update_button_states()
            
            # Trigger auto-save on parent to persist directory choice
            if hasattr(self.parent(), '_auto_save_settings'):
                try:
                    self.parent()._auto_save_settings()
                except Exception as e:
                    logger.warning(f"Failed to auto-save settings from batch dialog: {e}")

    def remove_selected(self):
        """
        Remove selected files from the batch list.
        """
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return

        for item in reversed(selected_items):
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            del self.file_paths[row]

        self.update_file_count()
        self.update_button_states()

    def clear_files(self):
        """
        Clear all files from the batch list.
        """
        self.file_list.clear()
        self.file_paths.clear()
        self.update_file_count()
        self.update_button_states()

    def update_file_count(self):
        """
        Update the file count label and color based on number of files selected.
        """
        count = len(self.file_paths)
        self.file_count_label.setText(
            f"{count} file{'s' if count != 1 else ''} selected"
        )

        color = get_file_count_color(count)
        self.file_count_label.setStyleSheet(f"color: {color};")

    def update_button_states(self):
        """
        Update enabled/disabled state of action buttons based on current state.
        """
        has_files = len(self.file_paths) > 0
        is_running = self.worker is not None and self.worker.isRunning()
        has_results = self.batch_result is not None

        self.add_files_btn.setEnabled(not is_running)
        self.remove_selected_btn.setEnabled(not is_running and has_files)
        self.clear_all_btn.setEnabled(not is_running and has_files)
        self.analyze_btn.setEnabled(not is_running and has_files)
        self.cancel_btn.setEnabled(is_running)
        self.view_results_btn.setEnabled(has_results)

    def start_analysis(self):
        """
        Start the batch analysis process in a worker thread.
        """
        if not self.file_paths:
            QMessageBox.warning(self, "No Files", "Please add files to analyze.")
            return

        # Reset progress
        self.progress_bar.setMaximum(len(self.file_paths))
        self.progress_bar.setValue(0)
        style_label(self.status_label, "info")  # Updated to use style_label with type
        self.status_label.setText("Starting analysis...")

        # Create and start worker thread
        self.worker = BatchAnalysisWorker(
            self.batch_service, self.file_paths.copy(), self.params
        )

        # Connect worker signals
        self.worker.progress.connect(self.on_progress)
        self.worker.file_complete.connect(self.on_file_complete)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_error)

        # Start analysis
        self.worker.start()
        self.update_button_states()
        logger.info(f"Started batch analysis of {len(self.file_paths)} files")

    def cancel_analysis(self):
        """
        Cancel the running batch analysis.
        """
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
            style_label(
                self.status_label, "warning"
            )  # Updated to use style_label with type
            self.status_label.setText("Analysis cancelled")
            self.update_button_states()

    def on_progress(self, completed, total, current_file):
        """
        Handle progress updates from worker thread.

        Args:
            completed (int): Number of files completed.
            total (int): Total number of files.
            current_file (str): Name of current file being processed.
        """
        self.progress_bar.setValue(completed)
        style_label(self.status_label, "info")  # Updated to use style_label with type
        self.status_label.setText(f"Processing {current_file} ({completed}/{total})")

    def on_file_complete(self, result: FileAnalysisResult):
        """
        Handle completion of individual file analysis.

        Args:
            result (FileAnalysisResult): Result of file analysis.
        """
        status = "✔" if result.success else "✗"
        logger.debug(f"{status} Completed: {result.base_name}")

    def on_analysis_finished(self, result: BatchAnalysisResult):
        """
        Handle completion of batch analysis.

        Args:
            result (BatchAnalysisResult): Result of batch analysis.
        """
        self.batch_result = result

        success_count = len(result.successful_results)
        fail_count = len(result.failed_results)
        total_time = result.processing_time

        if fail_count > 0:
            status_msg = f"Complete: {success_count} succeeded, {fail_count} failed in {total_time:.1f}s"
            style_label(
                self.status_label, "warning"
            )  # Updated to use style_label with type
        else:
            status_msg = (
                f"Complete: {success_count} files analyzed in {total_time:.1f}s"
            )
            style_label(
                self.status_label, "success"
            )  # Updated to use style_label with type

        self.status_label.setText(status_msg)
        self.update_button_states()
        logger.info(f"Batch analysis complete: {result.success_rate:.1f}% success rate")

    def on_error(self, error_msg):
        """
        Handle errors emitted from worker thread.

        Args:
            error_msg (str): Error message.
        """
        QMessageBox.critical(
            self, "Analysis Error", f"Batch analysis failed:\n{error_msg}"
        )
        style_label(self.status_label, "error")  # Updated to use style_label with type
        self.status_label.setText("Analysis failed")
        self.update_button_states()

    def view_results(self):
        """
        Open the batch results window to display analysis results.

        Raises:
            Exception: If results window fails to open.
        """
        if not self.batch_result:
            return

        try:
            results_window = BatchResultsWindow(
                self,
                self.batch_result,
                self.batch_service,
                self.parent().controller.data_service,
            )
            results_window.show()

        except Exception as e:
            logger.error(f"Failed to show results: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to display results:\n{str(e)}")
