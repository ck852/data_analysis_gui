# dialogs/batch_dialog.py

from pathlib import Path
from typing import List, Optional
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QListWidget, QProgressBar, QLabel, QCheckBox,
                             QDialogButtonBox, QMessageBox,
                             QAbstractItemView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from data_analysis_gui.gui_services import FileDialogService
from data_analysis_gui.core.models import FileAnalysisResult, BatchAnalysisResult
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)

# Default number of workers for parallel processing
DEFAULT_MAX_WORKERS = 4


class BatchAnalysisWorker(QThread):
    """Worker thread for batch analysis."""
    progress = pyqtSignal(int, int, str)
    file_complete = pyqtSignal(object)  # FileAnalysisResult
    finished = pyqtSignal(object)  # BatchAnalysisResult
    error = pyqtSignal(str)
    
    def __init__(self, batch_service, file_paths, params, parallel):
        super().__init__()
        self.batch_service = batch_service
        self.file_paths = file_paths
        self.params = params
        self.parallel = parallel
    
    def run(self):
        """Run batch analysis in thread."""
        try:
            # Set up progress callbacks
            self.batch_service.on_progress = lambda c, t, n: self.progress.emit(c, t, n)
            self.batch_service.on_file_complete = lambda r: self.file_complete.emit(r)
            
            result = self.batch_service.analyze_files(
                self.file_paths, 
                self.params, 
                self.parallel, 
                DEFAULT_MAX_WORKERS  # Use hardcoded default
            )
            
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}", exc_info=True)
            self.error.emit(str(e))


class BatchAnalysisDialog(QDialog):
    """Dialog for batch analysis with progress tracking."""
    
    def __init__(self, parent, batch_service, params):
        super().__init__(parent)
        self.batch_service = batch_service
        self.params = params
        self.file_paths = []
        self.worker = None
        self.batch_result = None
        
        # Initialize services
        self.file_dialog_service = FileDialogService()
        
        self.setWindowTitle("Batch Analysis")
        self.setModal(False)  # Non-modal to allow interaction
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # File list
        layout.addWidget(QLabel("Files to analyze:"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.file_list)
        
        # File count label
        self.file_count_label = QLabel("0 files selected")
        layout.addWidget(self.file_count_label)
        
        # File selection buttons
        file_button_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("Add Files...")
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.clear_all_btn = QPushButton("Clear All")
        file_button_layout.addWidget(self.add_files_btn)
        file_button_layout.addWidget(self.remove_selected_btn)
        file_button_layout.addWidget(self.clear_all_btn)
        file_button_layout.addStretch()
        layout.addLayout(file_button_layout)
        
        # Processing options (simplified - only parallel checkbox)
        options_layout = QHBoxLayout()
        self.parallel_checkbox = QCheckBox("Parallel Processing")
        self.parallel_checkbox.setChecked(True)
        options_layout.addWidget(self.parallel_checkbox)
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Progress section
        layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_box = QDialogButtonBox()
        self.analyze_btn = QPushButton("Start Analysis")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.view_results_btn = QPushButton("View Results")
        self.view_results_btn.setEnabled(False)
        self.close_btn = QPushButton("Close")
        
        button_box.addButton(self.analyze_btn, QDialogButtonBox.AcceptRole)
        button_box.addButton(self.cancel_btn, QDialogButtonBox.RejectRole)
        button_box.addButton(self.view_results_btn, QDialogButtonBox.ActionRole)
        button_box.addButton(self.close_btn, QDialogButtonBox.RejectRole)
        layout.addWidget(button_box)
        
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
        """Add files to the batch list."""
        file_types = (
            "Data files (*.mat *.abf);;"
            "MAT files (*.mat);;"
            "ABF files (*.abf);;"
            "All files (*.*)"
        )
        
        # Get default directory from parent if available
        default_dir = None
        if hasattr(self.parent(), 'current_file_path'):
            current_path = self.parent().current_file_path
            if current_path:
                default_dir = str(Path(current_path).parent)
        
        # Get multiple files
        file_paths = self.file_dialog_service.get_import_paths(
            self, 
            "Select Files for Batch Analysis",
            default_dir,
            file_types
        )
        
        if file_paths:
            # Add to list, avoiding duplicates
            for file_path in file_paths:
                if file_path not in self.file_paths:
                    self.file_paths.append(file_path)
                    self.file_list.addItem(Path(file_path).name)
            
            self.update_file_count()
            self.update_button_states()
    
    def remove_selected(self):
        """Remove selected files from the batch list."""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        # Remove in reverse order to maintain indices
        for item in reversed(selected_items):
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            del self.file_paths[row]
        
        self.update_file_count()
        self.update_button_states()
    
    def clear_files(self):
        """Clear all files from the batch list."""
        self.file_list.clear()
        self.file_paths.clear()
        self.update_file_count()
        self.update_button_states()
    
    def update_file_count(self):
        """Update the file count label."""
        count = len(self.file_paths)
        self.file_count_label.setText(f"{count} file{'s' if count != 1 else ''} selected")
    
    def update_button_states(self):
        """Update button enabled states based on current state."""
        has_files = len(self.file_paths) > 0
        is_running = self.worker is not None and self.worker.isRunning()
        has_results = self.batch_result is not None
        
        self.add_files_btn.setEnabled(not is_running)
        self.remove_selected_btn.setEnabled(not is_running and has_files)
        self.clear_all_btn.setEnabled(not is_running and has_files)
        self.analyze_btn.setEnabled(not is_running and has_files)
        self.cancel_btn.setEnabled(is_running)
        self.view_results_btn.setEnabled(has_results)
        self.parallel_checkbox.setEnabled(not is_running)
    
    def start_analysis(self):
        """Start the batch analysis."""
        if not self.file_paths:
            QMessageBox.warning(self, "No Files", "Please add files to analyze.")
            return
        
        # Reset progress
        self.progress_bar.setMaximum(len(self.file_paths))
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting analysis...")
        
        # Create and start worker thread
        self.worker = BatchAnalysisWorker(
            self.batch_service,
            self.file_paths.copy(),  # Copy list to avoid modifications
            self.params,
            self.parallel_checkbox.isChecked()
        )
        
        # Connect worker signals
        self.worker.progress.connect(self.on_progress)
        self.worker.file_complete.connect(self.on_file_complete)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_error)
        
        # Start analysis
        self.worker.start()
        
        # Update UI state
        self.update_button_states()
        logger.info(f"Started batch analysis of {len(self.file_paths)} files")
    
    def cancel_analysis(self):
        """Cancel the running analysis."""
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
            self.status_label.setText("Analysis cancelled")
            self.update_button_states()
    
    def on_progress(self, completed, total, current_file):
        """Handle progress updates from worker."""
        self.progress_bar.setValue(completed)
        self.status_label.setText(f"Processing {current_file} ({completed}/{total})")
    
    def on_file_complete(self, result: FileAnalysisResult):
        """Handle completion of individual file."""
        status = "✓" if result.success else "✗"
        logger.debug(f"{status} Completed: {result.base_name}")
    
    def on_analysis_finished(self, result: BatchAnalysisResult):
        """Handle completion of batch analysis."""
        self.batch_result = result
        
        # Update status (no popup dialog)
        success_count = len(result.successful_results)
        fail_count = len(result.failed_results)
        total_time = result.processing_time
        
        if fail_count > 0:
            status_msg = (
                f"Complete: {success_count} succeeded, "
                f"{fail_count} failed in {total_time:.1f}s"
            )
        else:
            status_msg = (
                f"Complete: {success_count} files analyzed in {total_time:.1f}s"
            )
        
        self.status_label.setText(status_msg)
        self.update_button_states()
        logger.info(f"Batch analysis complete: {result.success_rate:.1f}% success rate")
    
    def on_error(self, error_msg):
        """Handle errors from worker."""
        QMessageBox.critical(self, "Analysis Error", f"Batch analysis failed:\n{error_msg}")
        self.status_label.setText("Analysis failed")
        self.update_button_states()
    
    def view_results(self):
        """Open the results window."""
        if not self.batch_result:
            return
        
        try:
            # Import here to avoid circular dependencies
            from data_analysis_gui.dialogs.batch_results_window import BatchResultsWindow
            from data_analysis_gui.services.plot_service import PlotService
            
            # Create plot service
            plot_service = PlotService()
            
            # Create and show results window
            results_window = BatchResultsWindow(
                self,
                self.batch_result,
                self.batch_service,
                plot_service,
                self.parent().controller.export_service  # Pass export service
            )
            results_window.show()
            
        except Exception as e:
            logger.error(f"Failed to show results: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to display results:\n{str(e)}")