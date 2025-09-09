# dialogs/batch_dialog.py

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QListWidget, QProgressBar, QLabel, QCheckBox,
                             QSpinBox, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class BatchAnalysisWorker(QThread):
    """Worker thread for batch analysis."""
    progress = pyqtSignal(int, int, str)
    file_complete = pyqtSignal(object)  # FileAnalysisResult
    finished = pyqtSignal(object)  # BatchAnalysisResult
    
    def __init__(self, batch_service, file_paths, params, parallel, max_workers):
        super().__init__()
        self.batch_service = batch_service
        self.file_paths = file_paths
        self.params = params
        self.parallel = parallel
        self.max_workers = max_workers
    
    def run(self):
        """Run batch analysis in thread."""
        self.batch_service.on_progress = lambda c, t, n: self.progress.emit(c, t, n)
        self.batch_service.on_file_complete = lambda r: self.file_complete.emit(r)
        
        result = self.batch_service.analyze_files(
            self.file_paths, self.params, 
            self.parallel, self.max_workers
        )
        
        self.finished.emit(result)


class BatchAnalysisDialog(QDialog):
    """Dialog for batch analysis with progress tracking."""
    
    def __init__(self, parent, batch_service, params):
        super().__init__(parent)
        self.batch_service = batch_service
        self.params = params
        self.file_paths = []
        self.worker = None
        
        self.setWindowTitle("Batch Analysis")
        self.setModal(False)  # Non-modal to allow interaction
        self.setMinimumWidth(600)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # File list
        self.file_list = QListWidget()
        layout.addWidget(QLabel("Files to analyze:"))
        layout.addWidget(self.file_list)
        
        # File selection buttons
        file_button_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("Add Files...")
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.clear_all_btn = QPushButton("Clear All")
        file_button_layout.addWidget(self.add_files_btn)
        file_button_layout.addWidget(self.remove_selected_btn)
        file_button_layout.addWidget(self.clear_all_btn)
        layout.addLayout(file_button_layout)
        
        # Processing options
        options_layout = QHBoxLayout()
        self.parallel_checkbox = QCheckBox("Parallel Processing")
        self.parallel_checkbox.setChecked(True)
        options_layout.addWidget(self.parallel_checkbox)
        
        options_layout.addWidget(QLabel("Max Workers:"))
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setRange(1, 16)
        self.workers_spinbox.setValue(4)
        options_layout.addWidget(self.workers_spinbox)
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
        self.view_results_btn = QPushButton("View Results")
        self.view_results_btn.setEnabled(False)
        
        button_box.addButton(self.analyze_btn, QDialogButtonBox.AcceptRole)
        button_box.addButton(self.cancel_btn, QDialogButtonBox.RejectRole)
        button_box.addButton(self.view_results_btn, QDialogButtonBox.ActionRole)
        layout.addWidget(button_box)
        
        # Connect signals
        self.add_files_btn.clicked.connect(self.add_files)
        self.remove_selected_btn.clicked.connect(self.remove_selected)
        self.clear_all_btn.clicked.connect(self.clear_files)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        self.view_results_btn.clicked.connect(self.view_results)