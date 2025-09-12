"""
Dialog for entering slow capacitance values for current density calculations.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, Optional
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTableWidget, QTableWidgetItem, QLabel, 
                             QDialogButtonBox, QMessageBox, QHeaderView,
                             QLineEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

from data_analysis_gui.core.models import BatchAnalysisResult
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class CurrentDensityDialog(QDialog):
    """Dialog for entering Cslow values for each file."""
    
    def __init__(self, parent, batch_result: BatchAnalysisResult):
        super().__init__(parent)
        self.batch_result = batch_result
        self.cslow_inputs = {}  # filename -> QLineEdit
        
        self.setWindowTitle("Current Density Analysis - Enter Cslow Values")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Enter slow capacitance (Cslow) values in picofarads (pF) for each file.\n"
            "Current density will be calculated as Current (pA) / Cslow (pF)."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Table for file names and Cslow inputs
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["File", "Cslow (pF)", "Status"])
        
        # Configure table
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(2, 80)
        
        # Populate table
        self._populate_table()
        
        layout.addWidget(self.table)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Default value button
        set_all_btn = QPushButton("Set All to:")
        self.default_input = QLineEdit("18.0")
        self.default_input.setMaximumWidth(80)
        self.default_input.setValidator(QDoubleValidator(0.01, 10000.0, 2))
        
        button_layout.addWidget(set_all_btn)
        button_layout.addWidget(self.default_input)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Connect signals
        set_all_btn.clicked.connect(self._set_all_values)
    
    def _populate_table(self):
        """Populate table with files and input fields."""
        # Get successful results sorted by filename
        results = sorted(
            self.batch_result.successful_results,
            key=lambda r: self._extract_number(r.base_name)
        )
        
        # Filter to only show selected files if selection state exists
        if hasattr(self.batch_result, 'selected_files') and self.batch_result.selected_files:
            results = [r for r in results if r.base_name in self.batch_result.selected_files]

        self.table.setRowCount(len(results))
        
        for row, result in enumerate(results):
            # File name (read-only)
            file_item = QTableWidgetItem(result.base_name)
            file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, file_item)
            
            # Cslow input
            cslow_input = QLineEdit()
            cslow_input.setValidator(QDoubleValidator(0.01, 10000.0, 2))
            cslow_input.textChanged.connect(lambda _, r=row: self._update_status(r))
            cslow_input.setStyleSheet("QLineEdit { padding: 2px; }")
            self.table.setCellWidget(row, 1, cslow_input)
            self.cslow_inputs[result.base_name] = cslow_input
            
            # Status (initially empty)
            status_item = QTableWidgetItem("")
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, status_item)
    
    def _extract_number(self, filename: str) -> int:
        """Extract number from filename for sorting."""
        import re
        match = re.search(r'_(\d+)', filename)
        if match:
            return int(match.group(1))
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])
        return 0
    
    def _update_status(self, row: int):
        """Update status indicator for a row."""
        cslow_input = self.table.cellWidget(row, 1)
        status_item = self.table.item(row, 2)
        
        if cslow_input and status_item:
            text = cslow_input.text().strip()
            if text and self._is_valid_number(text):
                status_item.setText("✓")
                status_item.setForeground(Qt.darkGreen)
            else:
                status_item.setText("")
    
    def _is_valid_number(self, text: str) -> bool:
        """Check if text is a valid positive number."""
        try:
            value = float(text)
            return value > 0
        except ValueError:
            return False
    
    def _set_all_values(self):
        """Set all Cslow values to the default value."""
        default_value = self.default_input.text().strip()
        if not self._is_valid_number(default_value):
            QMessageBox.warning(
                self, 
                "Invalid Value", 
                "Please enter a valid positive number for the default value."
            )
            return
        
        for cslow_input in self.cslow_inputs.values():
            cslow_input.setText(default_value)
    
    def _validate_and_accept(self):
        """Validate all inputs before accepting."""
        missing_files = []
        
        for filename, cslow_input in self.cslow_inputs.items():
            text = cslow_input.text().strip()
            if not text or not self._is_valid_number(text):
                missing_files.append(filename)
        
        if missing_files:
            QMessageBox.warning(
                self,
                "Missing Values",
                f"Please enter valid Cslow values for all files.\n"
                f"Missing: {len(missing_files)} file(s)"
            )
            return
        
        self.accept()
    
    def get_cslow_mapping(self) -> Dict[str, float]:
        """
        Get the mapping of filenames to Cslow values.
        
        Returns:
            Dictionary mapping filename to Cslow value in pF
        """
        mapping = {}
        
        for filename, cslow_input in self.cslow_inputs.items():
            text = cslow_input.text().strip()
            if text and self._is_valid_number(text):
                mapping[filename] = float(text)
        
        return mapping
    
    def keyPressEvent(self, event):
        """Handle keyboard events for copy/paste support."""
        if event.matches(event.StandardKey.Paste):
            self._handle_paste()
        else:
            super().keyPressEvent(event)
    
    def _handle_paste(self):
        """Handle paste operation for bulk input."""
        from PyQt5.QtWidgets import QApplication
        
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        
        if not text:
            return
        
        # Split by newlines and filter valid numbers
        lines = text.strip().split('\n')
        values = []
        
        for line in lines:
            line = line.strip()
            if self._is_valid_number(line):
                values.append(line)
        
        # Apply values to inputs starting from current focus
        current_widget = QApplication.focusWidget()
        start_index = 0
        
        # Find starting position if focus is on one of our inputs
        for i, (_, input_widget) in enumerate(self.cslow_inputs.items()):
            if input_widget == current_widget:
                start_index = i
                break
        
        # Apply values
        input_list = list(self.cslow_inputs.values())
        for i, value in enumerate(values):
            target_index = start_index + i
            if target_index < len(input_list):
                input_list[target_index].setText(value)