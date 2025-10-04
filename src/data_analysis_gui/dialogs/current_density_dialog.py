"""PatchBatch Electrophysiology Data Analysis Tool.

Dialog for entering slow capacitance values for current density calculations.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import re
from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QDoubleValidator
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
                            QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
                                )

from data_analysis_gui.config.logging import get_logger

# Updated imports for refactored themes.py
from data_analysis_gui.config.themes import (MODERN_COLORS, apply_compact_layout, style_button, apply_modern_theme, style_input_field,
                                             style_table_widget)
from data_analysis_gui.core.models import BatchAnalysisResult

logger = get_logger(__name__)

class CslowInputTable(QTableWidget):
    """
    Custom table widget that allows Tab key to jump directly between Cslow input fields,
    skipping the read-only File and Status columns.
    """
    
    def keyPressEvent(self, event):
        """Override key press to handle Tab navigation between input fields only."""
        if event.key() == Qt.Key.Key_Tab:
            # Get current focused widget
            current_widget = self.focusWidget()
            
            # Check if it's one of our Cslow input fields
            if isinstance(current_widget, QLineEdit):
                # Find current row
                for row in range(self.rowCount()):
                    if self.cellWidget(row, 1) == current_widget:
                        # Move to next row's Cslow input
                        next_row = (row + 1) % self.rowCount()
                        next_input = self.cellWidget(next_row, 1)
                        if next_input:
                            next_input.setFocus()
                            next_input.selectAll()
                        return
        
        elif event.key() == Qt.Key.Key_Backtab:  # Shift+Tab
            # Get current focused widget
            current_widget = self.focusWidget()
            
            # Check if it's one of our Cslow input fields
            if isinstance(current_widget, QLineEdit):
                # Find current row
                for row in range(self.rowCount()):
                    if self.cellWidget(row, 1) == current_widget:
                        # Move to previous row's Cslow input
                        prev_row = (row - 1) % self.rowCount()
                        prev_input = self.cellWidget(prev_row, 1)
                        if prev_input:
                            prev_input.setFocus()
                            prev_input.selectAll()
                        return
        
        # For all other keys, use default behavior
        super().keyPressEvent(event)

class CurrentDensityDialog(QDialog):
    """
    Dialog for entering Cslow (slow capacitance) values for each file.

    Allows users to input capacitance values in picofarads (pF) for current density calculations.
    Provides bulk entry, validation, and status feedback for each file.
    """

    def __init__(self, parent, batch_result: BatchAnalysisResult):
        """
        Initialize the dialog.

        Args:
            parent: Parent widget.
            batch_result (BatchAnalysisResult): Batch analysis result containing file info.
        """
        super().__init__(parent)
        self.batch_result = batch_result
        self.cslow_inputs = {}  # filename -> QLineEdit

        self.setModal(True)

        # Set window title before applying theme
        self.setWindowTitle("Current Density Analysis - Enter Cslow Values")

        self.init_ui()

        # Apply centralized styling from refactored themes.py
        apply_modern_theme(self)  # Only takes 1 argument now
        apply_compact_layout(self)

    def init_ui(self):
        """
        Initialize the dialog UI components and layout.
        """
        layout = QVBoxLayout(self)
        self.resize(600, 600)

        # Instructions
        instructions = QLabel(
            "Enter slow capacitance (Cslow) values in picofarads (pF) for each file.\n"
            "Current density will be calculated as Current (pA) / Cslow (pF)."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Table for file names and Cslow inputs
        self.table = CslowInputTable()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["File", "Cslow (pF)", "Status"])
        style_table_widget(self.table)  # Apply theme styling

        # Configure table geometry (layout, not style)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(2, 80)

        # Populate table
        self._populate_table()

        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()

        # Default value button
        set_all_btn = QPushButton("Set All to:")
        style_button(set_all_btn, "secondary")
        self.default_input = QLineEdit("18.0")
        self.default_input.setMaximumWidth(80)
        self.default_input.setValidator(QDoubleValidator(0.01, 10000.0, 2))
        style_input_field(self.default_input)  # Apply theme styling

        button_layout.addWidget(set_all_btn)
        button_layout.addWidget(self.default_input)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Connect signals
        set_all_btn.clicked.connect(self._set_all_values)

    def _populate_table(self):
        """
        Populate the table with file names and input fields for Cslow values.
        """
        results = sorted(
            self.batch_result.successful_results,
            key=lambda r: self._extract_number(r.base_name),
        )
        self.table.setRowCount(len(results))

        for row, result in enumerate(results):
            # File name (read-only)
            file_item = QTableWidgetItem(result.base_name)
            file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, file_item)

            # Cslow input
            cslow_input = QLineEdit()
            cslow_input.setValidator(QDoubleValidator(0.01, 10000.0, 2))
            style_input_field(cslow_input)  # Apply theme styling
            cslow_input.textChanged.connect(lambda _, r=row: self._update_status(r))
            self.table.setCellWidget(row, 1, cslow_input)
            self.cslow_inputs[result.base_name] = cslow_input

            # Status (initially empty)
            status_item = QTableWidgetItem("")
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, status_item)

    def _extract_number(self, filename: str) -> tuple:
        """
        Extract numeric identifiers from the filename for sorting.
        
        Handles formats like 'date_exp' (e.g., 250923_001) by sorting first by date,
        then by experiment number within each date.

        Args:
            filename (str): File name string.

        Returns:
            tuple: Tuple of numbers for hierarchical sorting, or (0,) if no numbers found.
        """
        # Try to extract numbers on both sides of underscore (e.g., "250923_001")
        # Returns tuple (date, experiment_num) for proper hierarchical sorting
        match = re.search(r"(\d+)_(\d+)", filename)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        
        # Fallback: extract all numbers and return as tuple for multi-level sorting
        numbers = re.findall(r"\d+", filename)
        if numbers:
            return tuple(int(n) for n in numbers)
        
        # No numbers found
        return (0,)

    def _update_status(self, row: int):
        """
        Update the status indicator for a given row based on input validity.

        Args:
            row (int): Row index in the table.
        """
        cslow_input = self.table.cellWidget(row, 1)
        status_item = self.table.item(row, 2)

        if cslow_input and status_item:
            text = cslow_input.text().strip()
            if text and self._is_valid_number(text):
                status_item.setText("✔")
                # Use theme color for success status
                status_item.setForeground(QColor(MODERN_COLORS["success"]))
            else:
                status_item.setText("")

    def _is_valid_number(self, text: str) -> bool:
        """
        Check if the provided text is a valid positive number.

        Args:
            text (str): Input text.

        Returns:
            bool: True if valid positive float, False otherwise.
        """
        try:
            value = float(text)
            return value > 0
        except ValueError:
            return False

    def _set_all_values(self):
        """
        Set all Cslow input fields to the default value entered by the user.
        """
        default_value = self.default_input.text().strip()
        if not self._is_valid_number(default_value):
            QMessageBox.warning(
                self,
                "Invalid Value",
                "Please enter a valid positive number for the default value.",
            )
            return

        for cslow_input in self.cslow_inputs.values():
            cslow_input.setText(default_value)

    def _validate_and_accept(self):
        """
        Validate all Cslow inputs before accepting the dialog.

        Shows a warning if any selected file is missing a valid value.
        """
        missing_files = []
        selected_files = getattr(self.batch_result, "selected_files", set())

        for filename, cslow_input in self.cslow_inputs.items():
            text = cslow_input.text().strip()
            if filename in selected_files:
                if not text or not self._is_valid_number(text):
                    missing_files.append(filename)

        if missing_files:
            QMessageBox.warning(
                self,
                "Missing Values",
                f"Please enter valid Cslow values for selected files.\n"
                f"Missing: {len(missing_files)} file(s)",
            )
            return

        self.accept()

    def get_cslow_mapping(self) -> Dict[str, float]:
        """
        Get the mapping of filenames to entered Cslow values.

        Returns:
            Dict[str, float]: Mapping of file names to capacitance values.
        """
        mapping = {}
        for filename, cslow_input in self.cslow_inputs.items():
            text = cslow_input.text().strip()
            if text and self._is_valid_number(text):
                mapping[filename] = float(text)
        return mapping

    def keyPressEvent(self, event):
        """
        Handle keyboard events for copy/paste support.

        Args:
            event: QKeyEvent instance.
        """
        # Check for paste shortcut: Ctrl+V on Windows/Linux, Cmd+V on Mac
        is_paste = (
            event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and event.key() == Qt.Key.Key_V
        ) or (
            event.modifiers() == Qt.KeyboardModifier.MetaModifier
            and event.key() == Qt.Key.Key_V
        )

        if is_paste:
            self._handle_paste()
        else:
            super().keyPressEvent(event)

    def _handle_paste(self):
        """
        Handle paste operation for bulk input of Cslow values from clipboard.
        """
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text:
            return

        lines = text.strip().split("\n")
        values = [line.strip() for line in lines if self._is_valid_number(line.strip())]

        current_widget = QApplication.focusWidget()
        start_index = 0
        input_list = list(self.cslow_inputs.values())

        for i, input_widget in enumerate(input_list):
            if input_widget == current_widget:
                start_index = i
                break

        for i, value in enumerate(values):
            target_index = start_index + i
            if target_index < len(input_list):
                input_list[target_index].setText(value)
