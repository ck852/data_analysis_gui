"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Widget for selecting sweeps to analyze.
To be used in dialogs that involve operations with individual sweeps.
"""

from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, 
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QLabel
)
from data_analysis_gui.widgets.custom_inputs import RangeInputLineEdit

class SweepSelectionWidget(QWidget):
    """
    Widget for selecting which sweeps to analyze.
    """
    
    def __init__(self, sweep_names: List[str], parent=None):
        super().__init__(parent)
        
        self.sweep_names = sweep_names
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI with selection mode and table."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Selection mode radio buttons
        mode_layout = QHBoxLayout()
        self.table_mode_radio = QRadioButton("Select from table")
        self.range_mode_radio = QRadioButton("Enter range:")
        self.table_mode_radio.setChecked(True)  # Default to table mode
        
        self.range_input = RangeInputLineEdit()
        self.range_input.setPlaceholderText("e.g., 1,2,3-15,20")
        self.range_input.setMaximumWidth(150)
        #self.range_input.setEnabled(False)  # Disabled by default
        
        # Install event filter to auto-select radio button on focus
        self.range_input.installEventFilter(self)
        
        mode_layout.addWidget(self.table_mode_radio)
        mode_layout.addWidget(self.range_mode_radio)
        mode_layout.addWidget(self.range_input)
        mode_layout.addStretch()
        
        layout.addLayout(mode_layout)
        
        # Add instruction label
        instruction_label = QLabel("Separate sweeps with commas (e.g., 1,2,3 or 3-15 or 1,2,5-10)")
        instruction_label.setStyleSheet("color: #666666; font-size: 10px; font-style: italic;")
        instruction_label.setIndent(20)  # Indent to align with range input
        layout.addWidget(instruction_label)
        
        # Table widget
        self.table = QTableWidget()
        self._init_table()
        layout.addWidget(self.table)
        
        # Connect mode change signals
        self.table_mode_radio.toggled.connect(self._on_mode_changed)
        
    def _init_table(self):
        """Initialize the table with sweep selection."""
        self.table.setRowCount(len(self.sweep_names))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Select", "Sweep"])
        
        # Configure headers
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        # Hide vertical header
        self.table.verticalHeader().setVisible(False)
        
        # Set alternating row colors
        self.table.setAlternatingRowColors(True)
        
        # Populate rows
        for i, sweep_name in enumerate(self.sweep_names):
            # Checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            self.table.setCellWidget(i, 0, checkbox)
            
            # Sweep name
            item = QTableWidgetItem(f"Sweep {sweep_name}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 1, item)
        
        # Removed setMaximumHeight to allow table to expand vertically
        
        # Connect signal to make entire row clickable
        self.table.cellClicked.connect(self._on_cell_clicked)
    
    def eventFilter(self, obj, event):
        """
        Event filter to auto-select range mode when clicking in range input.
        
        Args:
            obj: The object being filtered
            event: The event
            
        Returns:
            bool: True if event was handled, False otherwise
        """
        from PySide6.QtCore import QEvent
        
        if obj is self.range_input and event.type() == QEvent.Type.FocusIn:
            # When range input gets focus, select range mode radio button
            if not self.range_mode_radio.isChecked():
                self.range_mode_radio.setChecked(True)
        
        # Always return False to allow event to propagate normally
        return False

    def _on_mode_changed(self, table_mode_checked: bool):
        """Enable/disable table and range input based on mode."""
        self.table.setEnabled(table_mode_checked)
        #self.range_input.setEnabled(not table_mode_checked)
        
    def _on_cell_clicked(self, row: int, column: int):
        """Toggle checkbox when any cell in the row is clicked."""
        checkbox = self.table.cellWidget(row, 0)
        if checkbox:
            checkbox.setChecked(not checkbox.isChecked())
            
    def select_all(self, checked: bool = True):
        """Select or deselect all sweeps in table."""
        for i in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(i, 0)
            if checkbox:
                checkbox.setChecked(checked)
                
    def get_selected_sweeps(self) -> Tuple[List[str], List[str]]:
        """
        Get list of selected sweep names based on current mode.
        
        Returns:
            Tuple[List[str], List[str]]: (valid_sweep_names, invalid_sweep_numbers)
                - valid_sweep_names: List of sweep names that were found
                - invalid_sweep_numbers: List of sweep numbers that were requested but not found
        """
        if self.table_mode_radio.isChecked():
            # Table mode - return checked sweeps, no invalid numbers
            selected = []
            for i in range(self.table.rowCount()):
                checkbox = self.table.cellWidget(i, 0)
                if checkbox and checkbox.isChecked():
                    selected.append(self.sweep_names[i])
            return selected, []
        else:
            # Range mode - parse range input
            return self._parse_range_input()


    def _parse_range_input(self) -> Tuple[List[str], List[str]]:
        """
        Parse range input like '1,2,3-15,20' and return matching sweep names.
        
        Supports:
        - Single numbers: 1,2,3
        - Ranges: 3-15
        - Combined: 1,2,3-15,20-25
        
        Returns:
            Tuple[List[str], List[str]]: (valid_sweep_names, invalid_sweep_numbers)
        """
        range_text = self.range_input.text().strip()
        if not range_text:
            return [], []
        
        requested_numbers = set()
        
        try:
            # Split by comma and process each segment
            segments = range_text.split(',')
            
            for segment in segments:
                segment = segment.strip()
                
                if '-' in segment:
                    # Range format: "3-15"
                    parts = segment.split('-', 1)
                    if len(parts) != 2:
                        continue
                        
                    start_str = parts[0].strip()
                    end_str = parts[1].strip()
                    
                    if not start_str or not end_str:
                        continue
                    
                    start = int(start_str)
                    end = int(end_str)
                    
                    if start > end:
                        continue
                    
                    # Add all numbers in range
                    for num in range(start, end + 1):
                        requested_numbers.add(num)
                else:
                    # Single number: "1"
                    if segment:
                        requested_numbers.add(int(segment))
            
        except (ValueError, AttributeError):
            # Invalid format - return empty
            return [], []
        
        # Find matching sweeps and track invalid numbers
        valid_sweeps = []
        invalid_numbers = []
        
        # Build set of valid sweep numbers for faster lookup
        valid_sweep_numbers = set()
        for sweep_name in self.sweep_names:
            try:
                valid_sweep_numbers.add(int(sweep_name))
            except ValueError:
                # Skip non-numeric sweep names
                pass
        
        # Check each requested number
        for num in sorted(requested_numbers):
            if num in valid_sweep_numbers:
                # Find the actual sweep name (convert back to string)
                valid_sweeps.append(str(num))
            else:
                invalid_numbers.append(str(num))
        
        return valid_sweeps, invalid_numbers