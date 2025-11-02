"""
PatchBatch Electrophysiology Data Analysis Tool

Sweep Filter Dialog - Filter sweeps from the loaded dataset.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import Dict, List, Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QCheckBox, QLabel, QMessageBox
)
from PySide6.QtCore import Qt

from data_analysis_gui.widgets.sweep_select_list import SweepSelectionWidget
from data_analysis_gui.config.themes import style_button, style_checkbox, style_label, create_styled_button


class SweepFilterDialog(QDialog):
    """
    Dialog for filtering sweeps from the loaded dataset.
    
    Allows user to select which sweeps to keep and optionally reset
    the time axis so the first selected sweep becomes t=0.
    """
    
    def __init__(self, sweep_names: List[str], sweep_times: Dict[str, float], parent=None):
        """
        Initialize the sweep filter dialog.
        
        Args:
            sweep_names: List of sweep index strings
            sweep_times: Dictionary mapping sweep indices to time values (seconds)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.sweep_names = sweep_names
        self.sweep_times = sweep_times
        
        self.setWindowTitle("Filter Sweeps")
        self.setModal(True)
        self.resize(500, 600)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Instruction label
        instruction = QLabel("Select sweeps to keep in the dataset:")
        style_label(instruction, "normal")
        layout.addWidget(instruction)
        
        # Use existing SweepSelectionWidget
        self.sweep_widget = SweepSelectionWidget(self.sweep_names, self)
        layout.addWidget(self.sweep_widget)
        
        # Selection control buttons (NEW)
        button_row = QHBoxLayout()
        select_all_btn = create_styled_button("Select All", "secondary")
        select_none_btn = create_styled_button("Select None", "secondary")
        
        select_all_btn.clicked.connect(lambda: self.sweep_widget.select_all(True))
        select_none_btn.clicked.connect(lambda: self.sweep_widget.select_all(False))
        
        button_row.addWidget(select_all_btn)
        button_row.addWidget(select_none_btn)
        button_row.addStretch()
        layout.addLayout(button_row)
        
        # Reset time checkbox
        self.reset_time_cb = QCheckBox("Reset time to zero at first selected sweep")
        self.reset_time_cb.setToolTip(
            "Subtract the time value of the first selected sweep from all sweep times"
        )
        style_checkbox(self.reset_time_cb)
        layout.addWidget(self.reset_time_cb)
        
        # Warning label
        warning = QLabel(
            "⚠️ Warning: This will permanently modify the loaded dataset.\n"
            "Use 'Reload Original File' button to restore."
        )
        warning.setStyleSheet(
            "color: #C73E1D; font-style: italic; padding: 8px; "
            "background-color: #FFE5E5; border-radius: 3px;"
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        style_button(cancel_btn, "secondary")
        button_layout.addWidget(cancel_btn)
        
        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.clicked.connect(self._on_apply)
        style_button(self.apply_btn, "danger")
        button_layout.addWidget(self.apply_btn)
        
        layout.addLayout(button_layout)
    
    def _on_apply(self):
        """Handle apply button click with validation."""
        selected_sweeps, invalid = self.sweep_widget.get_selected_sweeps()
        
        # Check for invalid sweep numbers
        if invalid:
            QMessageBox.warning(
                self,
                "Invalid Sweeps",
                f"The following sweep numbers were not found:\n{', '.join(invalid)}\n\n"
                "Only valid sweeps will be kept."
            )
        
        # Check if any sweeps selected
        if not selected_sweeps:
            QMessageBox.warning(
                self,
                "No Sweeps Selected",
                "You must select at least one sweep to keep."
            )
            return
        
        # Confirm destructive action
        removed_count = len(self.sweep_names) - len(selected_sweeps)
        msg = f"This will remove {removed_count} sweeps from the dataset."
        if self.reset_time_cb.isChecked():
            msg += "\nTime values will be adjusted."
        msg += "\n\nContinue?"
        
        reply = QMessageBox.question(
            self,
            "Confirm Filter",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.accept()
    
    def get_filter_config(self) -> Dict:
        """
        Get the filter configuration based on user selections.
        
        Returns:
            dict: Configuration with keys:
                - selected_sweeps: List of sweep indices to keep
                - reset_time: Whether to reset time axis
                - time_offset: Time offset to subtract (seconds)
        """
        selected_sweeps, _ = self.sweep_widget.get_selected_sweeps()
        reset_time = self.reset_time_cb.isChecked()
        time_offset = 0.0
        
        if reset_time and selected_sweeps:
            # Get time of first selected sweep (numerically smallest)
            first_sweep = min(selected_sweeps, key=lambda x: int(x) if x.isdigit() else 0)
            time_offset = self.sweep_times.get(first_sweep, 0.0)
        
        return {
            'selected_sweeps': selected_sweeps,
            'reset_time': reset_time,
            'time_offset': time_offset
        }