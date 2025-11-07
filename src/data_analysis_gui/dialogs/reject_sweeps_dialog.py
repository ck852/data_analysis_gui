"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Dialog for rejecting sweeps from beginning/end of recording.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, 
    QCheckBox, QGroupBox, QPushButton
)
from PySide6.QtCore import Qt

from data_analysis_gui.config.themes import (
    create_styled_button, style_group_box, style_label
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class RejectSweepsDialog(QDialog):
    """
    Dialog for rejecting sweeps from beginning and/or end of recording.
    
    Provides a simple interface to skip initial equilibration sweeps and/or
    final rundown sweeps, with optional time axis recalibration.
    """
    
    def __init__(self, parent, file_name: str, total_sweeps: int):
        """
        Initialize the reject sweeps dialog.
        
        Args:
            parent: Parent widget
            file_name: Name of the loaded data file
            total_sweeps: Total number of sweeps in the file
        """
        super().__init__(parent)
        
        self.file_name = file_name
        self.total_sweeps = total_sweeps
        
        self.setWindowTitle("Reject Sweeps")
        self.setModal(True)
        
        self._init_ui()
        self._update_preview()
        
        # Connect signals
        self.skip_first_spin.valueChanged.connect(self._update_preview)
        self.skip_last_spin.valueChanged.connect(self._update_preview)
        
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # File info
        file_label = QLabel(f"<b>File:</b> {self.file_name}")
        style_label(file_label, "normal")
        layout.addWidget(file_label)
        
        sweep_count_label = QLabel(f"<b>Total sweeps:</b> {self.total_sweeps}")
        style_label(sweep_count_label, "normal")
        layout.addWidget(sweep_count_label)
        
        layout.addSpacing(10)
        
        # Rejection controls
        rejection_group = QGroupBox("Sweep Rejection")
        style_group_box(rejection_group)
        rejection_layout = QVBoxLayout(rejection_group)
        
        # Skip first
        first_layout = QHBoxLayout()
        first_label = QLabel("Skip first:")
        first_label.setMinimumWidth(80)
        self.skip_first_spin = QSpinBox()
        self.skip_first_spin.setMinimum(0)
        self.skip_first_spin.setMaximum(max(0, self.total_sweeps - 1))
        self.skip_first_spin.setValue(0)
        self.skip_first_spin.setSuffix(" sweeps")
        first_layout.addWidget(first_label)
        first_layout.addWidget(self.skip_first_spin)
        first_layout.addStretch()
        rejection_layout.addLayout(first_layout)
        
        # Skip last
        last_layout = QHBoxLayout()
        last_label = QLabel("Skip last:")
        last_label.setMinimumWidth(80)
        self.skip_last_spin = QSpinBox()
        self.skip_last_spin.setMinimum(0)
        self.skip_last_spin.setMaximum(max(0, self.total_sweeps - 1))
        self.skip_last_spin.setValue(0)
        self.skip_last_spin.setSuffix(" sweeps")
        last_layout.addWidget(last_label)
        last_layout.addWidget(self.skip_last_spin)
        last_layout.addStretch()
        rejection_layout.addLayout(last_layout)
        
        layout.addWidget(rejection_group)
        
        # Preview
        preview_group = QGroupBox("Preview")
        style_group_box(preview_group)
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel()
        style_label(self.preview_label, "normal")
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_group)
        
        # Time reset option
        self.reset_time_cb = QCheckBox("Reset time axis to start at 0")
        self.reset_time_cb.setToolTip(
            "Recalibrate sweep times so the first kept sweep becomes t=0"
        )
        layout.addWidget(self.reset_time_cb)
        
        # Warning label
        self.warning_label = QLabel()
        self.warning_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = create_styled_button("Cancel", "secondary", self)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        self.apply_btn = create_styled_button("Apply", "primary", self)
        self.apply_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.apply_btn)
        
        layout.addLayout(button_layout)
        
        # Let Qt auto-size based on content instead of forcing a size
        self.adjustSize()
        self.setMinimumWidth(450)
    
    def _update_preview(self):
        """Update the preview text and validation."""
        skip_first = self.skip_first_spin.value()
        skip_last = self.skip_last_spin.value()
        
        # Calculate resulting sweeps
        remaining = self.total_sweeps - skip_first - skip_last
        
        if remaining <= 0:
            # Invalid configuration
            self.preview_label.setText(
                f"<span style='color: #d32f2f;'><b>Invalid:</b> "
                f"All sweeps would be removed!</span>"
            )
            self.apply_btn.setEnabled(False)
            self.warning_label.setVisible(False)
            return
        
        # Valid configuration
        first_kept = skip_first
        last_kept = self.total_sweeps - skip_last - 1
        
        preview_text = (
            f"<b>Resulting analysis:</b> Sweeps {first_kept}-{last_kept} "
            f"({remaining} total)"
        )
        self.preview_label.setText(preview_text)
        self.apply_btn.setEnabled(True)
        
        # Show warning if rejecting sweeps
        if skip_first > 0 or skip_last > 0:
            self.warning_label.setText(
                "⚠️ This operation permanently removes sweeps from the current session. "
                "Reload the file to restore all sweeps."
            )
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)
    
    def get_rejection_params(self):
        """
        Get the rejection parameters from the dialog.
        
        Returns:
            tuple: (skip_first, skip_last, reset_time)
        """
        return (
            self.skip_first_spin.value(),
            self.skip_last_spin.value(),
            self.reset_time_cb.isChecked()
        )