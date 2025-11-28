"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Dialog for rejecting sweeps from beginning/end of recording.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QCheckBox, QGroupBox, QRadioButton
)


from data_analysis_gui.widgets.custom_inputs import SelectAllSpinBox
from data_analysis_gui.config.themes import (
    create_styled_button, style_group_box, style_label
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class RejectSweepsDialog(QDialog):
    """
    Dialog for rejecting sweeps from beginning and/or end of recording.
    
    Provides two input modes:
    1. Skip from beginning/end (specify counts)
    2. Keep sweep range (specify first and last sweep to keep)
    
    All sweep numbers are 1-indexed for consistency with MainWindow display and user expectations
    ("sweep 0" is not common parlance in electrophysiology).
    """
    
    def __init__(self, parent, file_name: str, total_sweeps: int):

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
        self.from_sweep_spin.valueChanged.connect(self._update_preview)
        self.to_sweep_spin.valueChanged.connect(self._update_preview)
        
        self.skip_mode_radio.toggled.connect(self._on_mode_changed)
        
        # Connect widget focus to radio button activation
        self.skip_first_spin.focusInEvent = self._create_focus_handler(
            self.skip_first_spin, self.skip_mode_radio
        )
        self.skip_last_spin.focusInEvent = self._create_focus_handler(
            self.skip_last_spin, self.skip_mode_radio
        )
        self.from_sweep_spin.focusInEvent = self._create_focus_handler(
            self.from_sweep_spin, self.range_mode_radio
        )
        self.to_sweep_spin.focusInEvent = self._create_focus_handler(
            self.to_sweep_spin, self.range_mode_radio
        )
        
    def _create_focus_handler(self, widget, radio_button):
        """Create a focus event handler that activates the corresponding radio button."""
        original_focus_in = widget.focusInEvent
        
        def focus_handler(event):
            # Activate the radio button when widget gets focus
            radio_button.setChecked(True)
            # Call the original focus handler
            original_focus_in(event)
        
        return focus_handler
        
    def _init_ui(self):
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
        
        # ===== Mode 1: Skip from beginning/end =====
        self.skip_mode_radio = QRadioButton("Skip from beginning/end")
        self.skip_mode_radio.setChecked(True)  # Default mode
        rejection_layout.addWidget(self.skip_mode_radio)
        
        skip_controls_layout = QVBoxLayout()
        skip_controls_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        
        # Skip first
        skip_first_row = QHBoxLayout()
        skip_first_label = QLabel("Skip first")
        skip_first_label.setMinimumWidth(70)
        self.skip_first_spin = SelectAllSpinBox()
        self.skip_first_spin.setDecimals(0)  # Integer only
        self.skip_first_spin.setMinimum(0)
        self.skip_first_spin.setMaximum(max(0, self.total_sweeps - 1))
        self.skip_first_spin.setValue(0)
        self.skip_first_spin.setMaximumWidth(80)
        skip_first_suffix = QLabel("sweeps")
        
        skip_first_row.addWidget(skip_first_label)
        skip_first_row.addWidget(self.skip_first_spin)
        skip_first_row.addWidget(skip_first_suffix)
        skip_first_row.addStretch()
        skip_controls_layout.addLayout(skip_first_row)
        
        # Skip last
        skip_last_row = QHBoxLayout()
        skip_last_label = QLabel("Skip last")
        skip_last_label.setMinimumWidth(70)
        self.skip_last_spin = SelectAllSpinBox()
        self.skip_last_spin.setDecimals(0)  # Integer only
        self.skip_last_spin.setMinimum(0)
        self.skip_last_spin.setMaximum(max(0, self.total_sweeps - 1))
        self.skip_last_spin.setValue(0)
        self.skip_last_spin.setMaximumWidth(80)
        skip_last_suffix = QLabel("sweeps")
        
        skip_last_row.addWidget(skip_last_label)
        skip_last_row.addWidget(self.skip_last_spin)
        skip_last_row.addWidget(skip_last_suffix)
        skip_last_row.addStretch()
        skip_controls_layout.addLayout(skip_last_row)
        
        rejection_layout.addLayout(skip_controls_layout)
        rejection_layout.addSpacing(10)
        
        # ===== Mode 2: Keep sweep range =====
        self.range_mode_radio = QRadioButton("Keep sweep range")
        rejection_layout.addWidget(self.range_mode_radio)
        
        range_controls_layout = QHBoxLayout()
        range_controls_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        
        # From sweep
        from_sweep_label = QLabel("From sweep")
        from_sweep_label.setMinimumWidth(90)
        self.from_sweep_spin = SelectAllSpinBox()
        self.from_sweep_spin.setDecimals(0)  # Integer only
        self.from_sweep_spin.setMinimum(1)
        self.from_sweep_spin.setMaximum(self.total_sweeps)
        self.from_sweep_spin.setValue(1)
        self.from_sweep_spin.setMaximumWidth(80)
        
        range_controls_layout.addWidget(from_sweep_label)
        range_controls_layout.addWidget(self.from_sweep_spin)
        range_controls_layout.addSpacing(20)
        
        # To sweep
        to_sweep_label = QLabel("To sweep")
        to_sweep_label.setMinimumWidth(70)
        self.to_sweep_spin = SelectAllSpinBox()
        self.to_sweep_spin.setDecimals(0)  # Integer only
        self.to_sweep_spin.setMinimum(1)
        self.to_sweep_spin.setMaximum(self.total_sweeps)
        self.to_sweep_spin.setValue(self.total_sweeps)
        self.to_sweep_spin.setMaximumWidth(80)
        
        range_controls_layout.addWidget(to_sweep_label)
        range_controls_layout.addWidget(self.to_sweep_spin)
        range_controls_layout.addStretch()
        
        rejection_layout.addLayout(range_controls_layout)
        
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
        
        # Let Qt auto-size based on content
        self.adjustSize()
        self.setMinimumWidth(500)
    
    def _on_mode_changed(self, checked: bool):
        """Handle switching between skip and range modes."""
        if checked:  # Skip mode activated
            # Sync range mode to skip mode
            self._sync_range_to_skip()
        else:  # Range mode activated
            # Sync skip mode to range mode
            self._sync_skip_to_range()
        
        self._update_preview()
    
    def _sync_range_to_skip(self):
        """Update range spinboxes based on current skip values."""
        skip_first = int(self.skip_first_spin.value())
        skip_last = int(self.skip_last_spin.value())
        
        # Calculate 1-indexed range
        from_sweep = skip_first + 1
        to_sweep = self.total_sweeps - skip_last
        
        # Update range spinboxes (block signals to prevent loops)
        self.from_sweep_spin.blockSignals(True)
        self.to_sweep_spin.blockSignals(True)
        
        self.from_sweep_spin.setValue(int(from_sweep))
        self.to_sweep_spin.setValue(int(to_sweep))
        
        self.from_sweep_spin.blockSignals(False)
        self.to_sweep_spin.blockSignals(False)
    
    def _sync_skip_to_range(self):
        """Update skip spinboxes based on current range values."""
        from_sweep = int(self.from_sweep_spin.value())
        to_sweep = int(self.to_sweep_spin.value())
        
        # Calculate skip counts (1-indexed to counts)
        skip_first = from_sweep - 1
        skip_last = self.total_sweeps - to_sweep
        
        # Update skip spinboxes (block signals to prevent loops)
        self.skip_first_spin.blockSignals(True)
        self.skip_last_spin.blockSignals(True)
        
        self.skip_first_spin.setValue(int(skip_first))
        self.skip_last_spin.setValue(int(skip_last))
        
        self.skip_first_spin.blockSignals(False)
        self.skip_last_spin.blockSignals(False)
    
    def _update_preview(self):
        """Update the preview text showing which sweeps will be kept."""
        # Get current values based on active mode (no auto-sync)
        if self.skip_mode_radio.isChecked():
            skip_first = int(self.skip_first_spin.value())
            skip_last = int(self.skip_last_spin.value())
        else:
            # Use range mode, convert to skip counts
            from_sweep = int(self.from_sweep_spin.value())
            to_sweep = int(self.to_sweep_spin.value())
            skip_first = from_sweep - 1
            skip_last = self.total_sweeps - to_sweep
        
        # Calculate resulting sweeps (1-indexed for display)
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
        
        # Valid configuration - display 1-indexed sweep numbers
        first_kept = skip_first + 1  # Convert to 1-indexed
        last_kept = self.total_sweeps - skip_last
        
        preview_text = (
            f"<b>Will analyze:</b> Sweeps {int(first_kept)}–{int(last_kept)} "
            f"({int(remaining)} total)"
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
        """Returns skip counts and reset time option."""
        # Check which mode is active and get the appropriate values
        if self.skip_mode_radio.isChecked():
            # Use skip mode values directly
            skip_first = int(self.skip_first_spin.value())
            skip_last = int(self.skip_last_spin.value())
        else:
            # Convert range mode to skip counts
            from_sweep = int(self.from_sweep_spin.value())
            to_sweep = int(self.to_sweep_spin.value())
            skip_first = from_sweep - 1
            skip_last = self.total_sweeps - to_sweep
        
        return (
            skip_first,
            skip_last,
            self.reset_time_cb.isChecked()
        )