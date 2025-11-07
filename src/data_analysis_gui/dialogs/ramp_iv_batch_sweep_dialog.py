"""
Ramp IV Batch Sweep Selection Dialog for PatchBatch

Simple dialog to let users choose how sweeps should be handled during
batch ramp IV analysis: use all sweeps from each file, or use the same
sweep selection across all files.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import List, Tuple, Optional
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QRadioButton,
    QLabel,
    QButtonGroup,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt

from data_analysis_gui.config.themes import apply_modern_theme, style_label
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class RampIVBatchSweepDialog(QDialog):
    """
    Dialog for selecting sweep handling mode in ramp IV batch analysis.

    Allows users to choose between:
    - Using all sweeps from each file
    - Using the same sweep selection for all files
    """

    def __init__(self, parent=None, current_sweep_selection: Optional[List[str]] = None):
        """
        Initialize the sweep selection dialog.

        Args:
            parent: Parent widget.
            current_sweep_selection: List of sweep indices from preview analysis.
        """
        super().__init__(parent)

        self.current_sweep_selection = current_sweep_selection or []
        self.selected_mode = "all"  # Default

        self.setWindowTitle("Batch Sweep Selection")
        self.setModal(True)
        self.setFixedWidth(450)

        self._init_ui()
        apply_modern_theme(self)

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Instructions
        instructions = QLabel(
            "Choose how sweeps should be selected during batch analysis:"
        )
        instructions.setWordWrap(True)
        style_label(instructions, "body")
        layout.addWidget(instructions)

        # Radio button group
        self.button_group = QButtonGroup(self)

        # Option 1: Use all sweeps
        self.all_sweeps_radio = QRadioButton("Use all sweeps from each file")
        self.all_sweeps_radio.setChecked(True)  # Default
        self.button_group.addButton(self.all_sweeps_radio, 0)
        layout.addWidget(self.all_sweeps_radio)

        # Option 1 description
        all_sweeps_desc = QLabel(
            "   Each file will be analyzed using all of its available sweeps."
        )
        all_sweeps_desc.setWordWrap(True)
        style_label(all_sweeps_desc, "caption")
        layout.addWidget(all_sweeps_desc)

        layout.addSpacing(10)

        # Option 2: Use same sweeps
        if self.current_sweep_selection:
            sweep_list = ", ".join(self.current_sweep_selection[:5])
            if len(self.current_sweep_selection) > 5:
                sweep_list += f", ... ({len(self.current_sweep_selection)} total)"
            same_sweeps_text = f"Use same sweep selection for all files: {sweep_list}"
        else:
            same_sweeps_text = "Use same sweep selection for all files"

        self.same_sweeps_radio = QRadioButton(same_sweeps_text)
        self.button_group.addButton(self.same_sweeps_radio, 1)
        layout.addWidget(self.same_sweeps_radio)

        # Option 2 description
        same_sweeps_desc = QLabel(
            "   All files will be analyzed using the same sweep indices.\n"
            "   If a sweep doesn't exist in a file, NaN values will be returned."
        )
        same_sweeps_desc.setWordWrap(True)
        style_label(same_sweeps_desc, "caption")
        layout.addWidget(same_sweeps_desc)

        layout.addSpacing(10)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_selection(self) -> Tuple[str, Optional[List[str]]]:
        """
        Get the selected sweep mode and sweep list.

        Returns:
            Tuple of (mode, selected_sweeps) where:
            - mode is "all" or "same"
            - selected_sweeps is the list of sweep indices if mode is "same", else None
        """
        if self.all_sweeps_radio.isChecked():
            return ("all", None)
        else:
            return ("same", self.current_sweep_selection)