"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

"""GUI services for handling presentation layer concerns."""

from .file_dialog_service import FileDialogService
from data_analysis_gui.gui_services.clipboard_service import ClipboardService
from data_analysis_gui.gui_services.main_range_coordinator import MainRangeCoordinator 

__all__ = ["FileDialogService", "ClipboardService", "MainRangeCoordinator"]
