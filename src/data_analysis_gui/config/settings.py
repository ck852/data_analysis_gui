"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module defines default configuration settings, analysis constants, file patterns,
and table headers for the PatchBatch application. All constants are grouped for clarity
and maintainability. No functions are defined in this module.
"""

from PySide6.QtCore import Qt


DEFAULT_SETTINGS = {
    "range1_start": 0,
    "range1_end": 400,
    "range2_start": 100,
    "range2_end": 500,
    "cslow_default": 18.0,
    "plot_figsize": (10, 6),
    "window_geometry": (100, 100, 1400, 900),
}
"""
dict: Default application settings for PatchBatch.

Keys:
    range1_start (int): Start index for analysis range 1.
    range1_end (int): End index for analysis range 1.
    range2_start (int): Start index for analysis range 2.
    range2_end (int): End index for analysis range 2.
    cslow_default (float): Default Cslow value in pF.
    plot_figsize (tuple): Default figure size for plots (width, height).
    window_geometry (tuple): Default main window geometry (x, y, width, height).
"""


ANALYSIS_CONSTANTS = {
    "hold_timer_interval": 150,
    "zoom_scale_factor": 1.1,
    "pan_cursor": Qt.CursorShape.ClosedHandCursor,
    "line_picker_tolerance": 5,
    "range_colors": {
        "analysis": {"line": "#2E7D32", "fill": (0.18, 0.49, 0.20, 0.2)},
        "background": {"line": "#1565C0", "fill": (0.08, 0.40, 0.75, 0.2)},
    },
}
"""
dict: Analysis-related constants for PatchBatch.

Keys:
    hold_timer_interval (int): Interval for hold timer in ms.
    zoom_scale_factor (float): Factor for zooming in/out in plots.
    pan_cursor (Qt.CursorShape): Cursor shape for panning.
    line_picker_tolerance (int): Pixel tolerance for line picking.
    range_colors (dict): Colors for analysis and background ranges.
        analysis (dict): Line and fill color for analysis range.
        background (dict): Line and fill color for background range.
"""


FILE_PATTERNS = {
    "csv_files": "CSV files (*.csv)",
    "png_files": "PNG files (*.png)",
}
"""
dict: File dialog patterns and extensions for supported file types.

Keys:
    csv_files (str): Pattern for CSV files.
    png_files (str): Pattern for PNG image files.
"""


TABLE_HEADERS = {
    "ranges": ["✖", "Name", "Start", "End", "Analysis", "BG", "Paired BG"],
    "results": [
        "File",
        "Data Trace",
        "Range",
        "Raw Value",
        "Background",
        "Corrected Value",
    ],
    "current_density_iv": ["File", "Include", "Cslow (pF)"],
}
"""
dict: Table column headers for various PatchBatch tables.

Keys:
    ranges (list): Headers for range selection table.
    results (list): Headers for results table.
    current_density_iv (list): Headers for current density IV table.
"""
