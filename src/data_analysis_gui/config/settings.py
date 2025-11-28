"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module defines default configuration settings, analysis constants, file patterns,
and table headers. The configuration settings/analysis constants may be something we want to 
expand in the future as part of a new feature that would allow users to save multiple presets 
of analysis settings for different experimental protocols.

In the meantime, this module should remain mostly unchanged.

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
Default application settings including analysis range indices, Cslow value,
plot dimensions, and window geometry (x, y, width, height).
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
Constants for plot interactions and visual styling. Includes timer intervals,
zoom/pan behavior, line picking tolerance, and color schemes for analysis and
background ranges.
"""


FILE_PATTERNS = {
    "csv_files": "CSV files (*.csv)",
    "png_files": "PNG files (*.png)",
}
"""File dialog filter patterns for CSV exports and PNG image saves."""


TABLE_HEADERS = {
    "ranges": ["✗", "Name", "Start", "End", "Analysis", "BG", "Paired BG"],
    "results": [
        "File",
        "Data Trace",
        "Range",
        "Raw",
        "Background",
        "Corrected",
    ],
    "current_density_iv": ["File", "Include", "Cslow (pF)"],
}
"""Column headers for the range selection table, results table, and current density IV table."""