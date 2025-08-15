from PyQt5.QtCore import Qt

# Default analysis parameters
DEFAULT_SETTINGS = {
    'range1_start': 0,
    'range1_end': 1000,
    'range2_start': 100,
    'range2_end': 500,
    'stimulus_period': 1000,
    'cslow_default': 18.0,
    'plot_figsize': (10, 6),
    'window_geometry': (100, 100, 1400, 900),
}

# Analysis constants
ANALYSIS_CONSTANTS = {
    'hold_timer_interval': 150,
    'zoom_scale_factor': 1.1,
    'pan_cursor': Qt.ClosedHandCursor,
    'line_picker_tolerance': 5,
    'range_colors': {
        'analysis': {'line': '#2E7D32', 'fill': (0.18, 0.49, 0.20, 0.2)},
        'background': {'line': '#1565C0', 'fill': (0.08, 0.40, 0.75, 0.2)}
    }
}

# File patterns and extensions
FILE_PATTERNS = {
    'mat_files': "MAT files (*.mat)",
    'csv_files': "CSV files (*.csv)",
    'png_files': "PNG files (*.png)",
}

# Table column headers
TABLE_HEADERS = {
    'ranges': ["✖", "Name", "Start", "End", "Analysis", "BG", "Paired BG"],
    'results': ["File", "Data Trace", "Range", "Raw Value", "Background", "Corrected Value"],
    'current_density_iv': ["File", "Include", "Cslow (pF)"],
}