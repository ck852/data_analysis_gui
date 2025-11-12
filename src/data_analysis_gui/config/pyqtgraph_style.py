"""
PatchBatch Electrophysiology Data Analysis Tool

PyQtGraph styling configuration for scientific plots in PatchBatch.

This module provides centralized styling for all PyQtGraph plots, including
colors, fonts, line styles, cursors, and interactive elements. Ensures
consistency across the application and complements the Qt widget theme.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import pyqtgraph as pg
from PySide6.QtGui import QColor, QPen, QBrush, QFont, QCursor, QPixmap, QPainter
from PySide6.QtCore import Qt
from typing import Dict, Any, Tuple

from data_analysis_gui.config.plot_style import COLOR_CYCLE

# ============================================================================
# COLOR CONSTANTS - Synchronized with application theme
# ============================================================================

PYQTGRAPH_COLORS = {
    # Plot background and surfaces
    "plot_background": "#FAFAFA",  # Light gray background
    "plot_surface": "#FFFFFF",  # White surface
    
    # Grid and axes
    "grid": "#E1E5E8",  # Light gray grid
    "axis_line": "#B0B0B0",  # Axis border color
    "zero_line": "#2D3436",  # Prominent zero-axis lines
    
    # Text colors
    "text_primary": "#000000",  # Black for primary text (axis labels, title)
    "text_secondary": "#2D3436",  # Near-black for secondary text (tick labels)
    
    # Cursor and interactive element colors
    "cursor_analysis": "#73AB84",  # Sage green for analysis ranges
    "cursor_background": "#1565C0",  # Deep blue for background ranges
    "cursor_interactive": "#73AB84",  # Sage green for interactive crosshair
}

# Reuse matplotlib color cycle for consistency across plot types
DATA_COLOR_CYCLE = COLOR_CYCLE

# ============================================================================
# TEXT AND FONT CONFIGURATION
# ============================================================================

PYQTGRAPH_FONTS = {
    "axis_label": {"size": 12, "bold": True},
    "tick_label": {"size": 10, "bold": False},
    "title": {"size": 12, "bold": False},
}

def get_axis_label_font() -> QFont:
    """Get QFont for axis labels."""
    font = QFont()
    font.setPointSize(PYQTGRAPH_FONTS["axis_label"]["size"])
    font.setBold(PYQTGRAPH_FONTS["axis_label"]["bold"])
    return font

def get_tick_label_font() -> QFont:
    """Get QFont for tick labels."""
    font = QFont()
    font.setPointSize(PYQTGRAPH_FONTS["tick_label"]["size"])
    font.setBold(PYQTGRAPH_FONTS["tick_label"]["bold"])
    return font

def get_title_font() -> QFont:
    """Get QFont for plot titles."""
    font = QFont()
    font.setPointSize(PYQTGRAPH_FONTS["title"]["size"])
    font.setBold(PYQTGRAPH_FONTS["title"]["bold"])
    return font

# ============================================================================
# LINE AND PEN STYLES
# ============================================================================

DEFAULT_LINE_WIDTH = 2.5
ZERO_AXIS_WIDTH = 0.8
CURSOR_PEN_WIDTH = 1.5
INTERACTIVE_CURSOR_WIDTH = 2

def get_data_line_pen(color: str, width: float = None) -> QPen:
    """
    Get QPen for data lines.
    
    Args:
        color: Hex color string
        width: Line width (defaults to DEFAULT_LINE_WIDTH)
    
    Returns:
        QPen configured for data lines
    """
    if width is None:
        width = DEFAULT_LINE_WIDTH
    return pg.mkPen(color=color, width=width)

def get_zero_axis_pen() -> QPen:
    """
    Get QPen for prominent zero-axis lines.
    
    Returns:
        QPen with dotted style for x=0 and y=0 gridlines
    """
    color = QColor(PYQTGRAPH_COLORS["zero_line"])
    color.setAlpha(int(255 * 0.4))  # 40% opacity
    return pg.mkPen(color=color, width=ZERO_AXIS_WIDTH, style=Qt.PenStyle.DotLine)

def get_cursor_pen(is_background: bool = False) -> QPen:
    """
    Get QPen for range cursor boundaries.
    
    Args:
        is_background: Whether this is a background range cursor
    
    Returns:
        QPen with dashed style for cursor boundaries
    """
    color = (PYQTGRAPH_COLORS["cursor_background"] if is_background 
             else PYQTGRAPH_COLORS["cursor_analysis"])
    return pg.mkPen(color=color, width=CURSOR_PEN_WIDTH, style=Qt.PenStyle.DashLine)

def get_cursor_brush(is_background: bool = False, alpha: float = 0.2) -> QBrush:
    """
    Get QBrush for range cursor fill.
    
    Args:
        is_background: Whether this is a background range cursor
        alpha: Opacity (0.0 to 1.0)
    
    Returns:
        QBrush with semi-transparent fill
    """
    color = QColor(PYQTGRAPH_COLORS["cursor_background"] if is_background 
                   else PYQTGRAPH_COLORS["cursor_analysis"])
    color.setAlpha(int(255 * alpha))
    return QBrush(color)

def get_interactive_cursor_pen() -> QPen:
    """
    Get QPen for temporary interactive guide lines.
    
    Returns:
        QPen with dotted style for interactive cursors
    """
    return pg.mkPen(
        color=PYQTGRAPH_COLORS["cursor_interactive"],
        width=INTERACTIVE_CURSOR_WIDTH,
        style=Qt.PenStyle.DotLine
    )

# ============================================================================
# MARKER CONFIGURATION
# ============================================================================

MARKER_SETTINGS = {
    "symbol": "o",
    "size": 4,
    "threshold": 100,  # Show markers if dataset has fewer than this many points
}

def should_show_markers(data_length: int) -> bool:
    """
    Determine if markers should be shown based on dataset size.
    
    Args:
        data_length: Number of data points
    
    Returns:
        True if markers should be displayed
    """
    return data_length < MARKER_SETTINGS["threshold"]

def get_marker_settings(color: str) -> Dict[str, Any]:
    """
    Get marker settings for plot items.
    
    Args:
        color: Hex color string for marker
    
    Returns:
        Dictionary with symbol, symbolSize, symbolPen, symbolBrush
    """
    return {
        "symbol": MARKER_SETTINGS["symbol"],
        "symbolSize": MARKER_SETTINGS["size"],
        "symbolPen": None,
        "symbolBrush": color,
    }

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

def get_plot_background_color() -> str:
    """Get background color for plot widgets."""
    return PYQTGRAPH_COLORS["plot_background"]

def get_grid_settings() -> Dict[str, Any]:
    """
    Get grid configuration settings.
    
    Returns:
        Dictionary with x, y, and alpha settings
    """
    return {
        "x": True,
        "y": True,
        "alpha": 0.3,
    }

# ============================================================================
# STYLING FUNCTIONS
# ============================================================================

def style_plot_widget(plot_widget: pg.PlotWidget) -> None:
    """
    Apply standard styling to a PlotWidget.
    
    Args:
        plot_widget: PyQtGraph PlotWidget to style
    """
    # Set background
    plot_widget.setBackground(get_plot_background_color())
    
    # Enable grid
    grid_settings = get_grid_settings()
    plot_widget.getPlotItem().showGrid(
        x=grid_settings["x"],
        y=grid_settings["y"],
        alpha=grid_settings["alpha"]
    )

def style_plot_item_text(plot_item: pg.PlotItem, 
                         title: str = None,
                         xlabel: str = None, 
                         ylabel: str = None) -> None:
    """
    Apply text styling (labels and title) to a PlotItem.
    
    Args:
        plot_item: PyQtGraph PlotItem to style
        title: Plot title (optional)
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
    """
    text_color = PYQTGRAPH_COLORS["text_primary"]
    
    if title:
        plot_item.setTitle(title, color=text_color, size=f"{PYQTGRAPH_FONTS['title']['size']}pt")
    
    if xlabel:
        plot_item.setLabel('bottom', xlabel, color=text_color)
        
    if ylabel:
        plot_item.setLabel('left', ylabel, color=text_color)
    
    # Style axis tick labels
    for axis_name in ['bottom', 'left']:
        axis = plot_item.getAxis(axis_name)
        style_axis_ticks(axis)

def style_axis_ticks(axis_item: pg.AxisItem) -> None:
    """
    Apply styling to axis tick labels.
    
    Args:
        axis_item: PyQtGraph AxisItem to style
    """
    tick_font = get_tick_label_font()
    axis_item.setStyle(tickFont=tick_font)
    
    # Set tick text color
    tick_color = PYQTGRAPH_COLORS["text_secondary"]
    axis_item.setTextPen(tick_color)

def add_zero_axis_lines(plot_item: pg.PlotItem) -> Tuple[pg.InfiniteLine, pg.InfiniteLine]:
    """
    Add prominent gridlines at x=0 and y=0.
    
    Args:
        plot_item: PyQtGraph PlotItem to add lines to
    
    Returns:
        Tuple of (horizontal_line, vertical_line)
    """
    pen = get_zero_axis_pen()
    
    # Horizontal line at y=0
    hline = pg.InfiniteLine(pos=0, angle=0, pen=pen)
    plot_item.addItem(hline)
    
    # Vertical line at x=0
    vline = pg.InfiniteLine(pos=0, angle=90, pen=pen)
    plot_item.addItem(vline)
    
    return hline, vline

# ============================================================================
# CURSOR STYLING
# ============================================================================

def get_cursor_color(is_background: bool = False) -> str:
    """
    Get color for range cursors.
    
    Args:
        is_background: Whether this is a background range cursor
    
    Returns:
        Hex color string
    """
    return (PYQTGRAPH_COLORS["cursor_background"] if is_background 
            else PYQTGRAPH_COLORS["cursor_analysis"])

def create_crosshair_cursor() -> QCursor:
    """
    Create custom green crosshair cursor for interactive range creation.
    
    Returns:
        QCursor with crosshair icon
    """
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setPen(QColor(PYQTGRAPH_COLORS["cursor_interactive"]))
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    # Draw crosshair with thicker lines
    pen = painter.pen()
    pen.setWidth(2)
    painter.setPen(pen)
    
    # Vertical line
    painter.drawLine(16, 4, 16, 28)
    # Horizontal line
    painter.drawLine(4, 16, 28, 16)
    
    # Draw center dot
    painter.setBrush(QColor(PYQTGRAPH_COLORS["cursor_interactive"]))
    painter.drawEllipse(14, 14, 4, 4)
    
    painter.end()
    
    return QCursor(pixmap, hotX=16, hotY=16)

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def setup_analysis_plot(plot_widget: pg.PlotWidget,
                       title: str = None,
                       xlabel: str = None,
                       ylabel: str = None) -> pg.PlotItem:
    """
    Complete setup for an analysis plot with all standard styling.
    
    Args:
        plot_widget: PlotWidget to configure
        title: Plot title (optional)
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
    
    Returns:
        Configured PlotItem
    """
    # Apply widget-level styling
    style_plot_widget(plot_widget)
    
    # Get plot item and style text
    plot_item = plot_widget.getPlotItem()
    style_plot_item_text(plot_item, title, xlabel, ylabel)
    
    return plot_item