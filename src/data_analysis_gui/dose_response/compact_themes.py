"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Compact theme system for concentration response modules.

This module provides a complete, isolated theming system optimized for the
concentration response dialog and related widgets. It imports colors and fonts
from themes.py for visual consistency while maintaining independent sizing logic.

All conc_resp_*.py and concentration_*.py modules should import ONLY from this
module, never directly from themes.py.
"""

from PySide6.QtWidgets import (
    QWidget, QTableWidget, QLabel, QPushButton, QComboBox, 
    QGroupBox, QLineEdit, QHBoxLayout
)
from PySide6.QtCore import Qt

# Import colors and fonts from global theme for consistency
from data_analysis_gui.config.themes import MODERN_COLORS, BASE_FONT, FONT_SIZES

# ============================================================================
# COMPACT SIZING CONSTANTS
# ============================================================================

COMPACT_HEIGHT = 26
COMPACT_PADDING = "2px 6px"
COMPACT_BORDER_RADIUS = "3px"

# ============================================================================
# BASE DIALOG/WINDOW STYLING
# ============================================================================

def apply_compact_theme(widget: QWidget) -> None:
    """
    Apply compact theme to a dialog or window.
    
    Provides base styling for dialogs using the compact theme system.
    Should be called once on the main dialog/window widget.
    
    Args:
        widget: Dialog or window widget to style
    """
    widget.setStyleSheet(f"""
        QDialog, QMainWindow {{
            background-color: {MODERN_COLORS['background']};
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
        }}
        
        QWidget {{
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
        }}
        
        QFrame[frameShape="4"] /* HLine */ {{
            color: {MODERN_COLORS['border']};
            max-height: 1px;
        }}
        
        QFrame[frameShape="5"] /* VLine */ {{
            color: {MODERN_COLORS['border']};
            max-width: 1px;
        }}
        
        QSplitter::handle {{
            background: {MODERN_COLORS['border']};
            width: 1px;
        }}
        QSplitter::handle:hover {{
            background: {MODERN_COLORS['primary']};
        }}
    """)

# ============================================================================
# TABLE WIDGET STYLING
# ============================================================================

def style_table(table: QTableWidget) -> None:
    """
    Apply compact styling to QTableWidget.
    
    Provides consistent styling for tables with minimal padding optimized
    for compact widget embedding. Rows auto-size based on content.
    
    Args:
        table: QTableWidget to style
    """
    table.setAlternatingRowColors(True)
    table.setStyleSheet(f"""
        QTableWidget {{
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: {COMPACT_BORDER_RADIUS};
            background-color: {MODERN_COLORS['background']};
            alternate-background-color: {MODERN_COLORS['surface']};
            gridline-color: {MODERN_COLORS['border']};
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
        }}
        
        QTableWidget::item {{
            padding: 2px;
            border: none;
        }}
        
        QTableWidget::item:selected {{
            background-color: {MODERN_COLORS['selected']};
            color: {MODERN_COLORS['text']};
        }}
        
        QTableWidget::item:hover {{
            background-color: {MODERN_COLORS['hover']};
        }}
        
        QHeaderView::section {{
            background-color: {MODERN_COLORS['surface']};
            color: {MODERN_COLORS['text']};
            border: none;
            border-right: 1px solid {MODERN_COLORS['border']};
            border-bottom: 2px solid {MODERN_COLORS['border']};
            padding: 6px 8px;
            font-weight: 600;
            font-size: {FONT_SIZES['normal']};
            text-align: center;
        }}
        
        QHeaderView::section:last {{
            border-right: none;
        }}
        
        QHeaderView::section:hover {{
            background-color: {MODERN_COLORS['hover']};
        }}
    """)

# ============================================================================
# INPUT WIDGET STYLING
# ============================================================================

def style_input(widget: QLineEdit) -> None:
    """
    Apply compact styling to input widgets.
    
    Args:
        widget: The input widget to style
    """
    widget.setFixedHeight(COMPACT_HEIGHT)
    
    widget.setStyleSheet(f"""
        QLineEdit {{
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
            padding: {COMPACT_PADDING};
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: {COMPACT_BORDER_RADIUS};
            background-color: {MODERN_COLORS['background']};
            color: {MODERN_COLORS['text']};
        }}
        QLineEdit:hover {{
            border-color: {MODERN_COLORS['primary']};
        }}
        QLineEdit:focus {{
            border-color: {MODERN_COLORS['focus']};
            outline: none;
        }}
        QLineEdit:disabled {{
            background-color: {MODERN_COLORS['disabled']};
            color: {MODERN_COLORS['text_muted']};
        }}
    """)

def style_combo(widget: QComboBox) -> None:
    """
    Apply compact styling to combo boxes.
    
    Args:
        widget: The combo box to style
    """
    widget.setFixedHeight(COMPACT_HEIGHT)
    
    widget.setStyleSheet(f"""
        QComboBox {{
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
            padding: {COMPACT_PADDING};
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: {COMPACT_BORDER_RADIUS};
            background-color: {MODERN_COLORS['background']};
            color: {MODERN_COLORS['text']};
        }}
        QComboBox:hover {{
            border-color: {MODERN_COLORS['primary']};
        }}
        QComboBox:disabled {{
            background-color: {MODERN_COLORS['disabled']};
            color: {MODERN_COLORS['text_muted']};
        }}
        QComboBox::drop-down {{
            width: 0px;
            border: none;
        }}
        QComboBox::drop-down:hover {{
            background: {MODERN_COLORS['hover']};
        }}
        QComboBox::down-arrow {{
            image: none;
            width: 0px;
            height: 0px;
            border: none;
        }}
        QComboBox QAbstractItemView {{
            border: 1px solid {MODERN_COLORS['border']};
            background-color: {MODERN_COLORS['background']};
            selection-background-color: {MODERN_COLORS['selected']};
            padding: 2px;
        }}
    """)

def style_button(widget: QPushButton, height: int = None) -> None:
    """
    Apply compact styling to buttons.
    
    Args:
        widget: The button to style
        height: Optional fixed height in pixels (defaults to COMPACT_HEIGHT)
    """
    btn_height = height if height is not None else COMPACT_HEIGHT
    widget.setFixedHeight(btn_height)
    
    widget.setStyleSheet(f"""
        QPushButton {{
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
            padding: {COMPACT_PADDING};
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: {COMPACT_BORDER_RADIUS};
            background-color: {MODERN_COLORS['surface']};
            color: {MODERN_COLORS['text']};
        }}
        QPushButton:hover {{
            background-color: {MODERN_COLORS['hover']};
        }}
        QPushButton:pressed {{
            background-color: {MODERN_COLORS['hover']};
        }}
        QPushButton:disabled {{
            background-color: {MODERN_COLORS['disabled']};
            color: {MODERN_COLORS['text_muted']};
        }}
    """)

# ============================================================================
# LABEL STYLING
# ============================================================================

def style_label(widget: QLabel, style_type: str = "normal") -> None:
    """
    Apply compact styling to labels with various style types.
    
    Args:
        widget: The label to style
        style_type: Style variant - 'normal', 'heading', 'muted', 'info', 
                    'success', 'warning', 'error'
    """
    styles = {
        "normal": {
            "color": MODERN_COLORS["text"],
            "size": FONT_SIZES["normal"],
            "weight": "normal",
        },
        "heading": {
            "color": MODERN_COLORS["text"],
            "size": FONT_SIZES["large"],
            "weight": "bold",
        },
        "muted": {
            "color": MODERN_COLORS["text_muted"],
            "size": FONT_SIZES["small"],
            "weight": "normal",
        },
        "info": {
            "color": MODERN_COLORS["primary"],
            "size": FONT_SIZES["normal"],
            "weight": "500",
        },
        "success": {
            "color": MODERN_COLORS["success"],
            "size": FONT_SIZES["normal"],
            "weight": "500",
        },
        "warning": {
            "color": MODERN_COLORS["warning"],
            "size": FONT_SIZES["normal"],
            "weight": "500",
        },
        "error": {
            "color": MODERN_COLORS["danger"],
            "size": FONT_SIZES["normal"],
            "weight": "500",
        },
    }
    
    style = styles.get(style_type, styles["normal"])
    
    widget.setStyleSheet(f"""
        QLabel {{
            color: {style['color']};
            {BASE_FONT}
            font-size: {style['size']};
            font-weight: {style['weight']};
        }}
    """)

# ============================================================================
# GROUP BOX STYLING
# ============================================================================

def style_group_box(widget: QGroupBox) -> None:
    """
    Apply compact styling to group boxes.
    
    Args:
        widget: The group box to style
    """
    widget.setStyleSheet(f"""
        QGroupBox {{
            {BASE_FONT}
            font-weight: 500;
            font-size: {FONT_SIZES['normal']};
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: 4px;
            margin-top: 6px;
            padding-top: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            background-color: {MODERN_COLORS['background']};
        }}
    """)

# ============================================================================
# BUTTON FACTORY WITH COLOR VARIANTS
# ============================================================================

def create_button(text: str, style_type: str = "secondary", 
                  width: int = None, height: int = None) -> QPushButton:
    """
    Create a styled QPushButton with compact styling and color variant.
    
    Args:
        text: Button text
        style_type: 'primary', 'secondary', 'success', 'warning', 'danger'
        width: Optional fixed width
        height: Optional fixed height
        
    Returns:
        Styled QPushButton
    """
    button = QPushButton(text)
    btn_height = height if height is not None else COMPACT_HEIGHT
    button.setFixedHeight(btn_height)
    
    if width is not None:
        button.setFixedWidth(width)
    
    styles = {
        "primary": {
            "bg": MODERN_COLORS["primary"],
            "hover": "#0066CC",
            "text": "white",
            "border": "none",
        },
        "secondary": {
            "bg": MODERN_COLORS["surface"],
            "hover": MODERN_COLORS["hover"],
            "text": MODERN_COLORS["text"],
            "border": f"1px solid {MODERN_COLORS['border']}",
        },
        "success": {
            "bg": MODERN_COLORS["success"],
            "hover": "#218838",
            "text": "white",
            "border": "none",
        },
        "danger": {
            "bg": MODERN_COLORS["danger"],
            "hover": "#C82333",
            "text": "white",
            "border": "none",
        },
        "warning": {
            "bg": MODERN_COLORS["warning"],
            "hover": "#E0A800",
            "text": MODERN_COLORS["text"],
            "border": "none",
        },
    }
    
    style = styles.get(style_type, styles["secondary"])
    
    button.setStyleSheet(f"""
        QPushButton {{
            background-color: {style['bg']};
            color: {style['text']};
            border: {style['border']};
            border-radius: {COMPACT_BORDER_RADIUS};
            padding: {COMPACT_PADDING};
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {style['hover']};
        }}
        QPushButton:pressed {{
            background-color: {style['hover']};
        }}
        QPushButton:disabled {{
            background-color: {MODERN_COLORS['disabled']};
            color: {MODERN_COLORS['text_muted']};
        }}
    """)
    
    return button

# ============================================================================
# ALIGNMENT HELPERS
# ============================================================================

def align_center(widget: QWidget) -> QWidget:
    """
    Center a widget horizontally in a minimal container.
    
    Args:
        widget: Widget to center
        
    Returns:
        Widget wrapped in centered container
    """
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.addWidget(widget)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    return container

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Re-export colors for convenience
    'MODERN_COLORS',
    # Constants
    'COMPACT_HEIGHT',
    # Main styling functions
    'apply_compact_theme',
    'style_table',
    'style_input',
    'style_combo',
    'style_button',
    'style_label',
    'style_group_box',
    # Utilities
    'create_button',
    'align_center',
]