"""
PatchBatch Electrophysiology Data Analysis Tool

This module provides centralized Qt theming utilities for the PatchBatch application.
It defines a modern, consistent look for all supported Qt widgets, including colors,
fonts, spacing, and interactive states. Functions are provided to apply styles to
individual widgets or the entire application, ensuring a cohesive user interface.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QApplication,
    QListWidget,
    QTableWidget,
    QProgressBar,
    QLabel,
    QCheckBox,
    QGroupBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QDialog,
    QMainWindow,
    QSplitter,
)
from PySide6.QtGui import QPalette, QColor

# ============================================================================
# SIMPLIFIED COLOR PALETTE - Only essential colors
# ============================================================================

MODERN_COLORS = {
    # Core action colors
    "primary": "#0084FF",
    "secondary": "#6C757D",
    "success": "#28A745",
    "warning": "#FFC107",
    "danger": "#DC3545",
    # UI base colors
    "background": "#FFFFFF",
    "surface": "#F8F9FA",
    "border": "#DEE2E6",
    "text": "#212529",
    "text_muted": "#6C757D",
    # Interactive states
    "hover": "#E9ECEF",
    "disabled": "#CCCCCC",
    "selected": "#E3F2FD",
    "focus": "#80BDFF",
}

# Simple sizing constants
WIDGET_SIZES = {
    "button_height": 20,
    "input_height": 20,
    "button_min_width": 40,
}

# Simple spacing values
SPACING = {"border_radius": "3px", "padding": "4px 8px", "margin": "2px"}

# Base font settings
BASE_FONT = "font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;"
FONT_SIZES = {"normal": "10pt", "large": "11pt", "small": "9pt"}

# ============================================================================
# CORE STYLING FUNCTIONS - Simplified API
# ============================================================================


def apply_modern_theme(widget: QWidget) -> None:
    """
    Applies the modern PatchBatch theme to a given Qt widget.

    Automatically detects the widget type and applies appropriate styling, including colors, fonts, and layout. For QMainWindow, also sets the application palette.

    Args:
        widget (QWidget): The widget to style. Can be any supported Qt widget.
    """
    if isinstance(widget, QMainWindow) or isinstance(widget, QDialog):
        widget.setStyleSheet(_get_base_stylesheet())

        # Set application palette if main window
        if isinstance(widget, QMainWindow):
            _apply_palette(widget)

    elif isinstance(widget, QPushButton):
        style_button(widget)
    elif isinstance(widget, (QLineEdit, QSpinBox, QDoubleSpinBox)):
        style_input_field(widget)
    elif isinstance(widget, QComboBox):
        style_combo_box(widget)
    elif isinstance(widget, QLabel):
        style_label(widget)
    elif isinstance(widget, QTableWidget):
        style_table_widget(widget)
    elif isinstance(widget, QListWidget):
        style_list_widget(widget)
    elif isinstance(widget, QProgressBar):
        style_progress_bar(widget)
    elif isinstance(widget, QGroupBox):
        style_group_box(widget)
    elif isinstance(widget, QCheckBox):
        style_checkbox(widget)
    elif isinstance(widget, QSplitter):
        # Use minimal splitter styling
        widget.setStyleSheet(
            f"""
            QSplitter::handle {{
                background: {MODERN_COLORS['border']};
                width: 1px;
            }}
            QSplitter::handle:hover {{
                background: {MODERN_COLORS['primary']};
            }}
        """
        )
    else:
        # Apply base stylesheet for unknown widgets
        widget.setStyleSheet(_get_base_stylesheet())


def style_button(button: QPushButton, style_type: str = "secondary") -> None:
    """
    Styles a QPushButton according to the specified type.

    Supports 'primary', 'secondary', 'accent' (success), 'danger', and 'warning' styles.

    Args:
        button (QPushButton): The button to style.
        style_type (str, optional): The style type. Defaults to "secondary".
    """
    styles = {
        "primary": {
            "bg": MODERN_COLORS["primary"],
            "hover": "#0066CC",
            "text": "white",
        },
        "secondary": {
            "bg": MODERN_COLORS["surface"],
            "hover": MODERN_COLORS["hover"],
            "text": MODERN_COLORS["text"],
            "border": f"1px solid {MODERN_COLORS['border']}",
        },
        "accent": {"bg": MODERN_COLORS["success"], "hover": "#218838", "text": "white"},
        "danger": {"bg": MODERN_COLORS["danger"], "hover": "#C82333", "text": "white"},
        "warning": {
            "bg": MODERN_COLORS["warning"],
            "hover": "#E0A800",
            "text": MODERN_COLORS["text"],
        },
    }

    style = styles.get(style_type, styles["secondary"])
    border = style.get("border", "none")

    button.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {style['bg']};
            color: {style['text']};
            border: {border};
            border-radius: {SPACING['border_radius']};
            padding: {SPACING['padding']};
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
            font-weight: 500;
            min-height: {WIDGET_SIZES['button_height']}px;
            min-width: {WIDGET_SIZES['button_min_width']}px;
        }}
        QPushButton:hover {{
            background-color: {style['hover']};
        }}
        QPushButton:pressed {{
            background-color: {style['hover']};
            padding: 7px 11px 5px 13px;
        }}
        QPushButton:disabled {{
            background-color: {MODERN_COLORS['disabled']};
            color: {MODERN_COLORS['text_muted']};
            border-color: {MODERN_COLORS['border']};
        }}
    """
    )


def style_input_field(widget: QWidget, invalid: bool = False) -> None:
    """
    Styles input fields (QLineEdit, QSpinBox, QDoubleSpinBox) with modern appearance.

    Handles both normal and invalid states, changing border and background color if invalid.

    Args:
        widget (QWidget): The input widget to style.
        invalid (bool, optional): Whether to apply invalid styling. Defaults to False.
    """
    bg_color = "#ffcccc" if invalid else MODERN_COLORS["background"]
    border_color = MODERN_COLORS["danger"] if invalid else MODERN_COLORS["border"]

    base_style = f"""
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            border: 1px solid {border_color};
            border-radius: {SPACING['border_radius']};
            padding: 4px 8px;
            background-color: {bg_color};
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
            min-height: {WIDGET_SIZES['input_height']}px;
            color: {MODERN_COLORS['text']};
        }}
        QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {MODERN_COLORS['primary'] if not invalid else MODERN_COLORS['danger']};
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {MODERN_COLORS['focus']};
            outline: none;
        }}
        QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            background-color: {MODERN_COLORS['disabled']};
            color: {MODERN_COLORS['text_muted']};
        }}
    """

    # Add spinbox-specific styling
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        base_style += f"""
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                width: 16px;
                background: {MODERN_COLORS['surface']};
                border: none;
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background: {MODERN_COLORS['hover']};
            }}
        """

    widget.setStyleSheet(base_style)


def style_combo_box(widget: QComboBox) -> None:
    """
    Styles a QComboBox with modern appearance, including dropdown and item view.

    Args:
        widget (QComboBox): The combo box to style.
    """
    widget.setStyleSheet(
        f"""
        QComboBox {{
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: {SPACING['border_radius']};
            padding: 4px 8px;
            background-color: {MODERN_COLORS['background']};
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
            min-height: {WIDGET_SIZES['input_height']}px;
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
    """
    )


def style_label(widget: QLabel, style_type: str = "normal") -> None:
    """
    Styles a QLabel according to the specified type.

    Supported types: 'normal', 'heading', 'subheading', 'muted', 'caption', 'info', 'success', 'warning', 'error'.

    Args:
        widget (QLabel): The label to style.
        style_type (str, optional): The style type. Defaults to "normal".
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
        "subheading": {
            "color": MODERN_COLORS["text"],
            "size": FONT_SIZES["normal"],
            "weight": "500",
        },
        "muted": {
            "color": MODERN_COLORS["text_muted"],
            "size": FONT_SIZES["small"],
            "weight": "normal",
        },
        "caption": {
            "color": MODERN_COLORS["text_muted"],
            "size": FONT_SIZES["small"],
            "weight": "normal",
            "style": "italic",
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

    widget.setStyleSheet(
        f"""
        QLabel {{
            color: {style['color']};
            {BASE_FONT}
            font-size: {style['size']};
            font-weight: {style['weight']};
            {f"font-style: {style.get('style', 'normal')};" if 'style' in style else ""}
        }}
    """
    )


def style_table_widget(widget: QTableWidget) -> None:
    """
    Styles a QTableWidget with alternating row colors and modern header appearance.

    Args:
        widget (QTableWidget): The table widget to style.
    """
    widget.setAlternatingRowColors(True)
    widget.setStyleSheet(
        f"""
        QTableWidget {{
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: {SPACING['border_radius']};
            background-color: {MODERN_COLORS['background']};
            alternate-background-color: {MODERN_COLORS['surface']};
            gridline-color: {MODERN_COLORS['border']};
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
        }}
        QTableWidget::item {{
            padding: 4px;
        }}
        QTableWidget::item:selected {{
            background-color: {MODERN_COLORS['selected']};
            color: {MODERN_COLORS['text']};
        }}
        QHeaderView::section {{
            background-color: {MODERN_COLORS['surface']};
            border: none;
            border-bottom: 2px solid {MODERN_COLORS['border']};
            padding: 4px 8px;
            font-weight: 500;
        }}
    """
    )


def style_list_widget(widget: QListWidget) -> None:
    """
    Styles a QListWidget with modern appearance, including hover and selection states.

    Args:
        widget (QListWidget): The list widget to style.
    """
    widget.setStyleSheet(
        f"""
        QListWidget {{
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: {SPACING['border_radius']};
            background-color: {MODERN_COLORS['background']};
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
        }}
        QListWidget::item {{
            padding: 4px;
            border-bottom: 1px solid {MODERN_COLORS['surface']};
        }}
        QListWidget::item:selected {{
            background-color: {MODERN_COLORS['selected']};
            color: {MODERN_COLORS['text']};
        }}
        QListWidget::item:hover {{
            background-color: {MODERN_COLORS['hover']};
        }}
    """
    )


def style_progress_bar(widget: QProgressBar) -> None:
    """
    Styles a QProgressBar with modern appearance.

    Args:
        widget (QProgressBar): The progress bar to style.
    """
    widget.setStyleSheet(
        f"""
        QProgressBar {{
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: {SPACING['border_radius']};
            text-align: center;
            {BASE_FONT}
            font-size: {FONT_SIZES['small']};
            background-color: {MODERN_COLORS['surface']};
            min-height: 20px;
            max-height: 20px;
        }}
        QProgressBar::chunk {{
            background-color: {MODERN_COLORS['primary']};
            border-radius: 2px;
        }}
    """
    )


def style_group_box(widget: QGroupBox) -> None:
    """
    Styles a QGroupBox with modern appearance, including title and border.

    Args:
        widget (QGroupBox): The group box to style.
    """
    widget.setStyleSheet(
        f"""
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
    """
    )


def style_checkbox(widget: QCheckBox) -> None:
    """
    Styles a QCheckBox with modern appearance, including indicator and hover states.

    Args:
        widget (QCheckBox): The checkbox to style.
    """
    widget.setStyleSheet(
        f"""
        QCheckBox {{
            spacing: 4px;
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
            color: {MODERN_COLORS['text']};
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: 3px;
            background-color: {MODERN_COLORS['background']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {MODERN_COLORS['primary']};
            border-color: {MODERN_COLORS['primary']};
        }}
        QCheckBox::indicator:hover {{
            border-color: {MODERN_COLORS['primary']};
        }}
        QCheckBox::indicator:disabled {{
            background-color: {MODERN_COLORS['disabled']};
        }}
    """
    )


def apply_compact_layout(widget: QWidget, spacing: int = 8, margin: int = 10) -> None:
    """
    Applies compact spacing and margins to a widget's layout, if present.

    Args:
        widget (QWidget): The widget whose layout will be updated.
        spacing (int, optional): Spacing between items. Defaults to 8.
        margin (int, optional): Margin around the layout. Defaults to 10.
    """
    if widget.layout():
        widget.layout().setSpacing(spacing)
        widget.layout().setContentsMargins(margin, margin, margin, margin)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_styled_button(
    text: str, style_type: str = "secondary", parent: QWidget = None
) -> QPushButton:
    """
    Creates a QPushButton with the specified text and style, and returns it.

    Args:
        text (str): The button text.
        style_type (str, optional): The style type. Defaults to "secondary".
        parent (QWidget, optional): The parent widget. Defaults to None.

    Returns:
        QPushButton: The styled button instance.
    """
    button = QPushButton(text, parent)
    style_button(button, style_type)
    return button


def _get_base_stylesheet() -> str:
    """
    Returns the base stylesheet string for dialogs and main windows.

    Returns:
        str: The base stylesheet.
    """
    return f"""
        QDialog, QMainWindow {{
            background-color: {MODERN_COLORS['background']};
            {BASE_FONT}
            font-size: {FONT_SIZES['normal']};
        }}
        
        QWidget {{
            {BASE_FONT}
        }}
        
        QFrame[frameShape="4"] /* HLine */ {{
            color: {MODERN_COLORS['border']};
            max-height: 1px;
        }}
        
        QFrame[frameShape="5"] /* VLine */ {{
            color: {MODERN_COLORS['border']};
            max-width: 1px;
        }}
        
        /* Input field base styling - includes disabled state */
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            border: 1px solid {MODERN_COLORS['border']};
            border-radius: {SPACING['border_radius']};
            padding: 4px 8px;
            background-color: {MODERN_COLORS['background']};
            font-size: {FONT_SIZES['normal']};
            min-height: {WIDGET_SIZES['input_height']}px;
            color: {MODERN_COLORS['text']};
        }}
        
        QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {MODERN_COLORS['primary']};
        }}
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {MODERN_COLORS['focus']};
            outline: none;
        }}
        
        QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            background-color: {MODERN_COLORS['disabled']};
            color: {MODERN_COLORS['text_muted']};
            border-color: {MODERN_COLORS['border']};
        }}
    """

def _apply_palette(widget: QWidget) -> None:
    """
    Applies the PatchBatch color palette to a widget or QApplication.

    Args:
        widget (QWidget or QApplication): The widget or application to apply the palette to.
    """
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(MODERN_COLORS["background"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(MODERN_COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(MODERN_COLORS["background"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(MODERN_COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(MODERN_COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(MODERN_COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(MODERN_COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(MODERN_COLORS["selected"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(MODERN_COLORS["text"]))

    if hasattr(widget, "setPalette"):
        widget.setPalette(palette)
    elif isinstance(widget, QApplication):
        widget.setPalette(palette)


def apply_theme_to_application(app: QApplication) -> None:
    """
    Applies the PatchBatch theme to the entire QApplication instance.

    Args:
        app (QApplication): The application instance to style.
    """
    app.setStyle("Fusion")
    _apply_palette(app)
    app.setStyleSheet(_get_base_stylesheet())


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# Keep these for existing code that uses the old function names
# ============================================================================

style_main_window = apply_modern_theme
style_dialog = apply_modern_theme
style_splitter = apply_modern_theme
style_toolbar = lambda w: None  # No-op for compatibility
style_menu_bar = lambda w: None
style_menu = lambda w: None
style_status_bar = lambda w: None
style_input_invalid = lambda w: style_input_field(w, invalid=True)
style_input_valid = style_input_field
style_spinbox_with_arrows = style_input_field
style_combo_simple = style_combo_box
style_scroll_area = apply_modern_theme
apply_modern_style = apply_modern_theme
style_status_label = style_label
apply_widget_palette = apply_modern_theme
get_theme_stylesheet = _get_base_stylesheet


# Color utility functions for compatibility
def get_file_count_color(count: int, max_count: int = 10) -> str:
    """
    Returns a color string based on the file count relative to max_count.

    Args:
        count (int): The current file count.
        max_count (int, optional): The maximum file count. Defaults to 10.

    Returns:
        str: The color string for the count status.
    """
    return (
        MODERN_COLORS["success"]
        if count >= max_count // 2
        else MODERN_COLORS["warning"]
    )


def get_status_color(status: str) -> str:
    """
    Returns a color string based on the status string.

    Args:
        status (str): The status type (e.g., 'success', 'warning', 'error', 'info', 'muted').

    Returns:
        str: The color string for the status.
    """
    status_map = {
        "success": MODERN_COLORS["success"],
        "warning": MODERN_COLORS["warning"],
        "error": MODERN_COLORS["danger"],
        "info": MODERN_COLORS["primary"],
        "muted": MODERN_COLORS["text_muted"],
    }
    return status_map.get(status.lower(), MODERN_COLORS["text"])


def get_selection_summary_color(selected: int, total: int) -> str:
    """
    Returns a color string for selection summary based on selected and total counts.

    Args:
        selected (int): Number of selected items.
        total (int): Total number of items.

    Returns:
        str: The color string for the selection summary.
    """
    if selected == 0:
        return MODERN_COLORS["warning"]
    elif selected == total:
        return MODERN_COLORS["success"]
    else:
        return MODERN_COLORS["primary"]


# Typography and spacing exports for compatibility
TYPOGRAPHY = {"font_family": BASE_FONT}


# Button style getter for compatibility
def get_button_style(style_type: str = "secondary") -> str:
    """
    Returns the stylesheet string for a button of the specified style type.

    Args:
        style_type (str, optional): The style type. Defaults to "secondary".

    Returns:
        str: The stylesheet string for the button.
    """
    button = QPushButton()
    style_button(button, style_type)
    return button.styleSheet()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Core colors and constants
    "MODERN_COLORS",
    "WIDGET_SIZES",
    "SPACING",
    # Main styling functions
    "apply_modern_theme",
    "style_button",
    "style_input_field",
    "style_combo_box",
    "style_label",
    "style_table_widget",
    "style_list_widget",
    "style_progress_bar",
    "style_group_box",
    "style_checkbox",
    "apply_compact_layout",
    # Utility functions
    "create_styled_button",
    "apply_theme_to_application",
    # Backward compatibility
    "style_main_window",
    "style_dialog",
    "get_button_style",
    "get_file_count_color",
    "get_status_color",
    "get_selection_summary_color",
    "TYPOGRAPHY",
]
