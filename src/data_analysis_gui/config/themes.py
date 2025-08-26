from PyQt5.QtCore import Qt

THEMES = {
    "Light": {
        "background": "#f5f5f5",
        "group_border": "#cccccc",
        "button_bg": "#4CAF50",
        "button_fg": "white",
        "button_disabled_bg": "#cccccc",
        "button_disabled_fg": "#666666",
        "button_hover": "#45a049",
        "text_color": "#333",
        "status_bar_bg": "#e0e0e0",
        "toolbar_bg": "#f0f0f0",
    },
    "Dark": {
        "background": "#2e2e2e",
        "group_border": "#555555",
        "button_bg": "#5a9b5d",
        "button_fg": "white",
        "button_disabled_bg": "#444444",
        "button_disabled_fg": "#888888",
        "button_hover": "#6cad6f",
        "text_color": "#dddddd",
        "status_bar_bg": "#1e1e1e",
        "toolbar_bg": "#3a3a3a",
    },
    "Oceanic Blue": {
        "background": "#f0f8ff",
        "group_border": "#b0c4de",
        "button_bg": "#4682b4",
        "button_fg": "white",
        "button_disabled_bg": "#b0c4de",
        "button_disabled_fg": "#666666",
        "button_hover": "#5a9bd8",
        "text_color": "#000033",
        "status_bar_bg": "#add8e6",
        "toolbar_bg": "#ddeeff",
    },
    "Forest Green": {
        "background": "#f5fff5",
        "group_border": "#8fbc8f",
        "button_bg": "#2e8b57",
        "button_fg": "white",
        "button_disabled_bg": "#a9d8a9",
        "button_disabled_fg": "#666666",
        "button_hover": "#3cb371",
        "text_color": "#003300",
        "status_bar_bg": "#98fb98",
        "toolbar_bg": "#e0f8e0",
    },
    "High Contrast": {
        "background": "black",
        "group_border": "#f0e68c",
        "button_bg": "#f0e68c",
        "button_fg": "black",
        "button_disabled_bg": "#555555",
        "button_disabled_fg": "#aaaaaa",
        "button_hover": "#fffacd",
        "text_color": "#f0e68c",
        "status_bar_bg": "#333333",
        "toolbar_bg": "#222222",
    },
    "Slate Gray": {
        "background": "#f8f9fa",
        "group_border": "#adb5bd",
        "button_bg": "#495057",
        "button_fg": "white",
        "button_disabled_bg": "#ced4da",
        "button_disabled_fg": "#6c757d",
        "button_hover": "#6c757d",
        "text_color": "#212529",
        "status_bar_bg": "#dee2e6",
        "toolbar_bg": "#e9ecef",
    },
    "Vintage Wine": {
        "background": "#fff0f5",
        "group_border": "#dda0dd",
        "button_bg": "#800020",
        "button_fg": "white",
        "button_disabled_bg": "#e6cfe6",
        "button_disabled_fg": "#8d6c8d",
        "button_hover": "#990033",
        "text_color": "#4b002b",
        "status_bar_bg": "#e6e6fa",
        "toolbar_bg": "#f8f0f8",
    },
    "Sunrise Orange": {
        "background": "#fffaf0",
        "group_border": "#ffcc99",
        "button_bg": "#ff7f50",
        "button_fg": "white",
        "button_disabled_bg": "#ffdcb2",
        "button_disabled_fg": "#8b5a2b",
        "button_hover": "#ff6347",
        "text_color": "#8b4513",
        "status_bar_bg": "#ffe4b5",
        "toolbar_bg": "#fff0e1",
    },
}

def get_theme_stylesheet(theme):
    """Generate Qt stylesheet from theme dictionary."""
    
    return f"""
    QMainWindow {{
        background-color: {theme['background']};
    }}
    QGroupBox {{
        font-weight: bold;
        border: 2px solid {theme['group_border']};
        border-radius: 5px;
        margin: 5px;
        padding-top: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
    }}
    QPushButton {{
        background-color: {theme['button_bg']};
        color: {theme['button_fg']};
        border: none;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: bold;
    }}
    QPushButton:disabled {{
        background-color: {theme['button_disabled_bg']};
        color: {theme['button_disabled_fg']};
    }}
    QPushButton:hover {{
        background-color: {theme['button_hover']};
    }}
    QPushButton:pressed {{
        background-color: #3d8b40;
    }}
    QLabel {{
        color: {theme['text_color']};
    }}
    QCheckBox {{
        color: {theme['text_color']};
    }}
    QStatusBar {{
        background-color: {theme['status_bar_bg']};
        color: {theme['text_color']};
    }}
    QToolBar {{
        background-color: {theme['toolbar_bg']};
        border: 1px solid {theme['group_border']};
        spacing: 3px;
    }}
    QComboBox, QDoubleSpinBox, QSpinBox {{
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 4px;
        background-color: white;
    }}
    """