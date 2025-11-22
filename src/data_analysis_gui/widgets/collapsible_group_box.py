"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Collapsible group box widget with animated expand/collapse.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont, QCursor

from data_analysis_gui.config.themes import MODERN_COLORS
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class CollapsibleGroupBox(QWidget):
    """
    A collapsible group box with animated expand/collapse functionality.
    
    Features:
        - Clickable header with title and toggle arrow
        - Smooth animation when expanding/collapsing
        - Styled to match modern theme
        - Content area accepts any widget layout
    
    Signals:
        toggled(bool): Emitted when collapsed state changes (True = expanded)
    """
    
    toggled = Signal(bool)
    
    def __init__(self, title: str = "", parent=None):
        """
        Initialize the collapsible group box.
        
        Args:
            title: Title text for the header
            parent: Parent widget
        """
        super().__init__(parent)
        
        self._is_collapsed = False
        self._title = title
        
        self._init_ui()
        self._apply_styling()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header frame (clickable)
        self.header_frame = QFrame()
        self.header_frame.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.header_frame.mousePressEvent = self._on_header_clicked
        
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(8)
        
        # Toggle arrow
        self.arrow_label = QLabel("▼")
        arrow_font = QFont()
        arrow_font.setPointSize(10)
        self.arrow_label.setFont(arrow_font)
        header_layout.addWidget(self.arrow_label)
        
        # Title label
        self.title_label = QLabel(self._title)
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        main_layout.addWidget(self.header_frame)
        
        # Content container (the collapsible part)
        self.content_widget = QFrame()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(4)
        
        main_layout.addWidget(self.content_widget)
        
        # Animation
        self.animation = QPropertyAnimation(self.content_widget, b"maximumHeight")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
    
    def _apply_styling(self):
        """Apply modern theme styling."""
        self.setStyleSheet(f"""
            CollapsibleGroupBox {{
                background-color: transparent;
            }}
            
            QFrame#header {{
                background-color: {MODERN_COLORS['surface']};
                border: 1px solid {MODERN_COLORS['border']};
                border-radius: 4px;
            }}
            
            QFrame#header:hover {{
                background-color: {MODERN_COLORS['hover']};
            }}
            
            QFrame#content {{
                background-color: {MODERN_COLORS['background']};
                border: 1px solid {MODERN_COLORS['border']};
                border-top: none;
                border-radius: 0px 0px 4px 4px;
            }}
        """)
        
        self.header_frame.setObjectName("header")
        self.content_widget.setObjectName("content")
    
    def _on_header_clicked(self, event):
        """Handle header click to toggle collapse state."""
        self.toggle()
    
    def toggle(self):
        """Toggle between collapsed and expanded states."""
        if self._is_collapsed:
            self.expand()
        else:
            self.collapse()
    
    def collapse(self):
        """Collapse the content area with animation."""
        if self._is_collapsed:
            return
        
        self._is_collapsed = True
        self.arrow_label.setText("▶")
        
        # Animate collapse
        self.animation.setStartValue(self.content_widget.height())
        self.animation.setEndValue(0)
        self.animation.start()

        # Hide content when collapsed
        self.content_widget.setVisible(False)
        self.content_widget.setMaximumHeight(0)
        
        self.toggled.emit(False)
        logger.debug(f"Collapsed section: {self._title}")
    
    def expand(self):
        """Expand the content area with animation."""
        if not self._is_collapsed:
            return
        
        self._is_collapsed = False
        self.arrow_label.setText("▼")

        # Show content when expanding
        self.content_widget.setVisible(True)
        
        # Get the natural height of content
        self.content_widget.setMaximumHeight(16777215)  # Remove constraint temporarily
        content_height = self.content_widget.sizeHint().height()
        
        # Animate expand
        self.animation.setStartValue(0)
        self.animation.setEndValue(content_height)
        self.animation.start()
        
        self.toggled.emit(True)
        logger.debug(f"Expanded section: {self._title}")
    
    def set_collapsed(self, collapsed: bool):
        """
        Set collapsed state without animation.
        
        Args:
            collapsed: True to collapse, False to expand
        """
        if collapsed:
            self._is_collapsed = False  # Set to opposite so collapse() works
            self.arrow_label.setText("▶")
            self.content_widget.setMaximumHeight(0)
            self._is_collapsed = True
        else:
            self._is_collapsed = True  # Set to opposite so expand() works
            self.arrow_label.setText("▼")
            self.content_widget.setMaximumHeight(16777215)
            self._is_collapsed = False
        
        self.toggled.emit(not collapsed)
    
    def is_collapsed(self) -> bool:
        """Return whether the section is currently collapsed."""
        return self._is_collapsed
    
    def set_title(self, title: str):
        """
        Set the title text.
        
        Args:
            title: New title text
        """
        self._title = title
        self.title_label.setText(title)
    
    def get_content_layout(self):
        """
        Get the layout for adding content widgets.
        
        Returns:
            QVBoxLayout where content should be added
        """
        return self.content_layout
    
    def add_widget(self, widget: QWidget):
        """
        Convenience method to add a widget to the content area.
        
        Args:
            widget: Widget to add
        """
        self.content_layout.addWidget(widget)