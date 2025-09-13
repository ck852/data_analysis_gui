# widgets/custom_toolbar.py
"""
Streamlined navigation toolbar for matplotlib plots.
Provides essential zoom/pan functionality with a modern appearance.
"""

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5.QtWidgets import QToolBar, QAction, QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor


class StreamlinedNavigationToolbar(NavigationToolbar2QT):
    """
    A cleaner, more modern navigation toolbar for matplotlib plots.
    Keeps only essential functions and matches GUI styling.
    """
    
    # Signal for when zoom/pan state changes
    mode_changed = pyqtSignal(str)  # 'zoom', 'pan', or 'none'
    
    def __init__(self, canvas, parent=None):
        """
        Initialize the streamlined toolbar.
        
        Args:
            canvas: The matplotlib canvas
            parent: Parent widget
        """
        # Store the canvas reference before calling parent init
        self._canvas = canvas
        
        # Call parent constructor
        super().__init__(canvas, parent)
        
        # Apply custom styling
        self._apply_styling()
        
        # Track current mode
        self.current_mode = 'none'
    
    def _init_toolbar(self):
        """
        Override to create only the tools we want.
        This method is called by the parent __init__.
        """
        # Clear any default items
        self.clear()
        
        # Add only the tools we want, in the order we want
        self._add_streamlined_tools()
        
        # Add a stretch to push everything to the left
        self.addStretch()
        
        # Add a subtle label for current mode
        self.mode_label = QLabel("")
        self.mode_label.setStyleSheet("""
            QLabel {
                color: #606060;
                font-size: 9px;
                margin: 0px 10px;
            }
        """)
        self.addWidget(self.mode_label)
    
    def _add_streamlined_tools(self):
        """Add only essential navigation tools with custom icons."""
        # Home (reset view)
        self.home_action = QAction("Reset", self)
        self.home_action.setToolTip("Reset to original view")
        self.home_action.triggered.connect(self.home)
        self.home_action.setIcon(self._create_icon('home'))
        self.addAction(self.home_action)
        
        # Back/Forward navigation
        self.back_action = QAction("Back", self)
        self.back_action.setToolTip("Back to previous view")
        self.back_action.triggered.connect(self.back)
        self.back_action.setIcon(self._create_icon('back'))
        self.addAction(self.back_action)
        
        self.forward_action = QAction("Forward", self)
        self.forward_action.setToolTip("Forward to next view")
        self.forward_action.triggered.connect(self.forward)
        self.forward_action.setIcon(self._create_icon('forward'))
        self.addAction(self.forward_action)
        
        self.addSeparator()
        
        # Pan
        self.pan_action = QAction("Pan", self)
        self.pan_action.setToolTip("Pan axes with left mouse, zoom with right")
        self.pan_action.setCheckable(True)
        self.pan_action.triggered.connect(self.pan)
        self.pan_action.setIcon(self._create_icon('pan'))
        self.addAction(self.pan_action)
        
        # Zoom
        self.zoom_action = QAction("Zoom", self)
        self.zoom_action.setToolTip("Zoom to rectangle")
        self.zoom_action.setCheckable(True)
        self.zoom_action.triggered.connect(self.zoom)
        self.zoom_action.setIcon(self._create_icon('zoom'))
        self.addAction(self.zoom_action)
        
        self.addSeparator()
        
        # Save (optional - can be removed if export is handled elsewhere)
        self.save_action = QAction("Save", self)
        self.save_action.setToolTip("Save the figure")
        self.save_action.triggered.connect(self.save_figure)
        self.save_action.setIcon(self._create_icon('save'))
        self.addAction(self.save_action)
    
    def _create_icon(self, icon_type: str) -> QIcon:
        """
        Create simple, modern icons programmatically.
        
        Args:
            icon_type: Type of icon to create
            
        Returns:
            QIcon object
        """
        # Create a pixmap for the icon
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Icon color
        color = QColor('#606060')
        painter.setPen(color)
        painter.setBrush(color)
        
        if icon_type == 'home':
            # Simple house shape
            painter.drawLine(8, 4, 3, 9)
            painter.drawLine(8, 4, 13, 9)
            painter.drawRect(5, 9, 6, 5)
        
        elif icon_type == 'back':
            # Left arrow
            painter.drawLine(5, 8, 11, 4)
            painter.drawLine(5, 8, 11, 12)
            painter.drawLine(5, 8, 13, 8)
        
        elif icon_type == 'forward':
            # Right arrow
            painter.drawLine(11, 8, 5, 4)
            painter.drawLine(11, 8, 5, 12)
            painter.drawLine(11, 8, 3, 8)
        
        elif icon_type == 'pan':
            # Hand/move icon
            painter.drawLine(8, 3, 8, 13)
            painter.drawLine(3, 8, 13, 8)
            painter.drawLine(5, 5, 8, 3)
            painter.drawLine(11, 5, 8, 3)
            painter.drawLine(5, 11, 8, 13)
            painter.drawLine(11, 11, 8, 13)
        
        elif icon_type == 'zoom':
            # Magnifying glass
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(4, 4, 7, 7)
            painter.drawLine(10, 10, 13, 13)
        
        elif icon_type == 'save':
            # Floppy disk / save icon
            painter.drawRect(3, 3, 10, 10)
            painter.fillRect(5, 3, 6, 4, QColor('white'))
            painter.fillRect(9, 4, 2, 2, color)
        
        painter.end()
        
        return QIcon(pixmap)
    
    def _apply_styling(self):
        """Apply custom styling to match the GUI."""
        self.setStyleSheet("""
            QToolBar {
                background-color: #F5F5F5;
                border: none;
                border-bottom: 1px solid #D0D0D0;
                padding: 2px;
                spacing: 2px;
            }
            
            QToolBar::separator {
                background-color: #D0D0D0;
                width: 1px;
                margin: 4px 6px;
            }
            
            QToolButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 4px;
                margin: 1px;
            }
            
            QToolButton:hover {
                background-color: #E0E0E0;
                border: 1px solid #C0C0C0;
            }
            
            QToolButton:pressed {
                background-color: #D0D0D0;
                border: 1px solid #B0B0B0;
            }
            
            QToolButton:checked {
                background-color: #D8E4F0;
                border: 1px solid #2E86AB;
            }
        """)
        
        # Make the toolbar more compact
        self.setIconSize(self.iconSize() * 0.8)
        self.setMovable(False)
    
    def pan(self, *args):
        """Override pan to update mode indicator."""
        super().pan(*args)
        if self._actions['pan'].isChecked():
            self.current_mode = 'pan'
            self.mode_label.setText("Pan Mode")
            self.zoom_action.setChecked(False)
        else:
            self.current_mode = 'none'
            self.mode_label.setText("")
        self.mode_changed.emit(self.current_mode)
    
    def zoom(self, *args):
        """Override zoom to update mode indicator."""
        super().zoom(*args)
        if self._actions['zoom'].isChecked():
            self.current_mode = 'zoom'
            self.mode_label.setText("Zoom Mode")
            self.pan_action.setChecked(False)
        else:
            self.current_mode = 'none'
            self.mode_label.setText("")
        self.mode_changed.emit(self.current_mode)
    
    def home(self, *args):
        """Override home to clear mode."""
        super().home(*args)
        self.pan_action.setChecked(False)
        self.zoom_action.setChecked(False)
        self.current_mode = 'none'
        self.mode_label.setText("")
        self.mode_changed.emit(self.current_mode)


class MinimalNavigationToolbar(QWidget):
    """
    An even more minimal toolbar with just zoom/pan toggle.
    For use in dialogs and secondary windows.
    """
    
    mode_changed = pyqtSignal(str)
    
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.toolbar = NavigationToolbar2QT(canvas, self)
        self.toolbar.setVisible(False)  # Hide the actual toolbar
        
        # Create minimal UI
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Just zoom and pan buttons
        self.zoom_btn = self._create_tool_button("Zoom", "zoom")
        self.pan_btn = self._create_tool_button("Pan", "pan")
        self.reset_btn = self._create_tool_button("Reset", "reset")
        
        layout.addWidget(QLabel("Tools:"))
        layout.addWidget(self.zoom_btn)
        layout.addWidget(self.pan_btn)
        layout.addWidget(self.reset_btn)
        layout.addStretch()
        
        # Connect buttons
        self.zoom_btn.clicked.connect(self._toggle_zoom)
        self.pan_btn.clicked.connect(self._toggle_pan)
        self.reset_btn.clicked.connect(self._reset_view)
        
        self.current_mode = 'none'
    
    def _create_tool_button(self, text: str, mode: str):
        """Create a styled tool button."""
        from PyQt5.QtWidgets import QPushButton
        
        btn = QPushButton(text)
        btn.setCheckable(mode != 'reset')
        btn.setMaximumHeight(24)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #F0F0F0;
                border: 1px solid #C0C0C0;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
            }
            QPushButton:checked {
                background-color: #D8E4F0;
                border-color: #2E86AB;
            }
        """)
        return btn
    
    def _toggle_zoom(self):
        """Toggle zoom mode."""
        if self.zoom_btn.isChecked():
            self.toolbar.zoom()
            self.pan_btn.setChecked(False)
            self.current_mode = 'zoom'
        else:
            self.toolbar.zoom()  # Toggle off
            self.current_mode = 'none'
        self.mode_changed.emit(self.current_mode)
    
    def _toggle_pan(self):
        """Toggle pan mode."""
        if self.pan_btn.isChecked():
            self.toolbar.pan()
            self.zoom_btn.setChecked(False)
            self.current_mode = 'pan'
        else:
            self.toolbar.pan()  # Toggle off
            self.current_mode = 'none'
        self.mode_changed.emit(self.current_mode)
    
    def _reset_view(self):
        """Reset to home view."""
        self.toolbar.home()
        self.zoom_btn.setChecked(False)
        self.pan_btn.setChecked(False)
        self.current_mode = 'none'
        self.mode_changed.emit(self.current_mode)