"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module provides streamlined navigation toolbars for matplotlib plots
within a Qt-based GUI. It includes modern, minimal toolbars with essential
zoom/pan functionality and custom styling.
"""

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QAction, QIcon, QPixmap, QPainter, QColor, QFont

# Import centralized configuration from plot_style
from data_analysis_gui.config.plot_style import TOOLBAR_CONFIG, get_toolbar_style

import logging

logger = logging.getLogger(__name__)


class StreamlinedNavigationToolbar(NavigationToolbar):
    """
    StreamlinedNavigationToolbar provides a modern, minimal navigation toolbar
    for matplotlib plots in Qt GUIs. Only essential navigation tools are included,
    with custom icons and styling to match the application's appearance.

    Signals:
        mode_changed (str): Emitted when the zoom/pan mode changes ('zoom', 'pan', or 'none').
        reset_requested: Emitted when the Reset button is clicked to autofit to current data.
        plot_saved (str): Emitted when plot is saved (file path).

    Args:
        canvas: The matplotlib canvas to control.
        parent: Optional parent widget.
    """

    # Signal for when zoom/pan state changes
    mode_changed = Signal(str)  # 'zoom', 'pan', or 'none'

    # Signal for when plot is saved
    plot_saved = Signal(str)  # file path

    # Signal for reset/autofit request
    reset_requested = Signal()

    # Remove unnecessary default tool items
    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),  # Separator
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),  # Separator
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    )

    def __init__(self, canvas, parent=None, file_dialog_service=None):
        """
        Initialize the streamlined navigation toolbar.

        Args:
            canvas: The matplotlib canvas to attach the toolbar to.
            parent: Optional parent widget.
        """
        # Store the canvas reference before calling parent init
        self._canvas = canvas

        # Store file dialog service for save functionality
        self._file_dialog_service = file_dialog_service

        # Initialize current mode early
        self.current_mode = "none"

        # Initialize mode_label as None first
        self.mode_label = None

        # Call parent constructor
        super().__init__(canvas, parent)

        # Apply custom styling from centralized configuration
        self._apply_styling()

    def _init_toolbar(self):
        """
        Set up the toolbar with only essential navigation tools.
        Removes default items and adds custom actions and mode indicator.
        """
        # Clear any default items
        self.clear()

        # Add only the tools we want, in the order we want
        self._add_streamlined_tools()

        # Add a stretch to push everything to the left
        self.addStretch()

        # Add a subtle label for current mode with proper font size
        self.mode_label = QLabel("")
        self.mode_label.setStyleSheet(
            f"""
            QLabel {{
                color: #606060;
                font-size: {TOOLBAR_CONFIG['mode_label_font_size']}px;
                margin: 0px 10px;
            }}
        """
        )
        self.addWidget(self.mode_label)

        # Remove any unwanted default actions that may still exist
        self._remove_unwanted_actions()

    def save_figure(self, *args):
        """
        Override save_figure to use FileDialogService for directory memory.
        
        If file_dialog_service is available, uses it to remember the last save directory.
        Otherwise falls back to matplotlib's default save dialog.
        
        Emits plot_saved signal after successful save.
        """
        if self._file_dialog_service:
            # Use our file dialog service with directory memory
            from PySide6.QtWidgets import QMessageBox
            
            # Define supported file formats
            file_types = (
                "PNG files (*.png);;"
                "PDF files (*.pdf);;"
                "SVG files (*.svg);;"
                "JPEG files (*.jpg *.jpeg);;"
                "All files (*.*)"
            )
            
            # Get save path using the service
            file_path = self._file_dialog_service.get_export_path(
                parent=self.parent() or self._canvas.parent(),
                suggested_name="plot.png",
                file_types=file_types,
                dialog_type="save_plot"
            )
            
            if file_path:
                try:
                    # Save the figure using matplotlib
                    self.canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Plot saved to: {file_path}")
                    
                    # Emit signal after successful save
                    self.plot_saved.emit(file_path)
                    
                except Exception as e:
                    logger.error(f"Failed to save plot: {e}")
                    # Show error dialog
                    QMessageBox.critical(
                        self.parent() or self._canvas.parent(),
                        "Save Error",
                        f"Failed to save plot:\n{str(e)}"
                    )
        else:
            # Fallback to matplotlib's default save dialog
            super().save_figure(*args)

    def _remove_unwanted_actions(self):
        """
        Remove any unwanted actions that matplotlib might have added.
        This is a safety net in case toolitems override doesn't fully work.
        """
        # Get all actions currently in the toolbar
        all_actions = self.actions()
        
        # Define the text/tooltips of actions we want to keep
        keep_texts = {'Reset', 'Back', 'Forward', 'Pan', 'Zoom', 'Save'}
        
        # Remove any action that's not in our keep list and isn't a separator
        for action in all_actions:
            if action.isSeparator():
                continue
            action_text = action.text()
            if action_text and action_text not in keep_texts:
                self.removeAction(action)
                action.setVisible(False)

    def _add_streamlined_tools(self):
        """
        Add essential navigation actions (reset, back, forward, pan, zoom, save)
        with custom icons and styling.
        """

        # Create font for toolbar actions
        toolbar_font = QFont()
        toolbar_font.setPointSize(TOOLBAR_CONFIG["button_font_size"])

        # Reset (autofit to current data)
        self.home_action = QAction("Reset", self)
        self.home_action.setToolTip("Fit to current data")
        self.home_action.triggered.connect(self.home)
        self.home_action.setIcon(self._create_icon("home"))
        self.home_action.setFont(toolbar_font)
        self.addAction(self.home_action)

        # Back/Forward navigation
        self.back_action = QAction("Back", self)
        self.back_action.setToolTip("Back to previous view")
        self.back_action.triggered.connect(self.back)
        self.back_action.setIcon(self._create_icon("back"))
        self.back_action.setFont(toolbar_font)
        self.addAction(self.back_action)

        self.forward_action = QAction("Forward", self)
        self.forward_action.setToolTip("Forward to next view")
        self.forward_action.triggered.connect(self.forward)
        self.forward_action.setIcon(self._create_icon("forward"))
        self.forward_action.setFont(toolbar_font)
        self.addAction(self.forward_action)

        self.addSeparator()

        # Pan
        self.pan_action = QAction("Pan", self)
        self.pan_action.setToolTip("Pan axes with left mouse, zoom with right")
        self.pan_action.setCheckable(True)
        self.pan_action.triggered.connect(self.pan)
        self.pan_action.setIcon(self._create_icon("pan"))
        self.pan_action.setFont(toolbar_font)
        self.addAction(self.pan_action)

        # Zoom
        self.zoom_action = QAction("Zoom", self)
        self.zoom_action.setToolTip("Zoom to rectangle")
        self.zoom_action.setCheckable(True)
        self.zoom_action.triggered.connect(self.zoom)
        self.zoom_action.setIcon(self._create_icon("zoom"))
        self.zoom_action.setFont(toolbar_font)
        self.addAction(self.zoom_action)

        self.addSeparator()

        # Save (optional - can be removed if export is handled elsewhere)
        self.save_action = QAction("Save", self)
        self.save_action.setToolTip("Save the figure")
        self.save_action.triggered.connect(self.save_figure)
        self.save_action.setIcon(self._create_icon("save"))
        self.save_action.setFont(toolbar_font)
        self.addAction(self.save_action)

    def _create_icon(self, icon_type: str) -> QIcon:
        """
        Programmatically create simple, modern icons for toolbar actions.

        Args:
            icon_type (str): The type of icon to create ('home', 'back', 'forward', 'pan', 'zoom', 'save').

        Returns:
            QIcon: The generated icon.
        """
        # Use icon size from centralized configuration
        icon_size = TOOLBAR_CONFIG["icon_size"]

        # Create a pixmap for the icon
        pixmap = QPixmap(icon_size, icon_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Icon color
        color = QColor("#606060")
        painter.setPen(color)
        painter.setBrush(color)

        # Scale factor for larger icons
        scale = icon_size / 16.0  # Original designs were for 16x16

        if icon_type == "home":
            # Simple house shape (scaled)
            painter.drawLine(
                int(8 * scale), int(4 * scale), int(3 * scale), int(9 * scale)
            )
            painter.drawLine(
                int(8 * scale), int(4 * scale), int(13 * scale), int(9 * scale)
            )
            painter.drawRect(
                int(5 * scale), int(9 * scale), int(6 * scale), int(5 * scale)
            )

        elif icon_type == "back":
            # Left arrow (scaled)
            painter.drawLine(
                int(5 * scale), int(8 * scale), int(11 * scale), int(4 * scale)
            )
            painter.drawLine(
                int(5 * scale), int(8 * scale), int(11 * scale), int(12 * scale)
            )
            painter.drawLine(
                int(5 * scale), int(8 * scale), int(13 * scale), int(8 * scale)
            )

        elif icon_type == "forward":
            # Right arrow (scaled)
            painter.drawLine(
                int(11 * scale), int(8 * scale), int(5 * scale), int(4 * scale)
            )
            painter.drawLine(
                int(11 * scale), int(8 * scale), int(5 * scale), int(12 * scale)
            )
            painter.drawLine(
                int(11 * scale), int(8 * scale), int(3 * scale), int(8 * scale)
            )

        elif icon_type == "pan":
            # Hand/move icon (scaled)
            painter.drawLine(
                int(8 * scale), int(3 * scale), int(8 * scale), int(13 * scale)
            )
            painter.drawLine(
                int(3 * scale), int(8 * scale), int(13 * scale), int(8 * scale)
            )
            painter.drawLine(
                int(5 * scale), int(5 * scale), int(8 * scale), int(3 * scale)
            )
            painter.drawLine(
                int(11 * scale), int(5 * scale), int(8 * scale), int(3 * scale)
            )
            painter.drawLine(
                int(5 * scale), int(11 * scale), int(8 * scale), int(13 * scale)
            )
            painter.drawLine(
                int(11 * scale), int(11 * scale), int(8 * scale), int(13 * scale)
            )

        elif icon_type == "zoom":
            # Magnifying glass (scaled)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                int(4 * scale), int(4 * scale), int(7 * scale), int(7 * scale)
            )
            painter.drawLine(
                int(10 * scale), int(10 * scale), int(13 * scale), int(13 * scale)
            )

        elif icon_type == "save":
            # Floppy disk / save icon (scaled)
            painter.drawRect(
                int(3 * scale), int(3 * scale), int(10 * scale), int(10 * scale)
            )
            painter.fillRect(
                int(5 * scale),
                int(3 * scale),
                int(6 * scale),
                int(4 * scale),
                QColor("white"),
            )
            painter.fillRect(
                int(9 * scale), int(4 * scale), int(2 * scale), int(2 * scale), color
            )

        painter.end()

        return QIcon(pixmap)

    def release_zoom(self, event):
        """
        Override release_zoom to automatically disable zoom mode after a successful zoom.
        
        If the user performs an actual zoom operation (changes the view), zoom mode is
        automatically turned off. If no zoom occurs (e.g., click without drag), zoom
        mode stays active for the next attempt.
        
        Args:
            event: Matplotlib mouse button release event.
        """
        # Capture current axis limits before zoom operation
        old_xlim = self.canvas.figure.axes[0].get_xlim() if self.canvas.figure.axes else None
        old_ylim = self.canvas.figure.axes[0].get_ylim() if self.canvas.figure.axes else None
        
        # Perform the actual zoom operation
        super().release_zoom(event)
        
        # Check if limits changed (i.e., zoom actually occurred)
        if old_xlim is not None and old_ylim is not None and self.canvas.figure.axes:
            new_xlim = self.canvas.figure.axes[0].get_xlim()
            new_ylim = self.canvas.figure.axes[0].get_ylim()
            
            # Compare limits - if they changed, a zoom occurred
            zoom_occurred = (old_xlim != new_xlim or old_ylim != new_ylim)
            
            if zoom_occurred:
                # Automatically disable zoom mode by calling zoom() to toggle it off
                # This properly handles both UI state and matplotlib's internal state
                if self._actions["zoom"].isChecked():
                    self.zoom()  # Toggle zoom mode off
                    logger.debug("Zoom completed - automatically disabled zoom mode")
            else:
                # No zoom occurred (e.g., click without drag), keep zoom mode active
                logger.debug("No zoom change detected - keeping zoom mode active")

    def _apply_styling(self):
        """
        Apply custom styling and configuration to the toolbar, including icon size,
        button style, and disabling toolbar movement.
        """
        # Get stylesheet from centralized configuration
        self.setStyleSheet(get_toolbar_style())

        # Set icon size using configuration
        self.setIconSize(
            QSize(TOOLBAR_CONFIG["icon_size"], TOOLBAR_CONFIG["icon_size"])
        )
        self.setMovable(False)

        # Set toolbar button style to show text beside icons
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

    def pan(self, *args):
        """
        Override pan action to update the mode indicator and emit mode_changed signal.
        Ensures only one of pan/zoom is active at a time.
        """
        super().pan(*args)

        # Check if mode_label exists before using it
        if hasattr(self, "mode_label") and self.mode_label:
            if self._actions["pan"].isChecked():
                self.current_mode = "pan"
                self.mode_label.setText("Pan Mode")
                # Uncheck zoom action if it's checked
                if hasattr(self, "zoom_action"):
                    self.zoom_action.setChecked(False)
            else:
                self.current_mode = "none"
                self.mode_label.setText("")
        else:
            # Just update the mode without touching the label
            if self._actions["pan"].isChecked():
                self.current_mode = "pan"
            else:
                self.current_mode = "none"

        self.mode_changed.emit(self.current_mode)

    def zoom(self, *args):
        """
        Override zoom action to update the mode indicator and emit mode_changed signal.
        Ensures only one of pan/zoom is active at a time.
        """
        super().zoom(*args)

        # Check if mode_label exists before using it
        if hasattr(self, "mode_label") and self.mode_label:
            if self._actions["zoom"].isChecked():
                self.current_mode = "zoom"
                self.mode_label.setText("Zoom Mode")
                # Uncheck pan action if it's checked
                if hasattr(self, "pan_action"):
                    self.pan_action.setChecked(False)
            else:
                self.current_mode = "none"
                self.mode_label.setText("")
        else:
            # Just update the mode without touching the label
            if self._actions["zoom"].isChecked():
                self.current_mode = "zoom"
            else:
                self.current_mode = "none"

        self.mode_changed.emit(self.current_mode)

    def home(self, *args):
        """
        Override home action to emit reset_requested signal for autofitting
        to current data. Also clears pan/zoom modes and updates mode indicator.
        """
        # Uncheck both pan and zoom actions
        if hasattr(self, "pan_action"):
            self.pan_action.setChecked(False)
        if hasattr(self, "zoom_action"):
            self.zoom_action.setChecked(False)

        self.current_mode = "none"

        # Update label if it exists
        if hasattr(self, "mode_label") and self.mode_label:
            self.mode_label.setText("")

        # Emit signal for PlotManager to handle autofit
        self.reset_requested.emit()

        # Emit mode change
        self.mode_changed.emit(self.current_mode)


class MinimalNavigationToolbar(QWidget):
    """
    MinimalNavigationToolbar provides a highly simplified toolbar for dialogs or
    secondary windows, with only zoom, pan, and reset controls.

    Signals:
        mode_changed (str): Emitted when the zoom/pan mode changes.

    Args:
        canvas: The matplotlib canvas to control.
        parent: Optional parent widget.
    """

    mode_changed = Signal(str)

    def __init__(self, canvas, parent=None):
        """
        Initialize the minimal navigation toolbar with zoom, pan, and reset buttons.

        Args:
            canvas: The matplotlib canvas to attach the toolbar to.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.canvas = canvas
        self.toolbar = NavigationToolbar(canvas, self)
        self.toolbar.setVisible(False)  # Hide the actual toolbar

        # Create minimal UI
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Just zoom and pan buttons
        self.zoom_btn = self._create_tool_button("Zoom", "zoom")
        self.pan_btn = self._create_tool_button("Pan", "pan")
        self.reset_btn = self._create_tool_button("Reset", "reset")

        # Create label with proper font size from configuration
        tools_label = QLabel("Tools:")
        tools_label.setStyleSheet(f"font-size: {TOOLBAR_CONFIG['button_font_size']}px;")

        layout.addWidget(tools_label)
        layout.addWidget(self.zoom_btn)
        layout.addWidget(self.pan_btn)
        layout.addWidget(self.reset_btn)
        layout.addStretch()

        # Connect buttons
        self.zoom_btn.clicked.connect(self._toggle_zoom)
        self.pan_btn.clicked.connect(self._toggle_pan)
        self.reset_btn.clicked.connect(self._reset_view)

        self.current_mode = "none"

    def _create_tool_button(self, text: str, mode: str):
        """
        Create a styled QPushButton for toolbar actions.

        Args:
            text (str): Button label.
            mode (str): Action mode ('zoom', 'pan', or 'reset').

        Returns:
            QPushButton: The configured button.
        """
        from PySide6.QtWidgets import QPushButton

        btn = QPushButton(text)
        btn.setCheckable(mode != "reset")
        btn.setMaximumHeight(TOOLBAR_CONFIG["button_min_height"])
        btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #F0F0F0;
                border: 1px solid #C0C0C0;
                border-radius: 3px;
                padding: {TOOLBAR_CONFIG['button_padding']};
                font-size: {TOOLBAR_CONFIG['button_font_size']}px;
                font-weight: 500;
                min-height: {TOOLBAR_CONFIG['button_min_height'] - 4}px;
            }}
            QPushButton:hover {{
                background-color: #E0E0E0;
            }}
            QPushButton:checked {{
                background-color: #D8E4F0;
                border-color: #2E86AB;
            }}
        """
        )
        return btn

    def _toggle_zoom(self):
        """
        Toggle zoom mode on the toolbar and update button states.
        Emits mode_changed signal.
        """
        if self.zoom_btn.isChecked():
            self.toolbar.zoom()
            self.pan_btn.setChecked(False)
            self.current_mode = "zoom"
        else:
            self.toolbar.zoom()  # Toggle off
            self.current_mode = "none"
        self.mode_changed.emit(self.current_mode)

    def _toggle_pan(self):
        """
        Toggle pan mode on the toolbar and update button states.
        Emits mode_changed signal.
        """
        if self.pan_btn.isChecked():
            self.toolbar.pan()
            self.zoom_btn.setChecked(False)
            self.current_mode = "pan"
        else:
            self.toolbar.pan()  # Toggle off
            self.current_mode = "none"
        self.mode_changed.emit(self.current_mode)

    def _reset_view(self):
        """
        Reset the plot view to its original state and clear pan/zoom modes.
        Emits mode_changed signal.
        """
        self.toolbar.home()
        self.zoom_btn.setChecked(False)
        self.pan_btn.setChecked(False)
        self.current_mode = "none"
        self.mode_changed.emit(self.current_mode)
