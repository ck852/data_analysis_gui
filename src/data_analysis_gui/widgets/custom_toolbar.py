"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Defines plot toolbars for MainWindow, batch analysis results window, and current density results window.
Enables custom, trimmed matplotlib navigation toolbars with necessary functionality only.
"""

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QAction, QIcon, QPixmap, QPainter, QColor, QFont

from data_analysis_gui.config.plot_style import TOOLBAR_CONFIG, get_toolbar_style

import logging

logger = logging.getLogger(__name__)


class StreamlinedNavigationToolbar(NavigationToolbar):
    """
    Customized matplotlib navigation toolbar with necessary tools only. Designed for MainWindow 
    where it is used by PlotManager.
    
    Automatically disables zoom mode after completing a zoom operation for better UX.
    """

    mode_changed = Signal(str)
    plot_saved = Signal(str)
    reset_requested = Signal()

    toolitems = (
        ('Fit to Data', 'Reset original view', 'home', 'home'),
        (None, None, None, None),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        (None, None, None, None),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),
    )

    def __init__(self, canvas, parent=None, file_dialog_service=None):
        self._canvas = canvas
        self._file_dialog_service = file_dialog_service
        self.current_mode = "none"
        self.mode_label = None

        super().__init__(canvas, parent)
        self._apply_styling()

    def _init_toolbar(self):
        """Build toolbar with only essential navigation tools."""
        self.clear()
        self._add_streamlined_tools()
        self.addStretch()

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
        self._remove_unwanted_actions()

    def save_figure(self, *args):
        """Save plot using FileDialogService if available, otherwise use matplotlib default."""
        if self._file_dialog_service:
            from PySide6.QtWidgets import QMessageBox
            
            file_types = (
                "PNG files (*.png);;"
                "PDF files (*.pdf);;"
                "SVG files (*.svg);;"
                "JPEG files (*.jpg *.jpeg);;"
                "All files (*.*)"
            )
            
            file_path = self._file_dialog_service.get_export_path(
                parent=self.parent() or self._canvas.parent(),
                suggested_name="plot.png",
                file_types=file_types,
                dialog_type="save_plot"
            )
            
            if file_path:
                try:
                    self.canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Plot saved to: {file_path}")
                    self.plot_saved.emit(file_path)
                except Exception as e:
                    logger.error(f"Failed to save plot: {e}")
                    QMessageBox.critical(
                        self.parent() or self._canvas.parent(),
                        "Save Error",
                        f"Failed to save plot:\n{str(e)}"
                    )
        else:
            super().save_figure(*args)

    def _remove_unwanted_actions(self):
        """Safety net to remove any unwanted matplotlib default actions."""
        all_actions = self.actions()
        keep_texts = {'Reset', 'Back', 'Forward', 'Pan', 'Zoom', 'Save'}
        
        for action in all_actions:
            if action.isSeparator():
                continue
            action_text = action.text()
            if action_text and action_text not in keep_texts:
                self.removeAction(action)
                action.setVisible(False)

    def _add_streamlined_tools(self):
        toolbar_font = QFont()
        toolbar_font.setPointSize(TOOLBAR_CONFIG["button_font_size"])

        self.home_action = QAction("Reset", self)
        self.home_action.setToolTip("Fit to current data")
        self.home_action.triggered.connect(self.home)
        self.home_action.setIcon(self._create_icon("home"))
        self.home_action.setFont(toolbar_font)
        self.addAction(self.home_action)

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

        self.pan_action = QAction("Pan", self)
        self.pan_action.setToolTip("Pan axes with left mouse, zoom with right")
        self.pan_action.setCheckable(True)
        self.pan_action.triggered.connect(self.pan)
        self.pan_action.setIcon(self._create_icon("pan"))
        self.pan_action.setFont(toolbar_font)
        self.addAction(self.pan_action)

        self.zoom_action = QAction("Zoom", self)
        self.zoom_action.setToolTip("Zoom to rectangle")
        self.zoom_action.setCheckable(True)
        self.zoom_action.triggered.connect(self.zoom)
        self.zoom_action.setIcon(self._create_icon("zoom"))
        self.zoom_action.setFont(toolbar_font)
        self.addAction(self.zoom_action)

        self.addSeparator()

        self.save_action = QAction("Save", self)
        self.save_action.setToolTip("Save the figure")
        self.save_action.triggered.connect(self.save_figure)
        self.save_action.setIcon(self._create_icon("save"))
        self.save_action.setFont(toolbar_font)
        self.addAction(self.save_action)

    def _create_icon(self, icon_type: str) -> QIcon:
        """Generate simple vector icons for toolbar buttons."""
        icon_size = TOOLBAR_CONFIG["icon_size"]
        pixmap = QPixmap(icon_size, icon_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color = QColor("#606060")
        painter.setPen(color)
        painter.setBrush(color)

        scale = icon_size / 16.0

        if icon_type == "home":
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
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                int(4 * scale), int(4 * scale), int(7 * scale), int(7 * scale)
            )
            painter.drawLine(
                int(10 * scale), int(10 * scale), int(13 * scale), int(13 * scale)
            )

        elif icon_type == "save":
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
        Auto-disable zoom mode after completing a zoom operation.
        Keeps mode active if user clicks without dragging.
        """
        old_xlim = self.canvas.figure.axes[0].get_xlim() if self.canvas.figure.axes else None
        old_ylim = self.canvas.figure.axes[0].get_ylim() if self.canvas.figure.axes else None
        
        super().release_zoom(event)
        
        if old_xlim is not None and old_ylim is not None and self.canvas.figure.axes:
            new_xlim = self.canvas.figure.axes[0].get_xlim()
            new_ylim = self.canvas.figure.axes[0].get_ylim()
            
            zoom_occurred = (old_xlim != new_xlim or old_ylim != new_ylim)
            
            if zoom_occurred:
                if self._actions["zoom"].isChecked():
                    self.zoom()
                    logger.debug("Zoom completed - automatically disabled zoom mode")
            else:
                logger.debug("No zoom change detected - keeping zoom mode active")

    def _apply_styling(self):
        self.setStyleSheet(get_toolbar_style())
        self.setIconSize(
            QSize(TOOLBAR_CONFIG["icon_size"], TOOLBAR_CONFIG["icon_size"])
        )
        self.setMovable(False)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

    def pan(self, *args):
        """Toggle pan mode and update UI state."""
        super().pan(*args)

        if hasattr(self, "mode_label") and self.mode_label:
            if self._actions["pan"].isChecked():
                self.current_mode = "pan"
                self.mode_label.setText("Pan Mode")
                if hasattr(self, "zoom_action"):
                    self.zoom_action.setChecked(False)
            else:
                self.current_mode = "none"
                self.mode_label.setText("")
        else:
            if self._actions["pan"].isChecked():
                self.current_mode = "pan"
            else:
                self.current_mode = "none"

        self.mode_changed.emit(self.current_mode)

    def zoom(self, *args):
        """Toggle zoom mode and update UI state."""
        super().zoom(*args)

        if hasattr(self, "mode_label") and self.mode_label:
            if self._actions["zoom"].isChecked():
                self.current_mode = "zoom"
                self.mode_label.setText("Zoom Mode")
                if hasattr(self, "pan_action"):
                    self.pan_action.setChecked(False)
            else:
                self.current_mode = "none"
                self.mode_label.setText("")
        else:
            if self._actions["zoom"].isChecked():
                self.current_mode = "zoom"
            else:
                self.current_mode = "none"

        self.mode_changed.emit(self.current_mode)

    def home(self, *args):
        """Reset view and clear any active pan/zoom modes."""
        if hasattr(self, "pan_action"):
            self.pan_action.setChecked(False)
        if hasattr(self, "zoom_action"):
            self.zoom_action.setChecked(False)

        self.current_mode = "none"

        if hasattr(self, "mode_label") and self.mode_label:
            self.mode_label.setText("")

        self.reset_requested.emit()
        self.mode_changed.emit(self.current_mode)


class MinimalNavigationToolbar(QWidget):
    """Simplified toolbar for dialogs - just zoom, pan, and reset."""

    mode_changed = Signal(str)

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.toolbar = NavigationToolbar(canvas, self)
        self.toolbar.setVisible(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.zoom_btn = self._create_tool_button("Zoom", "zoom")
        self.pan_btn = self._create_tool_button("Pan", "pan")
        self.reset_btn = self._create_tool_button("Reset", "reset")

        tools_label = QLabel("Tools:")
        tools_label.setStyleSheet(f"font-size: {TOOLBAR_CONFIG['button_font_size']}px;")

        layout.addWidget(tools_label)
        layout.addWidget(self.zoom_btn)
        layout.addWidget(self.pan_btn)
        layout.addWidget(self.reset_btn)
        layout.addStretch()

        self.zoom_btn.clicked.connect(self._toggle_zoom)
        self.pan_btn.clicked.connect(self._toggle_pan)
        self.reset_btn.clicked.connect(self._reset_view)

        self.current_mode = "none"

    def _create_tool_button(self, text: str, mode: str):
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
        if self.zoom_btn.isChecked():
            self.toolbar.zoom()
            self.pan_btn.setChecked(False)
            self.current_mode = "zoom"
        else:
            self.toolbar.zoom()
            self.current_mode = "none"
        self.mode_changed.emit(self.current_mode)

    def _toggle_pan(self):
        if self.pan_btn.isChecked():
            self.toolbar.pan()
            self.zoom_btn.setChecked(False)
            self.current_mode = "pan"
        else:
            self.toolbar.pan()
            self.current_mode = "none"
        self.mode_changed.emit(self.current_mode)

    def _reset_view(self):
        self.toolbar.home()
        self.zoom_btn.setChecked(False)
        self.pan_btn.setChecked(False)
        self.current_mode = "none"
        self.mode_changed.emit(self.current_mode)