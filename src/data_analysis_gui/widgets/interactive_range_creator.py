"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Interactive range creation for concentration-response analysis.

Manages click-to-define range mode on matplotlib plots, allowing users to
click start/end positions directly on the canvas to create analysis ranges.
"""

from typing import Optional
from PySide6.QtWidgets import QPushButton, QLabel
from PySide6.QtGui import QCursor, QPixmap, QPainter, QColor
from PySide6.QtCore import Qt

from data_analysis_gui.config.themes import style_button, style_label
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class InteractiveRangeCreator:
    """
    Manages interactive range creation mode for matplotlib-based analysis dialogs.
    
    Allows users to click on plot to define range start/end positions, with
    visual feedback including custom cursor, temporary guide lines, and status updates.
    Supports creating regular ranges, background ranges, and paired background ranges.
    """
    
    def __init__(self, canvas, ax, range_table, status_label):
        """
        Initialize the interactive range creator.
        
        Args:
            canvas: Matplotlib FigureCanvas for event handling
            ax: Matplotlib axis to draw on
            range_table: ConcentrationRangeTable widget for adding ranges
            status_label: QLabel for displaying status messages
        """
        self.canvas = canvas
        self.ax = ax
        self.range_table = range_table
        self.status_label = status_label
        
        # Mode state
        self._mode_active = False
        self._start_position: Optional[float] = None
        self._is_background = False
        self._target_row: Optional[int] = None  # For paired background creation
        self._active_button: Optional[QPushButton] = None
        
        # Visual feedback
        self._original_cursor: Optional[QCursor] = None
        self._temp_start_line = None
        
        # Button references (will be set during setup)
        self.add_range_btn: Optional[QPushButton] = None
        self.add_bg_range_btn: Optional[QPushButton] = None
        self.add_paired_bg_btn: Optional[QPushButton] = None
        
        # Connect matplotlib click events
        self.canvas.mpl_connect('button_press_event', self._handle_plot_click)
    
    def setup_buttons(self):
        """
        Setup button references and connect to toggle handlers.
        
        Call this after the range_table has been initialized and buttons exist.
        This disconnects the default handlers and connects to our toggle methods.
        """
        # Get button references from range_table
        self.add_range_btn = self.range_table.add_range_btn
        self.add_bg_range_btn = self.range_table.add_bg_range_btn
        
        # Find paired background button in the table's layout
        for i in range(self.range_table.layout().count()):
            item = self.range_table.layout().itemAt(i)
            if item and item.layout():
                for j in range(item.layout().count()):
                    widget_item = item.layout().itemAt(j)
                    if widget_item and widget_item.widget():
                        widget = widget_item.widget()
                        if isinstance(widget, QPushButton) and widget.text() == "Add Paired Background Range":
                            self.add_paired_bg_btn = widget
                            break
        
        # Disconnect default handlers
        self.add_range_btn.clicked.disconnect()
        self.add_bg_range_btn.clicked.disconnect()
        if self.add_paired_bg_btn:
            self.add_paired_bg_btn.clicked.disconnect()
        
        # Connect to our toggle handlers
        self.add_range_btn.clicked.connect(
            lambda: self.toggle_range_mode(is_background=False)
        )
        self.add_bg_range_btn.clicked.connect(
            lambda: self.toggle_range_mode(is_background=True)
        )
        if self.add_paired_bg_btn:
            self.add_paired_bg_btn.clicked.connect(
                self.toggle_paired_background_mode
            )
        
        logger.debug("Interactive range creator buttons configured")
    
    @property
    def is_active(self) -> bool:
        """Whether range creation mode is currently active."""
        return self._mode_active
    
    def toggle_range_mode(self, is_background: bool):
        """
        Toggle range creation mode or cancel if this button started the mode.
        
        Args:
            is_background: Whether creating a background range
        """
        button = self.add_bg_range_btn if is_background else self.add_range_btn
        
        if self._mode_active and self._active_button == button:
            # This button started the mode, so cancel it
            self.cancel_mode()
        elif not self._mode_active:
            # Start new mode
            self._active_button = button
            self.start_mode(is_background=is_background, active_button=button)
    
    def toggle_paired_background_mode(self):
        """
        Toggle paired background range creation mode.
        
        This mode creates a background range and automatically pairs it
        to the most recent analysis range.
        """
        if self._mode_active and self._active_button == self.add_paired_bg_btn:
            # This button started the mode, so cancel it
            self.cancel_mode()
            return
        
        if self._mode_active:
            # Some other button is active, do nothing
            return
        
        # Find last non-background range to pair to
        target_row = self._find_last_analysis_range_row()
        
        if target_row is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self.canvas.parentWidget(),
                "No Range to Pair",
                "Add an analysis range first before creating a paired background range."
            )
            return
        
        # Store target row and enter background creation mode
        self._target_row = target_row
        self._active_button = self.add_paired_bg_btn
        self.start_mode(is_background=True, active_button=self.add_paired_bg_btn)
        
        # Update status to indicate pairing
        self.status_label.setText(
            "Click on plot for PAIRED Background Range START position (or click button to cancel)"
        )
        style_label(self.status_label, "info")
        
        logger.info(f"Entered paired background creation mode for row {target_row}")
    
    def start_mode(self, is_background: bool, active_button: QPushButton):
        """
        Enter range creation mode - click on plot to define start/end.
        
        Args:
            is_background: Whether creating a background range
            active_button: The button that initiated this mode
        """
        self._mode_active = True
        self._is_background = is_background
        self._start_position = None
        
        # Create and set green crosshair cursor
        self._set_crosshair_cursor()
        
        # Update UI feedback
        range_type = "Background Range" if is_background else "Range"
        self.status_label.setText(
            f"Click on plot for {range_type} START position (or click button to cancel)"
        )
        style_label(self.status_label, "info")
        
        # Update button appearance to show cancellation option
        active_button.setText("✖ Cancel")
        style_button(active_button, "warning")
        
        logger.info(f"Entered range creation mode (background={is_background})")
    
    def cancel_mode(self):
        """Cancel range creation mode and restore normal state."""
        self._mode_active = False
        self._start_position = None
        self._target_row = None
        self._active_button = None
        
        # Remove temporary start line if it exists
        if self._temp_start_line:
            try:
                self._temp_start_line.remove()
            except:
                pass
            self._temp_start_line = None
            self.canvas.draw_idle()
        
        # Restore cursor
        self._restore_cursor()
        
        # Restore button appearance
        self._restore_button_appearance()
        
        # Update status (will be set by dialog if data is loaded)
        self.status_label.setText("Range creation cancelled")
        style_label(self.status_label, "muted")
        
        logger.info("Cancelled range creation mode")
    
    def _handle_plot_click(self, event):
        """
        Handle matplotlib button press events for range creation.
        
        Args:
            event: Matplotlib button press event
        """
        # Only handle left clicks in creation mode with valid x data
        if not self._mode_active or event.xdata is None or event.button != 1:
            return
        
        if self._start_position is None:
            # First click - set start position
            self._start_position = event.xdata
            
            # Update status
            range_type = self._get_range_type_label()
            self.status_label.setText(
                f"{range_type} START: {event.xdata:.2f}s - Click for END position"
            )
            style_label(self.status_label, "info")
            
            # Draw temporary guide line
            self._draw_temp_start_line(event.xdata)
            
            logger.debug(f"Range start set to {event.xdata:.2f}s")
        
        else:
            # Second click - set end position and create range
            self._complete_range_creation(event.xdata)
    
    def _complete_range_creation(self, end_position: float):
        """
        Complete range creation with the end position.
        
        Args:
            end_position: X-axis position for range end
        """
        start = self._start_position
        end = end_position
        
        # Ensure start < end (swap if necessary)
        if end < start:
            start, end = end, start
        
        # Remove temporary line
        if self._temp_start_line:
            try:
                self._temp_start_line.remove()
            except:
                pass
            self._temp_start_line = None
            self.canvas.draw_idle()
        
        # Create the range with specified times
        self.range_table.add_range_row_with_times(
            start_time=start,
            end_time=end,
            is_background=self._is_background
        )
        
        # Handle paired background setup
        if self._target_row is not None:
            self._setup_background_pairing()
        
        # Update status
        range_type = self._get_range_type_label()
        self.status_label.setText(
            f"{range_type} range created: {start:.2f}s - {end:.2f}s"
        )
        style_label(self.status_label, "success")
        
        logger.info(f"Created {range_type.lower()} range: {start:.2f}s - {end:.2f}s")
        
        # Exit creation mode
        self.cancel_mode()
    
    def _setup_background_pairing(self):
            """
            Setup automatic pairing for a newly created background range.
            
            Pairs the most recently created background range to the target analysis range.
            """
            # Get the newly created background range's internal ID (hidden column 1)
            new_bg_row = self.range_table.table.rowCount() - 1
            bg_id_widget = self.range_table.table.cellWidget(new_bg_row, 1)
            
            if bg_id_widget and self._target_row is not None:
                internal_id = bg_id_widget.text()
                display_name = self.range_table._format_background_display(internal_id)
                
                # Set the target range's paired dropdown to this background
                paired_combo = self.range_table.table.cellWidget(self._target_row, 7)
                if paired_combo:
                    paired_combo.setCurrentText(display_name)
                    logger.info(f"Auto-paired background '{internal_id}' to row {self._target_row}")
    
    def _find_last_analysis_range_row(self) -> Optional[int]:
            """
            Find the most recent non-background range row.
            
            Returns:
                Row index of last analysis range, or None if not found
            """
            from PySide6.QtWidgets import QCheckBox
            
            for row in range(self.range_table.table.rowCount() - 1, -1, -1):
                bg_widget = self.range_table.table.cellWidget(row, 6)
                if bg_widget:
                    checkbox = bg_widget.findChild(QCheckBox)
                    if checkbox and not checkbox.isChecked():
                        return row
            
            return None
    
    def _get_range_type_label(self) -> str:
        """
        Get descriptive label for current range type.
        
        Returns:
            Human-readable string describing the range type
        """
        if self._target_row is not None:
            return "Paired Background"
        elif self._is_background:
            return "Background"
        else:
            return "Analysis"
    
    def _set_crosshair_cursor(self):
        """Set custom green crosshair cursor for range creation mode."""
        # Create green crosshair cursor (sage green matching analysis cursors)
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setPen(QColor("#73AB84"))  # Sage green from plot_style.py
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
        painter.setBrush(QColor("#73AB84"))
        painter.drawEllipse(14, 14, 4, 4)
        
        painter.end()
        
        # Store original cursor and set new one
        self._original_cursor = self.canvas.cursor()
        self.canvas.setCursor(QCursor(pixmap, hotX=16, hotY=16))
    
    def _restore_cursor(self):
        """Restore the original canvas cursor."""
        if self._original_cursor:
            self.canvas.setCursor(self._original_cursor)
            self._original_cursor = None
        else:
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
    
    def _draw_temp_start_line(self, x_position: float):
        """
        Draw temporary vertical line at start position for visual feedback.
        
        Args:
            x_position: X-axis position for the guide line
        """
        self._temp_start_line = self.ax.axvline(
            x_position,
            color="#73AB84",
            linestyle=":",
            linewidth=2,
            alpha=0.5
        )
        self.canvas.draw_idle()
    
    def _restore_button_appearance(self):
        """Restore all add range buttons to normal appearance."""
        if self.add_range_btn:
            self.add_range_btn.setText("Add Range")
            style_button(self.add_range_btn, "secondary")
        
        if self.add_bg_range_btn:
            self.add_bg_range_btn.setText("Add Background Range")
            style_button(self.add_bg_range_btn, "secondary")
        
        if self.add_paired_bg_btn:
            self.add_paired_bg_btn.setText("Add Paired Background Range")
            style_button(self.add_paired_bg_btn, "secondary")