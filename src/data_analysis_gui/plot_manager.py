from __future__ import annotations

"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module provides improved plot management for PatchBatch.

Features:
- Uses Qt signals for decoupling from the main window.
- PlotManager handles matplotlib visualization and emits signals for plot interactions.
"""

import logging
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

from data_analysis_gui.config.plot_style import (
    apply_plot_style,
    format_sweep_plot,
    get_line_styles,
)
from data_analysis_gui.widgets.custom_toolbar import StreamlinedNavigationToolbar


# Set up a logger for better debugging
logger = logging.getLogger(__name__)


class PlotManager(QObject):
    """
    Manages all interactive plotting operations for the application.

    Responsibilities:
    - Encapsulates a Matplotlib Figure, canvas, and toolbar.
    - Handles sweep plots, range lines, and user interactions.
    - Emits Qt signals for plot updates and line state changes.
    - Maintains no knowledge of external widgets or main window.
    """

    # Define signals for plot interactions
    # Signal: (action, line_id, value)
    # Actions: 'dragged', 'added', 'removed', 'centered'
    line_state_changed = Signal(str, str, float)

    # Signal for plot updates
    plot_updated = Signal()

    def __init__(self, figure_size: Tuple[int, int] = (8, 6)):
        """
        Initialize the PlotManager with modern styling and interactive components.

        Args:
            figure_size: Tuple specifying the initial figure size (width, height).
        """
        super().__init__()

        # Apply modern plot style globally
        apply_plot_style()

        # Get line styles for consistent appearance
        self.line_styles = get_line_styles()

        # 1. Matplotlib components setup with styled figure
        self.figure: Figure = Figure(figsize=figure_size, facecolor="#FAFAFA")
        self.canvas: FigureCanvas = FigureCanvas(self.figure)
        self.ax: Axes = self.figure.add_subplot(111)

        # Use the streamlined toolbar instead of standard NavigationToolbar
        self.toolbar: StreamlinedNavigationToolbar = StreamlinedNavigationToolbar(
            self.canvas, None
        )

        # Create the plot widget
        self.plot_widget: QWidget = QWidget()
        plot_layout: QVBoxLayout = QVBoxLayout(self.plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # 2. Range lines management
        self.range_lines: List[Line2D] = []
        self._line_ids: Dict[Line2D, str] = {}
        self._initialize_range_lines()

        # NEW: Cursor text management
        self._cursor_texts: Dict[str, Any] = {}  # Maps line_id to Text object
        self._current_time_data: Optional[np.ndarray] = None
        self._current_y_data: Optional[np.ndarray] = None
        self._current_channel_type: Optional[str] = None
        self._current_units: str = "pA"  # Default units
        
        # Track axis limits to detect zoom/pan
        self._last_xlim: Optional[Tuple[float, float]] = None
        self._last_ylim: Optional[Tuple[float, float]] = None

        # 3. Interactivity state
        self.dragging_line: Optional[Line2D] = None

        # 4. Connect interactive events
        self._connect_events()

        # 5. Apply initial styling to axes
        self._style_axes()

    def _style_axes(self):
        """
        Apply modern styling to the plot axes, including font sizes and colors.
        """
        self.ax.set_facecolor("#FAFBFC")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_linewidth(0.8)
        self.ax.spines["bottom"].set_linewidth(0.8)
        self.ax.spines["left"].set_color("#B0B0B0")
        self.ax.spines["bottom"].set_color("#B0B0B0")

        # Use the increased font sizes from plot_style
        self.ax.tick_params(
            axis="both",
            which="major",
            labelsize=9,
            colors="#606060",
            length=4,
            width=0.8,
        )

        self.ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="#E1E5E8")
        self.ax.set_axisbelow(True)

    def get_plot_widget(self) -> QWidget:
        """
        Returns the QWidget containing the plot canvas and toolbar.

        Returns:
            QWidget: The plot widget for embedding in Qt layouts.
        """
        return self.plot_widget

    def _connect_events(self) -> None:
        """
        Connect mouse events to handlers for interactive line dragging.
        """
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("motion_notify_event", self._on_drag)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # Connect to draw event to update cursor text after zoom/pan
        self.canvas.mpl_connect("draw_event", self._on_draw)

    def _initialize_range_lines(self) -> None:
        """
        Initialize default range lines with modern styling and emit signals.
        """
        self.range_lines.clear()
        self._line_ids.clear()

        # Use styled colors for range lines
        range1_style = self.line_styles["range1"]

        line1 = self.ax.axvline(
            150,
            color=range1_style["color"],
            linestyle=range1_style["linestyle"],
            linewidth=range1_style["linewidth"],
            alpha=range1_style["alpha"],
            picker=5,
        )
        line2 = self.ax.axvline(
            500,
            color=range1_style["color"],
            linestyle=range1_style["linestyle"],
            linewidth=range1_style["linewidth"],
            alpha=range1_style["alpha"],
            picker=5,
        )

        self.range_lines.extend([line1, line2])
        self._line_ids[line1] = "range1_start"
        self._line_ids[line2] = "range1_end"

        self.line_state_changed.emit("added", "range1_start", 150)
        self.line_state_changed.emit("added", "range1_end", 500)

        logger.debug("Initialized styled range lines.")

    def update_sweep_plot(
        self,
        t: np.ndarray,
        y: np.ndarray,
        channel: int,
        sweep_index: int,
        channel_type: str,
        channel_config: Optional[dict] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> None:
        """
        Update the plot with new sweep data and styling.

        Args:
            t: Time array.
            y: Data array (2D).
            channel: Channel index to plot.
            sweep_index: Index of the sweep.
            channel_type: Type of channel.
            channel_config: Optional channel configuration.
            title: Optional plot title.
            x_label: Optional x-axis label.
            y_label: Optional y-axis label.
        """
        self.ax.clear()

        # Plot with modern styling
        line_style = self.line_styles["primary"]
        self.ax.plot(
            t,
            y[:, channel],
            color=line_style["color"],
            linewidth=line_style["linewidth"],
            alpha=line_style["alpha"],
        )

        # Apply sweep-specific formatting with custom labels if provided
        if title or x_label or y_label:
            # Use custom labels/title
            from data_analysis_gui.config.plot_style import style_axis

            style_axis(self.ax, title=title, xlabel=x_label, ylabel=y_label)
            self.ax.set_facecolor("#FAFBFC")  # Keep sweep plot background
        else:
            # Use default formatting
            format_sweep_plot(self.ax, sweep_index, channel_type)

        # Store current plot data for cursor text
        self._current_time_data = t
        self._current_y_data = y[:, channel]
        self._current_channel_type = channel_type
        
        # Get units from channel_config or use defaults
        if channel_config:
            self._current_units = channel_config.get("current_units", "pA")
        else:
            self._current_units = "pA"

        # Restore range lines with consistent styling
        for line in self.range_lines:
            self.ax.add_line(line)
        
        # Create cursor text labels for all current cursors
        for line, line_id in self._line_ids.items():
            x_position = line.get_xdata()[0]
            self._create_cursor_text(line_id, x_position)

        # Apply padding for better visualization
        self.ax.relim()
        self.ax.autoscale_view(tight=True)
        self.ax.margins(x=0.02, y=0.05)

        # Initialize axis limits tracking for zoom/pan detection
        self._last_xlim = self.ax.get_xlim()
        self._last_ylim = self.ax.get_ylim()

        self.figure.tight_layout(pad=1.0)
        self.redraw()
        self.plot_updated.emit()
        logger.info(f"Updated plot for sweep {sweep_index}, channel {channel}.")

    def update_range_lines(
        self,
        start1: float,
        end1: float,
        use_dual_range: bool = False,
        start2: Optional[float] = None,
        end2: Optional[float] = None,
    ) -> None:
        """
        Update the positions and visibility of range lines.

        Args:
            start1: Start position for range 1.
            end1: End position for range 1.
            use_dual_range: Whether to show a second range.
            start2: Start position for range 2 (if dual range).
            end2: End position for range 2 (if dual range).
        """
        # Get style configurations
        range1_style = self.line_styles["range1"]
        range2_style = self.line_styles["range2"]

        # Ensure we have at least 2 lines for Range 1
        if len(self.range_lines) < 2:
            if len(self.range_lines) == 0:
                line1 = self.ax.axvline(
                    start1,
                    color=range1_style["color"],
                    linestyle=range1_style["linestyle"],
                    linewidth=range1_style["linewidth"],
                    alpha=range1_style["alpha"],
                    picker=5,
                )
                line2 = self.ax.axvline(
                    end1,
                    color=range1_style["color"],
                    linestyle=range1_style["linestyle"],
                    linewidth=range1_style["linewidth"],
                    alpha=range1_style["alpha"],
                    picker=5,
                )
                self.range_lines.extend([line1, line2])
                self._line_ids[line1] = "range1_start"
                self._line_ids[line2] = "range1_end"
                
                # Create text for new lines
                self._create_cursor_text("range1_start", start1)
                self._create_cursor_text("range1_end", end1)
            elif len(self.range_lines) == 1:
                line2 = self.ax.axvline(
                    end1,
                    color=range1_style["color"],
                    linestyle=range1_style["linestyle"],
                    linewidth=range1_style["linewidth"],
                    alpha=range1_style["alpha"],
                    picker=5,
                )
                self.range_lines.append(line2)
                self._line_ids[line2] = "range1_end"
                
                # Create text for new line
                self._create_cursor_text("range1_end", end1)
        else:
            # Update existing Range 1 lines
            self.range_lines[0].set_xdata([start1, start1])
            self.range_lines[1].set_xdata([end1, end1])
            
            # Update cursor text positions
            self._update_cursor_text("range1_start", start1)
            self._update_cursor_text("range1_end", end1)

        has_second_range = len(self.range_lines) == 4

        if use_dual_range and start2 is not None and end2 is not None:
            if not has_second_range:
                # Add Range 2 lines with different styling
                line3 = self.ax.axvline(
                    start2,
                    color=range2_style["color"],
                    linestyle=range2_style["linestyle"],
                    linewidth=range2_style["linewidth"],
                    alpha=range2_style["alpha"],
                    picker=5,
                )
                line4 = self.ax.axvline(
                    end2,
                    color=range2_style["color"],
                    linestyle=range2_style["linestyle"],
                    linewidth=range2_style["linewidth"],
                    alpha=range2_style["alpha"],
                    picker=5,
                )

                self.range_lines.extend([line3, line4])
                self._line_ids[line3] = "range2_start"
                self._line_ids[line4] = "range2_end"

                self.line_state_changed.emit("added", "range2_start", start2)
                self.line_state_changed.emit("added", "range2_end", end2)
                
                # Create text for new Range 2 lines
                self._create_cursor_text("range2_start", start2)
                self._create_cursor_text("range2_end", end2)
            else:
                # Update existing Range 2 lines
                self.range_lines[2].set_xdata([start2, start2])
                self.range_lines[3].set_xdata([end2, end2])
                
                # Update cursor text positions
                self._update_cursor_text("range2_start", start2)
                self._update_cursor_text("range2_end", end2)
        elif not use_dual_range and has_second_range:
            # Remove Range 2 lines and their text
            line4 = self.range_lines.pop()
            line3 = self.range_lines.pop()

            line3_id = self._line_ids.get(line3, "range2_start")
            line4_id = self._line_ids.get(line4, "range2_end")
            
            self.line_state_changed.emit(
                "removed",
                line3_id,
                line3.get_xdata()[0],
            )
            self.line_state_changed.emit(
                "removed", line4_id, line4.get_xdata()[0]
            )

            # Remove text labels
            self._remove_cursor_text(line3_id)
            self._remove_cursor_text(line4_id)

            if line3 in self._line_ids:
                del self._line_ids[line3]
            if line4 in self._line_ids:
                del self._line_ids[line4]

            if line3.axes:
                line3.remove()
            if line4.axes:
                line4.remove()

        self.redraw()
        logger.debug("Updated range lines with modern styling.")

    def center_nearest_cursor(self) -> Tuple[Optional[str], Optional[float]]:
        """
        Center the nearest range line to the horizontal center of the plot view.

        Returns:
            Tuple[str, float]: The line ID and new x-position, or (None, None).
        """
        if not self.range_lines or not self.ax.has_data():
            logger.warning("Cannot center cursor: No range lines or data available.")
            return None, None

        x_min, x_max = self.ax.get_xlim()
        center_x = (x_min + x_max) / 2

        # Find the line closest to the center of the view
        distances = [abs(line.get_xdata()[0] - center_x) for line in self.range_lines]
        nearest_idx = int(np.argmin(distances))
        nearest_line = self.range_lines[nearest_idx]
        line_id = self._line_ids.get(nearest_line, f"line_{nearest_idx}")

        # Move the line
        nearest_line.set_xdata([center_x, center_x])

        logger.info(f"Centered nearest cursor to x={center_x:.2f}.")

        # Emit signal about the centering
        self.line_state_changed.emit("centered", line_id, center_x)
        
        # Update cursor text to new position
        self._update_cursor_text(line_id, center_x)

        self.redraw()

        return line_id, center_x

    # --- Mouse Interaction Handlers ---

    def _on_pick(self, event) -> None:
        """
        Handle pick events to initiate dragging a range line.

        Args:
            event: Matplotlib pick event.
        """
        if isinstance(event.artist, Line2D) and event.artist in self.range_lines:
            self.dragging_line = event.artist
            logger.debug(
                f"Picked line: {self._line_ids.get(self.dragging_line, 'unknown')}."
            )

    def _on_drag(self, event) -> None:
        """
        Handle mouse motion events to drag a selected range line.

        Args:
            event: Matplotlib motion event.
        """
        if self.dragging_line and event.xdata is not None:
            x_pos = float(event.xdata)
            self.dragging_line.set_xdata([x_pos, x_pos])

            # Emit signal about the drag
            line_id = self._line_ids.get(self.dragging_line, "unknown")
            self.line_state_changed.emit("dragged", line_id, x_pos)
            
            # Update cursor text to follow the cursor
            self._update_cursor_text(line_id, x_pos)

            self.redraw()

    def _on_release(self, event) -> None:
        """
        Handle mouse release events to conclude a drag operation.

        Args:
            event: Matplotlib button release event.
        """
        if self.dragging_line:
            line_id = self._line_ids.get(self.dragging_line, "unknown")
            x_pos = self.dragging_line.get_xdata()[0]
            logger.debug(f"Released line {line_id} at x={x_pos:.2f}.")
            self.dragging_line = None

    def clear(self) -> None:
        """
        Clear the plot axes and reset range lines to defaults.
        """
        # Clear axes - this removes all artists including lines
        self.ax.clear()

        # Reset line tracking (don't try to remove already-removed lines)
        self.range_lines.clear()
        self._line_ids.clear()
        
        # Clear cursor text tracking
        self._cursor_texts.clear()
        self._current_time_data = None
        self._current_y_data = None
        self._current_channel_type = None

        # Re-add default range lines
        line1 = self.ax.axvline(
            150, color="green", linestyle="-", picker=5, linewidth=2
        )
        line2 = self.ax.axvline(
            500, color="green", linestyle="-", picker=5, linewidth=2
        )

        self.range_lines.extend([line1, line2])
        self._line_ids[line1] = "range1_start"
        self._line_ids[line2] = "range1_end"

        self.line_state_changed.emit("added", "range1_start", 150)
        self.line_state_changed.emit("added", "range1_end", 500)

        self.redraw()
        self.plot_updated.emit()
        logger.info("Plot cleared.")

    def redraw(self) -> None:
        """
        Force a redraw of the plot canvas.
        """
        self.canvas.draw_idle()

    def toggle_dual_range(self, enabled: bool, start2: float, end2: float) -> None:
        """
        Toggle dual range visualization.

        Args:
            enabled: Whether to enable dual range.
            start2: Start position for range 2.
            end2: End position for range 2.
        """
        if enabled:
            # Get current range 1 values
            start1 = self.range_lines[0].get_xdata()[0] if self.range_lines else 150
            end1 = (
                self.range_lines[1].get_xdata()[0] if len(self.range_lines) > 1 else 500
            )

            # Update with dual range
            self.update_range_lines(start1, end1, True, start2, end2)
        else:
            # Get current range 1 values
            start1 = self.range_lines[0].get_xdata()[0] if self.range_lines else 150
            end1 = (
                self.range_lines[1].get_xdata()[0] if len(self.range_lines) > 1 else 500
            )

            # Update without dual range
            self.update_range_lines(start1, end1, False, None, None)

    def get_line_positions(self) -> Dict[str, float]:
        """
        Get current positions of all range lines.

        Returns:
            Dict[str, float]: Mapping of line IDs to their x positions.
        """
        positions = {}
        for line, line_id in self._line_ids.items():
            positions[line_id] = line.get_xdata()[0]
        return positions

    def _sample_y_value_nearest(self, x_position: float) -> Optional[float]:
        """
        Find the nearest y-value from the plot data at the given x-position.
        
        Args:
            x_position: X-coordinate (time in ms) to sample at.
        
        Returns:
            float: Y-value at nearest data point, or None if no data available.
        """
        if self._current_time_data is None or self._current_y_data is None:
            return None
        
        if len(self._current_time_data) == 0 or len(self._current_y_data) == 0:
            return None
        
        # Find index of nearest time point
        idx = np.argmin(np.abs(self._current_time_data - x_position))
        
        # Return corresponding y-value
        return float(self._current_y_data[idx])


    def _create_cursor_text(self, line_id: str, x_position: float) -> None:
        """
        Create a text label for a cursor line showing the y-value at its position.
        
        Args:
            line_id: Identifier for the cursor line.
            x_position: X-coordinate of the cursor.
        """
        # Sample y-value at cursor position
        y_value = self._sample_y_value_nearest(x_position)
        
        if y_value is None:
            return
        
        # Determine units based on current channel type
        if self._current_channel_type == "Voltage":
            unit = "mV"
            formatted_value = f"{y_value:.1f} {unit}"
        else:
            unit = self._current_units
            formatted_value = f"{y_value:.1f} {unit}"
        
        # Position text near top of plot
        y_min, y_max = self.ax.get_ylim()
        text_y = y_max - (y_max - y_min) * 0.05  # 5% from top
        
        # Create text object
        text = self.ax.text(
            x_position, text_y, formatted_value,
            ha='center', va='top',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor='gray', alpha=0.9)
        )
        
        # Store reference
        self._cursor_texts[line_id] = text
        
        logger.debug(f"Created cursor text for {line_id} at x={x_position:.2f}, y={y_value:.2f}")


    def _update_cursor_text(self, line_id: str, x_position: float) -> None:
        """
        Update the position and value of a cursor text label.
        
        Args:
            line_id: Identifier for the cursor line.
            x_position: New x-coordinate of the cursor.
        """
        if line_id not in self._cursor_texts:
            return
        
        # Sample new y-value
        y_value = self._sample_y_value_nearest(x_position)
        
        if y_value is None:
            return
        
        # Determine units
        if self._current_channel_type == "Voltage":
            unit = "mV"
            formatted_value = f"{y_value:.1f} {unit}"
        else:
            unit = self._current_units
            formatted_value = f"{y_value:.1f} {unit}"
        
        # Update text content and position
        text = self._cursor_texts[line_id]
        text.set_text(formatted_value)
        
        # Keep y-position near top of plot (recalculate in case of zoom)
        y_min, y_max = self.ax.get_ylim()
        text_y = y_max - (y_max - y_min) * 0.05
        
        text.set_position((x_position, text_y))


    def _remove_cursor_text(self, line_id: str) -> None:
        """
        Remove a cursor text label.
        
        Args:
            line_id: Identifier for the cursor line.
        """
        if line_id in self._cursor_texts:
            text = self._cursor_texts[line_id]
            text.remove()
            del self._cursor_texts[line_id]
            logger.debug(f"Removed cursor text for {line_id}")


    def _update_all_cursor_text_positions(self) -> None:
        """
        Update the y-position of all cursor text labels based on current axis limits.
        Called after zoom/pan operations to keep text visible.
        """
        if not self._cursor_texts:
            return
        
        # Get current axis limits
        y_min, y_max = self.ax.get_ylim()
        text_y = y_max - (y_max - y_min) * 0.05  # 5% from top
        
        # Update position for each text label
        for line_id, text in self._cursor_texts.items():
            # Get current x-position from the text
            x_position = text.get_position()[0]
            
            # Update y-position to keep text near top
            text.set_position((x_position, text_y))
        
        logger.debug("Updated all cursor text positions after zoom/pan")


    def _on_draw(self, event) -> None:
        """
        Handle draw events to update cursor text positions after zoom/pan.
        
        Args:
            event: Matplotlib draw event.
        """
        # Check if axis limits have changed
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        limits_changed = (
            self._last_xlim != current_xlim or 
            self._last_ylim != current_ylim
        )
        
        if limits_changed and self._cursor_texts:
            # Update cursor text positions to match new limits
            self._update_all_cursor_text_positions()
            
            # Store new limits
            self._last_xlim = current_xlim
            self._last_ylim = current_ylim