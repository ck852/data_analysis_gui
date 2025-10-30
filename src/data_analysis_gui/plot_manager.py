from __future__ import annotations

"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module provides plot management for PatchBatch.

Features:
- Uses Qt signals for decoupling from the main window.
- PlotManager handles matplotlib visualization and emits signals for plot interactions.
- Coordinates ViewStateManager and CursorManager for focused responsibility separation.
"""

import logging
from typing import Optional, Tuple, Dict

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

from data_analysis_gui.config.plot_style import (
    apply_plot_style,
    format_sweep_plot,
    get_line_styles,
)
from data_analysis_gui.widgets.custom_toolbar import StreamlinedNavigationToolbar
from data_analysis_gui.core.view_state_manager import ViewStateManager
from data_analysis_gui.widgets.cursor_manager import CursorManager
from data_analysis_gui.widgets.axis_zoom_controller import AxisZoomController

# Set up a logger for better debugging
logger = logging.getLogger(__name__)


class PlotManager(QObject):
    """
    Manages all interactive plotting operations for the application.

    Responsibilities:
    - Encapsulates a Matplotlib Figure, canvas, and toolbar.
    - Coordinates ViewStateManager for axis limit state.
    - Coordinates CursorManager for cursor/text management.
    - Handles sweep plots and user interactions.
    - Emits Qt signals for plot updates and line state changes.
    """

    # Define signals for plot interactions
    # Signal: (action, line_id, value)
    # Actions: 'dragged', 'added', 'removed', 'centered'
    line_state_changed = Signal(str, str, float)

    # Signal for plot updates
    plot_updated = Signal()

    # Clickable welcome message
    welcome_clicked = Signal() 


    def __init__(self, figure_size: Tuple[int, int] = (8, 6), file_dialog_service=None):
        """
        Initialize the PlotManager with modern styling and interactive components.

        Args:
            figure_size: Tuple specifying the initial figure size (width, height).
            file_dialog_service: Optional FileDialogService for persistent directory memory.
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

        # Use the streamlined toolbar with file_dialog_service
        self.toolbar: StreamlinedNavigationToolbar = StreamlinedNavigationToolbar(
            self.canvas, None, file_dialog_service=file_dialog_service
        )

        # Create the plot widget
        self.plot_widget: QWidget = QWidget()
        plot_layout: QVBoxLayout = QVBoxLayout(self.plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # 2. Helper components for state and cursor management
        self.view_manager = ViewStateManager()
        self.cursor_manager = CursorManager(self.ax)

        # Axis zoom controller
        self.axis_zoom_controller = AxisZoomController(self.figure, self.ax)

        # 3. Initialize range lines (but don't add to axes yet)
        self._initialize_range_lines()

        # 4. Connect interactive events
        self._connect_events()

        # 5. Apply initial styling to axes
        self._style_axes()

        # Data bounds with margins for zoom limiting (calculated from actual data)
        self._data_bounds_x: Optional[Tuple[float, float]] = None
        self._data_bounds_y: Optional[Tuple[float, float]] = None

        # Track current channel type for per-channel view state management
        self._current_channel_type: str = 'Voltage'

        # Track which channels have been auto-fitted for the current file
        self._autofitted_channels: set = set()

        # Connect toolbar reset signal to autofit method
        self.toolbar.reset_requested.connect(self.autofit_to_data)

        # Welcome message state
        self._welcome_text = None
        self._welcome_click_cid = None
        
        # Show welcome message on startup
        self.show_welcome_message()

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
        # Use styled colors for range lines
        range1_style = self.line_styles["range1"]

        # Create Range 1 cursors via CursorManager
        line1 = self.cursor_manager.create_cursor(
            line_id="range1_start",
            position=0,
            color=range1_style["color"],
            linestyle=range1_style["linestyle"],
            linewidth=range1_style["linewidth"],
            alpha=range1_style["alpha"]
        )
        
        line2 = self.cursor_manager.create_cursor(
            line_id="range1_end",
            position=500,
            color=range1_style["color"],
            linestyle=range1_style["linestyle"],
            linewidth=range1_style["linewidth"],
            alpha=range1_style["alpha"]
        )

        # Add lines to axes
        self.ax.add_line(line1)
        self.ax.add_line(line2)

        # Emit signals for initial positions
        self.line_state_changed.emit("added", "range1_start", 0)
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
            channel_type: Type of channel ('Voltage' or 'Current').
            channel_config: Optional channel configuration.
            title: Optional plot title.
            x_label: Optional x-axis label.
            y_label: Optional y-axis label.
        """
        # Store current channel type for per-channel view state management
        self._current_channel_type = channel_type
        
        # Calculate data bounds with margins for zoom limiting
        # Same margins as autofit_to_data: 2% for X, 5% for Y
        x_min, x_max = float(np.min(t)), float(np.max(t))
        y_min, y_max = float(np.min(y[:, channel])), float(np.max(y[:, channel]))
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_margin = x_range * 0.02
        y_margin = y_range * 0.05
        
        self._data_bounds_x = (x_min - x_margin, x_max + x_margin)
        self._data_bounds_y = (y_min - y_margin, y_max + y_margin)
        
        logger.debug(f"Updated data bounds: X={self._data_bounds_x}, Y={self._data_bounds_y}")
        
        # Clear zoom buttons BEFORE clearing axes
        self.axis_zoom_controller.clear_buttons()

        # 1. Clear axes
        self.ax.clear()

        # 2. Plot data with modern styling
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

        # 3. Give CursorManager the plot data for text labels
        units = "pA"
        if channel_config:
            units = channel_config.get("current_units", "pA")
        
        self.cursor_manager.set_plot_data(
            time_data=t,
            y_data=y[:, channel],
            channel_type=channel_type,
            units=units
        )

        # 4. Re-add cursor Line2D objects from CursorManager
        for line in self.cursor_manager.get_all_lines():
            self.ax.add_line(line)

        # 5. Handle view limits with per-channel view state
        current_view = self.view_manager.get_current_view(channel_type)

        if current_view is not None:
            # Restore the preserved zoom/pan state for this channel
            xlim, ylim = current_view
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            logger.debug(f"Restored {channel_type} view: X={xlim}, Y={ylim}")
        else:
            # First time viewing this channel - autoscale
            self.ax.relim()
            self.ax.autoscale_view(tight=True)
            self.ax.margins(x=0.02, y=0.05)
            
            # Store the autoscaled limits as current view for this channel
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.view_manager.update_current_view(xlim, ylim, channel_type)
            logger.debug(f"Set initial {channel_type} view: X={xlim}, Y={ylim}")

        # 6. CursorManager recreates text labels with new data
        self.cursor_manager.recreate_all_text_labels(self.ax)

        # Create zoom buttons AFTER tight_layout
        self.axis_zoom_controller.create_buttons(self._on_axis_zoom)
        self.redraw()
        
        # 7. Auto-fit if this is the first time viewing this channel
        if channel_type not in self._autofitted_channels:
            logger.info(f"Auto-fitting {channel_type} channel on first view")
            self.autofit_to_data()
            self._autofitted_channels.add(channel_type)
        
        self.plot_updated.emit()
        logger.info(f"Updated plot for sweep {sweep_index}, channel {channel} ({channel_type}).")


    def reset_for_new_file(self) -> None:
        """
        Reset plot state for a new file load.
        
        Clears view state for all channels so first sweep will autoscale
        and establish fresh views. Called from MainWindow when loading a new file.
        """
        self.view_manager.reset()
        self._current_channel_type = 'Voltage'
        self._autofitted_channels.clear()  # Reset auto-fit tracking for new file
        logger.info("Reset plot manager for new file (all channel views cleared, auto-fit tracking reset)")


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

        # Get current cursor positions to check what exists
        current_positions = self.cursor_manager.get_cursor_positions()
        
        # Update Range 1 positions
        if "range1_start" in current_positions:
            self.cursor_manager.update_cursor_position("range1_start", start1)
        else:
            # Create if missing (shouldn't happen in normal operation)
            line = self.cursor_manager.create_cursor(
                "range1_start",
                start1,
                color=range1_style["color"],
                linestyle=range1_style["linestyle"],
                linewidth=range1_style["linewidth"],
                alpha=range1_style["alpha"]
            )
            self.ax.add_line(line)
            self.cursor_manager.recreate_all_text_labels(self.ax)
        
        if "range1_end" in current_positions:
            self.cursor_manager.update_cursor_position("range1_end", end1)
        else:
            # Create if missing
            line = self.cursor_manager.create_cursor(
                "range1_end",
                end1,
                color=range1_style["color"],
                linestyle=range1_style["linestyle"],
                linewidth=range1_style["linewidth"],
                alpha=range1_style["alpha"]
            )
            self.ax.add_line(line)
            self.cursor_manager.recreate_all_text_labels(self.ax)

        # Handle Range 2
        has_range2 = "range2_start" in current_positions
        
        if use_dual_range and start2 is not None and end2 is not None:
            if not has_range2:
                # Add Range 2 lines
                line3 = self.cursor_manager.create_cursor(
                    "range2_start",
                    start2,
                    color=range2_style["color"],
                    linestyle=range2_style["linestyle"],
                    linewidth=range2_style["linewidth"],
                    alpha=range2_style["alpha"]
                )
                line4 = self.cursor_manager.create_cursor(
                    "range2_end",
                    end2,
                    color=range2_style["color"],
                    linestyle=range2_style["linestyle"],
                    linewidth=range2_style["linewidth"],
                    alpha=range2_style["alpha"]
                )
                
                self.ax.add_line(line3)
                self.ax.add_line(line4)
                
                # Recreate text labels for new cursors
                self.cursor_manager.recreate_all_text_labels(self.ax)

                self.line_state_changed.emit("added", "range2_start", start2)
                self.line_state_changed.emit("added", "range2_end", end2)
            else:
                # Update existing Range 2 positions
                self.cursor_manager.update_cursor_position("range2_start", start2)
                self.cursor_manager.update_cursor_position("range2_end", end2)
                
        elif not use_dual_range and has_range2:
            # Remove Range 2 lines
            range2_start_pos = current_positions.get("range2_start", 0)
            range2_end_pos = current_positions.get("range2_end", 0)
            
            self.cursor_manager.remove_cursor("range2_start")
            self.cursor_manager.remove_cursor("range2_end")
            
            self.line_state_changed.emit("removed", "range2_start", range2_start_pos)
            self.line_state_changed.emit("removed", "range2_end", range2_end_pos)

        # NEW: Recreate zoom buttons after line updates
        # (only if they already existed - don't create if plot not initialized)
        if self.axis_zoom_controller.has_buttons():
            self.axis_zoom_controller.clear_buttons()
            self.axis_zoom_controller.create_buttons(self._on_axis_zoom)

        self.redraw()
        logger.debug("Updated range lines with modern styling.")

    def center_nearest_cursor(self) -> Tuple[Optional[str], Optional[float]]:
        """
        Center the nearest range line to the horizontal center of the plot view.

        Returns:
            Tuple[str, float]: The line ID and new x-position, or (None, None).
        """
        cursor_positions = self.cursor_manager.get_cursor_positions()
        
        if not cursor_positions or not self.ax.has_data():
            logger.warning("Cannot center cursor: No cursors or data available.")
            return None, None

        x_min, x_max = self.ax.get_xlim()
        center_x = (x_min + x_max) / 2

        # Find the cursor closest to the center of the view
        nearest_line_id = None
        min_distance = float('inf')
        
        for line_id, position in cursor_positions.items():
            distance = abs(position - center_x)
            if distance < min_distance:
                min_distance = distance
                nearest_line_id = line_id

        if nearest_line_id is None:
            return None, None

        # Move the cursor
        self.cursor_manager.update_cursor_position(nearest_line_id, center_x)

        logger.info(f"Centered nearest cursor to x={center_x:.2f}.")

        # Emit signal about the centering
        self.line_state_changed.emit("centered", nearest_line_id, center_x)

        self.redraw()

        return nearest_line_id, center_x

    def show_welcome_message(self) -> None:
        """Display a clickable welcome message on the empty plot at startup."""
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        self._welcome_text = self.ax.text(
            0.5, 0.5,
            'Open a data file to begin analysis',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16,
            color='#4A90E2',
            transform=self.ax.transAxes,
            picker=True  # Make it pickable
        )
        
        # Make entire plot clickable during welcome state
        self._welcome_click_cid = self.canvas.mpl_connect(
            'button_press_event', 
            self._on_welcome_click
        )
        
        self.redraw()
        logger.info("Displayed welcome message on empty plot (entire plot is clickable)")

    def _on_welcome_click(self, event) -> None:
        """
        Handle mouse clicks during welcome state.
        
        When the welcome message is displayed, any click on the plot area
        will emit the welcome_clicked signal to open the file dialog.
        
        Args:
            event: Matplotlib button press event.
        """
        # Only respond if we're in welcome state
        if self._welcome_text is not None:
            logger.debug("Welcome plot clicked - opening file dialog")
            self.welcome_clicked.emit()

    def clear_welcome_state(self) -> None:
        """
        Clear the welcome message and restore normal plot appearance.
        
        Called when the first data file is loaded. Removes the welcome text,
        disconnects the click handler, and restores axis decorations for normal plotting.
        """
        if self._welcome_text is not None:
            self._welcome_text.remove()
            self._welcome_text = None
            logger.debug("Removed welcome message")
        
        # Disconnect the welcome click handler
        if self._welcome_click_cid is not None:
            self.canvas.mpl_disconnect(self._welcome_click_cid)
            self._welcome_click_cid = None
            logger.debug("Disconnected welcome click handler")
        
        # Restore normal axis appearance (will be properly styled when data loads)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Restore spines (they'll be styled by _style_axes via update_sweep_plot)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(True)
        self.ax.spines['bottom'].set_visible(True)

    def autofit_to_data(self) -> None:
        """
        Autoscale axes to fit all data points in the currently displayed sweep.
        
        Performs a fresh autoscale on current sweep data and stores the result
        for the current channel type. Called when Reset button is clicked.
        """
        # Get data directly from cursor manager
        time_data = self.cursor_manager._current_time_data
        y_data = self.cursor_manager._current_y_data
        
        if time_data is None or y_data is None or len(time_data) == 0:
            logger.warning("Cannot autofit: No data currently available")
            return
        
        # Calculate limits directly from data
        x_min, x_max = float(np.min(time_data)), float(np.max(time_data))
        y_min, y_max = float(np.min(y_data)), float(np.max(y_data))
        
        # Add margins (2% for x, 5% for y)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_margin = x_range * 0.02
        y_margin = y_range * 0.05
        
        xlim = (x_min - x_margin, x_max + x_margin)
        ylim = (y_min - y_margin, y_max + y_margin)
        
        # Set limits directly
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        
        # Update cursor text positions for new view
        self.cursor_manager.update_all_text_positions(ylim)
        
        # Store as current view for the current channel type
        self.view_manager.update_current_view(xlim, ylim, self._current_channel_type)
        
        # Push onto toolbar's navigation stack
        self.toolbar.push_current()
        
        logger.info(f"Autofitted {self._current_channel_type} to data: X={xlim}, Y={ylim}")
        self.redraw()

    def reset_for_new_file(self) -> None:
        """
        Reset plot state for a new file load.
        
        Clears view state for all channels so first sweep will autoscale
        and establish fresh views. Called from MainWindow when loading a new file.
        """
        self.view_manager.reset()
        self._current_channel_type = 'Voltage'
        logger.info("Reset plot manager for new file (all channel views cleared)")

    # --- Mouse Interaction Handlers ---

    def _on_pick(self, event) -> None:
        """Handle pick events for dragging range lines or clicking welcome text."""
        # Check if welcome text was clicked
        if self._welcome_text is not None and event.artist == self._welcome_text:
            logger.debug("Welcome text clicked - opening file dialog")
            self.welcome_clicked.emit()
            return
        
        # Otherwise, handle cursor dragging
        line_id = self.cursor_manager.handle_pick(event.artist)
        if line_id:
            logger.debug(f"Picked cursor: {line_id}")

    def set_max_time_bound(self, max_time: float) -> None:
        """
        Set the maximum time bound for X-axis zoom limiting.
        
        Args:
            max_time: Maximum sweep time in milliseconds.
        """
        self._max_time_bound = max_time
        logger.debug(f"Set max time bound: {max_time:.2f} ms")

    def _on_axis_zoom(self, axis: str, direction: str) -> None:
        """
        Handle axis zoom button clicks with bounds limiting.
        
        Args:
            axis: 'x' or 'y' - which axis to zoom.
            direction: 'in' or 'out' - zoom direction.
        """
        # Get current limits for the specified axis
        if axis == 'x':
            current_limits = self.ax.get_xlim()
            max_bounds = self._data_bounds_x
        else:
            current_limits = self.ax.get_ylim()
            max_bounds = self._data_bounds_y
        
        # Calculate new limits using controller with bounds
        new_limits = self.axis_zoom_controller.calculate_zoom(
            axis, direction, current_limits, max_bounds=max_bounds
        )
        
        # Apply new limits to axes
        if axis == 'x':
            self.ax.set_xlim(new_limits)
        else:
            self.ax.set_ylim(new_limits)
        
        # Update cursor text positions
        current_ylim = self.ax.get_ylim()
        self.cursor_manager.update_all_text_positions(current_ylim)
        
        # Store new view state for current channel
        current_xlim = self.ax.get_xlim()
        self.view_manager.update_current_view(current_xlim, current_ylim, self._current_channel_type)
        
        self.redraw()

    def _on_drag(self, event) -> None:
        """
        Handle mouse motion events to drag a selected range line.

        Args:
            event: Matplotlib motion event.
        """
        if not self.cursor_manager.is_dragging():
            return
        
        result = self.cursor_manager.update_drag(event.xdata)
        
        if result:
            line_id, new_position = result
            # Emit signal about the drag
            self.line_state_changed.emit("dragged", line_id, new_position)
            self.redraw()

    def _on_release(self, event) -> None:
        """
        Handle mouse release events to conclude a drag operation.

        Args:
            event: Matplotlib button release event.
        """
        line_id = self.cursor_manager.release_drag()
        if line_id:
            positions = self.cursor_manager.get_cursor_positions()
            x_pos = positions.get(line_id, 0)
            logger.debug(f"Released cursor {line_id} at x={x_pos:.2f}.")
            
            # NEW: Emit signal that drag is complete
            self.line_state_changed.emit("released", line_id, x_pos)

    def _on_draw(self, event) -> None:
        """
        Handle draw events to update cursor text positions after zoom/pan.
        
        Args:
            event: Matplotlib draw event.
        """
        # Get current axis limits
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        # Check if view has changed for the current channel
        if self.view_manager.has_view_changed(current_xlim, current_ylim, self._current_channel_type):
            # Update cursor text positions to match new limits
            self.cursor_manager.update_all_text_positions(current_ylim)
            
            # Store new limits for the current channel
            self.view_manager.update_current_view(current_xlim, current_ylim, self._current_channel_type)

    def clear(self) -> None:
        """
        Clear the plot axes and reset range lines to defaults.
        """
        # NEW: Clear zoom buttons before clearing axes
        self.axis_zoom_controller.clear_buttons()

        # Clear axes - this removes all artists including lines
        self.ax.clear()

        # Clear cursor manager's plot data
        self.cursor_manager.clear_plot_data()

        # Remove all cursors from tracking
        for line_id in list(self.cursor_manager.get_cursor_positions().keys()):
            self.cursor_manager.remove_cursor(line_id)

        # Re-initialize default range lines
        self._initialize_range_lines()

        self.redraw()
        self.plot_updated.emit()
        logger.info("Plot cleared.")

    def redraw(self) -> None:
        """
        Force a redraw of the plot canvas.
        """
        self.canvas.draw()

    def toggle_dual_range(self, enabled: bool, start2: float, end2: float) -> None:
        """
        Toggle dual range visualization.

        Args:
            enabled: Whether to enable dual range.
            start2: Start position for range 2.
            end2: End position for range 2.
        """
        positions = self.cursor_manager.get_cursor_positions()
        
        if enabled:
            # Get current range 1 values
            start1 = positions.get("range1_start", 150)
            end1 = positions.get("range1_end", 500)

            # Update with dual range
            self.update_range_lines(start1, end1, True, start2, end2)
        else:
            # Get current range 1 values
            start1 = positions.get("range1_start", 150)
            end1 = positions.get("range1_end", 500)

            # Update without dual range
            self.update_range_lines(start1, end1, False, None, None)

    def get_line_positions(self) -> Dict[str, float]:
        """
        Get current positions of all range lines.

        Returns:
            Dict[str, float]: Mapping of line IDs to their x positions.
        """
        return self.cursor_manager.get_cursor_positions()