from __future__ import annotations

"""
PatchBatch Electrophysiology Data Analysis Tool - Plot Manager

This module defines the MainWindow plot. It manages interactive matplotlib plotting for electrophysiology sweep visualization.
Coordinates the matplotlib canvas with specialized managers for cursors (CursorManager),
per-channel view state (ViewStateManager), and axis zoom controls (AxisZoomController).
Emits Qt signals when user interactions occur - MainRangeCoordinator handles synchronization
between cursor positions and ControlPanel spinboxes.

This module creates and styles the plot, handles mouse events (dragging cursors, zooming, panning),
manages cursor lifecycle (creation, positioning, removal), and maintains per-channel axis limits.
CursorManager handles the matplotlib Line2D and Text objects independently, while PlotManager
coordinates the higher-level interactions and signal emission.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

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
from data_analysis_gui.gui_services.view_state_manager import ViewStateManager
from data_analysis_gui.widgets.cursor_manager import CursorManager
from data_analysis_gui.widgets.axis_zoom_controller import AxisZoomController

logger = logging.getLogger(__name__)


class PlotManager(QObject):
    """
    Interactive plot manager for sweep data with draggable cursors and per-channel view memory.
    
    Mouse events trigger CursorManager methods that return values; PlotManager emits Qt signals
    with these values for MainRangeCoordinator to handle spinbox synchronization. Auto-fits once 
    per channel on first view, then remembers zoom/pan state when switching between channels.
    """

    # Actions: 'dragged', 'added', 'removed', 'centered', 'released'
    line_state_changed = Signal(str, str, float)  # action, line_id, value
    plot_updated = Signal()
    welcome_clicked = Signal()


    def __init__(self, figure_size: Tuple[int, int] = (8, 6), file_dialog_service=None):

        super().__init__()

        apply_plot_style()
        self.line_styles = get_line_styles()

        # Matplotlib setup
        self.figure: Figure = Figure(figsize=figure_size, facecolor="#FAFAFA")
        self.canvas: FigureCanvas = FigureCanvas(self.figure)
        self.ax: Axes = self.figure.add_subplot(111)

        self.toolbar: StreamlinedNavigationToolbar = StreamlinedNavigationToolbar(
            self.canvas, None, file_dialog_service=file_dialog_service
        )

        self.plot_widget: QWidget = QWidget()
        plot_layout: QVBoxLayout = QVBoxLayout(self.plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # State managers
        self.view_manager = ViewStateManager()
        self.cursor_manager = CursorManager(self.ax)
        self.axis_zoom_controller = AxisZoomController(self.figure, self.ax)

        # Initialize range lines (but don't add to axes yet)
        self._initialize_range_lines()
        self._connect_events()
        self._style_axes()

        # Data bounds for zoom limiting (calculated with margins from actual data)
        self._data_bounds_x: Optional[Tuple[float, float]] = None
        self._data_bounds_y: Optional[Tuple[float, float]] = None

        self._current_channel_type: str = 'Voltage'
        self._autofitted_channels: set = set()

        self.toolbar.reset_requested.connect(self.autofit_to_data)

        # Welcome message state
        self._welcome_text = None
        self._welcome_click_cid = None
        
        # Show welcome message on startup
        self.show_welcome_message()

    def _style_axes(self):

        self.ax.set_facecolor("#FAFBFC")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_linewidth(0.8)
        self.ax.spines["bottom"].set_linewidth(0.8)
        self.ax.spines["left"].set_color("#B0B0B0")
        self.ax.spines["bottom"].set_color("#B0B0B0")

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

        return self.plot_widget

    def _connect_events(self) -> None:

        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("motion_notify_event", self._on_drag)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        # Update cursor text after zoom/pan
        self.canvas.mpl_connect("draw_event", self._on_draw)

    def _initialize_range_lines(self) -> None:
        """Creates initial Range 1 cursors and adds them to the plot."""
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
        Plots new sweep data and restores the saved view state for this channel type.
        
        Auto-fits to data on first view of each channel type, then remembers zoom/pan 
        state when switching. Custom labels override default formatting. Updates cursor 
        text labels to match new data and units.
        """
        self._current_channel_type = channel_type
        
        # Calculate bounds with 2% X margin, 5% Y margin for zoom limiting
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

        # Clear axes
        self.ax.clear()

        # Plot with styling
        line_style = self.line_styles["primary"]
        self.ax.plot(
            t,
            y[:, channel],
            color=line_style["color"],
            linewidth=line_style["linewidth"],
            alpha=line_style["alpha"],
        )

        if title or x_label or y_label:
            from data_analysis_gui.config.plot_style import style_axis
            style_axis(self.ax, title=title, xlabel=x_label, ylabel=y_label)
            self.ax.set_facecolor("#FAFBFC")
        else:
            format_sweep_plot(self.ax, sweep_index, channel_type)

        # Give CursorManager plot data for text labels
        units = "pA"
        if channel_config:
            units = channel_config.get("current_units", "pA")
        
        self.cursor_manager.set_plot_data(
            time_data=t,
            y_data=y[:, channel],
            channel_type=channel_type,
            units=units
        )

        # Re-snap cursors to new data
        for line_id, position in self.cursor_manager.get_cursor_positions().items():
            self.cursor_manager.update_cursor_position(line_id, position)

        # Re-add cursor lines
        for line in self.cursor_manager.get_all_lines():
            self.ax.add_line(line)

        # Restore or initialize view for this channel
        current_view = self.view_manager.get_current_view(channel_type)

        if current_view is not None:
            xlim, ylim = current_view
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            logger.debug(f"Restored {channel_type} view: X={xlim}, Y={ylim}")
        else:
            # First time - autoscale and store
            self.ax.relim()
            self.ax.autoscale_view(tight=True)
            self.ax.margins(x=0.02, y=0.05)
            
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.view_manager.update_current_view(xlim, ylim, channel_type)
            logger.debug(f"Set initial {channel_type} view: X={xlim}, Y={ylim}")

        self.cursor_manager.recreate_all_text_labels(self.ax)
        self.axis_zoom_controller.create_buttons(self._on_axis_zoom)
        self.redraw()
        
        # Auto-fit first time viewing this channel
        if channel_type not in self._autofitted_channels:
            logger.info(f"Auto-fitting {channel_type} channel on first view")
            self.autofit_to_data()
            self._autofitted_channels.add(channel_type)
        
        self.plot_updated.emit()
        logger.info(f"Updated plot for sweep {sweep_index}, channel {channel} ({channel_type}).")


    def reset_for_new_file(self) -> None:
        """Clears all saved view states so new file data gets fresh auto-fitting."""
        self.view_manager.reset()
        self._current_channel_type = 'Voltage'
        self._autofitted_channels.clear()
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
        Updates cursor positions and toggles Range 2 cursors on/off.
        
        Always updates Range 1. When use_dual_range=True, creates Range 2 cursors if 
        they don't exist or updates their positions if they do. When False, removes 
        Range 2 cursors if present.
        """
        range1_style = self.line_styles["range1"]
        range2_style = self.line_styles["range2"]

        current_positions = self.cursor_manager.get_cursor_positions()
        
        # Update Range 1
        if "range1_start" in current_positions:
            self.cursor_manager.update_cursor_position("range1_start", start1)
        else:
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

        has_range2 = "range2_start" in current_positions
        
        if use_dual_range and start2 is not None and end2 is not None:
            if not has_range2:
                # Add Range 2
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
                
                self.cursor_manager.recreate_all_text_labels(self.ax)

                self.line_state_changed.emit("added", "range2_start", start2)
                self.line_state_changed.emit("added", "range2_end", end2)
            else:
                self.cursor_manager.update_cursor_position("range2_start", start2)
                self.cursor_manager.update_cursor_position("range2_end", end2)
                
        elif not use_dual_range and has_range2:
            # Remove Range 2
            range2_start_pos = current_positions.get("range2_start", 0)
            range2_end_pos = current_positions.get("range2_end", 0)
            
            self.cursor_manager.remove_cursor("range2_start")
            self.cursor_manager.remove_cursor("range2_end")
            
            self.line_state_changed.emit("removed", "range2_start", range2_start_pos)
            self.line_state_changed.emit("removed", "range2_end", range2_end_pos)

        # Recreate zoom buttons if they exist
        if self.axis_zoom_controller.has_buttons():
            self.axis_zoom_controller.clear_buttons()
            self.axis_zoom_controller.create_buttons(self._on_axis_zoom)

        self.redraw()
        logger.debug("Updated range lines.")

    def center_nearest_cursor(self) -> Tuple[Optional[str], Optional[float]]:
        """
        Snaps the cursor closest to the current view center to the exact center X position.
        
        Returns (line_id, new_position) for the moved cursor, or (None, None) if no cursors exist.
        """
        cursor_positions = self.cursor_manager.get_cursor_positions()
        
        if not cursor_positions or not self.ax.has_data():
            logger.warning("Cannot center cursor: No cursors or data available.")
            return None, None

        x_min, x_max = self.ax.get_xlim()
        center_x = (x_min + x_max) / 2

        # Find closest cursor
        nearest_line_id = None
        min_distance = float('inf')
        
        for line_id, position in cursor_positions.items():
            distance = abs(position - center_x)
            if distance < min_distance:
                min_distance = distance
                nearest_line_id = line_id

        if nearest_line_id is None:
            return None, None

        self.cursor_manager.update_cursor_position(nearest_line_id, center_x)

        logger.info(f"Centered nearest cursor to x={center_x:.2f}.")

        self.line_state_changed.emit("centered", nearest_line_id, center_x)

        self.redraw()

        return nearest_line_id, center_x

    def show_welcome_message(self) -> None:
        """Displays clickable startup message before data is loaded."""
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
            picker=True
        )
        
        # Make entire plot clickable
        self._welcome_click_cid = self.canvas.mpl_connect(
            'button_press_event', 
            self._on_welcome_click
        )
        
        self.redraw()
        logger.info("Displayed welcome message on empty plot (entire plot is clickable)")

    def _on_welcome_click(self, event) -> None:
        """Emits welcome_clicked signal when user clicks the empty plot."""
        if self._welcome_text is not None:
            logger.debug("Welcome plot clicked - opening file dialog")
            self.welcome_clicked.emit()

    def clear_welcome_state(self) -> None:
        """Removes welcome message and restores normal axis styling."""
        if self._welcome_text is not None:
            self._welcome_text.remove()
            self._welcome_text = None
            logger.debug("Removed welcome message")
        
        if self._welcome_click_cid is not None:
            self.canvas.mpl_disconnect(self._welcome_click_cid)
            self._welcome_click_cid = None
            logger.debug("Disconnected welcome click handler")
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(True)
        self.ax.spines['bottom'].set_visible(True)

    def autofit_to_data(self) -> None:
        """
        Fits axes to current data with 2% X margin and 5% Y margin.
        
        Called automatically on first view of each channel type, or manually via reset button.
        Saves the new view state for this channel type.
        """
        time_data = self.cursor_manager._current_time_data
        y_data = self.cursor_manager._current_y_data
        
        if time_data is None or y_data is None or len(time_data) == 0:
            logger.warning("Cannot autofit: No data currently available")
            return
        
        x_min, x_max = float(np.min(time_data)), float(np.max(time_data))
        y_min, y_max = float(np.min(y_data)), float(np.max(y_data))
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_margin = x_range * 0.02
        y_margin = y_range * 0.05
        
        xlim = (x_min - x_margin, x_max + x_margin)
        ylim = (y_min - y_margin, y_max + y_margin)
        
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        
        self.cursor_manager.update_all_text_positions(ylim)
        self.view_manager.update_current_view(xlim, ylim, self._current_channel_type)
        self.toolbar.push_current()
        
        logger.info(f"Autofitted {self._current_channel_type} to data: X={xlim}, Y={ylim}")
        self.redraw()

    # --- Mouse event handlers ---

    def _on_pick(self, event) -> None:
        """Handles clicks on welcome text or initiates cursor dragging."""
        if self._welcome_text is not None and event.artist == self._welcome_text:
            logger.debug("Welcome text clicked - opening file dialog")
            self.welcome_clicked.emit()
            return
        
        line_id = self.cursor_manager.handle_pick(event.artist)
        if line_id:
            logger.debug(f"Picked cursor: {line_id}")

    def set_max_time_bound(self, max_time: float) -> None:
        """Sets maximum X-axis limit for zoom controls (in milliseconds)."""
        self._max_time_bound = max_time
        logger.debug(f"Set max time bound: {max_time:.2f} ms")

    def _on_axis_zoom(self, axis: str, direction: str) -> None:
        """
        Handles zoom button clicks with bounds checking.
        
        Won't zoom beyond data bounds (with margins). Updates cursor text positions 
        and saves new view state for current channel type.
        """
        if axis == 'x':
            current_limits = self.ax.get_xlim()
            max_bounds = self._data_bounds_x
        else:
            current_limits = self.ax.get_ylim()
            max_bounds = self._data_bounds_y
        
        new_limits = self.axis_zoom_controller.calculate_zoom(
            axis, direction, current_limits, max_bounds=max_bounds
        )
        
        if axis == 'x':
            self.ax.set_xlim(new_limits)
        else:
            self.ax.set_ylim(new_limits)
        
        current_ylim = self.ax.get_ylim()
        self.cursor_manager.update_all_text_positions(current_ylim)
        
        current_xlim = self.ax.get_xlim()
        self.view_manager.update_current_view(current_xlim, current_ylim, self._current_channel_type)
        
        self.redraw()

    def _on_drag(self, event) -> None:

        if not self.cursor_manager.is_dragging():
            return
        
        result = self.cursor_manager.update_drag(event.xdata)
        
        if result:
            line_id, new_position = result
            self.line_state_changed.emit("dragged", line_id, new_position)
            self.redraw()

    def _on_release(self, event) -> None:

        line_id = self.cursor_manager.release_drag()
        if line_id:
            positions = self.cursor_manager.get_cursor_positions()
            x_pos = positions.get(line_id, 0)
            logger.debug(f"Released cursor {line_id} at x={x_pos:.2f}.")
            
            self.line_state_changed.emit("released", line_id, x_pos)

    def _on_draw(self, event) -> None:
        """
        Updates cursor text positions after matplotlib zoom/pan operations.
        
        Saves new view state if it changed. Connected to matplotlib's 'draw_event'.
        """
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        if self.view_manager.has_view_changed(current_xlim, current_ylim, self._current_channel_type):
            self.cursor_manager.update_all_text_positions(current_ylim)
            self.view_manager.update_current_view(current_xlim, current_ylim, self._current_channel_type)

    def clear(self) -> None:
        """Clears plot, removes all cursors, then recreates default Range 1 cursors."""
        self.axis_zoom_controller.clear_buttons()
        self.ax.clear()
        self.cursor_manager.clear_plot_data()

        # Remove all cursors
        for line_id in list(self.cursor_manager.get_cursor_positions().keys()):
            self.cursor_manager.remove_cursor(line_id)

        self._initialize_range_lines()

        self.redraw()
        self.plot_updated.emit()
        logger.info("Plot cleared.")

    def redraw(self) -> None:

        self.canvas.draw()

    def toggle_dual_range(self, enabled: bool, start2: float, end2: float) -> None:
        """Adds or removes Range 2 cursors while preserving Range 1 positions."""
        positions = self.cursor_manager.get_cursor_positions()
        
        if enabled:
            start1 = positions.get("range1_start", 150)
            end1 = positions.get("range1_end", 500)
            self.update_range_lines(start1, end1, True, start2, end2)
        else:
            start1 = positions.get("range1_start", 150)
            end1 = positions.get("range1_end", 500)
            self.update_range_lines(start1, end1, False, None, None)

    def get_line_positions(self) -> Dict[str, float]:
        """Returns current X positions of all cursors (e.g., {'range1_start': 150.0, ...})."""
        return self.cursor_manager.get_cursor_positions()