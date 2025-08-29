"""
Improved plot management module for MAT File Sweep Analyzer.

This version enhances type safety, separates concerns more cleanly,
and uses a consistent, object-oriented approach for all plotting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# Use TYPE_CHECKING to avoid runtime circular dependencies and allow type hints
# for heavy GUI components, which is good practice for CI environments.
if TYPE_CHECKING:
    from PyQt5.QtWidgets import QDoubleSpinBox, QVBoxLayout, QWidget
    from matplotlib.backend_bases import MouseEvent, PickEvent

# Set up a logger for better debugging
logger = logging.getLogger(__name__)


class PlotManager:
    """
    Manages all plotting operations, including sweep plots, range lines, and interactions.

    This class encapsulates a Matplotlib Figure and its associated canvas and toolbar,
    providing a clear interface for plotting data and handling user interactions
    within a PyQt5 application.
    """

    def __init__(self, parent_widget: "QWidget", figure_size: tuple[int, int] = (8, 6)):
        """
        Initializes the plot manager.

        Args:
            parent_widget: The parent QWidget that will contain the plot and toolbar.
            figure_size: A tuple representing the (width, height) of the figure in inches.
        """
        # We import Qt classes here for runtime use. This is a concession to the
        # structure where PlotManager builds its own QWidget.
        from PyQt5.QtWidgets import QVBoxLayout, QWidget

        self.parent_widget: "QWidget" = parent_widget

        # 1. Matplotlib components setup
        self.figure: Figure = Figure(figsize=figure_size, tight_layout=True)
        self.canvas: FigureCanvas = FigureCanvas(self.figure)
        self.ax: Axes = self.figure.add_subplot(111)
        self.toolbar: NavigationToolbar = NavigationToolbar(self.canvas, self.parent_widget)

        # Create the main plot widget that will be returned to the main GUI
        self.plot_widget: "QWidget" = QWidget()
        plot_layout: "QVBoxLayout" = QVBoxLayout(self.plot_widget)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # 2. Range lines management
        self.range_lines: list[Line2D] = []
        self._initialize_range_lines()  # Initialize with default lines

        # 3. Interactivity state
        self.dragging_line: Optional[Line2D] = None
        self.line_spinbox_map: dict[Line2D, "QDoubleSpinBox"] = {}
        self.drag_callback: Optional[Callable[[Line2D, float], None]] = None

        # 4. Connect interactive events
        self._connect_events()

    def get_plot_widget(self) -> "QWidget":
        """Returns the QWidget containing the plot canvas and toolbar."""
        return self.plot_widget

    def _connect_events(self) -> None:
        """Connects mouse events to their respective handlers for interactivity."""
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("motion_notify_event", self._on_drag)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _initialize_range_lines(self) -> None:
        """Initializes default range lines on the plot."""
        for line in self.range_lines:
            line.remove()
        self.range_lines.clear()

        # Add two default green lines
        self.range_lines.extend([
            self.ax.axvline(150, color='green', linestyle='-', picker=5, linewidth=2),
            self.ax.axvline(500, color='green', linestyle='-', picker=5, linewidth=2)
        ])
        logger.debug("Initialized default range lines.")

    def update_sweep_plot(
        self,
        t: np.ndarray,
        y: np.ndarray,
        channel: int,
        sweep_index: int,
        channel_type: str,
        channel_config: Optional[dict] = None,  # Added for API compatibility
    ) -> None:
        """
        Updates the plot with new sweep data.

        Args:
            t: Time data array (X-axis).
            y: Voltage/Current data array (Y-axis).
            channel: The index of the channel to plot from the `y` array.
            sweep_index: The index/number of the sweep.
            channel_type: The type of channel ("Voltage" or "Current").
            channel_config: (Ignored) Maintained for compatibility.
        """
        self.ax.clear()

        self.ax.plot(t, y[:, channel], linewidth=2)

        unit = "mV" if channel_type == "Voltage" else "pA"
        self.ax.set_title(f"Sweep {sweep_index} - {channel_type}")
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel(f"{channel_type} ({unit})")
        self.ax.grid(True, which='both', linestyle='--', alpha=0.5)

        # Restore range lines, which are removed by ax.clear()
        for line in self.range_lines:
            self.ax.add_line(line)

        # Autoscale and add padding for better visualization
        self.ax.relim()
        self.ax.autoscale_view(tight=True)
        self.ax.margins(x=0.02, y=0.05)

        self.redraw()
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
        Updates the positions of the draggable range lines efficiently.

        Args:
            start1: The start position for the first range (green lines).
            end1: The end position for the first range (green lines).
            use_dual_range: Flag to enable the second range (red lines).
            start2: The start position for the second range.
            end2: The end position for the second range.
        """
        if len(self.range_lines) < 2:
            self._initialize_range_lines()

        self.range_lines[0].set_xdata([start1, start1])
        self.range_lines[1].set_xdata([end1, end1])

        has_second_range = len(self.range_lines) == 4
        if use_dual_range and start2 is not None and end2 is not None:
            if not has_second_range:
                self.range_lines.extend([
                    self.ax.axvline(start2, color='red', linestyle='-', picker=5, linewidth=2),
                    self.ax.axvline(end2, color='red', linestyle='-', picker=5, linewidth=2)
                ])
            else:
                self.range_lines[2].set_xdata([start2, start2])
                self.range_lines[3].set_xdata([end2, end2])
        elif not use_dual_range and has_second_range:
            self.range_lines.pop(3).remove()
            self.range_lines.pop(2).remove()

        self.redraw()
        logger.debug("Updated range lines.")

    def update_line_spinbox_map(self, spinboxes: dict[str, "QDoubleSpinBox"]) -> None:
        """
        Updates the mapping between range lines and their corresponding QDoubleSpinBox widgets.

        Args:
            spinboxes: A dictionary mapping keys like 'start1', 'end1' to their widgets.
        """
        self.line_spinbox_map.clear()
        key_to_line_idx = {'start1': 0, 'end1': 1, 'start2': 2, 'end2': 3}

        for key, spinbox in spinboxes.items():
            idx = key_to_line_idx.get(key)
            if idx is not None and idx < len(self.range_lines):
                line = self.range_lines[idx]
                self.line_spinbox_map[line] = spinbox
        logger.debug("Updated line-spinbox map.")

    def set_drag_callback(self, callback: Callable[[Line2D, float], None]) -> None:
        """
        Sets a callback function to be invoked when a range line is dragged.

        Args:
            callback: A function that accepts the dragged line (Line2D) and its new x-position.
        """
        self.drag_callback = callback

    def center_nearest_cursor(self) -> tuple[Optional[Line2D], Optional[float]]:
        """
        Finds the horizontal center of the current plot view and moves the nearest
        range line to that position.

        Also invokes the drag callback to notify the application of the change.

        Returns:
            A tuple containing the Line2D object that was moved and its new
            x-position, or (None, None) if no action was taken.
        """
        if not self.range_lines or not self.ax.has_data():
            logger.warning("Cannot center cursor: No range lines or data available.")
            return None, None

        x_min, x_max = self.ax.get_xlim()
        center_x = (x_min + x_max) / 2

        # Find the line closest to the center of the view using numpy for efficiency
        distances = [abs(line.get_xdata()[0] - center_x) for line in self.range_lines]
        nearest_line = self.range_lines[int(np.argmin(distances))]

        # Move the line
        nearest_line.set_xdata([center_x, center_x])

        logger.info(f"Centered nearest cursor to x={center_x:.2f}.")

        # If a callback is connected, notify it of the change so UI can update
        if self.drag_callback:
            self.drag_callback(nearest_line, center_x)

        self.redraw()

        return nearest_line, center_x

    # --- Mouse Interaction Handlers ---

    def _on_pick(self, event: "PickEvent") -> None:
        """Handles pick events to initiate dragging a line."""
        if isinstance(event.artist, Line2D) and event.artist in self.range_lines:
            self.dragging_line = event.artist
            logger.debug(f"Picked line: {self.dragging_line.get_label()}.")

    def _on_drag(self, event: "MouseEvent") -> None:
        """Handles mouse motion events to drag a selected line."""
        if self.dragging_line and event.xdata is not None:
            x_pos = float(event.xdata)
            self.dragging_line.set_xdata([x_pos, x_pos])

            if self.drag_callback:
                self.drag_callback(self.dragging_line, x_pos)

            self.redraw()

    def _on_release(self, event: "MouseEvent") -> None:
        """Handles mouse release events to conclude a drag operation."""
        if self.dragging_line:
            logger.debug(f"Released line at x={self.dragging_line.get_xdata()[0]:.2f}.")
            self.dragging_line = None

    def clear(self) -> None:
        """Clears the plot axes completely."""
        self.ax.clear()
        self._initialize_range_lines()  # Re-add the lines after clearing
        self.redraw()
        logger.info("Plot cleared.")

    def redraw(self) -> None:
        """Forces a redraw of the plot canvas."""
        self.canvas.draw_idle()

    def update_lines_from_values(
        self,
        start1: float,
        end1: float,
        use_dual_range: bool = False,
        start2: Optional[float] = None,
        end2: Optional[float] = None,
    ) -> None:
        """
        Updates range line positions without recreating them.
        This method maintains compatibility with the existing main_window.py interface.
        
        Args:
            start1: Start position for range 1
            end1: End position for range 1
            use_dual_range: Whether dual range is active
            start2: Start position for range 2 (if dual range)
            end2: End position for range 2 (if dual range)
        """
        # Delegate to the main update method
        self.update_range_lines(start1, end1, use_dual_range, start2, end2)

    def toggle_dual_range(self, enabled: bool, start2: float, end2: float) -> None:
        """
        Toggle dual range visualization.
        This method maintains compatibility with the existing main_window.py interface.
        
        Args:
            enabled: Whether to enable dual range
            start2: Start position for range 2
            end2: End position for range 2
        """
        if enabled:
            # Add second range lines if not present
            if len(self.range_lines) == 2:
                self.range_lines.extend([
                    self.ax.axvline(start2, color='red', linestyle='-', picker=5, linewidth=2),
                    self.ax.axvline(end2, color='red', linestyle='-', picker=5, linewidth=2)
                ])
                logger.debug("Added dual range lines.")
        else:
            # Remove second range lines if present
            if len(self.range_lines) == 4:
                self.range_lines[3].remove()
                self.range_lines[2].remove()
                self.range_lines = self.range_lines[:2]
                logger.debug("Removed dual range lines.")
        
        self.redraw()

    def clear_plot(self) -> None:
        """Alias for clear() to maintain backward compatibility."""
        self.clear()


class BatchPlotter:
    """
    Provides static methods for creating non-interactive batch plots.

    By making these static, we separate the concern of creating one-off batch plots
    from the stateful management of the main interactive plot.
    """

    @staticmethod
    def create_figure(
        x_label: str,
        y_label: str,
        title: Optional[str] = None,
        figsize: tuple[int, int] = (10, 6)
    ) -> tuple[Figure, Axes]:
        """
        Creates a new Figure and Axes for a batch plot.

        Args:
            x_label: The label for the x-axis.
            y_label: The label for the y-axis.
            title: The plot title. If None, a default is generated.
            figsize: A tuple for the figure size (width, height) in inches.

        Returns:
            A tuple containing the created Matplotlib Figure and Axes.
        """
        fig = Figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(111)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title or f"{y_label} vs {x_label}")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        return fig, ax

    @staticmethod
    def plot_data(
        ax: Axes, x_data: np.ndarray, y_data: np.ndarray, label: str,
        marker: str = 'o-', **kwargs
    ) -> None:
        """
        Plots a data series on the given Axes.

        Args:
            ax: The Matplotlib Axes to plot on.
            x_data: The data for the x-axis.
            y_data: The data for the y-axis.
            label: The legend label for the data series.
            marker: The marker and line style.
            **kwargs: Additional keyword arguments passed to ax.plot().
        """
        if x_data.size > 0 and y_data.size > 0:
            ax.plot(x_data, y_data, marker, label=label, **kwargs)
        else:
            logger.warning(f"Attempted to plot empty data for label '{label}'.")

    @staticmethod
    def finalize_plot(fig: Figure, ax: Axes) -> None:
        """
        Finalizes a batch plot by adding a legend if applicable.

        Args:
            fig: The Matplotlib Figure.
            ax: The Matplotlib Axes.
        """
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        logger.info("Finalized batch plot.")



