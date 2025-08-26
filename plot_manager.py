"""
Plot management module for MAT File Sweep Analyzer
Handles all plotting operations, range lines, and plot interactions
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QVBoxLayout

from utils import format_voltage_label


class PlotManager:
    """Manages all plotting operations including sweep plots, range lines, and interactions"""
    
    def __init__(self, parent_widget, figure_size=(8, 6)):
        """
        Initialize the plot manager
        
        Args:
            parent_widget: The parent QWidget that will contain the plot
            figure_size: Tuple of (width, height) for the figure
        """
        self.parent_widget = parent_widget
        
        # Create matplotlib components
        self.figure = Figure(figsize=figure_size)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, parent_widget)
        
        # Range lines management
        self.range_lines = []
        self.dragging_line = None
        self.line_spinbox_map = {}
        
        # Callback for spinbox updates during dragging
        self.drag_callback = None
        
        # Connect mouse events
        self._connect_events()
        
        # Initialize with default range lines
        self._initialize_range_lines()
    
    def get_plot_widget(self):
        """
        Create and return a widget containing the plot and toolbar
        
        Returns:
            QWidget containing the complete plot setup
        """
        from PyQt5.QtWidgets import QWidget
        
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        return plot_widget
    
    def _connect_events(self):
        """Connect mouse events for interactivity"""
        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)
    
    def _initialize_range_lines(self):
        """Initialize default range lines"""
        self.range_lines = [
            self.ax.axvline(150, color='green', linestyle='-', picker=5),
            self.ax.axvline(500, color='green', linestyle='-', picker=5)
        ]
    
    def update_sweep_plot(self, t, y, channel, sweep_index, channel_type, channel_config):
        """
        Update the plot to show sweep data
        
        Args:
            t: Time array
            y: Data array
            channel: Physical channel number to plot
            sweep_index: Index/number of the sweep
            channel_type: Type of channel ("Voltage" or "Current")
            channel_config: ChannelConfiguration object for proper labeling
        """
        self.ax.clear()
        
        # Plot the data
        self.ax.plot(t, y[:, channel], linewidth=2)
        
        # Set labels and title
        self.ax.set_title(f"Sweep {sweep_index} - {channel_type}")
        self.ax.set_xlabel("Time (ms)")
        
        # Use proper units based on channel type
        unit = "mV" if channel_type == "Voltage" else "pA"
        self.ax.set_ylabel(f"{channel_type} ({unit})")
        
        self.ax.grid(True, alpha=0.3)
        
        # Force autoscaling on the data
        self.ax.relim()
        self.ax.autoscale_view(tight=True)
        
        # Get the data limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Add 5% padding to y-axis
        y_range = ylim[1] - ylim[0]
        y_padding = y_range * 0.05
        
        # Restore the y-limits with padding
        self.ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)
        
        # Set x-limits to data range with small padding
        x_range = xlim[1] - xlim[0]
        x_padding = x_range * 0.02
        self.ax.set_xlim(xlim[0] - x_padding, xlim[1] + x_padding)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_range_lines(self, start1, end1, use_dual_range=False, start2=None, end2=None):
        """
        Update range lines on the plot
        
        Args:
            start1: Start position for range 1
            end1: End position for range 1
            use_dual_range: Whether to show dual range lines
            start2: Start position for range 2 (if dual range)
            end2: End position for range 2 (if dual range)
        """
        # Clear existing range lines from the plot
        for line in self.range_lines:
            if line in self.ax.lines:
                line.remove()
        
        # Create new range lines
        self.range_lines = [
            self.ax.axvline(start1, color='green', linestyle='-', picker=5, linewidth=2),
            self.ax.axvline(end1, color='green', linestyle='-', picker=5, linewidth=2)
        ]
        
        # Add second range lines if enabled
        if use_dual_range and start2 is not None and end2 is not None:
            self.range_lines.append(
                self.ax.axvline(start2, color='red', linestyle='-', picker=5, linewidth=2)
            )
            self.range_lines.append(
                self.ax.axvline(end2, color='red', linestyle='-', picker=5, linewidth=2)
            )
        
        self.canvas.draw()
    
    def set_drag_callback(self, callback):
        """
        Set a callback function to be called when lines are dragged
        
        Args:
            callback: Function that takes (line, new_x_value) as arguments
        """
        self.drag_callback = callback
    
    def update_line_spinbox_map(self, spinboxes):
        """
        Update the mapping between range lines and spinboxes
        
        Args:
            spinboxes: Dict with keys 'start1', 'end1', and optionally 'start2', 'end2'
                      Values are the spinbox widgets
        """
        self.line_spinbox_map = {}
        
        if len(self.range_lines) >= 2:
            self.line_spinbox_map[self.range_lines[0]] = spinboxes['start1']
            self.line_spinbox_map[self.range_lines[1]] = spinboxes['end1']
        
        if len(self.range_lines) == 4 and 'start2' in spinboxes:
            self.line_spinbox_map[self.range_lines[2]] = spinboxes['start2']
            self.line_spinbox_map[self.range_lines[3]] = spinboxes['end2']
    
    def update_lines_from_values(self, start1, end1, use_dual_range=False, start2=None, end2=None):
        """
        Update range line positions without recreating them
        
        Args:
            start1: Start position for range 1
            end1: End position for range 1
            use_dual_range: Whether dual range is active
            start2: Start position for range 2 (if dual range)
            end2: End position for range 2 (if dual range)
        """
        if not self.range_lines:
            return
        
        # Update Range 1 lines
        if len(self.range_lines) >= 2:
            self.range_lines[0].set_xdata([start1, start1])
            self.range_lines[1].set_xdata([end1, end1])
        
        # Update Range 2 lines if present
        if use_dual_range and len(self.range_lines) == 4 and start2 is not None and end2 is not None:
            self.range_lines[2].set_xdata([start2, start2])
            self.range_lines[3].set_xdata([end2, end2])
        
        self.canvas.draw()
    
    def toggle_dual_range(self, enabled, start2, end2):
        """
        Toggle dual range visualization
        
        Args:
            enabled: Whether to enable dual range
            start2: Start position for range 2
            end2: End position for range 2
        """
        if enabled:
            # Add second range lines if not present
            if len(self.range_lines) == 2:
                self.range_lines.append(
                    self.ax.axvline(start2, color='red', linestyle='-', picker=5)
                )
                self.range_lines.append(
                    self.ax.axvline(end2, color='red', linestyle='-', picker=5)
                )
                self.canvas.draw()
        else:
            # Remove second range lines if present
            if len(self.range_lines) == 4:
                self.range_lines[2].remove()
                self.range_lines[3].remove()
                self.range_lines = self.range_lines[:2]
                self.canvas.draw()
    
    def center_nearest_cursor(self):
        """
        Find the horizontal center of the plot view and move the nearest cursor line to it
        
        Returns:
            Tuple of (line_moved, new_position) or (None, None) if no action taken
        """
        if not self.range_lines or not self.ax.has_data():
            return None, None
        
        # Get the center of the current x-axis view
        x_min, x_max = self.ax.get_xlim()
        center_x = (x_min + x_max) / 2
        
        # Find the cursor line nearest to the center
        nearest_line = None
        min_distance = float('inf')
        
        for line in self.range_lines:
            line_pos_x = line.get_xdata()[0]
            distance = abs(line_pos_x - center_x)
            if distance < min_distance:
                min_distance = distance
                nearest_line = line
        
        # Move the nearest line
        if nearest_line:
            nearest_line.set_xdata([center_x, center_x])
            self.canvas.draw()
            
            # Return which line was moved and the new position
            return nearest_line, center_x
        
        return None, None
    
    # Mouse interaction methods
    def on_pick(self, event):
        """Handle pick events for draggable lines"""
        if event.artist in self.range_lines:
            self.dragging_line = event.artist
    
    def on_drag(self, event):
        """
        Handle drag events for range lines
        
        Returns:
            The spinbox that should be updated and its new value, or (None, None)
        """
        if self.dragging_line and event.xdata is not None:
            x = event.xdata
            self.dragging_line.set_xdata([x, x])
            self.canvas.draw()
            
            # Call the callback if set
            if self.drag_callback:
                self.drag_callback(self.dragging_line, x)
            
            # Return the spinbox to update if mapped (for backward compatibility)
            if self.dragging_line in self.line_spinbox_map:
                return self.line_spinbox_map[self.dragging_line], x
        
        return None, None
    
    def on_release(self, event):
        """Handle mouse release events"""
        self.dragging_line = None
    
    # Batch plotting methods
    def create_batch_figure(self, x_label, y_label, figsize=(10, 6)):
        """
        Create a new figure for batch plotting
        
        Args:
            x_label: Label for x-axis
            y_label: Label for y-axis
            figsize: Figure size tuple
            
        Returns:
            Tuple of (figure, axes)
        """
        import matplotlib.pyplot as plt
        
        batch_fig = plt.figure(figsize=figsize)
        batch_ax = batch_fig.add_subplot(111)
        batch_ax.set_xlabel(x_label)
        batch_ax.set_ylabel(y_label)
        batch_ax.set_title(f"{y_label} vs {x_label}")
        batch_ax.grid(True, alpha=0.3)
        
        return batch_fig, batch_ax
    
    def plot_batch_data(self, ax, x_data, y_data, label, marker='o-', 
                       y_data2=None, label2=None, marker2='s--'):
        """
        Plot data on batch axes
        
        Args:
            ax: Matplotlib axes to plot on
            x_data: X-axis data
            y_data: Y-axis data for range 1
            label: Label for range 1 data
            marker: Marker style for range 1
            y_data2: Optional Y-axis data for range 2
            label2: Label for range 2 data
            marker2: Marker style for range 2
        """
        if len(x_data) > 0 and len(y_data) > 0:
            ax.plot(x_data, y_data, marker, label=label)
            
            if y_data2 is not None and len(y_data2) > 0 and label2:
                ax.plot(x_data, y_data2, marker2, label=label2)
    
    def finalize_batch_plot(self, fig, ax):
        """
        Finalize batch plot with legend and layout
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes
        """
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
            fig.tight_layout()
    
    def clear_plot(self):
        """Clear the current plot"""
        self.ax.clear()
        self.canvas.draw()
    
    def redraw(self):
        """Force a redraw of the canvas"""
        self.canvas.draw()