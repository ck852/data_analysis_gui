"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Axis Zoom Controller

Manages axis-specific zoom buttons for matplotlib plots, providing independent
zoom controls for X and Y axes. Extracted from PlotManager to provide focused
zoom functionality without Qt dependencies.

This module follows the architecture pattern established by CursorManager and
ViewStateManager. It manages matplotlib Button widgets and calculates zoom
transformations, but does not directly manipulate axes - instead returning
new limits for the coordinator (PlotManager) to apply.
"""

import logging
from typing import Optional, Tuple, Callable, List

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.widgets import Button

logger = logging.getLogger(__name__)


class AxisZoomController:
    """
    Manages axis-specific zoom buttons for matplotlib plots.
    
    This class handles:
    - Creating and positioning matplotlib Button widgets for X+/X-/Y+/Y-
    - Calculating new axis limits for zoom in/out operations
    - Properly cleaning up button event handlers to prevent conflicts
    - Returning calculated limits for coordinator to apply
    
    Not a QObject - returns values for the coordinator (PlotManager) to handle.
    Requires Figure and Axes references for creating matplotlib widgets.
    
    Zoom behavior:
    - Zoom in: Reduces visible range by 20% (factor = 0.8)
    - Zoom out: Increases visible range by 25% (factor = 1.25)
    - These factors are symmetric: 5 clicks in gives same result as 5 clicks out
    - Zoom is centered on current view's midpoint
    
    Example Usage:
        >>> zoom_controller = AxisZoomController(fig, ax)
        >>> 
        >>> # After plotting and tight_layout, create buttons
        >>> zoom_controller.create_buttons(on_zoom_callback)
        >>> 
        >>> # Before clearing axes, remove buttons
        >>> zoom_controller.clear_buttons()
        >>> ax.clear()
        >>> 
        >>> # Calculate new limits (doesn't apply them)
        >>> new_xlim = zoom_controller.calculate_zoom('x', 'in', current_xlim)
        >>> ax.set_xlim(new_xlim)
    
    Future Feature Hooks:
        - Configurable zoom factors: Add zoom_in_factor/zoom_out_factor parameters
        - Custom button positioning: Add position parameters to create_buttons()
        - Zoom to data bounds: Add method to calculate limits based on data range
    """
    
    # Zoom factors (reciprocals to ensure symmetric behavior)
    ZOOM_IN_FACTOR = 0.8   # Reduce range by 20%
    ZOOM_OUT_FACTOR = 1.25  # Increase range by 25% (1 / 0.8)
    
    def __init__(self, figure: Figure, ax: Axes):
        """
        Initialize axis zoom controller with figure and axes references.
        
        Args:
            figure: Matplotlib figure to create button axes on.
            ax: Main plot axes for zoom operations.
        """
        self._figure = figure
        self._ax = ax
        
        # Button storage: list of Button widget references
        self._buttons: List[Button] = []
        
        # Callback for zoom actions (set during create_buttons)
        self._on_zoom_callback: Optional[Callable[[str, str], None]] = None
    
    def create_buttons(self, on_zoom_callback: Callable[[str, str], None]) -> None:
        """
        Create axis zoom buttons and add them to the figure.
        
        Must be called AFTER tight_layout() to avoid layout conflicts.
        Buttons are positioned in the bottom-left corner of the plot.
        
        Layout:
        - X buttons: Horizontal pair at bottom-left (X- left, X+ right)
        - Y buttons: Vertical pair to the left of X buttons (Y- bottom, Y+ top)
        
        Args:
            on_zoom_callback: Function(axis, direction) called when button clicked.
                            axis is 'x' or 'y', direction is 'in' or 'out'.
        """
        # Clear any existing buttons first
        self.clear_buttons()
        
        # Store callback
        self._on_zoom_callback = on_zoom_callback
        
        # Button styling to match application theme
        button_props = {
            'color': '#F0F0F0',
            'hovercolor': '#E0E0E0',
        }
        
        # === X-axis buttons (bottom-left corner, horizontal) ===
        x_button_width = 0.025
        x_button_height = 0.035
        x_left_position = 0.05  # Left side of plot area
        x_spacing = 0.002  # Gap between buttons
        x_y_position = 0.0  # Just above X-axis label area
        
        # X- button (zoom out)
        ax_xminus = self._figure.add_axes([
            x_left_position,
            x_y_position,
            x_button_width,
            x_button_height
        ])
        btn_xminus = Button(ax_xminus, 'X-', **button_props)
        btn_xminus.label.set_fontsize(8)
        btn_xminus.label.set_weight('normal')
        btn_xminus.on_clicked(lambda event: self._handle_button_click('x', 'out'))
        self._buttons.append(btn_xminus)
        
        # X+ button (zoom in)
        ax_xplus = self._figure.add_axes([
            x_left_position + x_button_width + x_spacing,
            x_y_position,
            x_button_width,
            x_button_height
        ])
        btn_xplus = Button(ax_xplus, 'X+', **button_props)
        btn_xplus.label.set_fontsize(8)
        btn_xplus.label.set_weight('normal')
        btn_xplus.on_clicked(lambda event: self._handle_button_click('x', 'in'))
        self._buttons.append(btn_xplus)
        
        # === Y-axis buttons (bottom-left corner, vertical) ===
        y_button_width = 0.025
        y_button_height = 0.035
        y_x_position = 0.01  # Left edge, before X buttons
        y_bottom_position = 0.05  # Aligned with plot area
        y_spacing = 0.002  # Gap between buttons
        
        # Y- button (zoom out)
        ax_yminus = self._figure.add_axes([
            y_x_position,
            y_bottom_position,
            y_button_width,
            y_button_height
        ])
        btn_yminus = Button(ax_yminus, 'Y-', **button_props)
        btn_yminus.label.set_fontsize(8)
        btn_yminus.label.set_weight('normal')
        btn_yminus.on_clicked(lambda event: self._handle_button_click('y', 'out'))
        self._buttons.append(btn_yminus)
        
        # Y+ button (zoom in)
        ax_yplus = self._figure.add_axes([
            y_x_position,
            y_bottom_position + y_button_height + y_spacing,
            y_button_width,
            y_button_height
        ])
        btn_yplus = Button(ax_yplus, 'Y+', **button_props)
        btn_yplus.label.set_fontsize(8)
        btn_yplus.label.set_weight('normal')
        btn_yplus.on_clicked(lambda event: self._handle_button_click('y', 'in'))
        self._buttons.append(btn_yplus)
        
        logger.debug(f"Created {len(self._buttons)} axis zoom buttons")
    
    def clear_buttons(self) -> None:
        """
        Properly clean up axis zoom buttons by disconnecting event handlers
        before removing axes.
        
        This prevents "Other artist currently being used" errors that occur
        when matplotlib tries to handle events for removed axes.
        
        Should be called BEFORE ax.clear() in plot update cycle.
        """
        for button in self._buttons:
            try:
                # Disconnect the button's event handler
                if hasattr(button, 'disconnect_events'):
                    button.disconnect_events()
                
                # Remove the button's axes from the figure
                if hasattr(button, 'ax') and button.ax:
                    button.ax.remove()
            except Exception as e:
                # Log but don't crash if cleanup fails
                logger.debug(f"Error cleaning up zoom button: {e}")
        
        self._buttons.clear()
        logger.debug("Cleared axis zoom buttons")
    
    def calculate_zoom(
        self,
        axis: str,
        direction: str,
        current_limits: Tuple[float, float],
        max_bounds: Optional[Tuple[float, float]] = None  # ADD THIS PARAMETER
    ) -> Tuple[float, float]:
        """
        Calculate new axis limits for zoom operation with optional bounds clamping.
        
        Does not apply the limits - returns them for coordinator to apply.
        Zoom is centered on the current view's midpoint.
        
        Args:
            axis: 'x' or 'y' - which axis to zoom.
            direction: 'in' (reduce range) or 'out' (increase range).
            current_limits: Current axis limits as (min, max) tuple.
            max_bounds: Optional (min, max) to clamp result within data bounds.
        
        Returns:
            New axis limits as (min, max) tuple, clamped to max_bounds if provided.
        
        Raises:
            ValueError: If axis is not 'x' or 'y', or direction is not 'in' or 'out'.
        """
        if axis not in ('x', 'y'):
            raise ValueError(f"Invalid axis: {axis}. Must be 'x' or 'y'.")
        
        if direction not in ('in', 'out'):
            raise ValueError(f"Invalid direction: {direction}. Must be 'in' or 'out'.")
        
        current_min, current_max = current_limits
        
        # Calculate center and current range
        center = (current_min + current_max) / 2
        current_range = current_max - current_min
        
        # Calculate new range based on zoom direction
        if direction == 'in':
            new_range = current_range * self.ZOOM_IN_FACTOR
        else:  # 'out'
            new_range = current_range * self.ZOOM_OUT_FACTOR
        
        # Calculate new limits centered on current view
        new_min = center - new_range / 2
        new_max = center + new_range / 2
        
        # NEW: Clamp to max bounds if provided
        if max_bounds is not None:
            bounds_min, bounds_max = max_bounds
            
            # If zooming out would exceed bounds, clamp to bounds
            if new_min < bounds_min:
                new_min = bounds_min
            if new_max > bounds_max:
                new_max = bounds_max
            
            # Ensure we don't have invalid range after clamping
            if new_max <= new_min:
                # If clamping created invalid range, just use the bounds
                new_min, new_max = bounds_min, bounds_max
                logger.debug(f"Zoom clamped to data bounds: [{new_min:.2f}, {new_max:.2f}]")
        
        logger.debug(
            f"Calculated {axis}-axis zoom {direction}: "
            f"[{current_min:.2f}, {current_max:.2f}] -> [{new_min:.2f}, {new_max:.2f}]"
            + (f" (clamped to {max_bounds})" if max_bounds else "")
        )
        
        return (new_min, new_max)
    
    def _handle_button_click(self, axis: str, direction: str) -> None:
        """
        Internal handler for button clicks.
        
        Delegates to the coordinator's callback rather than directly
        manipulating axes. This maintains separation of concerns.
        
        Args:
            axis: 'x' or 'y'.
            direction: 'in' or 'out'.
        """
        if self._on_zoom_callback:
            self._on_zoom_callback(axis, direction)
        else:
            logger.warning(
                f"Zoom button clicked ({axis}, {direction}) but no callback set"
            )
    
    def has_buttons(self) -> bool:
        """
        Check if buttons are currently created.
        
        Returns:
            True if buttons exist, False otherwise.
        """
        return len(self._buttons) > 0