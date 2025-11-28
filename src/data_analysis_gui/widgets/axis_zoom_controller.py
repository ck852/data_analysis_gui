"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Manages axis-specific zoom buttons for matplotlib plots. Provides independent
zoom controls for X and Y axes without direct Qt dependencies.

Follows the same coordinator pattern as CursorManager and ViewStateManager:
calculates zoom transformations but returns new limits rather than applying
them directly.
"""

import logging
from typing import Optional, Tuple, Callable, List

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.widgets import Button

logger = logging.getLogger(__name__)


class AxisZoomController:
    """
    Manages X+/X-/Y+/Y- zoom buttons on matplotlib figures.
    
    Buttons are matplotlib widgets that need to be recreated when clearing axes.
    Zoom is 20% in, 25% out (symmetric).
    """
    
    ZOOM_IN_FACTOR = 0.8   # Reduce range by 20%
    ZOOM_OUT_FACTOR = 1.25  # Increase by 25% (1 / 0.8 for symmetry)
    
    def __init__(self, figure: Figure, ax: Axes):
        self._figure = figure
        self._ax = ax
        self._buttons: List[Button] = []
        self._on_zoom_callback: Optional[Callable[[str, str], None]] = None
    
    def create_buttons(self, on_zoom_callback: Callable[[str, str], None]) -> None:
        """
        Create axis zoom buttons and add them to the figure.
        
        Call AFTER tight_layout() to avoid layout conflicts.
        
        Layout:
        - X buttons: Horizontal pair at bottom-left (X- left, X+ right)
        - Y buttons: Vertical pair to the left of X buttons (Y- bottom, Y+ top)
        
        Callback receives (axis, direction) where axis is 'x'/'y' and 
        direction is 'in'/'out'.
        """
        self.clear_buttons()
        self._on_zoom_callback = on_zoom_callback
        
        button_props = {
            'color': '#F0F0F0',
            'hovercolor': '#E0E0E0',
        }
        
        # X-axis buttons (bottom-left corner, horizontal)
        x_button_width = 0.035
        x_button_height = 0.055
        x_left_position = 0.04
        x_spacing = 0.002
        x_y_position = 0.0
        
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
        
        # Y-axis buttons (bottom-left corner, vertical)
        y_button_width = 0.035
        y_button_height = 0.055
        y_x_position = 0.005
        y_bottom_position = 0.2
        y_spacing = 0.002
        
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
        Clean up buttons by disconnecting event handlers before removing axes.
        
        Prevents "Other artist currently being used" errors when matplotlib
        tries to handle events for removed axes. Call before ax.clear().
        """
        for button in self._buttons:
            try:
                if hasattr(button, 'disconnect_events'):
                    button.disconnect_events()
                
                if hasattr(button, 'ax') and button.ax:
                    button.ax.remove()
            except Exception as e:
                logger.debug(f"Error cleaning up zoom button: {e}")
        
        self._buttons.clear()
        logger.debug("Cleared axis zoom buttons")
    
    def calculate_zoom(
        self,
        axis: str,
        direction: str,
        current_limits: Tuple[float, float],
        max_bounds: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        Calculate new axis limits for zoom operation, optionally clamped to bounds.
        
        Zoom is centered on the current view's midpoint. Returns limits without
        applying them - coordinator handles application.
        """
        if axis not in ('x', 'y'):
            raise ValueError(f"Invalid axis: {axis}. Must be 'x' or 'y'.")
        
        if direction not in ('in', 'out'):
            raise ValueError(f"Invalid direction: {direction}. Must be 'in' or 'out'.")
        
        current_min, current_max = current_limits
        center = (current_min + current_max) / 2
        current_range = current_max - current_min
        
        if direction == 'in':
            new_range = current_range * self.ZOOM_IN_FACTOR
        else:
            new_range = current_range * self.ZOOM_OUT_FACTOR
        
        new_min = center - new_range / 2
        new_max = center + new_range / 2
        
        if max_bounds is not None:
            bounds_min, bounds_max = max_bounds
            
            if new_min < bounds_min:
                new_min = bounds_min
            if new_max > bounds_max:
                new_max = bounds_max
            
            # Ensure valid range after clamping
            if new_max <= new_min:
                new_min, new_max = bounds_min, bounds_max
                logger.debug(f"Zoom clamped to data bounds: [{new_min:.2f}, {new_max:.2f}]")
        
        logger.debug(
            f"Calculated {axis}-axis zoom {direction}: "
            f"[{current_min:.2f}, {current_max:.2f}] -> [{new_min:.2f}, {new_max:.2f}]"
            + (f" (clamped to {max_bounds})" if max_bounds else "")
        )
        
        return (new_min, new_max)
    
    def _handle_button_click(self, axis: str, direction: str) -> None:
        """Route button clicks to coordinator's callback."""
        if self._on_zoom_callback:
            self._on_zoom_callback(axis, direction)
        else:
            logger.warning(
                f"Zoom button clicked ({axis}, {direction}) but no callback set"
            )
    
    def has_buttons(self) -> bool:
        """Check if buttons are currently created."""
        return len(self._buttons) > 0