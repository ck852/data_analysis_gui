"""
PatchBatch Electrophysiology Data Analysis Tool - Cursor Manager

Manages interactive cursor lines and text labels for plot analysis ranges.
Returns values rather than emitting signals to enable bidirectional synchronization
without feedback loops between cursor positions and ControlPanel spinboxes.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import logging
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text

logger = logging.getLogger(__name__)


class CursorManager:
    """
    Manages interactive vertical cursor lines with snap-to-data positioning.
    
    Returns values instead of emitting Qt signals. This design allows MainRangeCoordinator
    to control when and how spinboxes are updated, preventing feedback loops. If cursors
    emitted signals directly, a drag operation would update the spinbox, which would
    trigger its valueChanged signal, which would update the cursor again, creating
    infinite recursion.
    
    Cursors automatically snap to the nearest time point in loaded data. When users type
    values in spinboxes, cursors snap to actual data points, then the coordinator updates
    spinboxes to display the snapped value.
    """
    
    def __init__(self, ax: Axes):
        self._ax = ax
        self._cursors: Dict[str, Line2D] = {}
        self._cursor_texts: Dict[str, Text] = {}
        
        # Plot data for snap-to-data and text labels
        self._current_time_data: Optional[np.ndarray] = None
        self._current_y_data: Optional[np.ndarray] = None
        self._current_channel_type: Optional[str] = None
        self._current_units: str = "pA"
        
        self._dragging_line_id: Optional[str] = None
    
    def create_cursor(
        self,
        line_id: str,
        position: float,
        color: str = 'green',
        linestyle: str = '-',
        linewidth: float = 2,
        alpha: float = 1.0
    ) -> Line2D:
        """
        Create vertical cursor line at position. Line2D is created but not added
        to axes - caller must add it explicitly for control over lifecycle.
        """
        line = Line2D(
            [position, position],
            [0, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            picker=5,
            transform=self._ax.get_xaxis_transform()
        )
        
        self._cursors[line_id] = line
        logger.debug(f"Created cursor '{line_id}' at position {position:.2f}")
        return line
    
    def remove_cursor(self, line_id: str) -> None:
        """Remove cursor line and text label from plot and tracking."""
        if line_id in self._cursors:
            line = self._cursors[line_id]
            try:
                line.remove()
            except (ValueError, AttributeError) as e:
                logger.debug(f"Line '{line_id}' already removed: {e}")
            del self._cursors[line_id]
        
        if line_id in self._cursor_texts:
            text = self._cursor_texts[line_id]
            try:
                text.remove()
            except (ValueError, AttributeError) as e:
                logger.debug(f"Text '{line_id}' already removed: {e}")
            del self._cursor_texts[line_id]
    
    def update_cursor_position(self, line_id: str, position: float) -> None:
        """Update cursor to snapped position and refresh text label if present."""
        if line_id not in self._cursors:
            logger.warning(f"Cannot update position: cursor '{line_id}' not found")
            return
        
        snapped_position = self._snap_to_nearest_time(position)
        
        line = self._cursors[line_id]
        line.set_xdata([snapped_position, snapped_position])
        
        if line_id in self._cursor_texts:
            self._update_cursor_text(line_id, snapped_position)
        
        logger.debug(f"Updated cursor '{line_id}' to position {snapped_position:.2f}")
    
    def get_all_lines(self) -> List[Line2D]:
        """Get all cursor Line2D objects for re-adding after axes.clear()."""
        return [self._cursors[line_id] for line_id in sorted(self._cursors.keys())]
    
    def get_cursor_positions(self) -> Dict[str, float]:
        """Get current x-positions of all cursors."""
        positions = {}
        for line_id, line in self._cursors.items():
            positions[line_id] = line.get_xdata()[0]
        return positions
    
    def get_cursor_line(self, line_id: str) -> Optional[Line2D]:
        """Get Line2D object for specific cursor."""
        return self._cursors.get(line_id)
    
    def set_plot_data(
        self,
        time_data: np.ndarray,
        y_data: np.ndarray,
        channel_type: str,
        units: str = "pA"
    ) -> None:
        """Store plot data for text labels and snap-to-data functionality."""
        self._current_time_data = time_data
        self._current_y_data = y_data
        self._current_channel_type = channel_type
        self._current_units = units
        logger.debug(f"Stored plot data: {len(time_data)} points, {channel_type}, {units}")
    
    def clear_plot_data(self) -> None:
        """Clear stored plot data."""
        self._current_time_data = None
        self._current_y_data = None
        self._current_channel_type = None
    
    def _sample_y_value_nearest(self, x_position: float) -> Optional[float]:
        """Sample y-value at nearest data point to x-position."""
        if self._current_time_data is None or self._current_y_data is None:
            return None
        
        if len(self._current_time_data) == 0 or len(self._current_y_data) == 0:
            return None
        
        idx = np.argmin(np.abs(self._current_time_data - x_position))
        return float(self._current_y_data[idx])
    
    def recreate_all_text_labels(self, ax: Axes) -> None:
        """
        Recreate text labels after axes.clear(). Matplotlib removes all artists
        on clear, requiring recreation while maintaining cursor state.
        """
        for line_id, text in self._cursor_texts.items():
            try:
                text.remove()
            except (ValueError, AttributeError, NotImplementedError):
                pass
        
        self._cursor_texts.clear()
        
        for line_id, line in self._cursors.items():
            x_position = line.get_xdata()[0]
            self._create_cursor_text(line_id, x_position, ax)
        
        logger.debug(f"Recreated {len(self._cursors)} text labels")
    
    def _create_cursor_text(self, line_id: str, x_position: float, ax: Axes) -> None:
        """Create text label showing data value at cursor position."""
        y_value = self._sample_y_value_nearest(x_position)
        
        if y_value is None:
            logger.debug(f"No data available for text label '{line_id}'")
            return
        
        if self._current_channel_type == "Voltage":
            unit = "mV"
            formatted_value = f"{y_value:.1f} {unit}"
        else:
            unit = self._current_units
            formatted_value = f"{y_value:.1f} {unit}"
        
        y_min, y_max = ax.get_ylim()
        text_y = y_max - (y_max - y_min) * 0.05
        
        text = ax.text(
            x_position, text_y, formatted_value,
            ha='center', va='top',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='gray', alpha=0.9)
        )
        
        self._cursor_texts[line_id] = text
        logger.debug(f"Created text label for '{line_id}' at x={x_position:.2f}, y={y_value:.2f}")
    
    def _update_cursor_text(self, line_id: str, x_position: float) -> None:
        """Update text label position and value for cursor."""
        if line_id not in self._cursor_texts:
            return
        
        y_value = self._sample_y_value_nearest(x_position)
        if y_value is None:
            return
        
        if self._current_channel_type == "Voltage":
            unit = "mV"
            formatted_value = f"{y_value:.1f} {unit}"
        else:
            unit = self._current_units
            formatted_value = f"{y_value:.1f} {unit}"
        
        text = self._cursor_texts[line_id]
        text.set_text(formatted_value)
        
        y_min, y_max = self._ax.get_ylim()
        text_y = y_max - (y_max - y_min) * 0.05
        text.set_position((x_position, text_y))
    
    def update_all_text_positions(self, ylim: Tuple[float, float]) -> None:
        """
        Reposition text labels after axis limit changes (zoom/pan).
        Does not resample data values, only adjusts vertical positioning.
        """
        if not self._cursor_texts:
            return
        
        y_min, y_max = ylim
        text_y = y_max - (y_max - y_min) * 0.05
        
        for line_id, text in self._cursor_texts.items():
            x_position = text.get_position()[0]
            text.set_position((x_position, text_y))
        
        logger.debug(f"Updated {len(self._cursor_texts)} text positions for new ylim")
    
    def handle_pick(self, artist: Any) -> Optional[str]:
        """Check if picked artist is a cursor. Returns line_id if yes, None otherwise."""
        if not isinstance(artist, Line2D):
            return None
        
        for line_id, line in self._cursors.items():
            if line is artist:
                self._dragging_line_id = line_id
                logger.debug(f"Picked cursor '{line_id}'")
                return line_id
        
        return None
    
    def update_drag(self, xdata: Optional[float]) -> Optional[Tuple[str, float]]:
        """
        Update cursor position during drag with snap-to-data. Returns (line_id, 
        snapped_position) for coordinator to emit signal and update spinbox.
        """
        if not self._dragging_line_id or xdata is None:
            return None
        
        line_id = self._dragging_line_id
        snapped_position = self._snap_to_nearest_time(float(xdata))
        
        self.update_cursor_position(line_id, snapped_position)
        
        return (line_id, snapped_position)
    
    def release_drag(self) -> Optional[str]:
        """End drag operation and return released cursor line_id."""
        if self._dragging_line_id:
            line_id = self._dragging_line_id
            logger.debug(f"Released cursor '{line_id}'")
            self._dragging_line_id = None
            return line_id
        return None
    
    def is_dragging(self) -> bool:
        """Check if currently dragging a cursor."""
        return self._dragging_line_id is not None
    
    def _snap_to_nearest_time(self, position: float) -> float:
        """
        Snap position to nearest time point in loaded data. Returns original 
        position if no data loaded. Critical for ensuring cursors and spinboxes
        reference actual data points.
        """
        if self._current_time_data is None or len(self._current_time_data) == 0:
            return position
        
        idx = np.argmin(np.abs(self._current_time_data - position))
        snapped_position = float(self._current_time_data[idx])
        
        logger.debug(f"Snapped position {position:.2f} to {snapped_position:.2f}")
        return snapped_position