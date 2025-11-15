"""
PatchBatch Electrophysiology Data Analysis Tool - Cursor Manager

Manages interactive cursor lines and their text labels for plot analysis ranges.
Extracted from PlotManager to provide focused cursor/text management without
Qt dependencies. Returns values rather than emitting signals - the coordinator
(PlotManager) handles signal emission.

Functions are well documented for clarity (bidirectional signaling between spinboxes and cursors is complex! Effective 
state management and avoiding feedback loops is critical). Stable now, consider migrating to PyQtGraph in the future if
further optimizations/improvements are desired.

CursorManager handles cursor positioning and snap-to-data behavior independently. PlotManager coordinates cursor movements with
control panel spinboxes via MainRangeCoordinator - CursorManager returns values rather
than directly interacting with the coordination layer.

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
    Manages interactive vertical cursor lines and text labels for matplotlib plots.
    
    Handles cursor creation, positioning with snap-to-data, text label updates,
    and mouse interaction tracking. Does not emit Qt signals - returns values
    for PlotManager to handle. Cursors automatically snap to nearest time point
    in loaded data when dragged or positioned programmatically.
    """
    
    def __init__(self, ax: Axes):
        """
        Initialize cursor manager with axes reference.
        
        Args:
            ax: Matplotlib axes to create artists on.
        """
        self._ax = ax
        
        # Cursor line storage: line_id -> Line2D
        self._cursors: Dict[str, Line2D] = {}
        
        # Text label storage: line_id -> Text
        self._cursor_texts: Dict[str, Text] = {}
        
        # Plot data for sampling y-values at cursor positions
        self._current_time_data: Optional[np.ndarray] = None
        self._current_y_data: Optional[np.ndarray] = None
        self._current_channel_type: Optional[str] = None
        self._current_units: str = "pA"
        
        # Drag state
        self._dragging_line_id: Optional[str] = None
    
    # ========================================================================
    # Cursor Lifecycle
    # ========================================================================
    
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
        Create a vertical cursor line at the specified position.
        
        The Line2D is created but NOT added to axes - caller must add it.
        This allows explicit control over when lines are added/removed.
        
        Args:
            line_id: Unique identifier for this cursor.
            position: X-coordinate for the vertical line.
            color: Line color.
            linestyle: Line style ('-', '--', etc.).
            linewidth: Line width.
            alpha: Line transparency.
        
        Returns:
            Line2D object (not yet added to axes).
        """
        # Create Line2D manually without adding to axes
        # Use axes transform to span full y-axis regardless of data limits
        line = Line2D(
            [position, position],  # xdata - same x for vertical line
            [0, 1],  # ydata in axes coordinates (0=bottom, 1=top)
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            picker=5,
            transform=self._ax.get_xaxis_transform()  # x in data coords, y in axes coords
        )
        
        self._cursors[line_id] = line
        logger.debug(f"Created cursor '{line_id}' at position {position:.2f}")
        return line
    
    def remove_cursor(self, line_id: str) -> None:
        """
        Remove cursor line and associated text label from plot and tracking.
        
        Args:
            line_id: Cursor identifier
        """
        # Remove line
        if line_id in self._cursors:
            line = self._cursors[line_id]
            try:
                # Try to remove the line from axes
                # This may raise ValueError if line is not in axes
                line.remove()
                logger.debug(f"Removed cursor line '{line_id}' from axes")
            except (ValueError, AttributeError) as e:
                # Line may not be in axes or already removed - not an error
                logger.debug(f"Line '{line_id}' not in axes or already removed: {e}")
            
            del self._cursors[line_id]
            logger.debug(f"Removed cursor '{line_id}' from tracking")
        
        # Remove text label
        if line_id in self._cursor_texts:
            text = self._cursor_texts[line_id]
            try:
                text.remove()
                logger.debug(f"Removed text label for '{line_id}'")
            except (ValueError, AttributeError) as e:
                logger.debug(f"Text for '{line_id}' not in axes or already removed: {e}")
            
            del self._cursor_texts[line_id]
    
    def update_cursor_position(self, line_id: str, position: float) -> None:
        """
        Update cursor position with automatic snap-to-data.
        
        Position is snapped to nearest time point in loaded data. Updates both
        the cursor line and text label if present.
        
        Args:
            line_id: Cursor identifier
            position: New x-coordinate (will be snapped)
        """
        if line_id not in self._cursors:
            logger.warning(f"Cannot update position: cursor '{line_id}' not found")
            return
        
        # Snap position to nearest data point
        snapped_position = self._snap_to_nearest_time(position)
        
        line = self._cursors[line_id]
        line.set_xdata([snapped_position, snapped_position])
        
        # Update text label if exists
        if line_id in self._cursor_texts:
            self._update_cursor_text(line_id, snapped_position)
        
        logger.debug(f"Updated cursor '{line_id}' to position {snapped_position:.2f}")
    
    def get_all_lines(self) -> List[Line2D]:
        """
        Get all cursor Line2D objects for re-adding after axes.clear().
        
        Returns:
            List of Line2D objects sorted by line_id
        """
        return [self._cursors[line_id] for line_id in sorted(self._cursors.keys())]
    
    def get_cursor_positions(self) -> Dict[str, float]:
        """
        Get current x-positions of all cursors.
        
        Returns:
            Dictionary mapping line_id to x-position.
        """
        positions = {}
        for line_id, line in self._cursors.items():
            positions[line_id] = line.get_xdata()[0]
        return positions
    
    def get_cursor_line(self, line_id: str) -> Optional[Line2D]:
        """
        Gete2D object for a specific cursor.
        
        Args:
            line_id: Identifier of cursor.
        
        Returns:
            Line2D object or None if not found.
        """
        return self._cursors.get(line_id)
    
    # ========================================================================
    # Plot Data Management
    # ========================================================================
    
    def set_plot_data(
        self,
        time_data: np.ndarray,
        y_data: np.ndarray,
        channel_type: str,
        units: str = "pA"
    ) -> None:
        """
        Store plot data for cursor text labels and snap-to-data functionality.
        
        Must be called before creating text labels. Enables sampling actual
        data values at cursor positions and snapping cursors to time points.
        
        Args:
            time_data: Time array (x-values)
            y_data: Data array (y-values)
            channel_type: 'Voltage' or 'Current'
            units: Current channel units (e.g., 'pA', 'nA')
        """
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
        """
        Sample y-value at nearest data point to x-position.
        
        Args:
            x_position: X-coordinate to sample at
        
        Returns:
            Y-value at nearest data point or None if no data available
        """
        if self._current_time_data is None or self._current_y_data is None:
            return None
        
        if len(self._current_time_data) == 0 or len(self._current_y_data) == 0:
            return None
        
        # Find index of nearest time point
        idx = np.argmin(np.abs(self._current_time_data - x_position))
        
        # Return corresponding y-value
        return float(self._current_y_data[idx])
    
    # ========================================================================
    # Text Label Management
    # ========================================================================
    
    def recreate_all_text_labels(self, ax: Axes) -> None:
        """
        Recreate text labels for all cursors using current plot data.
        
        Called after axes.clear() or when toggling cursor visibility.
        Samples y-values at each cursor position and creates new Text objects.
        
        Args:
            ax: Axes to create text on (may differ from stored _ax after clear).
        """
        # Remove existing Text objects from axes before clearing references
        for line_id, text in self._cursor_texts.items():
            try:
                text.remove()
                logger.debug(f"Removed existing text label for '{line_id}'")
            except (ValueError, AttributeError, NotImplementedError):
                # Text already removed by ax.clear() - this is expected, no need to log
                pass
        
        # Clear text references
        self._cursor_texts.clear()
        
        # Create new text for each cursor
        for line_id, line in self._cursors.items():
            x_position = line.get_xdata()[0]
            self._create_cursor_text(line_id, x_position, ax)
        
        logger.debug(f"Recreated {len(self._cursors)} text labels")
    
    def _create_cursor_text(self, line_id: str, x_position: float, ax: Axes) -> None:
        """
        Create text label showing data value at cursor position.
        
        Args:
            line_id: Cursor identifier
            x_position: X-coordinate of cursor
            ax: Axes to create text on
        """
        # Sample y-value at cursor position
        y_value = self._sample_y_value_nearest(x_position)
        
        if y_value is None:
            logger.debug(f"No data available for text label '{line_id}'")
            return
        
        # Determine units based on channel type
        if self._current_channel_type == "Voltage":
            unit = "mV"
            formatted_value = f"{y_value:.1f} {unit}"
        else:
            unit = self._current_units
            formatted_value = f"{y_value:.1f} {unit}"
        
        # Position text near top of plot
        y_min, y_max = ax.get_ylim()
        text_y = y_max - (y_max - y_min) * 0.05  # 5% from top
        
        # Create text object
        text = ax.text(
            x_position, text_y, formatted_value,
            ha='center', va='top',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='gray', alpha=0.9)
        )
        
        # Store reference
        self._cursor_texts[line_id] = text
        
        logger.debug(f"Created text label for '{line_id}' at x={x_position:.2f}, y={y_value:.2f}")
    
    def _update_cursor_text(self, line_id: str, x_position: float) -> None:
        """
        Update text label position and value for a cursor.
        
        Args:
            line_id: Cursor identifier
            x_position: New x-coordinate
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
        
        # Keep y-position near top of plot
        y_min, y_max = self._ax.get_ylim()
        text_y = y_max - (y_max - y_min) * 0.05
        
        text.set_position((x_position, text_y))
    
    def update_all_text_positions(self, ylim: Tuple[float, float]) -> None:
        """
        Reposition all text labels after axis limit changes.
        
        Called after zoom/pan to keep labels near top of view. Does not
        resample data values, only adjusts vertical positioning.
        
        Args:
            ylim: New y-axis limits (min, max)
        """
        if not self._cursor_texts:
            return
        
        y_min, y_max = ylim
        text_y = y_max - (y_max - y_min) * 0.05  # 5% from top
        
        # Update position for each text label
        for line_id, text in self._cursor_texts.items():
            # Get current x-position from the text
            x_position = text.get_position()[0]
            
            # Update y-position to keep text near top
            text.set_position((x_position, text_y))
        
        logger.debug(f"Updated {len(self._cursor_texts)} text positions for new ylim")
    
    # ========================================================================
    # Mouse Interaction
    # ========================================================================
    
    def handle_pick(self, artist: Any) -> Optional[str]:
        """
        Check if a picked artist is one of our cursors.
        
        Called from PlotManager's pick_event handler. Returns line_id if
        the picked artist is a cursor, allowing PlotManager to initiate drag.
        
        Args:
            artist: Matplotlib artist from pick event
        
        Returns:
            line_id if artist is a cursor, None otherwise
        """
        if not isinstance(artist, Line2D):
            return None
        
        # Check if this Line2D is one of our cursors
        for line_id, line in self._cursors.items():
            if line is artist:
                self._dragging_line_id = line_id
                logger.debug(f"Picked cursor '{line_id}'")
                return line_id
        
        return None
    
    def update_drag(self, xdata: Optional[float]) -> Optional[Tuple[str, float]]:
        """
        Update cursor position during drag operation with snap-to-data.
        
        Position is automatically snapped to nearest time point in loaded data.
        Returns snapped position for PlotManager to emit signal. This method
        updates the Line2D and text label, then returns the information
        needed for signal emission.
        
        Args:
            xdata: X-coordinate from mouse event (None if outside axes)
        
        Returns:
            Tuple of (line_id, snapped_position) if dragging, None otherwise
        """
        if not self._dragging_line_id or xdata is None:
            return None
        
        line_id = self._dragging_line_id
        
        # Snap position to nearest data point
        snapped_position = self._snap_to_nearest_time(float(xdata))
        
        # Update cursor position (handles both line and text)
        # Note: This will snap again inside update_cursor_position, but that's
        # idempotent - snapping an already-snapped value returns the same value
        self.update_cursor_position(line_id, snapped_position)
        
        return (line_id, snapped_position)
    
    def release_drag(self) -> Optional[str]:
        """
        End drag operation and clear drag state.
        
        Returns:
            line_id of released cursor or None if not dragging
        """
        if self._dragging_line_id:
            line_id = self._dragging_line_id
            logger.debug(f"Released cursor '{line_id}'")
            self._dragging_line_id = None
            return line_id
        return None
    
    def is_dragging(self) -> bool:
        """
        Check if currently dragging a cursor.
        
        Returns:
            True if dragging, False otherwise
        """
        return self._dragging_line_id is not None
    
    def _snap_to_nearest_time(self, position: float) -> float:
        """
        Snap position to nearest time point in loaded data.
        
        Returns original position unchanged if no data is loaded.
        
        Args:
            position: Target x-coordinate
        
        Returns:
            Snapped position (nearest time point) or original if no data
        """
        if self._current_time_data is None or len(self._current_time_data) == 0:
            return position  # No data loaded - bypass snapping
        
        # Find index of nearest time point
        idx = np.argmin(np.abs(self._current_time_data - position))
        
        # Return the actual time value from data
        snapped_position = float(self._current_time_data[idx])
        
        logger.debug(f"Snapped position {position:.2f} to {snapped_position:.2f}")
        return snapped_position