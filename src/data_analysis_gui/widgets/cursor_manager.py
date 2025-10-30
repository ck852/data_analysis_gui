"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Cursor Manager

Manages interactive cursor lines and their text labels for plot analysis ranges.
Extracted from PlotManager to provide focused cursor/text management without
Qt dependencies. Returns values rather than emitting signals - the coordinator
(PlotManager) handles signal emission.

This module is part of Phase 2 of the plot manager refactoring. It consolidates
all cursor Line2D objects, text labels, and mouse interaction logic.
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
    Manages cursor lines (Line2D) and their text labels (Text) for plot analysis.
    
    This class handles:
    - Creating and removing vertical cursor lines
    - Tracking cursor positions
    - Sampling plot data to show values at cursor positions
    - Creating and updating text labels
    - Mouse interaction (pick/drag/release) without emitting Qt signals
    
    Not a QObject - returns values for the coordinator (PlotManager) to handle.
    Requires an Axes reference for creating matplotlib artists.
    
    Example Usage:
        >>> cursor_mgr = CursorManager(ax)
        >>> # Create cursors with style
        >>> line = cursor_mgr.create_cursor('range1_start', 150, color='green')
        >>> ax.add_line(line)
        >>> 
        >>> # After plotting data, store it for text labels
        >>> cursor_mgr.set_plot_data(time_ms, data_array, 'Voltage', 'pA')
        >>> cursor_mgr.recreate_all_text_labels(ax)
        >>> 
        >>> # Handle mouse interaction
        >>> line_id = cursor_mgr.handle_pick(event.artist)
        >>> if line_id:
        ...     result = cursor_mgr.update_drag(event.xdata)
        ...     if result:
        ...         line_id, new_pos = result
    
    Future Feature Hooks:
        - Snap-to-data: Add snap logic in update_drag() before returning position
        - Multiple cursor sets: Add group parameter to create_cursor()
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
        Remove a cursor line and its text label.
        
        Removes the Line2D from axes (if attached) and Text object,
        then removes from internal tracking.
        
        Args:
            line_id: Identifier of cursor to remove.
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
        Update a cursor's x-position with snap-to-data.
        
        Position is automatically snapped to nearest time point in loaded data.
        Updates both the Line2D position and the text label (if exists).
        
        Args:
            line_id: Identifier of cursor to move.
            position: New x-coordinate (will be snapped).
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
            List of Line2D objects in consistent order (sorted by line_id).
        """
        return [self._cursors[line_id] for line_id in sorted(self._cursors.keys())]
    
    def get_cursor_positions(self) -> Dict[str, float]:
        """
        Get current positions of all cursors.
        
        Returns:
            Dictionary mapping line_id to x-position.
        """
        positions = {}
        for line_id, line in self._cursors.items():
            positions[line_id] = line.get_xdata()[0]
        return positions
    
    def get_cursor_line(self, line_id: str) -> Optional[Line2D]:
        """
        Get the Line2D object for a specific cursor.
        
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
        Store plot data for sampling y-values at cursor positions.
        
        Must be called before creating text labels. This data is used to
        sample the actual data values at each cursor position.
        
        Args:
            time_data: Time array (x-values).
            y_data: Data array (y-values).
            channel_type: 'Voltage' or 'Current'.
            units: Units for current channel (e.g., 'pA', 'nA').
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
        Find the nearest y-value from plot data at given x-position.
        
        Uses nearest-neighbor sampling to find the data point closest to
        the cursor position.
        
        Args:
            x_position: X-coordinate to sample at.
        
        Returns:
            Y-value at nearest data point, or None if no data available.
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
        
        Called after axes.clear() and new plot data is available, OR when
        adding/removing cursor sets (e.g., toggling dual range).
        Samples y-values at each cursor position and creates Text objects.
        
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
        Create a text label for a cursor showing y-value at its position.
        
        Args:
            line_id: Identifier for the cursor.
            x_position: X-coordinate of the cursor.
            ax: Axes to create text on.
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
        Update position and value of a cursor text label.
        
        Args:
            line_id: Identifier for the cursor.
            x_position: New x-coordinate.
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
        Update y-position of all text labels based on new axis limits.
        
        Called after zoom/pan to keep text labels visible near top of view.
        Does not resample y-values, only repositions vertically.
        
        Args:
            ylim: New y-axis limits as (min, max) tuple.
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
            artist: The matplotlib artist from the pick event.
        
        Returns:
            line_id if artist is a cursor, None otherwise.
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
            xdata: New x-coordinate from mouse event (None if outside axes).
        
        Returns:
            (line_id, snapped_position) tuple if dragging, None otherwise.
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
            line_id of released cursor, or None if not dragging.
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
            True if dragging, False otherwise.
        """
        return self._dragging_line_id is not None
    
    def _snap_to_nearest_time(self, position: float) -> float:
        """
        Snap position to nearest time point in loaded data.
        
        If no data is loaded, returns the original position unchanged.
        Uses numpy's argmin for fast nearest-neighbor search.
        
        Args:
            position: Target x-coordinate to snap.
        
        Returns:
            Snapped position (nearest time point in data), or original if no data.
        """
        if self._current_time_data is None or len(self._current_time_data) == 0:
            return position  # No data loaded - bypass snapping
        
        # Find index of nearest time point
        idx = np.argmin(np.abs(self._current_time_data - position))
        
        # Return the actual time value from data
        snapped_position = float(self._current_time_data[idx])
        
        logger.debug(f"Snapped position {position:.2f} to {snapped_position:.2f}")
        return snapped_position