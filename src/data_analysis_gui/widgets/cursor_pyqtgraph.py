"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

PyQtGraph-based cursor system for interactive range definition.

Provides draggable range cursors using PyQtGraph's LinearRegionItem for
concentration-response analysis and other interactive plotting needs.
"""

import pyqtgraph as pg
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor

from data_analysis_gui.config.pyqtgraph_style import (
    get_cursor_pen,
    get_cursor_brush,
    get_cursor_color,
)

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class PyQtGraphRangeCursor(QObject):
    """
    Manages a single draggable range cursor using PyQtGraph's LinearRegionItem.
    
    Each cursor represents one analysis range with two draggable boundaries
    and a shaded region between them. Emits signals when boundaries are moved.
    
    Signals:
        position_changed(str, str, float): Emitted when boundary moves
            (range_id, boundary ('start' or 'end'), new_value)
    """
    
    position_changed = Signal(str, str, float)
    
    def __init__(self, plot_item, range_id: str, start_value: float, end_value: float, is_background: bool = False):
        """
        Initialize a range cursor.
        
        Args:
            plot_item: PyQtGraph PlotItem to add cursor to
            range_id: Unique identifier for this range
            start_value: Start boundary position
            end_value: End boundary position
            is_background: Whether this is a background range (affects color)
        """
        super().__init__()
        
        self.plot_item = plot_item
        self.range_id = range_id
        self.is_background = is_background
        
        # Get styled pen and brush from centralized styling
        pen = get_cursor_pen(is_background)
        brush = get_cursor_brush(is_background, alpha=0.2)
        
        self.region = pg.LinearRegionItem(
            values=[start_value, end_value],
            brush=brush,
            pen=pen,
            movable=True
        )
        
        # Add to plot
        self.plot_item.addItem(self.region)
        
        # Connect signal
        self.region.sigRegionChanged.connect(self._on_region_changed)
        
        # Store current values
        self._start_value = start_value
        self._end_value = end_value
        
        range_type = "background" if is_background else "analysis"
        logger.debug(f"Created {range_type} cursor '{range_id}': [{start_value:.2f}, {end_value:.2f}]")
    
    def _on_region_changed(self):
        """
        Handle region change from user dragging.
        
        Always emits position_changed signals for both boundaries with sorted values,
        ensuring Start is always <= End regardless of which cursor was dragged.
        """
        new_start, new_end = self.region.getRegion()  # PyQtGraph returns sorted [min, max]
        
        # Always emit both boundaries with sorted values
        # This ensures Start is always the smaller value, End is always the larger value
        if abs(new_start - self._start_value) > 1e-6 or abs(new_end - self._end_value) > 1e-6:
            self._start_value = new_start
            self._end_value = new_end
            
            # Emit both boundaries so spinboxes update to sorted values
            self.position_changed.emit(self.range_id, 'start', new_start)
            self.position_changed.emit(self.range_id, 'end', new_end)
    
    def update_position(self, start_value: float, end_value: float):
        """
        Update the cursor position programmatically.
        
        Args:
            start_value: New start boundary position
            end_value: New end boundary position
        """
        # Block signals to prevent triggering _on_region_changed
        self.region.sigRegionChanged.disconnect(self._on_region_changed)
        
        self.region.setRegion([start_value, end_value])
        self._start_value = start_value
        self._end_value = end_value
        
        # Reconnect signal
        self.region.sigRegionChanged.connect(self._on_region_changed)
        
        logger.debug(f"Updated cursor '{self.range_id}' position: [{start_value:.2f}, {end_value:.2f}]")
    
    def remove(self):
        """Remove the cursor from the plot."""
        try:
            self.plot_item.removeItem(self.region)
            logger.debug(f"Removed cursor '{self.range_id}' from plot")
        except Exception as e:
            logger.warning(f"Error removing cursor '{self.range_id}': {e}")
    
    def get_region(self):
        """
        Get the current region values.
        
        Returns:
            tuple: (start_value, end_value)
        """
        return self.region.getRegion()

class PyQtGraphCursorManager(QObject):
    """
    Manages all range cursors for a plot.
    
    Handles creation, removal, and updates of multiple range cursors.
    Stores cursor data for recreation after plot clearing.
    
    Signals:
        range_position_changed(str, str, float): Forwarded from individual cursors
            (range_id, boundary, new_value)
    """
    
    range_position_changed = Signal(str, str, float)
    
    def __init__(self, plot_item, plot_widget):
        """
        Initialize the cursor manager.
        
        Args:
            plot_item: PyQtGraph PlotItem for adding cursors
            plot_widget: PyQtGraph PlotWidget (parent widget)
        """
        super().__init__()
        
        self.plot_item = plot_item
        self.plot_widget = plot_widget
        
        # Storage for cursors
        self.cursors = {}  # {range_id: PyQtGraphRangeCursor}
        
        logger.debug("PyQtGraphCursorManager initialized")
    
    def add_range_pair(self, range_id: str, start_val: float, end_val: float, is_background: bool = False):
        """
        Add a new range cursor.
        
        Args:
            range_id: Unique identifier for this range
            start_val: Start boundary position
            end_val: End boundary position
            is_background: Whether this is a background range
        """
        if range_id in self.cursors:
            logger.warning(f"Range '{range_id}' already exists, updating position instead")
            self.update_range_position(range_id, start_val, end_val)
            return
        
        # Create new cursor
        cursor = PyQtGraphRangeCursor(
            self.plot_item,
            range_id,
            start_val,
            end_val,
            is_background
        )
        
        # Connect signal to forward to manager's signal
        cursor.position_changed.connect(self._forward_position_change)
        
        # Store cursor
        self.cursors[range_id] = cursor
        
        range_type = "background" if is_background else "analysis"
        logger.info(f"Added {range_type} range cursor '{range_id}': [{start_val:.2f}, {end_val:.2f}]")
    
    def remove_range_pair(self, range_id: str):
        """
        Remove a range cursor.
        
        Args:
            range_id: Identifier of the range to remove
        """
        if range_id not in self.cursors:
            logger.warning(f"Attempted to remove non-existent cursor '{range_id}'")
            return
        
        # Remove cursor
        cursor = self.cursors[range_id]
        cursor.remove()
        
        # Delete from storage
        del self.cursors[range_id]
        
        logger.info(f"Removed cursor '{range_id}'")
    
    def update_range_position(self, range_id: str, start_val: float, end_val: float):
        """
        Update the position of an existing cursor.
        
        Args:
            range_id: Identifier of the range to update
            start_val: New start boundary position
            end_val: New end boundary position
        """
        if range_id not in self.cursors:
            logger.warning(f"Attempted to update non-existent cursor '{range_id}'")
            return
        
        cursor = self.cursors[range_id]
        cursor.update_position(start_val, end_val)
        
        logger.debug(f"Updated cursor '{range_id}' position: [{start_val:.2f}, {end_val:.2f}]")
    
    def recreate_all(self):
        """
        Recreate all cursors after plot has been cleared.
        
        Stores cursor data, removes old cursors, and creates new ones.
        Called after plot.clear() in the dialog.
        """
        if not self.cursors:
            logger.debug("No cursors to recreate")
            return
        
        # Collect data from existing cursors
        cursor_data = []
        for range_id, cursor in self.cursors.items():
            start_val, end_val = cursor.get_region()
            cursor_data.append({
                'range_id': range_id,
                'start_val': start_val,
                'end_val': end_val,
                'is_background': cursor.is_background
            })
        
        # Clear existing cursors (they're already removed from plot by plot.clear())
        self.cursors.clear()
        
        # Recreate all cursors
        for data in cursor_data:
            self.add_range_pair(
                data['range_id'],
                data['start_val'],
                data['end_val'],
                data['is_background']
            )
        
        logger.info(f"Recreated {len(cursor_data)} cursors after plot clear")
    
    def _forward_position_change(self, range_id: str, boundary: str, new_value: float):
        """
        Forward position change signal from individual cursor to manager's signal.
        
        Args:
            range_id: Identifier of the range that changed
            boundary: 'start' or 'end'
            new_value: New boundary position
        """
        self.range_position_changed.emit(range_id, boundary, new_value)