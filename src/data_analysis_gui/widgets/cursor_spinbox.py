"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Cursor-Spinbox synchronization manager for interactive matplotlib plots.
"""
# ===============================================================
# For any dialog that needs draggable cursors synced with spinboxes (example):
#
# self.cursor_manager = CursorSpinboxManager(self.ax, self.canvas)
# self.cursor_manager.add_cursor("my_cursor", self.my_spinbox, 100.0, color="red")
# ===============================================================

from PySide6.QtCore import QObject, Signal
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

class CursorSpinbox(QObject):
    """
    Manages bidirectional sync between draggable matplotlib cursors and QSpinBoxes.
    """
    
    def __init__(self, ax, canvas):
        super().__init__()
        self.ax = ax
        self.canvas = canvas
        self.cursors = {}
        self.dragging_cursor = None
        self.shaded_region = None
        
        # Connect matplotlib events
        self.canvas.mpl_connect('pick_event', self._on_pick)
        self.canvas.mpl_connect('motion_notify_event', self._on_drag)
        self.canvas.mpl_connect('button_release_event', self._on_release)
    
    def add_cursor(self, cursor_id, spinbox, initial_position, color="#73AB84", 
                   linestyle="-", linewidth=2, alpha=0.7):
        """Add a draggable cursor linked to a spinbox."""
        line = self.ax.axvline(
            initial_position, 
            color=color, 
            linestyle=linestyle,
            linewidth=linewidth, 
            alpha=alpha, 
            picker=5
        )
        
        self.cursors[cursor_id] = {
            'line': line,
            'spinbox': spinbox,
            'color': color
        }
        
        # When spinbox changes, update cursor
        spinbox.valueChanged.connect(
            lambda value, cid=cursor_id: self._update_cursor_from_spinbox(cid, value)
        )
    
    def enable_shading(self, alpha=0.1):
        """Enable shaded region between the first two cursors."""
        if len(self.cursors) >= 2:
            cursor_ids = list(self.cursors.keys())
            start_pos = self.cursors[cursor_ids[0]]['spinbox'].value()
            end_pos = self.cursors[cursor_ids[1]]['spinbox'].value()
            color = self.cursors[cursor_ids[0]]['color']
            
            self.shaded_region = {
                'patch': self.ax.axvspan(start_pos, end_pos, alpha=alpha, color=color),
                'start_id': cursor_ids[0],
                'end_id': cursor_ids[1],
                'alpha': alpha
            }
    
    def recreate_shading_after_clear(self):
        """
        Recreate the shaded region patch after axes.clear() has been called.
        Call this from the dialog after clearing axes.
        """
        if self.shaded_region:
            start_pos = self.cursors[self.shaded_region['start_id']]['spinbox'].value()
            end_pos = self.cursors[self.shaded_region['end_id']]['spinbox'].value()
            color = self.cursors[self.shaded_region['start_id']]['color']
            
            self.shaded_region['patch'] = self.ax.axvspan(
                start_pos, end_pos, 
                alpha=self.shaded_region['alpha'], 
                color=color
            )

    def _update_shading(self):
        """Update the shaded region position if enabled."""
        if self.shaded_region:
            start_pos = self.cursors[self.shaded_region['start_id']]['spinbox'].value()
            end_pos = self.cursors[self.shaded_region['end_id']]['spinbox'].value()
            
            # Set old patch invisible
            self.shaded_region['patch'].set_visible(False)
            
            # Create new patch
            color = self.cursors[self.shaded_region['start_id']]['color']
            self.shaded_region['patch'] = self.ax.axvspan(
                start_pos, end_pos, 
                alpha=self.shaded_region['alpha'], 
                color=color
            )
    
    def _update_cursor_from_spinbox(self, cursor_id, value):
        """Update cursor position when spinbox changes."""
        line = self.cursors[cursor_id]['line']
        line.set_xdata([value, value])
        self._update_shading()
        self.canvas.draw_idle()
    
    def _on_pick(self, event):
        """Handle cursor pick event."""
        if isinstance(event.artist, Line2D):
            for cursor_id, data in self.cursors.items():
                if event.artist == data['line']:
                    self.dragging_cursor = cursor_id
                    break
    
    def _on_drag(self, event):
        """Handle cursor drag."""
        if self.dragging_cursor and event.xdata is not None:
            cursor_data = self.cursors[self.dragging_cursor]
            line = cursor_data['line']
            spinbox = cursor_data['spinbox']
            
            # Update line position
            x_pos = float(event.xdata)
            line.set_xdata([x_pos, x_pos])
            
            # Update spinbox (block signals to avoid triggering dialog's _update_plot)
            spinbox.blockSignals(True)
            spinbox.setValue(x_pos)
            spinbox.blockSignals(False)
            
            # DON'T update shading during drag - too finicky
            # It will update on release or when spinbox changes from typing
            
            self.canvas.draw_idle()

    def _on_release(self, event):
        """Handle mouse release."""
        if self.dragging_cursor:
            self.dragging_cursor = None
            # Update shading after drag completes
            self._update_shading()
            self.canvas.draw_idle()

class ConcRespCursors(QObject):
    """
    Manages draggable range pairs for concentration-response analysis.
    
    Each range pair consists of start/end boundary lines with shaded region between them.
    Supports analysis ranges (green) and background ranges (blue) with visual distinction.
    """
    
    # Color scheme
    ANALYSIS_COLOR = "#73AB84"  # Sage green (from plot_style.py)
    BACKGROUND_COLOR = "#1565C0"  # Deep blue

    # Signal emitted when a range boundary is dragged
    range_position_changed = Signal(str, str, float)  # range_id, boundary ('start' or 'end'), new_value
    
    def __init__(self, ax, canvas):
        """
        Initialize the range manager.
        
        Args:
            ax: Matplotlib axis to draw on
            canvas: FigureCanvas for event handling
        """
        super().__init__()
        self.ax = ax
        self.canvas = canvas
        
        # Storage for range pairs
        # {range_id: {
        #     'start_line': Line2D,
        #     'end_line': Line2D, 
        #     'patch': Rectangle,
        #     'color': str,
        #     'is_background': bool
        # }}
        self.ranges = {}
        
        # Dragging state
        self.dragging_range = None  # (range_id, 'start' or 'end')
        
        # Connect matplotlib events
        self.canvas.mpl_connect('pick_event', self._on_pick)
        self.canvas.mpl_connect('motion_notify_event', self._on_drag)
        self.canvas.mpl_connect('button_release_event', self._on_release)
    
    def add_range_pair(
        self, 
        range_id: str, 
        start_val: float, 
        end_val: float, 
        is_background: bool = False
    ):
        """
        Add a new range pair with shading.
        
        Args:
            range_id: Unique identifier for this range
            start_val: Start time value
            end_val: End time value
            is_background: Whether this is a background range (affects color)
        """
        if range_id in self.ranges:
            # Already exists, update instead
            self.update_range_position(range_id, start_val, end_val)
            return
        
        # Determine color based on range type
        color = self.BACKGROUND_COLOR if is_background else self.ANALYSIS_COLOR
        
        # Create shaded patch
        ylim = self.ax.get_ylim()
        patch = self.ax.add_patch(
            plt.Rectangle(
                (start_val, ylim[0]),
                end_val - start_val,
                ylim[1] - ylim[0],
                facecolor=color,
                alpha=0.2,
                edgecolor='none',
                zorder=1
            )
        )
        
        # Create boundary lines
        start_line = self.ax.axvline(
            start_val,
            color=color,
            linestyle='--',
            linewidth=1.5,
            picker=5,
            alpha=0.7
        )
        
        end_line = self.ax.axvline(
            end_val,
            color=color,
            linestyle='--',
            linewidth=1.5,
            picker=5,
            alpha=0.7
        )
        
        # Store range data
        self.ranges[range_id] = {
            'start_line': start_line,
            'end_line': end_line,
            'patch': patch,
            'color': color,
            'is_background': is_background,
            'start_val': start_val,
            'end_val': end_val
        }
        
        self.canvas.draw_idle()
    
    def remove_range_pair(self, range_id: str):
        """
        Remove a range pair and all its visual elements.
        
        Args:
            range_id: Identifier of the range to remove
        """
        if range_id not in self.ranges:
            return
        
        range_data = self.ranges[range_id]
        
        # Remove visual elements
        try:
            range_data['start_line'].remove()
        except:
            pass
        
        try:
            range_data['end_line'].remove()
        except:
            pass
        
        try:
            range_data['patch'].remove()
        except:
            pass
        
        # Remove from storage
        del self.ranges[range_id]
        
        self.canvas.draw_idle()
    
    def update_range_position(self, range_id: str, start_val: float, end_val: float):
        """
        Update the position of an existing range pair.
        
        Args:
            range_id: Identifier of the range to update
            start_val: New start time value
            end_val: New end time value
        """
        if range_id not in self.ranges:
            return
        
        range_data = self.ranges[range_id]
        
        # Update line positions
        range_data['start_line'].set_xdata([start_val, start_val])
        range_data['end_line'].set_xdata([end_val, end_val])
        
        # Update patch position and width
        ylim = self.ax.get_ylim()
        range_data['patch'].set_xy((start_val, ylim[0]))
        range_data['patch'].set_width(end_val - start_val)
        range_data['patch'].set_height(ylim[1] - ylim[0])
        
        # Store new values
        range_data['start_val'] = start_val
        range_data['end_val'] = end_val
        
        self.canvas.draw_idle()
    
    def get_dragged_range(self) -> tuple:
        """
        Get the currently dragging range information.
        
        Returns:
            Tuple of (range_id, boundary) where boundary is 'start' or 'end',
            or None if not dragging
        """
        return self.dragging_range
    
    def recreate_patches_after_clear(self):
        """
        Recreate all patches and lines after axes.clear() has been called.
        Call this from the dialog after clearing axes for replotting.
        """
        # Collect all range data before recreating
        ranges_to_recreate = []
        for range_id, data in self.ranges.items():
            ranges_to_recreate.append({
                'range_id': range_id,
                'start_val': data['start_val'],
                'end_val': data['end_val'],
                'is_background': data['is_background']
            })
        
        # Clear storage
        self.ranges.clear()
        
        # Recreate all ranges
        for range_info in ranges_to_recreate:
            self.add_range_pair(
                range_info['range_id'],
                range_info['start_val'],
                range_info['end_val'],
                range_info['is_background']
            )
    
    def _on_pick(self, event):
        """Handle line pick event to start dragging."""
        if not isinstance(event.artist, Line2D):
            return
        
        # Find which range this line belongs to
        for range_id, data in self.ranges.items():
            if event.artist == data['start_line']:
                self.dragging_range = (range_id, 'start')
                return
            elif event.artist == data['end_line']:
                self.dragging_range = (range_id, 'end')
                return
    
    def _on_drag(self, event):
        """Handle mouse drag to update range boundary position."""
        if self.dragging_range is None or event.xdata is None:
            return
        
        range_id, boundary = self.dragging_range
        
        if range_id not in self.ranges:
            return
        
        range_data = self.ranges[range_id]
        
        # Update the appropriate boundary
        if boundary == 'start':
            new_start = float(event.xdata)
            # Update line position
            range_data['start_line'].set_xdata([new_start, new_start])
            # Update patch
            ylim = self.ax.get_ylim()
            range_data['patch'].set_xy((new_start, ylim[0]))
            range_data['patch'].set_width(range_data['end_val'] - new_start)
            range_data['start_val'] = new_start
            # Emit signal
            self.range_position_changed.emit(range_id, 'start', new_start)
        else:  # boundary == 'end'
            new_end = float(event.xdata)
            # Update line position
            range_data['end_line'].set_xdata([new_end, new_end])
            # Update patch width
            range_data['patch'].set_width(new_end - range_data['start_val'])
            range_data['end_val'] = new_end
            # Emit signal
            self.range_position_changed.emit(range_id, 'end', new_end)
        
        self.canvas.draw_idle()
    
    def _on_release(self, event):
        """Handle mouse release to end dragging."""
        if self.dragging_range:
            self.dragging_range = None
            self.canvas.draw_idle()