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

from PySide6.QtCore import QObject
from matplotlib.lines import Line2D


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