"""
PatchBatch Electrophysiology Data Analysis Tool

Coordinates synchronization between ControlPanel range spinboxes and PlotManager
cursor positions.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import logging

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class MainRangeCoordinator(QObject):
    """
    Mediates range value synchronization between ControlPanel spinboxes and
    PlotManager cursors. Neither component knows about the other.
    """
    
    # Pass-through signals from ControlPanel
    analysis_requested = Signal()
    export_requested = Signal()
    settings_changed = Signal()
    
    def __init__(self, control_panel, plot_manager):
        super().__init__()
        
        self.control_panel = control_panel
        self.plot_manager = plot_manager
        
        self._spinbox_to_cursor_map = {
            "start1": "range1_start",
            "end1": "range1_end",
            "start2": "range2_start",
            "end2": "range2_end"
        }
        
        self._connect_signals()
        logger.info("MainRangeCoordinator initialized")
    
    def _connect_signals(self):
        """Wire up synchronization and pass-through signals."""
        # ControlPanel -> Coordinator
        self.control_panel.dual_range_toggled.connect(self._on_dual_range_toggled)
        self.control_panel.range_values_changed.connect(self._sync_spinboxes_to_cursors)
        
        self.control_panel.analysis_requested.connect(self.analysis_requested.emit)
        self.control_panel.export_requested.connect(self.export_requested.emit)
        
        # PlotManager -> Coordinator
        self.plot_manager.line_state_changed.connect(self._on_cursor_moved)
        
        self._connect_spinbox_editing_signals()
        logger.debug("Connected all range coordination signals")
    
    def _connect_spinbox_editing_signals(self):
        """Connect editingFinished signals for snap-back behavior."""
        spinboxes = self.control_panel.get_range_spinboxes()
        for spinbox_key, spinbox in spinboxes.items():
            spinbox.editingFinished.connect(self._on_spinbox_editing_finished)
        logger.debug(f"Connected editingFinished for {len(spinboxes)} spinboxes")
    
    # Spinbox -> Cursor sync
    
    def _sync_spinboxes_to_cursors(self):
        """Update cursor positions from spinbox values."""
        vals = self.control_panel.get_range_values()
        
        self.plot_manager.update_range_lines(
            vals["range1_start"],
            vals["range1_end"],
            vals["use_dual_range"],
            vals.get("range2_start"),
            vals.get("range2_end"),
        )
        logger.debug("Synced spinboxes -> cursors")
    
    def _on_spinbox_editing_finished(self):
        """Update spinbox to show actual cursor position after snapping."""
        positions = self.plot_manager.get_line_positions()
        spinboxes = self.control_panel.get_range_spinboxes()
        
        for spinbox_key, spinbox in spinboxes.items():
            line_id = self._spinbox_to_cursor_map.get(spinbox_key)
            if line_id and line_id in positions:
                spinbox.blockSignals(True)
                spinbox.setValue(positions[line_id])
                spinbox.blockSignals(False)
        
        logger.debug("Spinbox editing finished - snapped to cursor positions")
    
    # Cursor -> Spinbox sync
    
    def _on_cursor_moved(self, action: str, line_id: str, position: float):
        """Update spinbox when user drags a cursor."""
        if action == "dragged":
            self._sync_cursor_to_spinbox(line_id, position)
        
        elif action == "centered":
            self._sync_cursor_to_spinbox(line_id, position)
            logger.debug("Cursor centered - triggering settings save")
            self.settings_changed.emit()
        
        elif action == "released":
            logger.debug(f"Cursor '{line_id}' drag completed - triggering settings save")
            self.settings_changed.emit()
    
    def _sync_cursor_to_spinbox(self, line_id: str, position: float):
        """Update a single spinbox from cursor position, blocking signals."""
        if line_id is None or position is None:
            return
        
        spinbox_key = None
        for key, cursor_id in self._spinbox_to_cursor_map.items():
            if cursor_id == line_id:
                spinbox_key = key
                break
        
        if spinbox_key:
            self.control_panel.update_range_value_silent(spinbox_key, position)
            logger.debug(f"Synced cursor '{line_id}' -> spinbox '{spinbox_key}' = {position:.2f}")
    
    def sync_cursors_to_spinboxes(self):
        """Update all spinbox values to match current cursor positions."""
        positions = self.plot_manager.get_line_positions()
        spinboxes = self.control_panel.get_range_spinboxes()
        
        for spinbox_key, spinbox in spinboxes.items():
            line_id = self._spinbox_to_cursor_map.get(spinbox_key)
            if line_id and line_id in positions:
                spinbox.blockSignals(True)
                spinbox.setValue(positions[line_id])
                spinbox.blockSignals(False)
        
        logger.debug("Synced all cursors -> spinboxes")
    
    # Dual range
    
    def _on_dual_range_toggled(self, enabled: bool):
        """Show or hide Range 2 cursors."""
        if enabled:
            vals = self.control_panel.get_range_values()
            start2 = vals.get("range2_start", 600)
            end2 = vals.get("range2_end", 900)
            self.plot_manager.toggle_dual_range(True, start2, end2)
            logger.debug(f"Enabled dual range: Range 2 [{start2}, {end2}]")
        else:
            self.plot_manager.toggle_dual_range(False, 0, 0)
            logger.debug("Disabled dual range")
        
        self._connect_spinbox_editing_signals()