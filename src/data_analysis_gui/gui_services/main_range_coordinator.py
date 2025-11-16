"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Main Range Coordinator

Coordinates bidirectional synchronization between ControlPanel range spinboxes
and PlotManager cursor positions. When a user drags a cursor, the spinbox updates.
When they type in a spinbox, the cursor moves. Neither component needs to know
about the other - this coordinator mediates all communication.

Extracted from MainWindow to simplify that class and encapsulate the complex
synchronization logic. Prevents feedback loops through careful signal blocking.
"""

import logging

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class MainRangeCoordinator(QObject):
    """
    Mediates range value synchronization between ControlPanel spinboxes and PlotManager cursors.
    
    Neither the ControlPanel nor PlotManager know about each other - they only communicate
    through this coordinator. This prevents tight coupling and makes both components
    independently testable.
    
    Pass-through signals (analysis_requested, export_requested) are provided so MainWindow
    doesn't need to connect to ControlPanel directly.
    """
    
    # Pass-through signals from ControlPanel for MainWindow convenience
    analysis_requested = Signal()
    export_requested = Signal()

    settings_changed = Signal()
    
    def __init__(self, control_panel, plot_manager):
        """
        Args:
            control_panel: ControlPanel widget with range spinboxes.
            plot_manager: PlotManager with cursor lines.
        """
        super().__init__()
        
        self.control_panel = control_panel
        self.plot_manager = plot_manager
        
        # Mapping between spinbox keys and cursor line IDs
        self._spinbox_to_cursor_map = {
            "start1": "range1_start",
            "end1": "range1_end",
            "start2": "range2_start",
            "end2": "range2_end"
        }
        
        self._connect_signals()
        
        logger.info("MainRangeCoordinator initialized")
    
    def _connect_signals(self):
        """Wire up all bidirectional synchronization and pass-through signals."""
        # ControlPanel → Coordinator
        self.control_panel.dual_range_toggled.connect(self._on_dual_range_toggled)
        self.control_panel.range_values_changed.connect(self._sync_spinboxes_to_cursors)
        
        # Pass-through signals for MainWindow convenience
        self.control_panel.analysis_requested.connect(self.analysis_requested.emit)
        self.control_panel.export_requested.connect(self.export_requested.emit)
        
        # PlotManager → Coordinator
        self.plot_manager.line_state_changed.connect(self._on_cursor_moved)
        
        # Spinbox editingFinished → Coordinator (for snap-back behavior)
        self._connect_spinbox_editing_signals()
        
        logger.debug("Connected all range coordination signals")
    
    def _connect_spinbox_editing_signals(self):
        """
        Connect editingFinished signals for snap-back behavior.
        
        When user finishes editing a spinbox (Enter or focus loss), the spinbox
        updates to show where the cursor actually landed after snapping to data.
        """
        spinboxes = self.control_panel.get_range_spinboxes()
        
        for spinbox_key, spinbox in spinboxes.items():
            spinbox.editingFinished.connect(self._on_spinbox_editing_finished)
        
        logger.debug(f"Connected editingFinished for {len(spinboxes)} spinboxes")
    
    # =========================================================================
    # Spinbox → Cursor Synchronization
    # =========================================================================
    
    def _sync_spinboxes_to_cursors(self):
        """
        Update cursor positions when user types in spinboxes.
        
        Cursors snap to nearest time index data points (handled by PlotManager).
        Called continuously as user types for real-time feedback.
        """
        vals = self.control_panel.get_range_values()
        
        self.plot_manager.update_range_lines(
            vals["range1_start"],
            vals["range1_end"],
            vals["use_dual_range"],
            vals.get("range2_start"),
            vals.get("range2_end"),
        )
        
        logger.debug("Synced spinboxes → cursors")
    
    def _on_spinbox_editing_finished(self):
        """
        Update spinbox to show actual cursor position after user finishes editing.
        
        Provides visual feedback that cursor snapped to a data point rather than
        the exact typed value.
        """
        # Get actual cursor positions from plot
        positions = self.plot_manager.get_line_positions()
        
        # Get spinboxes (only active ones based on dual range state)
        spinboxes = self.control_panel.get_range_spinboxes()
        
        # Update each spinbox to match its cursor position
        for spinbox_key, spinbox in spinboxes.items():
            line_id = self._spinbox_to_cursor_map.get(spinbox_key)
            if line_id and line_id in positions:
                # Block signals to prevent recursion
                spinbox.blockSignals(True)
                spinbox.setValue(positions[line_id])
                spinbox.blockSignals(False)
        
        logger.debug("Spinbox editing finished - snapped to cursor positions")
    
    # =========================================================================
    # Cursor → Spinbox Synchronization
    # =========================================================================
    
    def _on_cursor_moved(self, action: str, line_id: str, position: float):
        """
        Update spinbox when user drags a cursor. Triggers auto-save on drag completion.
        
        Args:
            action: "dragged" (during), "centered" (after center operation), or "released" (after drag).
            line_id: Which cursor moved (e.g., "range1_start").
            position: New position in ms.
        """
        if action == "dragged":
            # During drag - update spinbox silently (no auto-save)
            self._sync_cursor_to_spinbox(line_id, position)
        
        elif action == "centered":
            # After centering - update spinbox and trigger save
            self._sync_cursor_to_spinbox(line_id, position)
            logger.debug("Cursor centered - triggering settings save")
            self.settings_changed.emit()
        
        elif action == "released":
            # After drag completes - trigger save
            logger.debug(f"Cursor '{line_id}' drag completed - triggering settings save")
            self.settings_changed.emit()
    
    def _sync_cursor_to_spinbox(self, line_id: str, position: float):
        """Update a single spinbox from its cursor position, blocking signals to prevent loops."""
        if line_id is None or position is None:
            return
        
        # Find corresponding spinbox key
        spinbox_key = None
        for key, cursor_id in self._spinbox_to_cursor_map.items():
            if cursor_id == line_id:
                spinbox_key = key
                break
        
        if spinbox_key:
            # Block signals to prevent feedback loop
            self.control_panel.update_range_value_silent(spinbox_key, position)
            logger.debug(f"Synced cursor '{line_id}' → spinbox '{spinbox_key}' = {position:.2f}")
    
    def sync_cursors_to_spinboxes(self):
        """
        Update all spinbox values to match current cursor positions.
        
        Called by MainWindow after loading a new sweep to ensure spinboxes
        reflect where cursors actually snapped.
        """
        # Get actual cursor positions
        positions = self.plot_manager.get_line_positions()
        
        # Get spinboxes (only active ones)
        spinboxes = self.control_panel.get_range_spinboxes()
        
        # Update each spinbox to match its cursor position
        for spinbox_key, spinbox in spinboxes.items():
            line_id = self._spinbox_to_cursor_map.get(spinbox_key)
            if line_id and line_id in positions:
                # Block signals to prevent recursion
                spinbox.blockSignals(True)
                spinbox.setValue(positions[line_id])
                spinbox.blockSignals(False)
        
        logger.debug("Synced all cursors → spinboxes")
    
    # =========================================================================
    # Dual Range Coordination
    # =========================================================================
    
    def _on_dual_range_toggled(self, enabled: bool):
        """
        Show/hide Range 2 cursors when dual range checkbox is toggled.
        
        Args:
            enabled: True to show Range 2 cursors, False to hide them.
        """
        if enabled:
            # Get Range 2 values from control panel
            vals = self.control_panel.get_range_values()
            start2 = vals.get("range2_start", 600)
            end2 = vals.get("range2_end", 900)
            
            # Show Range 2 cursors
            self.plot_manager.toggle_dual_range(True, start2, end2)
            logger.debug(f"Enabled dual range: Range 2 [{start2}, {end2}]")
        else:
            # Hide Range 2 cursors
            self.plot_manager.toggle_dual_range(False, 0, 0)
            logger.debug("Disabled dual range")
        
        # Reconnect editingFinished signals (spinboxes may have changed)
        self._connect_spinbox_editing_signals()