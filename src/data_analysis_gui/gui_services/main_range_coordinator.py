"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Main Range Coordinator

Coordinates bidirectional synchronization between ControlPanel range spinboxes
and PlotManager cursor positions. Extracted from MainWindow to provide focused
responsibility for range value coordination without Qt widget dependencies.

This service follows the established pattern of extracting coordination logic
into focused components (similar to ViewStateManager, CursorManager pattern).
"""

import logging
from typing import Optional, Dict

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class MainRangeCoordinator(QObject):
    """
    Coordinates range values between ControlPanel spinboxes and PlotManager cursors.
    
    Responsibilities:
    - Bidirectional synchronization (spinboxes ↔ cursors)
    - Dual range cursor visibility coordination
    - Signal routing between ControlPanel and PlotManager
    - Change detection and propagation
    
    This class acts as a mediator between ControlPanel (which owns the spinboxes)
    and PlotManager (which owns the cursor lines). Neither component needs to
    know about the other - they only know about this coordinator.
    
    Example Usage:
        >>> coordinator = MainRangeCoordinator(control_panel, plot_manager)
        >>> # Coordinator automatically handles all synchronization
        >>> # No manual sync calls needed from MainWindow
    
    Signals:
        All signals are pass-through for MainWindow convenience:
        - analysis_requested: Forwarded from ControlPanel
        - export_requested: Forwarded from ControlPanel
    """
    
    # Pass-through signals from ControlPanel for MainWindow convenience
    analysis_requested = Signal()
    export_requested = Signal()
    
    def __init__(self, control_panel, plot_manager):
        """
        Initialize the range coordinator.
        
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
        """Connect all coordination signals."""
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
        Connect editingFinished signals from all range spinboxes.
        
        This enables snap-back behavior: when user finishes editing a spinbox,
        it updates to show the actual cursor position (which has snapped to data).
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
        Synchronize cursor positions from spinbox values.
        
        Called when user types in spinboxes. Updates cursors in real-time
        as the user types. The cursors will snap to nearest data points
        (handled by PlotManager's CursorManager).
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
        Handle editingFinished signal from range spinboxes.
        
        When user finishes editing (loses focus or presses Enter), update the
        spinbox to show the actual cursor position (which has already snapped).
        This provides visual feedback that the cursor snapped to a data point.
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
        Handle cursor movement events from PlotManager.
        
        When user drags a cursor, update the corresponding spinbox to show
        the new position in real-time.
        
        Args:
            action: Type of cursor action (e.g., "dragged", "centered").
            line_id: Identifier for the cursor line.
            position: New position value for the cursor.
        """
        if action == "dragged" or action == "centered":
            self._sync_cursor_to_spinbox(line_id, position)
    
    def _sync_cursor_to_spinbox(self, line_id: str, position: float):
        """
        Synchronize a single cursor position to its corresponding spinbox.
        
        Args:
            line_id: Cursor line identifier (e.g., "range1_start").
            position: New position value.
        """
        if line_id is None or position is None:
            return
        
        # Find corresponding spinbox key
        spinbox_key = None
        for key, cursor_id in self._spinbox_to_cursor_map.items():
            if cursor_id == line_id:
                spinbox_key = key
                break
        
        if spinbox_key:
            self.control_panel.update_range_value(spinbox_key, position)
            logger.debug(f"Synced cursor '{line_id}' → spinbox '{spinbox_key}' = {position:.2f}")
    
    def sync_cursors_to_spinboxes(self):
        """
        Update all spinbox values to match current cursor positions.
        
        Called after cursors have snapped to ensure spinboxes display actual positions.
        This is a public method that can be called by MainWindow when needed
        (e.g., after loading a new sweep).
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
        Handle dual range checkbox toggle.
        
        Coordinates cursor visibility in PlotManager with the checkbox state.
        When enabled, shows Range 2 cursors. When disabled, hides them.
        
        Args:
            enabled: True if dual range is enabled, False otherwise.
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