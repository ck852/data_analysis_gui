"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

View State Manager

Manages axis limit state for plot views, specifically for preserving zoom/pan
state when changing between sweeps. Provides explicit state management for
current view (last known limits for change detection and preservation).
"""

from typing import Optional, Tuple


class ViewStateManager:
    """
    Manages axis limit state for plot views.
    
    Tracks current view state for zoom/pan preservation across sweep changes.
    The "Reset" button now performs fresh autoscaling rather than restoring
    a stored home view.
    
    This class is a pure Python state manager with no matplotlib or Qt dependencies.
    It replaces the manual _last_xlim and _last_ylim tracking previously done
    in PlotManager.
    
    Example Usage:
        >>> view_manager = ViewStateManager()
        >>> # After initial plot
        >>> view_manager.update_current_view(xlim=(0, 100), ylim=(-50, 50))
        >>> 
        >>> # Later, check if view changed
        >>> new_xlim = ax.get_xlim()
        >>> new_ylim = ax.get_ylim()
        >>> if view_manager.has_view_changed(new_xlim, new_ylim):
        ...     # Handle view change (e.g., reposition text)
        ...     view_manager.update_current_view(new_xlim, new_ylim)
    
    Future Feature Hooks:
        - Per-sweep view storage: Add dict mapping sweep_id to view tuples
        - Zoom calculations: Add methods that operate on current_view for
          calculating zoom in/out by factor
    """
    
    def __init__(self):
        """Initialize view state manager with no views set."""
        self._current_xlim: Optional[Tuple[float, float]] = None
        self._current_ylim: Optional[Tuple[float, float]] = None
    
    def update_current_view(self, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
        """
        Update the stored current view limits.
        
        This should be called after detecting a view change (via has_view_changed)
        to store the new limits as the "last known" state for future change detection.
        Typically called after handling zoom/pan events.
        
        Args:
            xlim: X-axis limits as (min, max) tuple.
            ylim: Y-axis limits as (min, max) tuple.
        """
        self._current_xlim = xlim
        self._current_ylim = ylim
    
    def get_current_view(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get the stored current view limits.
        
        Returns:
            Tuple of (xlim, ylim) if view is set, None otherwise.
        """
        if self._current_xlim is None or self._current_ylim is None:
            return None
        return (self._current_xlim, self._current_ylim)
    
    def has_view_changed(self, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> bool:
        """
        Check if the provided limits differ from the stored current view.
        
        This is used to detect zoom/pan operations by comparing axes limits
        to the last known stored limits. Returns True on first call (when no
        view is stored yet) or when limits differ.
        
        Usage pattern:
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()
            if view_manager.has_view_changed(current_xlim, current_ylim):
                # Handle the change
                view_manager.update_current_view(current_xlim, current_ylim)
        
        Args:
            xlim: X-axis limits to compare.
            ylim: Y-axis limits to compare.
        
        Returns:
            True if limits differ from stored current view or if no view is stored,
            False if limits match stored current view exactly.
        """
        if self._current_xlim is None or self._current_ylim is None:
            return True
        
        return self._current_xlim != xlim or self._current_ylim != ylim
    
    def reset(self) -> None:
        """
        Clear all stored view state.
        
        Called when loading a new file to ensure the first sweep
        establishes a fresh view via autoscaling. After reset,
        get_current_view() will return None, triggering autoscale
        on the next plot update.
        """
        self._current_xlim = None
        self._current_ylim = None