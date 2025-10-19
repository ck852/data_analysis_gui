"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

View State Manager

Manages axis limit state for plot views, replacing manual xlim/ylim tracking.
Provides explicit state management for home view (initial autoscaled view) and
current view (last known limits for change detection).

This module is part of Phase 1 of the plot manager refactoring. It extracts
view state management into a focused component with clear responsibilities.
"""

from typing import Optional, Tuple


class ViewStateManager:
    """
    Manages axis limit state for plot views.
    
    Tracks two types of view state:
    - Home view: The initial autoscaled view, used for reset operations
    - Current view: The last known view limits, used for change detection
    
    This class is a pure Python state manager with no matplotlib or Qt dependencies.
    It replaces the manual _last_xlim and _last_ylim tracking previously done
    in PlotManager.
    
    Example Usage:
        >>> view_manager = ViewStateManager()
        >>> # After initial plot
        >>> view_manager.set_home_view(xlim=(0, 100), ylim=(-50, 50))
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
        self._home_xlim: Optional[Tuple[float, float]] = None
        self._home_ylim: Optional[Tuple[float, float]] = None
        self._current_xlim: Optional[Tuple[float, float]] = None
        self._current_ylim: Optional[Tuple[float, float]] = None
    
    def set_home_view(self, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
        """
        Store the home view limits (typically after initial autoscale).
        
        The home view represents the default/reset view of the plot and is set
        after matplotlib performs initial autoscaling. This view can be restored
        later via reset_to_home().
        
        Args:
            xlim: X-axis limits as (min, max) tuple.
            ylim: Y-axis limits as (min, max) tuple.
        """
        self._home_xlim = xlim
        self._home_ylim = ylim
    
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
    
    def get_home_view(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get the stored home view limits.
        
        Returns:
            Tuple of (xlim, ylim) if home view is set, None otherwise.
        """
        if self._home_xlim is None or self._home_ylim is None:
            return None
        return (self._home_xlim, self._home_ylim)
    
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
    
    def reset_to_home(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Reset current view to home view.
        
        This copies the home view limits into the current view and returns them.
        Useful when implementing a "reset zoom" operation - the caller can then
        apply these limits to the axes.
        
        Returns:
            The home view as (xlim, ylim) if set, None if home view not set.
        """
        if self._home_xlim is None or self._home_ylim is None:
            return None
        
        self._current_xlim = self._home_xlim
        self._current_ylim = self._home_ylim
        return (self._home_xlim, self._home_ylim)
    
    def reset(self) -> None:
        """
        Clear all stored view state.
        
        Called when loading a new file to ensure the first sweep
        establishes a fresh home view via autoscaling. After reset,
        get_current_view() will return None, triggering autoscale
        on the next plot update.
        """
        self._home_xlim = None
        self._home_ylim = None
        self._current_xlim = None
        self._current_ylim = None