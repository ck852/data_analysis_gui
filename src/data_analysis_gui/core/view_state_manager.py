"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Manages plot view state (axis limits) independently for Voltage and Current channels.
Each channel maintains its own zoom/pan state, allowing users to switch between
channels without losing their view settings.
"""

from typing import Optional, Tuple
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ViewStateManager:
    """
    Preserves independent axis limits for Voltage and Current channels.
    
    When users zoom into a region of interest on one channel, switching to the
    other channel and back will restore the original zoom level rather than
    forcing an autoscale.
    """
    
    def __init__(self):
        """Initialize with separate storage for each channel type."""
        self._channel_views = {
            'Voltage': None,
            'Current': None
        }
        logger.debug("ViewStateManager initialized with per-channel storage")
    
    def update_current_view(
        self, 
        xlim: Tuple[float, float], 
        ylim: Tuple[float, float],
        channel_type: str
    ) -> None:
        """
        Store the current view state for a specific channel.
        
        Pass xlim/ylim as (min, max) tuples and specify which channel type
        ('Voltage' or 'Current') to store the view for.
        """
        if channel_type not in self._channel_views:
            logger.warning(f"Unknown channel type '{channel_type}', defaulting to 'Voltage'")
            channel_type = 'Voltage'
        
        self._channel_views[channel_type] = (xlim, ylim)
        logger.debug(f"Stored view for {channel_type}: X={xlim}, Y={ylim}")
    
    def get_current_view(self, channel_type: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Retrieve the stored view state for a specific channel.
        
        Returns (xlim, ylim) tuple if a view has been saved for this channel,
        or None if the channel should autoscale.
        """
        if channel_type not in self._channel_views:
            logger.warning(f"Unknown channel type '{channel_type}', defaulting to 'Voltage'")
            channel_type = 'Voltage'
        
        view = self._channel_views[channel_type]
        
        if view is not None:
            logger.debug(f"Retrieved view for {channel_type}: X={view[0]}, Y={view[1]}")
        else:
            logger.debug(f"No stored view for {channel_type} (will autoscale)")
        
        return view
    
    def has_view_changed(
        self, 
        current_xlim: Tuple[float, float], 
        current_ylim: Tuple[float, float],
        channel_type: str
    ) -> bool:
        """
        Check if the current view differs from what's stored for this channel.
        
        Used to detect zoom/pan operations that require updating cursor text
        positions or other view-dependent elements.
        """
        if channel_type not in self._channel_views:
            channel_type = 'Voltage'
        
        stored_view = self._channel_views[channel_type]
        
        if stored_view is None:
            return True
        
        stored_xlim, stored_ylim = stored_view
        changed = (current_xlim != stored_xlim) or (current_ylim != stored_ylim)
        
        if changed:
            logger.debug(f"View changed for {channel_type}")
        
        return changed
    
    def reset(self) -> None:
        """
        Clear view state for all channels.
        
        Called when loading a new file to ensure old zoom levels don't
        carry over to unrelated data.
        """
        self._channel_views = {
            'Voltage': None,
            'Current': None
        }
        logger.debug("Reset view state for all channels")