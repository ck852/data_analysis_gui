"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

"""
Channel definitions for electrophysiology data analysis.

This module provides a centralized way to manage channel assignments for voltage
and current data in electrophysiology recordings. It eliminates hardcoded channel
assumptions and allows for flexible channel configuration.
"""

from typing import Tuple, Optional, Dict


class ChannelDefinitions:
    """
    Centralized manager for channel assignments in electrophysiology data.

    Provides flexible configuration for voltage and current channel assignments, supporting swapping,
    manual configuration, and label/unit retrieval. Eliminates hardcoded channel assumptions and enables
    robust, production-ready channel management for analysis and UI integration.

    Attributes:
        _voltage_channel (int): Channel ID assigned to voltage data.
        _current_channel (int): Channel ID assigned to current data.
        _default_voltage_channel (int): Default channel for voltage (0).
        _default_current_channel (int): Default channel for current (1).

    Example:
        >>> channels = ChannelDefinitions()
        >>> channels.get_voltage_channel()
        0
        >>> channels.get_current_channel()
        1
        >>> channels.swap_channels()
        >>> channels.get_voltage_channel()
        1
        >>> channels.get_channel_label(0)
        'Current (pA)'
    """

    def __init__(self, voltage_channel: int = 0, current_channel: int = 1):
        """
        Initialize channel definitions with specified or default channel assignments.

        Args:
            voltage_channel (int, optional): Channel ID for voltage data (default: 0).
            current_channel (int, optional): Channel ID for current data (default: 1).

        Raises:
            ValueError: If the same channel is assigned to both voltage and current.
            ValueError: If channel IDs are negative.
        """
        # Store default values
        self._default_voltage_channel: int = 0
        self._default_current_channel: int = 1

        # Initialize channel assignments
        self._voltage_channel: int = voltage_channel
        self._current_channel: int = current_channel

        # Validate initial configuration
        self.validate()

        # Auto-detection support for WCP files
        self._auto_detected = False
        self._detected_voltage_units: Optional[str] = None
        self._detected_current_units: Optional[str] = None

    def get_available_types(self):
        """
        Get the list of selectable channel types for UI components.

        Returns:
            list[str]: List of available channel types ("Voltage", "Current").
        """
        return ["Voltage", "Current"]

    # Shims so existing UI code works without refactors
    def get_voltage_channel_id(self) -> int:
        """
        Shim for UI compatibility. Get the channel ID assigned to voltage data.

        Returns:
            int: Channel ID for voltage data.
        """
        return self.get_voltage_channel()

    def get_current_channel_id(self) -> int:
        """
        Shim for UI compatibility. Get the channel ID assigned to current data.

        Returns:
            int: Channel ID for current data.
        """
        return self.get_current_channel()

    def set_assignments(self, voltage_channel: int, current_channel: int):
        """
        Atomically set both voltage and current channel assignments with validation.

        Args:
            voltage_channel (int): Channel ID for voltage data.
            current_channel (int): Channel ID for current data.

        Raises:
            ValueError: If assignments are invalid.
        """
        old_v, old_c = self._voltage_channel, self._current_channel
        try:
            self._voltage_channel = voltage_channel
            self._current_channel = current_channel
            self.validate()
        except Exception:
            self._voltage_channel, self._current_channel = old_v, old_c
            raise

    def get_voltage_channel(self) -> int:
        """
        Get the channel ID currently assigned to voltage data.

        Returns:
            int: Channel ID for voltage data.
        """
        return self._voltage_channel

    def get_current_channel(self) -> int:
        """
        Get the channel ID currently assigned to current data.

        Returns:
            int: Channel ID for current data.
        """
        return self._current_channel

    def swap_channels(self) -> None:
        """
        Swap the voltage and current channel assignments.

        After calling, the channel previously assigned to voltage is now assigned to current, and vice versa.

        Example:
            >>> channels = ChannelDefinitions()
            >>> channels.get_voltage_channel()
            0
            >>> channels.swap_channels()
            >>> channels.get_voltage_channel()
            1
        """
        self._voltage_channel, self._current_channel = (
            self._current_channel,
            self._voltage_channel,
        )

    def set_voltage_channel(self, channel_id: int) -> None:
        """
        Manually set the channel ID for voltage data.

        Args:
            channel_id (int): Channel ID to assign to voltage data.

        Raises:
            ValueError: If channel_id is negative or already assigned to current.
        """
        if channel_id < 0:
            raise ValueError(f"Channel ID must be non-negative, got {channel_id}")

        if channel_id == self._current_channel:
            raise ValueError(
                f"Channel {channel_id} is already assigned to current data. "
                "Cannot assign the same channel to both voltage and current."
            )

        self._voltage_channel = channel_id

    def set_current_channel(self, channel_id: int) -> None:
        """
        Manually set the channel ID for current data.

        Args:
            channel_id (int): Channel ID to assign to current data.

        Raises:
            ValueError: If channel_id is negative or already assigned to voltage.
        """
        if channel_id < 0:
            raise ValueError(f"Channel ID must be non-negative, got {channel_id}")

        if channel_id == self._voltage_channel:
            raise ValueError(
                f"Channel {channel_id} is already assigned to voltage data. "
                "Cannot assign the same channel to both voltage and current."
            )

        self._current_channel = channel_id

    def get_channel_label(self, channel_id: int, include_units: bool = True) -> str:
        """
        Get the label for a specific channel based on its assignment.
        Uses auto-detected units if available, otherwise defaults.
        """
        if channel_id == self._voltage_channel:
            label = "Voltage"
            unit = self._detected_voltage_units or "mV"  # MODIFIED: Use detected or default
        elif channel_id == self._current_channel:
            label = "Current"
            unit = self._detected_current_units or "pA"  # MODIFIED: Use detected or default
        else:
            return f"Channel {channel_id}"

        if include_units:
            return f"{label} ({unit})"
        return label

    def is_swapped(self) -> bool:
        """
        Check if channels are in a non-default (swapped) configuration.

        Returns:
            bool: True if channels are swapped from default, False otherwise.

        Example:
            >>> channels = ChannelDefinitions()
            >>> channels.is_swapped()
            False
            >>> channels.swap_channels()
            >>> channels.is_swapped()
            True
        """
        return (
            self._voltage_channel != self._default_voltage_channel
            or self._current_channel != self._default_current_channel
        )

    def validate(self) -> bool:
        """
        Validate the current channel configuration.

        Ensures that voltage and current are assigned to different, non-negative channels.

        Returns:
            bool: True if configuration is valid.

        Raises:
            ValueError: If the same channel is assigned to both voltage and current, or if any channel ID is negative.
        """
        # Check for negative channel IDs
        if self._voltage_channel < 0:
            raise ValueError(
                f"Voltage channel ID must be non-negative, got {self._voltage_channel}"
            )

        if self._current_channel < 0:
            raise ValueError(
                f"Current channel ID must be non-negative, got {self._current_channel}"
            )

        # Check for duplicate assignments
        if self._voltage_channel == self._current_channel:
            raise ValueError(
                f"Voltage and current cannot be assigned to the same channel "
                f"(both assigned to channel {self._voltage_channel})"
            )

        return True

    def reset_to_defaults(self) -> None:
        """
        Reset channel assignments to their default values (Voltage: 0, Current: 1).
        """
        self._voltage_channel = self._default_voltage_channel
        self._current_channel = self._default_current_channel

    def get_configuration(self) -> Dict[str, int]:
        """
        Get the current channel configuration as a dictionary.

        Returns:
            Dict[str, int]: Dictionary with 'voltage' and 'current' keys mapping to their respective channel IDs.

        Example:
            >>> channels = ChannelDefinitions()
            >>> channels.get_configuration()
            {'voltage': 0, 'current': 1}
        """
        return {"voltage": self._voltage_channel, "current": self._current_channel}

    def set_configuration(self, config: Dict[str, int]) -> None:
        """
        Set channel configuration from a dictionary.

        Args:
            config (Dict[str, int]): Dictionary with 'voltage' and 'current' keys mapping to channel IDs.

        Raises:
            KeyError: If required keys are missing from config.
            ValueError: If configuration is invalid.

        Example:
            >>> channels = ChannelDefinitions()
            >>> channels.set_configuration({'voltage': 1, 'current': 0})
            >>> channels.get_voltage_channel()
            1
        """
        if "voltage" not in config or "current" not in config:
            raise KeyError(
                "Configuration must include both 'voltage' and 'current' keys"
            )

        # Store old values in case we need to rollback
        old_voltage = self._voltage_channel
        old_current = self._current_channel

        try:
            self._voltage_channel = config["voltage"]
            self._current_channel = config["current"]
            self.validate()
        except (ValueError, TypeError) as e:
            # Rollback on error
            self._voltage_channel = old_voltage
            self._current_channel = old_current
            raise ValueError(f"Invalid configuration: {e}")

    def get_channel_for_type(self, data_type: str) -> int:
        """
        Get the channel ID for a specific data type ('voltage' or 'current').

        Args:
            data_type (str): Either 'voltage' or 'current' (case-insensitive).

        Returns:
            int: Channel ID for the specified data type.

        Raises:
            ValueError: If data_type is not 'voltage' or 'current'.

        Example:
            >>> channels = ChannelDefinitions()
            >>> channels.get_channel_for_type('voltage')
            0
            >>> channels.get_channel_for_type('CURRENT')
            1
        """
        data_type_lower = data_type.lower()

        if data_type_lower == "voltage":
            return self._voltage_channel
        elif data_type_lower == "current":
            return self._current_channel
        else:
            raise ValueError(
                f"Invalid data type '{data_type}'. "
                "Must be 'voltage' or 'current' (case-insensitive)."
            )

    def get_type_for_channel(self, channel_id: int) -> Optional[str]:
        """
        Get the data type ('voltage' or 'current') for a specific channel ID.

        Args:
            channel_id (int): Channel ID to query.

        Returns:
            Optional[str]: 'voltage', 'current', or None if channel is unassigned.

        Example:
            >>> channels = ChannelDefinitions()
            >>> channels.get_type_for_channel(0)
            'voltage'
            >>> channels.get_type_for_channel(1)
            'current'
            >>> channels.get_type_for_channel(2)
            None
        """
        if channel_id == self._voltage_channel:
            return "voltage"
        elif channel_id == self._current_channel:
            return "current"
        else:
            return None

# =========================================================================
# WCP Auto-Detection Support
# =========================================================================

    def set_from_wcp_detection(
        self,
        voltage_channel: int,
        current_channel: int,
        voltage_units: str,
        current_units: str,
    ) -> None:
        """
        Configure channels from WCP file auto-detection.

        This method is called when WCP metadata has been analyzed and
        channel assignments have been automatically determined.

        Args:
            voltage_channel: Detected voltage channel index
            current_channel: Detected current channel index
            voltage_units: Detected voltage units (e.g., "mV")
            current_units: Detected current units (e.g., "pA", "µA")
        """
        self._voltage_channel = voltage_channel
        self._current_channel = current_channel
        self._detected_voltage_units = voltage_units
        self._detected_current_units = current_units
        self._auto_detected = True

    def is_auto_detected(self) -> bool:
        """
        Check if channel configuration was auto-detected from file metadata.

        Returns:
            True if auto-detected from WCP file, False if manual configuration
        """
        return self._auto_detected

    def get_detected_units(self) -> Dict[str, Optional[str]]:
        """
        Get the auto-detected units for voltage and current.

        Returns:
            Dictionary with 'voltage' and 'current' keys, or None if not detected
        """
        return {
            "voltage": self._detected_voltage_units,
            "current": self._detected_current_units,
        }

    def reset_to_manual(self) -> None:
        """
        Reset to manual configuration mode.

        Called when loading ABF/MAT files to use manual GUI settings.
        """
        self._auto_detected = False
        self._detected_voltage_units = None
        self._detected_current_units = None

    def __repr__(self) -> str:
        """
        Return a string representation of the channel configuration.

        Returns:
            str: String representation showing current channel assignments.
        """
        return (
            f"ChannelDefinitions(voltage_channel={self._voltage_channel}, "
            f"current_channel={self._current_channel})"
        )

    def __str__(self) -> str:
        """
        Return a human-readable string of the channel configuration.

        Returns:
            str: Human-readable string showing channel assignments and swap status.
        """
        status = "swapped" if self.is_swapped() else "default"
        return (
            f"Channel Configuration ({status}): "
            f"Voltage=Ch{self._voltage_channel}, Current=Ch{self._current_channel}"
        )
