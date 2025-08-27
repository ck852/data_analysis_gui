"""
Unit tests for ChannelDefinitions class using pytest.

Place this file in tests/test_channel_definitions.py
Run with: pytest tests/test_channel_definitions.py -v
"""

import pytest
import numpy as np
from data_analysis_gui.core.channel_definitions import ChannelDefinitions


class TestChannelDefinitions:
    """Test suite for ChannelDefinitions class."""
    
    def test_default_initialization(self):
        """Test that default initialization sets correct values."""
        channels = ChannelDefinitions()
        
        assert channels.get_voltage_channel() == 0
        assert channels.get_current_channel() == 1
        assert not channels.is_swapped()
        assert channels.validate()
    
    def test_custom_initialization(self):
        """Test initialization with custom channel assignments."""
        channels = ChannelDefinitions(voltage_channel=2, current_channel=3)
        
        assert channels.get_voltage_channel() == 2
        assert channels.get_current_channel() == 3
        assert channels.is_swapped()
    
    def test_initialization_with_duplicate_channels(self):
        """Test that initializing with duplicate channels raises error."""
        with pytest.raises(ValueError, match="cannot be assigned to the same channel"):
            ChannelDefinitions(voltage_channel=1, current_channel=1)
    
    def test_initialization_with_negative_channel(self):
        """Test that negative channel IDs raise error."""
        with pytest.raises(ValueError, match="must be non-negative"):
            ChannelDefinitions(voltage_channel=-1, current_channel=1)
    
    def test_swap_channels(self):
        """Test channel swapping functionality."""
        channels = ChannelDefinitions()
        
        # Initial state
        assert channels.get_voltage_channel() == 0
        assert channels.get_current_channel() == 1
        
        # After swap
        channels.swap_channels()
        assert channels.get_voltage_channel() == 1
        assert channels.get_current_channel() == 0
        assert channels.is_swapped()
        
        # Swap back
        channels.swap_channels()
        assert channels.get_voltage_channel() == 0
        assert channels.get_current_channel() == 1
        assert not channels.is_swapped()
    
    def test_set_voltage_channel(self):
        """Test manually setting voltage channel."""
        channels = ChannelDefinitions()
        
        channels.set_voltage_channel(5)
        assert channels.get_voltage_channel() == 5
        assert channels.is_swapped()
        
        # Test setting to current channel raises error
        with pytest.raises(ValueError, match="already assigned to current"):
            channels.set_voltage_channel(1)
        
        # Test negative channel raises error
        with pytest.raises(ValueError, match="must be non-negative"):
            channels.set_voltage_channel(-1)
    
    def test_set_current_channel(self):
        """Test manually setting current channel."""
        channels = ChannelDefinitions()
        
        channels.set_current_channel(7)
        assert channels.get_current_channel() == 7
        assert channels.is_swapped()
        
        # Test setting to voltage channel raises error
        with pytest.raises(ValueError, match="already assigned to voltage"):
            channels.set_current_channel(0)
        
        # Test negative channel raises error
        with pytest.raises(ValueError, match="must be non-negative"):
            channels.set_current_channel(-2)
    
    def test_get_channel_label(self):
        """Test getting channel labels."""
        channels = ChannelDefinitions()
        
        # Default configuration
        assert channels.get_channel_label(0) == "Voltage (mV)"
        assert channels.get_channel_label(1) == "Current (pA)"
        assert channels.get_channel_label(2) == "Channel 2"
        
        # Without units
        assert channels.get_channel_label(0, include_units=False) == "Voltage"
        assert channels.get_channel_label(1, include_units=False) == "Current"
        
        # After swap
        channels.swap_channels()
        assert channels.get_channel_label(0) == "Current (pA)"
        assert channels.get_channel_label(1) == "Voltage (mV)"
    
    def test_reset_to_defaults(self):
        """Test resetting to default configuration."""
        channels = ChannelDefinitions()
        
        # Modify configuration
        channels.set_voltage_channel(3)
        channels.set_current_channel(4)
        assert channels.is_swapped()
        
        # Reset
        channels.reset_to_defaults()
        assert channels.get_voltage_channel() == 0
        assert channels.get_current_channel() == 1
        assert not channels.is_swapped()
    
    def test_get_set_configuration(self):
        """Test getting and setting configuration as dictionary."""
        channels = ChannelDefinitions()
        
        # Get default configuration
        config = channels.get_configuration()
        assert config == {'voltage': 0, 'current': 1}
        
        # Set new configuration
        new_config = {'voltage': 5, 'current': 8}
        channels.set_configuration(new_config)
        assert channels.get_voltage_channel() == 5
        assert channels.get_current_channel() == 8
        assert channels.get_configuration() == new_config
        
        # Test invalid configuration (duplicate channels)
        with pytest.raises(ValueError, match="Invalid configuration"):
            channels.set_configuration({'voltage': 2, 'current': 2})
        
        # Verify configuration didn't change after error
        assert channels.get_configuration() == new_config
        
        # Test missing keys
        with pytest.raises(KeyError):
            channels.set_configuration({'voltage': 1})
    
    def test_get_channel_for_type(self):
        """Test getting channel by data type."""
        channels = ChannelDefinitions()
        
        assert channels.get_channel_for_type('voltage') == 0
        assert channels.get_channel_for_type('current') == 1
        
        # Test case insensitive
        assert channels.get_channel_for_type('VOLTAGE') == 0
        assert channels.get_channel_for_type('Current') == 1
        
        # Test invalid type
        with pytest.raises(ValueError, match="Invalid data type"):
            channels.get_channel_for_type('temperature')
    
    def test_get_type_for_channel(self):
        """Test getting data type by channel."""
        channels = ChannelDefinitions()
        
        assert channels.get_type_for_channel(0) == 'voltage'
        assert channels.get_type_for_channel(1) == 'current'
        assert channels.get_type_for_channel(2) is None
        assert channels.get_type_for_channel(99) is None
        
        # After swap
        channels.swap_channels()
        assert channels.get_type_for_channel(0) == 'current'
        assert channels.get_type_for_channel(1) == 'voltage'
    
    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        channels = ChannelDefinitions()
        
        # Test repr
        repr_str = repr(channels)
        assert "ChannelDefinitions" in repr_str
        assert "voltage_channel=0" in repr_str
        assert "current_channel=1" in repr_str
        
        # Test str
        str_output = str(channels)
        assert "Channel Configuration" in str_output
        assert "default" in str_output
        assert "Voltage=Ch0" in str_output
        assert "Current=Ch1" in str_output
        
        # After swap
        channels.swap_channels()
        str_output = str(channels)
        assert "swapped" in str_output
        assert "Voltage=Ch1" in str_output
        assert "Current=Ch0" in str_output
    
    def test_validate_method(self):
        """Test the validate method."""
        # Valid configuration
        channels = ChannelDefinitions()
        assert channels.validate() is True
        
        # Force invalid state and test validate
        channels._voltage_channel = 1
        channels._current_channel = 1
        with pytest.raises(ValueError, match="cannot be assigned to the same channel"):
            channels.validate()
        
        # Force negative channel
        channels._voltage_channel = -1
        channels._current_channel = 1
        with pytest.raises(ValueError, match="must be non-negative"):
            channels.validate()


class TestChannelDefinitionsIntegration:
    """Integration tests showing practical usage."""
    
    def test_with_numpy_data(self):
        """Test using channel definitions with numpy arrays."""
        # Simulate 2-channel data
        data = np.array([[1.0, 2.0],
                         [1.1, 2.1],
                         [1.2, 2.2]])
        
        channels = ChannelDefinitions()
        
        # Extract by channel
        voltage_data = data[:, channels.get_voltage_channel()]
        current_data = data[:, channels.get_current_channel()]
        
        assert voltage_data[0] == 1.0
        assert current_data[0] == 2.0
        
        # After swap
        channels.swap_channels()
        voltage_data = data[:, channels.get_voltage_channel()]
        current_data = data[:, channels.get_current_channel()]
        
        assert voltage_data[0] == 2.0
        assert current_data[0] == 1.0
    
    def test_multiple_instances(self):
        """Test that multiple instances maintain separate state."""
        channels1 = ChannelDefinitions()
        channels2 = ChannelDefinitions()
        
        # Modify one instance
        channels1.swap_channels()
        
        # Verify they're independent
        assert channels1.get_voltage_channel() == 1
        assert channels2.get_voltage_channel() == 0
        assert channels1.is_swapped()
        assert not channels2.is_swapped()