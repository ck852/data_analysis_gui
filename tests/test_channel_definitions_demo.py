"""
Tests for the ChannelDefinitions class.

Run this file directly to see examples of how the ChannelDefinitions class works.
For proper unit testing with pytest, see test_channel_definitions_pytest.py
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import the module
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from data_analysis_gui.core.channel_definitions import ChannelDefinitions


def demonstrate_basic_usage():
    """Demonstrate basic usage of ChannelDefinitions."""
    print("=" * 60)
    print("BASIC USAGE DEMONSTRATION")
    print("=" * 60)
    
    # Create with default configuration
    channels = ChannelDefinitions()
    print(f"\n1. Default configuration:")
    print(f"   {channels}")
    print(f"   Voltage channel: {channels.get_voltage_channel()}")
    print(f"   Current channel: {channels.get_current_channel()}")
    print(f"   Is swapped? {channels.is_swapped()}")
    
    # Get labels
    print(f"\n2. Channel labels:")
    print(f"   Channel 0: {channels.get_channel_label(0)}")
    print(f"   Channel 1: {channels.get_channel_label(1)}")
    print(f"   Channel 0 (no units): {channels.get_channel_label(0, include_units=False)}")
    
    # Swap channels
    channels.swap_channels()
    print(f"\n3. After swapping channels:")
    print(f"   {channels}")
    print(f"   Voltage channel: {channels.get_voltage_channel()}")
    print(f"   Current channel: {channels.get_current_channel()}")
    print(f"   Is swapped? {channels.is_swapped()}")
    print(f"   Channel 0: {channels.get_channel_label(0)}")
    print(f"   Channel 1: {channels.get_channel_label(1)}")
    
    # Reset to defaults
    channels.reset_to_defaults()
    print(f"\n4. After reset to defaults:")
    print(f"   {channels}")
    print(f"   Is swapped? {channels.is_swapped()}")


def demonstrate_manual_configuration():
    """Demonstrate manual channel configuration."""
    print("\n" + "=" * 60)
    print("MANUAL CONFIGURATION DEMONSTRATION")
    print("=" * 60)
    
    channels = ChannelDefinitions()
    
    # Manually set channels
    print(f"\n1. Setting voltage to channel 2:")
    channels.set_voltage_channel(2)
    print(f"   {channels}")
    
    print(f"\n2. Setting current to channel 3:")
    channels.set_current_channel(3)
    print(f"   {channels}")
    print(f"   Is swapped? {channels.is_swapped()}")
    
    # Get configuration as dictionary
    config = channels.get_configuration()
    print(f"\n3. Configuration as dictionary:")
    print(f"   {config}")
    
    # Set configuration from dictionary
    new_config = {'voltage': 5, 'current': 7}
    channels.set_configuration(new_config)
    print(f"\n4. After setting configuration from dictionary {new_config}:")
    print(f"   {channels}")
    
    # Get channel by type
    print(f"\n5. Getting channels by data type:")
    print(f"   Channel for 'voltage': {channels.get_channel_for_type('voltage')}")
    print(f"   Channel for 'CURRENT': {channels.get_channel_for_type('CURRENT')}")
    
    # Get type by channel
    print(f"\n6. Getting data type by channel:")
    print(f"   Type for channel 5: {channels.get_type_for_channel(5)}")
    print(f"   Type for channel 7: {channels.get_type_for_channel(7)}")
    print(f"   Type for channel 0: {channels.get_type_for_channel(0)}")


def demonstrate_error_handling():
    """Demonstrate error handling."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMONSTRATION")
    print("=" * 60)
    
    channels = ChannelDefinitions()
    
    # Try to set same channel for both
    print("\n1. Attempting to set voltage to channel already used by current:")
    try:
        channels.set_voltage_channel(1)  # Channel 1 is already current
    except ValueError as e:
        print(f"   Error caught: {e}")
    
    # Try to set negative channel
    print("\n2. Attempting to set negative channel ID:")
    try:
        channels.set_voltage_channel(-1)
    except ValueError as e:
        print(f"   Error caught: {e}")
    
    # Try invalid data type
    print("\n3. Attempting to get channel for invalid data type:")
    try:
        channels.get_channel_for_type('temperature')
    except ValueError as e:
        print(f"   Error caught: {e}")
    
    # Try invalid configuration
    print("\n4. Attempting to set invalid configuration:")
    try:
        invalid_config = {'voltage': 2, 'current': 2}  # Same channel for both
        channels.set_configuration(invalid_config)
    except ValueError as e:
        print(f"   Error caught: {e}")
        print(f"   Configuration remained unchanged: {channels.get_configuration()}")


def demonstrate_practical_usage():
    """Show how this would be used in practice with your data processing."""
    print("\n" + "=" * 60)
    print("PRACTICAL USAGE IN DATA PROCESSING")
    print("=" * 60)
    
    # Simulate loading data with 2 channels
    import numpy as np
    
    # Example: data array with shape (1000, 2) - 1000 time points, 2 channels
    mock_data = np.random.randn(1000, 2)
    
    channels = ChannelDefinitions()
    
    print("\n1. Processing with default configuration:")
    voltage_data = mock_data[:, channels.get_voltage_channel()]
    current_data = mock_data[:, channels.get_current_channel()]
    print(f"   Extracted voltage data from channel {channels.get_voltage_channel()}")
    print(f"   Extracted current data from channel {channels.get_current_channel()}")
    print(f"   Voltage data shape: {voltage_data.shape}")
    print(f"   Current data shape: {current_data.shape}")
    
    print("\n2. User swaps channels in GUI:")
    channels.swap_channels()
    voltage_data = mock_data[:, channels.get_voltage_channel()]
    current_data = mock_data[:, channels.get_current_channel()]
    print(f"   Now extracting voltage from channel {channels.get_voltage_channel()}")
    print(f"   Now extracting current from channel {channels.get_current_channel()}")
    
    print("\n3. Creating axis labels for plots:")
    for ch_id in range(2):
        label = channels.get_channel_label(ch_id)
        print(f"   Channel {ch_id} plot label: '{label}'")
    
    print("\n4. Checking if user has modified default configuration:")
    if channels.is_swapped():
        print("   ⚠️  Channels are swapped - showing warning in GUI")
    else:
        print("   ✓  Using default channel configuration")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_basic_usage()
    demonstrate_manual_configuration()
    demonstrate_error_handling()
    demonstrate_practical_usage()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)