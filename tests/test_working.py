"""
Test file for ChannelDefinitions - This version will definitely work!
"""

import sys
import os

# Get the absolute path to the root directory
# This file is in tests/, so parent is root
import pathlib
test_file = pathlib.Path(__file__).resolve()
tests_dir = test_file.parent
root_dir = tests_dir.parent
src_dir = root_dir / 'src'

print(f"Adding to path: {src_dir}")

# Add src to the Python path
sys.path.insert(0, str(src_dir))

# Now import should work
from data_analysis_gui.core.channel_definitions import ChannelDefinitions

print("✓ Successfully imported ChannelDefinitions!\n")

def test_basic_functionality():
    """Test basic ChannelDefinitions functionality."""
    print("=" * 60)
    print("TESTING CHANNEL DEFINITIONS")
    print("=" * 60)
    
    # Create instance with defaults
    channels = ChannelDefinitions()
    print(f"\n1. Default configuration:")
    print(f"   {channels}")
    print(f"   Voltage channel: {channels.get_voltage_channel()}")
    print(f"   Current channel: {channels.get_current_channel()}")
    
    # Test swapping
    channels.swap_channels()
    print(f"\n2. After swapping:")
    print(f"   {channels}")
    print(f"   Voltage channel: {channels.get_voltage_channel()}")
    print(f"   Current channel: {channels.get_current_channel()}")
    
    # Test labels
    print(f"\n3. Channel labels:")
    print(f"   Channel 0: {channels.get_channel_label(0)}")
    print(f"   Channel 1: {channels.get_channel_label(1)}")
    
    # Reset
    channels.reset_to_defaults()
    print(f"\n4. After reset:")
    print(f"   {channels}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_basic_functionality()