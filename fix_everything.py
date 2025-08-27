"""
Complete setup script to fix all import issues.
Run this from your root directory.
"""

import os
from pathlib import Path

def setup_everything():
    """Set up the complete core module structure."""
    
    print("=" * 60)
    print("COMPLETE SETUP SCRIPT")
    print("=" * 60)
    
    # Define all necessary paths
    root = Path.cwd()
    src = root / "src"
    dag = src / "data_analysis_gui"
    core = dag / "core"
    
    print(f"\nWorking from: {root}")
    
    # Step 1: Create all directories
    print("\n1. Creating directory structure...")
    core.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Created: {core}")
    
    # Step 2: Fix main __init__.py
    print("\n2. Updating main __init__.py...")
    main_init = dag / "__init__.py"
    main_init_content = """__version__ = '0.1.0'

# Import core module to make it accessible
from . import core
"""
    
    with open(main_init, 'w') as f:
        f.write(main_init_content)
    print(f"   ✓ Updated: {main_init}")
    
    # Step 3: Create core/__init__.py
    print("\n3. Creating core/__init__.py...")
    core_init = core / "__init__.py"
    core_init_content = """\"\"\"
Core business logic module for the data analysis GUI.
\"\"\"

from .channel_definitions import ChannelDefinitions

__all__ = ['ChannelDefinitions']
"""
    
    with open(core_init, 'w') as f:
        f.write(core_init_content)
    print(f"   ✓ Created: {core_init}")
    
    # Step 4: Create channel_definitions.py (if it doesn't exist)
    print("\n4. Checking channel_definitions.py...")
    channel_def = core / "channel_definitions.py"
    
    if not channel_def.exists():
        print("   Creating channel_definitions.py...")
        # Create a minimal version for testing
        channel_def_content = '''"""
Channel definitions for electrophysiology data analysis.
"""

from typing import Optional


class ChannelDefinitions:
    """Manages channel assignments for electrophysiology data."""
    
    def __init__(self, voltage_channel: int = 0, current_channel: int = 1):
        """Initialize with default or custom channel assignments."""
        self._voltage_channel = voltage_channel
        self._current_channel = current_channel
        self._default_voltage_channel = 0
        self._default_current_channel = 1
    
    def get_voltage_channel(self) -> int:
        """Get the channel ID for voltage data."""
        return self._voltage_channel
    
    def get_current_channel(self) -> int:
        """Get the channel ID for current data."""
        return self._current_channel
    
    def swap_channels(self) -> None:
        """Swap voltage and current channel assignments."""
        self._voltage_channel, self._current_channel = self._current_channel, self._voltage_channel
    
    def set_voltage_channel(self, channel_id: int) -> None:
        """Set the voltage channel."""
        if channel_id == self._current_channel:
            raise ValueError(f"Channel {channel_id} is already assigned to current")
        self._voltage_channel = channel_id
    
    def set_current_channel(self, channel_id: int) -> None:
        """Set the current channel."""
        if channel_id == self._voltage_channel:
            raise ValueError(f"Channel {channel_id} is already assigned to voltage")
        self._current_channel = channel_id
    
    def get_channel_label(self, channel_id: int, include_units: bool = True) -> str:
        """Get label for a channel."""
        if channel_id == self._voltage_channel:
            return "Voltage (mV)" if include_units else "Voltage"
        elif channel_id == self._current_channel:
            return "Current (pA)" if include_units else "Current"
        else:
            return f"Channel {channel_id}"
    
    def is_swapped(self) -> bool:
        """Check if channels are swapped from defaults."""
        return (self._voltage_channel != self._default_voltage_channel or 
                self._current_channel != self._default_current_channel)
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if self._voltage_channel == self._current_channel:
            raise ValueError("Voltage and current cannot use the same channel")
        return True
    
    def reset_to_defaults(self) -> None:
        """Reset to default configuration."""
        self._voltage_channel = self._default_voltage_channel
        self._current_channel = self._default_current_channel
    
    def get_configuration(self) -> dict:
        """Get configuration as dictionary."""
        return {
            'voltage': self._voltage_channel,
            'current': self._current_channel
        }
    
    def set_configuration(self, config: dict) -> None:
        """Set configuration from dictionary."""
        self._voltage_channel = config['voltage']
        self._current_channel = config['current']
        self.validate()
    
    def get_channel_for_type(self, data_type: str) -> int:
        """Get channel for a data type."""
        if data_type.lower() == 'voltage':
            return self._voltage_channel
        elif data_type.lower() == 'current':
            return self._current_channel
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def get_type_for_channel(self, channel_id: int) -> Optional[str]:
        """Get data type for a channel."""
        if channel_id == self._voltage_channel:
            return 'voltage'
        elif channel_id == self._current_channel:
            return 'current'
        return None
    
    def __repr__(self) -> str:
        return f"ChannelDefinitions(voltage={self._voltage_channel}, current={self._current_channel})"
    
    def __str__(self) -> str:
        status = "swapped" if self.is_swapped() else "default"
        return f"Channels ({status}): V=Ch{self._voltage_channel}, I=Ch{self._current_channel}"
'''
        
        with open(channel_def, 'w') as f:
            f.write(channel_def_content)
        print(f"   ✓ Created: {channel_def}")
    else:
        print(f"   ✓ Already exists: {channel_def}")
    
    # Step 5: Test the import
    print("\n5. Testing import...")
    import sys
    sys.path.insert(0, str(src))
    
    try:
        # Remove any cached imports
        if 'data_analysis_gui' in sys.modules:
            del sys.modules['data_analysis_gui']
        if 'data_analysis_gui.core' in sys.modules:
            del sys.modules['data_analysis_gui.core']
        if 'data_analysis_gui.core.channel_definitions' in sys.modules:
            del sys.modules['data_analysis_gui.core.channel_definitions']
        
        # Try fresh import
        from data_analysis_gui.core.channel_definitions import ChannelDefinitions
        
        # Test it
        ch = ChannelDefinitions()
        print(f"   ✓ Import successful!")
        print(f"   ✓ Created instance: {ch}")
        
        print("\n" + "=" * 60)
        print("✅ SETUP COMPLETE - Everything should work now!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python tests/test_channel_definitions_demo.py")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        print("\n" + "=" * 60)
        print("❌ SETUP FAILED - See error above")
        print("=" * 60)
        return False


if __name__ == "__main__":
    setup_everything()