"""
ABF (Axon Binary Format) Loader for PatchBatch

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

PHASE 1 ENHANCEMENT: Auto-detection of channel configuration from ABF metadata
"""

import struct
import logging
from pathlib import Path
from typing import Optional, Any, Union, Dict, List, Tuple
import numpy as np


logger = logging.getLogger(__name__)

from data_analysis_gui.core.dataset import ElectrophysiologyDataset

try:
    import pyabf
    PYABF_AVAILABLE = True
except ImportError:
    PYABF_AVAILABLE = False


# =============================================================================
# Binary Reading Utilities
# =============================================================================

def read_struct(f, struct_format, seek_to=-1):
    """Read structured data from file."""
    if seek_to >= 0:
        f.seek(seek_to)
    byte_count = struct.calcsize(struct_format)
    byte_string = f.read(byte_count)
    value = struct.unpack(struct_format, byte_string)
    return list(value)


def decode_string(byte_list):
    """Decode a list of bytestrings to regular strings."""
    return [s.decode('ascii', errors='ignore').strip('\x00').strip() for s in byte_list]


# =============================================================================
# Channel Auto-Detection
# =============================================================================

def _detect_channel_configuration(channel_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze channel info and determine voltage/current channel assignments.
    
    Args:
        channel_info: List of channel dicts with 'index', 'name', 'units', 'signal_type'
    
    Returns:
        Dict with keys:
            - voltage_channel: int (channel index for voltage)
            - current_channel: int (channel index for current)
            - voltage_units: str (detected units for voltage)
            - current_units: str (detected units for current)
            - valid: bool (True if detection was successful)
            - message: str (description of detection result)
    """
    voltage_channels = [ch for ch in channel_info if ch['signal_type'] == 'voltage']
    current_channels = [ch for ch in channel_info if ch['signal_type'] == 'current']
    
    # Case 1: Perfect detection - exactly 1 voltage and 1 current
    if len(voltage_channels) == 1 and len(current_channels) == 1:
        return {
            'voltage_channel': voltage_channels[0]['index'],
            'current_channel': current_channels[0]['index'],
            'voltage_units': voltage_channels[0]['units'],
            'current_units': current_channels[0]['units'],
            'valid': True,
            'message': f"Auto-detected: Ch.{voltage_channels[0]['index']} (voltage, {voltage_channels[0]['units']}), "
                      f"Ch.{current_channels[0]['index']} (current, {current_channels[0]['units']})"
        }
    
    # Case 2: Multiple voltage or current channels - use first of each
    if len(voltage_channels) >= 1 and len(current_channels) >= 1:
        logger.warning(
            f"Multiple channels detected: {len(voltage_channels)} voltage, {len(current_channels)} current. "
            f"Using first of each."
        )
        return {
            'voltage_channel': voltage_channels[0]['index'],
            'current_channel': current_channels[0]['index'],
            'voltage_units': voltage_channels[0]['units'],
            'current_units': current_channels[0]['units'],
            'valid': True,
            'message': f"Auto-detected (multiple channels): Ch.{voltage_channels[0]['index']} (voltage), "
                      f"Ch.{current_channels[0]['index']} (current)"
        }
    
    # Case 3: Missing voltage or current channel
    if len(voltage_channels) == 0:
        logger.error("No voltage channel detected in ABF file")
        return {
            'voltage_channel': 0,
            'current_channel': 1,
            'voltage_units': 'mV',
            'current_units': 'pA',
            'valid': False,
            'message': "Could not detect voltage channel - using default configuration"
        }
    
    if len(current_channels) == 0:
        logger.error("No current channel detected in ABF file")
        return {
            'voltage_channel': 0,
            'current_channel': 1,
            'voltage_units': 'mV',
            'current_units': 'pA',
            'valid': False,
            'message': "Could not detect current channel - using default configuration"
        }
    
    # Fallback - should not reach here
    logger.error("Unexpected channel configuration")
    return {
        'voltage_channel': 0,
        'current_channel': 1,
        'voltage_units': 'mV',
        'current_units': 'pA',
        'valid': False,
        'message': "Channel detection failed - using default configuration"
    }


# =============================================================================
# ABF1 Parser
# =============================================================================

class ABF1Parser:
    """Parser for ABF1 format files"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.file_handle = None
        self.metadata = {}
        
    def __enter__(self):
        self.file_handle = open(self.filepath, 'rb')
        self._parse_header()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
    
    def _parse_header(self):
        """Parse ABF1 file header"""
        f = self.file_handle
        
        # Verify signature
        file_signature = read_struct(f, "4s", 0)[0]
        if file_signature != b'ABF ':
            raise ValueError(f"Not an ABF1 file")
        
        # Basic info
        self.metadata['file_version'] = read_struct(f, "f", 4)[0]
        self.metadata['actual_episodes'] = read_struct(f, "i", 16)[0]
        
        # Sampling parameters
        self.metadata['num_adc_channels'] = read_struct(f, "h", 120)[0]
        self.metadata['adc_sample_interval'] = read_struct(f, "f", 122)[0]
        self.metadata['num_samples_per_episode'] = read_struct(f, "i", 138)[0]
        
        # Calculate sample rate
        num_channels = self.metadata['num_adc_channels']
        interval = self.metadata['adc_sample_interval']
        self.metadata['sample_rate'] = 1e6 / (interval * num_channels) if num_channels > 0 else 0
        
        # Synch array info (for sweep times)
        self.metadata['synch_array_ptr'] = read_struct(f, "i", 92)[0]
        self.metadata['synch_array_size'] = read_struct(f, "i", 96)[0]
        self.metadata['synch_time_unit'] = read_struct(f, "f", 130)[0]
        
        # Channel info
        self.metadata['adc_sampling_seq'] = read_struct(f, "16h", 410)
        channel_names_raw = read_struct(f, "10s" * 16, 442)
        channel_units_raw = read_struct(f, "8s" * 16, 602)
        
        self.metadata['channel_names'] = decode_string(channel_names_raw)
        self.metadata['channel_units'] = decode_string(channel_units_raw)
    
    def get_channel_info(self) -> List[Dict[str, Any]]:
        """Get organized channel information."""
        channels = []
        num_channels = self.metadata['num_adc_channels']
        adc_sampling_seq = self.metadata['adc_sampling_seq']
        channel_names = self.metadata['channel_names']
        channel_units = self.metadata['channel_units']
        
        for i in range(num_channels):
            channel_idx = adc_sampling_seq[i]
            name = channel_names[channel_idx]
            units = channel_units[channel_idx]
            
            # Identify signal type
            units_lower = units.lower()
            if 'mv' in units_lower or units_lower == 'v':
                signal_type = "voltage"
            elif any(u in units_lower for u in ['pa', 'na', 'µa', 'ua', 'ma', 'a']):
                signal_type = "current"
            else:
                signal_type = "unknown"
            
            channels.append({
                'index': i,
                'name': name,
                'units': units,
                'signal_type': signal_type
            })
        
        return channels
    
    def get_sweep_times(self) -> Dict[str, float]:
        """Extract sweep times."""
        sweep_times = {}
        num_episodes = self.metadata['actual_episodes']
        synch_array_ptr = self.metadata['synch_array_ptr']
        synch_array_size = self.metadata['synch_array_size']
        synch_time_unit = self.metadata['synch_time_unit']
        sample_rate = self.metadata['sample_rate']
        num_samples = self.metadata['num_samples_per_episode']
        
        # Check for variable-length sweeps
        has_synch_array = synch_array_ptr > 0 and synch_array_size > 0
        
        if has_synch_array:
            self.file_handle.seek(synch_array_ptr * 512)
            for sweep_num in range(num_episodes):
                synch_entry = read_struct(self.file_handle, "II")
                start_time_units = synch_entry[0]
                start_time_sec = start_time_units * synch_time_unit / 1e6
                sweep_times[str(sweep_num + 1)] = float(start_time_sec)
        else:
            # Fixed-length sweeps
            sweep_length_sec = num_samples / sample_rate if sample_rate > 0 else 0
            for sweep_num in range(num_episodes):
                start_time_sec = sweep_num * sweep_length_sec
                sweep_times[str(sweep_num + 1)] = float(start_time_sec)
        
        return sweep_times


# =============================================================================
# ABF2 Parser
# =============================================================================

class ABF2Parser:
    """Parser for ABF2 format files"""
    
    BLOCKSIZE = 512
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.file_handle = None
        self.metadata = {}
        
    def __enter__(self):
        self.file_handle = open(self.filepath, 'rb')
        self._parse_header()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
    
    def _parse_header(self):
        """Parse ABF2 file header"""
        f = self.file_handle
        
        # Verify signature
        file_signature = read_struct(f, "4s", 0)[0]
        if file_signature != b'ABF2':
            raise ValueError(f"Not an ABF2 file")
        
        # Basic info
        f.seek(0)
        file_signature = read_struct(f, "4s")
        file_version = read_struct(f, "4b")
        file_info_size = read_struct(f, "I")
        actual_episodes = read_struct(f, "I")[0]
        
        self.metadata['file_version'] = file_version[0]
        self.metadata['actual_episodes'] = actual_episodes
        
        # Section pointers
        protocol_section = read_struct(f, "IIl", 76)
        adc_section = read_struct(f, "IIl", 92)
        data_section = read_struct(f, "IIl", 236)
        synch_array_section = read_struct(f, "IIl", 316)
        
        self.metadata['adc_section'] = adc_section
        self.metadata['data_section'] = data_section
        self.metadata['synch_array_section'] = synch_array_section
        
        # Protocol section
        f.seek(protocol_section[0] * self.BLOCKSIZE)
        operation_mode = read_struct(f, "h")[0]
        adc_sequence_interval = read_struct(f, "f")[0]
        enable_file_compression = read_struct(f, "B")[0]
        f.seek(3, 1)
        file_compression_ratio = read_struct(f, "I")[0]
        synch_time_unit = read_struct(f, "f")[0]
        
        self.metadata['adc_sequence_interval'] = adc_sequence_interval
        self.metadata['synch_time_unit'] = synch_time_unit
        self.metadata['num_adc_channels'] = adc_section[2]
        self.metadata['sample_rate'] = 1e6 / adc_sequence_interval if adc_sequence_interval > 0 else 0
        
        # Get channel info from ADC section
        self._parse_adc_section()
    
    def _parse_adc_section(self):
        """Parse ADC section for channel info"""
        f = self.file_handle
        adc_section = self.metadata['adc_section']
        num_channels = adc_section[2]
        
        channel_names = []
        channel_units = []
        
        for i in range(num_channels):
            f.seek(adc_section[0] * self.BLOCKSIZE + i * 128 + 4)
            name_bytes = read_struct(f, "10s")[0]
            units_bytes = read_struct(f, "8s")[0]
            
            name = name_bytes.decode('ascii', errors='ignore').strip('\x00').strip()
            units = units_bytes.decode('ascii', errors='ignore').strip('\x00').strip()
            
            channel_names.append(name if name else f"Channel {i}")
            channel_units.append(units)
        
        self.metadata['channel_names'] = channel_names
        self.metadata['channel_units'] = channel_units
    
    def get_channel_info(self) -> List[Dict[str, Any]]:
        """Get organized channel information."""
        channels = []
        num_channels = self.metadata.get('num_adc_channels', 0)
        channel_names = self.metadata.get('channel_names', [])
        channel_units = self.metadata.get('channel_units', [])
        
        for i in range(num_channels):
            name = channel_names[i] if i < len(channel_names) else f"Channel {i}"
            units = channel_units[i] if i < len(channel_units) else ""
            
            # Identify signal type
            units_lower = units.lower()
            if 'mv' in units_lower or units_lower == 'v':
                signal_type = "voltage"
            elif any(u in units_lower for u in ['pa', 'na', 'µa', 'ua', 'ma', 'a']):
                signal_type = "current"
            else:
                signal_type = "unknown"
            
            channels.append({
                'index': i,
                'name': name,
                'units': units,
                'signal_type': signal_type
            })
        
        return channels
    
    def get_sweep_times(self) -> Dict[str, float]:
        """Extract sweep times."""
        sweep_times = {}
        num_episodes = self.metadata['actual_episodes']
        synch_array_section = self.metadata['synch_array_section']
        synch_time_unit = self.metadata['synch_time_unit']
        sample_rate = self.metadata['sample_rate']
        num_channels = self.metadata['num_adc_channels']
        data_section = self.metadata['data_section']
        
        # Check for variable-length sweeps
        has_synch_array = synch_array_section[2] > 0
        
        if has_synch_array:
            self.file_handle.seek(synch_array_section[0] * self.BLOCKSIZE)
            for sweep_num in range(num_episodes):
                synch_entry = read_struct(self.file_handle, "II")
                start_time_units = synch_entry[0]
                start_time_sec = start_time_units * synch_time_unit / 1e6
                sweep_times[str(sweep_num + 1)] = float(start_time_sec)
        else:
            # Fixed-length sweeps
            total_samples = data_section[2]
            if num_episodes > 0 and num_channels > 0:
                samples_per_sweep = total_samples // (num_episodes * num_channels)
                if sample_rate > 0:
                    sweep_length_sec = samples_per_sweep / sample_rate
                    for sweep_num in range(num_episodes):
                        start_time_sec = sweep_num * sweep_length_sec
                        sweep_times[str(sweep_num + 1)] = float(start_time_sec)
        
        return sweep_times


# =============================================================================
# Main Loading Function
# =============================================================================

def load_abf(
    file_path: Union[str, Path],
    validate_data: bool = True,
) -> "ElectrophysiologyDataset":
    """
    Load an ABF file into a standardized dataset with auto-detected channel configuration.

    Channel configuration is automatically detected from ABF metadata based on channel
    units and stored in the dataset metadata.

    Args:
        file_path: Path to the ABF file
        validate_data: If True, check for NaN/Inf values

    Returns:
        ElectrophysiologyDataset containing all sweeps from the ABF file with
        auto-detected channel configuration stored in metadata['channel_config']

    Raises:
        ImportError: If pyabf is not installed
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
        ValueError: If file is invalid or contains no data
    """
    if not PYABF_AVAILABLE:
        raise ImportError("pyabf required for ABF support. Install with: pip install pyabf")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ABF file not found: {file_path}")

    logger.info(f"Loading ABF file: {file_path.name}")

    # Extract metadata using binary parsing
    try:
        with open(file_path, 'rb') as f:
            file_signature = read_struct(f, "4s", 0)[0]
        
        if file_signature == b'ABF ':
            with ABF1Parser(str(file_path)) as parser:
                metadata = parser.metadata
                channel_info = parser.get_channel_info()
                sweep_times = parser.get_sweep_times()
                abf_version = 1
        elif file_signature == b'ABF2':
            with ABF2Parser(str(file_path)) as parser:
                metadata = parser.metadata
                channel_info = parser.get_channel_info()
                sweep_times = parser.get_sweep_times()
                abf_version = 2
        else:
            raise ValueError(f"Invalid ABF file")
        
        logger.info(f"ABF{abf_version}: {len(channel_info)} channels, {len(sweep_times)} sweeps")
        
    except Exception as e:
        raise IOError(f"Failed to parse ABF metadata: {e}")

    # Auto-detect channel configuration
    channel_config = _detect_channel_configuration(channel_info)
    logger.info(channel_config['message'])
    
    # Load data with pyabf
    try:
        abf = pyabf.ABF(str(file_path), loadData=True)
    except Exception as e:
        raise IOError(f"Failed to load ABF data: {e}")

    if abf.sweepCount == 0:
        raise ValueError("ABF file contains no sweeps")

    # Create dataset
    dataset = ElectrophysiologyDataset()

    # Store metadata
    dataset.metadata["format"] = "abf"
    dataset.metadata["source_file"] = str(file_path)
    dataset.metadata["abf_version"] = abf_version
    dataset.metadata["sampling_rate_hz"] = metadata.get('sample_rate', abf.sampleRate)
    dataset.metadata["channel_count"] = len(channel_info)
    dataset.metadata["channel_labels"] = [ch['name'] for ch in channel_info]
    dataset.metadata["channel_units"] = [ch['units'] for ch in channel_info]
    dataset.metadata["channel_types"] = [ch['signal_type'] for ch in channel_info]
    dataset.metadata["sweep_times"] = sweep_times
    
    # Store auto-detected channel configuration
    dataset.metadata["channel_config"] = channel_config

    # Load all sweeps
    for sweep_idx in range(abf.sweepCount):
        abf.setSweep(sweep_idx)
        time_s = abf.sweepX
        time_ms = time_s * 1000.0

        if validate_data and (np.any(np.isnan(time_ms)) or np.any(np.isinf(time_ms))):
            raise ValueError(f"Sweep {sweep_idx} contains invalid time values")

        # Load data for all channels
        data_matrix = np.zeros((len(time_ms), len(channel_info)), dtype=np.float32)
        
        for ch_idx in range(len(channel_info)):
            if len(channel_info) > 1:
                abf.setSweep(sweep_idx, channel=ch_idx)
            
            data_matrix[:, ch_idx] = abf.sweepY.astype(np.float32)

        if validate_data:
            if np.any(np.isnan(data_matrix)):
                logger.warning(f"Sweep {sweep_idx} contains NaN values")
            if np.any(np.isinf(data_matrix)):
                logger.warning(f"Sweep {sweep_idx} contains infinite values")

        # Add to dataset (1-based indexing)
        sweep_index = str(sweep_idx + 1)
        dataset.add_sweep(sweep_index, time_ms, data_matrix)

    if dataset.is_empty():
        raise ValueError("No valid sweeps loaded")

    logger.info(f"Successfully loaded {dataset.sweep_count()} sweeps from {file_path.name}")

    return dataset