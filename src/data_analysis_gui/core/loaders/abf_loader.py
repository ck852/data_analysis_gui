"""
ABF (Axon Binary Format) Loader for PatchBatch

Uses pyABF (https://github.com/swharden/pyABF) to load ABF data into ElectrophysiologyDataset. 

Concern that pyABF yields distinct time indexing per sweep when using ABF files exported by WinWCP (when compared to the
original WCP file time index).
 
We are using sweepTimesSec for the sweep start times and seeing times that are consistent with protocol duration,
but not the stimulus repeat period. This distinction means that a voltage protocol with 0.5s sweeps that repeats every
1s will yield sweep start times of 0s, 0.5s, 1.0s, 1.5s, etc. rather than 0s, 1s, 2s, etc. Since we only have access
to ABF files exported by WinWCP, we will proceed with this approach for now, but it may need to be revisited if other ABF files
(from pClamp or other software) behave differently.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

"""

import logging
from pathlib import Path
from typing import Optional, Any, Union, Dict, List
import numpy as np


logger = logging.getLogger(__name__)

from data_analysis_gui.core.dataset import ElectrophysiologyDataset

try:
    import pyabf
    PYABF_AVAILABLE = True
except ImportError:
    PYABF_AVAILABLE = False


# =============================================================================
# Channel Auto-Detection
# =============================================================================

def _detect_channel_configuration(channel_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze channel info and determine voltage/current channel assignments.
    
    I and V channels are identified based on units in channel metadata. The codebase has 
    been written for input files with one voltage and one current channel. This script includes
    fallbacks for input files with multiple or missing channels. User will be informed via 
    _check_channel_warnings() in MainWindow if their input file does not match expected format.
    """
    voltage_channels = [ch for ch in channel_info if ch['signal_type'] == 'voltage']
    current_channels = [ch for ch in channel_info if ch['signal_type'] == 'current']
    
    # Case 1: Perfect detection - exactly 1 voltage and 1 current
    if len(voltage_channels) == 1 and len(current_channels) == 1:
        return {
            'voltage_channel': voltage_channels[0]['index'],
            'current_channel': current_channels[0]['index'],
            'voltage_units': voltage_channels[0]['units'],
            'current_units': current_channels[0]['units'].replace('uA', 'μA').replace('ua', 'μA'),
            'valid': True,
            'warning_level': 'none',
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
            'current_units': current_channels[0]['units'].replace('uA', 'μA').replace('ua', 'μA'),
            'valid': True,
            'warning_level': 'info',
            'message': f"Multiple channels detected:\n"
                      f"• {len(voltage_channels)} voltage channel(s)\n"
                      f"• {len(current_channels)} current channel(s)\n\n"
                      f"Using Ch.{voltage_channels[0]['index']} (voltage) and "
                      f"Ch.{current_channels[0]['index']} (current).",
            'user_message': f"Multiple channels detected. Using Ch.{voltage_channels[0]['index']} (voltage) "
                           f"and Ch.{current_channels[0]['index']} (current)."
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
            'warning_level': 'error',
            'message': "No voltage channel detected in file.\n\n"
                      "Using default configuration (Ch.0 = voltage, Ch.1 = current).\n"
                      "Analysis results may be incorrect.",
            'user_message': "No voltage channel detected. Using default configuration."
        }
    
    if len(current_channels) == 0:
        logger.error("No current channel detected in ABF file")
        return {
            'voltage_channel': 0,
            'current_channel': 1,
            'voltage_units': 'mV',
            'current_units': 'pA',
            'valid': False,
            'warning_level': 'error',
            'message': "No current channel detected in file.\n\n"
                      "Using default configuration (Ch.0 = voltage, Ch.1 = current).\n"
                      "Analysis results may be incorrect.",
            'user_message': "No current channel detected. Using default configuration."
        }
    
    # Fallback - should not reach here
    logger.error("Unexpected channel configuration")
    return {
        'voltage_channel': 0,
        'current_channel': 1,
        'voltage_units': 'mV',
        'current_units': 'pA',
        'valid': False,
        'warning_level': 'error',
        'message': "Unexpected channel configuration encountered.\n\n"
                  "Using default configuration (Ch.0 = voltage, Ch.1 = current).\n"
                  "Analysis results may be incorrect.",
        'user_message': "Channel detection failed. Using default configuration."
    }


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

    # Load ABF file with pyabf
    try:
        abf = pyabf.ABF(str(file_path), loadData=True)
    except Exception as e:
        raise IOError(f"Failed to load ABF file: {e}")

    if abf.sweepCount == 0:
        raise ValueError("ABF file contains no sweeps")

    # Extract channel information from pyabf
    channel_info = []
    for i in range(abf.channelCount):
        name = abf.adcNames[i] if i < len(abf.adcNames) else f"Channel {i}"
        units = abf.adcUnits[i] if i < len(abf.adcUnits) else ""
        
        # Identify signal type based on units
        units_lower = units.lower()
        if 'mv' in units_lower or units_lower == 'v':
            signal_type = "voltage"
        elif any(u in units_lower for u in ['pa', 'na', 'µa', 'ua', 'ma', 'a']):
            signal_type = "current"
        else:
            signal_type = "unknown"
        
        channel_info.append({
            'index': i,
            'name': name,
            'units': units,
            'signal_type': signal_type
        })
    
    logger.info(f"ABF{abf.abfVersion}: {len(channel_info)} channels, {abf.sweepCount} sweeps")

    # Auto-detect channel configuration
    channel_config = _detect_channel_configuration(channel_info)
    logger.info(channel_config['message'])
    
    # Extract sweep times
    # Use sweepTimesSec which gives the start time of each sweep
    sweep_times = {}
    for sweep_idx in range(abf.sweepCount):
        sweep_times[str(sweep_idx + 1)] = float(abf.sweepTimesSec[sweep_idx])

    # Create dataset
    dataset = ElectrophysiologyDataset()

    # Store metadata
    dataset.metadata["format"] = "abf"
    dataset.metadata["source_file"] = str(file_path)
    dataset.metadata["abf_version"] = abf.abfVersion
    dataset.metadata["sampling_rate_hz"] = abf.sampleRate
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