"""
ABF (Axon Binary Format) Loader for PatchBatch

Uses pyABF (https://github.com/swharden/pyABF) to load ABF data into ElectrophysiologyDataset. 

Concern that pyABF yields incorrect time indexing per sweep when using ABF files exported by WinWCP (when compared to the
original WCP file time index).

We are using sweepTimesSec for the sweep start times and seeing times that are consistent with protocol duration,
but not the stimulus repeat period. This distinction means that a voltage protocol with 0.5s sweeps that repeats every
1s will yield sweep start times of 0s, 0.5s, 1.0s, 1.5s, etc. rather than 0s, 1s, 2s, etc. Since I only have access
to ABF files exported by WinWCP, and the electrophysiology data (I and V sweeps) appear unaffected, I'm keeping this as-is, 
but it may need to be revisited if other ABF files (from other software) behave differently. This is believed to be an issue 
with the ABF export process in WinWCP rather than pyABF itself.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import logging
from pathlib import Path
from typing import Any, Union, Dict, List
import numpy as np

logger = logging.getLogger(__name__)

from data_analysis_gui.core.dataset import ElectrophysiologyDataset

try:
    import pyabf
    PYABF_AVAILABLE = True
except ImportError:
    PYABF_AVAILABLE = False


def _detect_channel_configuration(channel_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Identify which channel is voltage and which is current based on units.
    
    Raises ValueError if the file doesn't have exactly one of each - there's no
    unambiguous way to map channels in that case.
    """
    voltage_channels = [ch for ch in channel_info if ch['signal_type'] == 'voltage']
    current_channels = [ch for ch in channel_info if ch['signal_type'] == 'current']
    
    if len(voltage_channels) != 1 or len(current_channels) != 1:
        raise ValueError(
            f"Cannot identify channels: found {len(voltage_channels)} voltage, "
            f"{len(current_channels)} current (need exactly 1 of each)"
        )
    
    v_ch = voltage_channels[0]
    i_ch = current_channels[0]
    
    logger.info(
        f"Mapped channels: Ch.{v_ch['index']} ({v_ch['units']}) → voltage, "
        f"Ch.{i_ch['index']} ({i_ch['units']}) → current"
    )
    
    return {
        'voltage_channel': v_ch['index'],
        'current_channel': i_ch['index'],
        'voltage_units': v_ch['units'],
        'current_units': i_ch['units'].replace('uA', 'μA').replace('ua', 'μA')
    }


def load_abf(
    file_path: Union[str, Path],
    validate_data: bool = True,
) -> "ElectrophysiologyDataset":
    """
    Load an ABF file into the standard dataset format.
    
    The file must have exactly one voltage and one current channel. The loader
    identifies which is which automatically based on units, so channel ordering
    doesn't matter.
    """
    if not PYABF_AVAILABLE:
        raise ImportError("pyabf required for ABF support. Install with: pip install pyabf")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ABF file not found: {file_path}")

    logger.info(f"Loading ABF file: {file_path.name}")

    try:
        abf = pyabf.ABF(str(file_path), loadData=True)
    except Exception as e:
        raise IOError(f"Failed to load ABF file: {e}")

    if abf.sweepCount == 0:
        raise ValueError("ABF file contains no sweeps")

    # Extract channel information
    channel_info = []
    for i in range(abf.channelCount):
        name = abf.adcNames[i] if i < len(abf.adcNames) else f"Channel {i}"
        units = abf.adcUnits[i] if i < len(abf.adcUnits) else ""
        
        # Classify by units
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

    # Identify voltage and current channels
    channel_config = _detect_channel_configuration(channel_info)
    
    # Extract sweep times
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
    dataset.metadata["channel_config"] = channel_config

    # Load sweeps
    for sweep_idx in range(abf.sweepCount):
        abf.setSweep(sweep_idx)
        time_s = abf.sweepX
        time_ms = time_s * 1000.0

        if validate_data and (np.any(np.isnan(time_ms)) or np.any(np.isinf(time_ms))):
            raise ValueError(f"Sweep {sweep_idx} contains invalid time values")

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

        sweep_index = str(sweep_idx + 1)
        dataset.add_sweep(sweep_index, time_ms, data_matrix)

    if dataset.is_empty():
        raise ValueError("No valid sweeps loaded")

    logger.info(f"Successfully loaded {dataset.sweep_count()} sweeps")
    return dataset