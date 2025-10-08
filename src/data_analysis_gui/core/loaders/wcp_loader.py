"""
WCP (WinWCP) File Loader for PatchBatch

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

PHASE 1 ENHANCEMENT: Auto-detection of channel configuration from WCP metadata
"""

import struct
import logging
from pathlib import Path
from typing import Optional, Any, Union, Dict, Tuple, List
import numpy as np

# Set up module logger
logger = logging.getLogger(__name__)

from data_analysis_gui.core.dataset import ElectrophysiologyDataset


# =============================================================================
# Channel Auto-Detection
# =============================================================================

def _detect_channel_configuration_wcp(channels: List[Any]) -> Dict[str, Any]:
    """
    Analyze WCP channel info and determine voltage/current channel assignments.
    
    Args:
        channels: List of WCPChannel objects with 'name' and 'units' attributes
    
    Returns:
        Dict with keys:
            - voltage_channel: int (channel index for voltage)
            - current_channel: int (channel index for current)
            - voltage_units: str (detected units for voltage)
            - current_units: str (detected units for current)
            - valid: bool (True if detection was successful)
            - message: str (description of detection result)
    """
    voltage_channels = []
    current_channels = []
    
    for i, ch in enumerate(channels):
        units_lower = ch.units.lower()
        
        # Identify voltage channels
        if 'mv' in units_lower or units_lower == 'v':
            voltage_channels.append({
                'index': i,
                'name': ch.name,
                'units': ch.units,
                'signal_type': 'voltage'
            })
        # Identify current channels
        elif any(u in units_lower for u in ['pa', 'na', 'µa', 'ua', 'ma', 'a']):
            current_channels.append({
                'index': i,
                'name': ch.name,
                'units': ch.units,
                'signal_type': 'current'
            })
    
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
        logger.error("No voltage channel detected in WCP file")
        return {
            'voltage_channel': 0,
            'current_channel': 1,
            'voltage_units': 'mV',
            'current_units': 'pA',
            'valid': False,
            'message': "Could not detect voltage channel - using default configuration"
        }
    
    if len(current_channels) == 0:
        logger.error("No current channel detected in WCP file")
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


def load_wcp(
    file_path: Union[str, Path],
    validate_data: bool = True,
) -> "ElectrophysiologyDataset":
    """
    Load a WCP (WinWCP) file into a standardized dataset with auto-detected channel configuration.

    This function reads WCP files and converts them to the ElectrophysiologyDataset
    format used throughout the application. WCP files contain actual sweep times 
    which are extracted and stored. Channel configuration is automatically detected
    from file metadata.

    Args:
        file_path: Path to the WCP file
        validate_data: If True, check for NaN/Inf values and warn about anomalies

    Returns:
        ElectrophysiologyDataset containing all sweeps from the WCP file with
        auto-detected channel configuration stored in metadata['channel_config']

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        IOError: If file cannot be read or is corrupted
        ValueError: If file structure is invalid or contains no data

    Example:
        >>> dataset = load_wcp('recording.wcp')
        >>> print(f"Loaded {dataset.sweep_count()} sweeps")
        >>> config = dataset.metadata['channel_config']
        >>> time_ms, data = dataset.get_sweep('1')
    """

    file_path = Path(file_path)

    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"WCP file not found: {file_path}")

    # Load WCP file
    logger.info(f"Loading WCP file: {file_path.name}")
    
    try:
        with WCPParser(str(file_path)) as wcp:
            # Auto-detect channel configuration
            channel_config = _detect_channel_configuration_wcp(wcp.file_header.channels)
            logger.info(channel_config['message'])
            
            # Create dataset
            dataset = ElectrophysiologyDataset()
            
            # Extract and store metadata
            dataset.metadata["format"] = "wcp"
            dataset.metadata["source_file"] = str(file_path)
            dataset.metadata["sampling_rate_hz"] = 1000.0 / wcp.file_header.dt if wcp.file_header.dt > 0 else None
            dataset.metadata["wcp_version"] = wcp.file_header.version
            dataset.metadata["channel_count"] = wcp.file_header.num_channels
            dataset.metadata["sweep_count"] = wcp.file_header.num_records
            
            # Store channel information
            channel_labels = [ch.name for ch in wcp.file_header.channels]
            channel_units = [ch.units for ch in wcp.file_header.channels]
            dataset.metadata["channel_labels"] = channel_labels
            dataset.metadata["channel_units"] = channel_units
            
            # Store auto-detected channel configuration
            dataset.metadata["channel_config"] = channel_config
            
            # Initialize sweep_times dictionary
            dataset.metadata["sweep_times"] = {}
            
            # Load all sweeps
            logger.debug(f"Loading {wcp.file_header.num_records} sweeps with {wcp.file_header.num_channels} channel(s)")
            
            for record_num in range(1, wcp.file_header.num_records + 1):
                try:
                    # Read sweep data and header
                    header, data = wcp.read_record(record_num, calibrated=True)
                    
                    # Get time axis in milliseconds
                    time_ms = wcp.get_time_axis(header) * 1000.0
                    
                    # Store actual sweep time (in seconds)
                    sweep_index = str(record_num)
                    dataset.metadata["sweep_times"][sweep_index] = float(header.time)
                    
                    # Validate data if requested
                    if validate_data:
                        if np.any(np.isnan(time_ms)):
                            raise ValueError(f"Sweep {record_num} contains NaN time values")
                        if np.any(np.isnan(data)):
                            logger.warning(f"Sweep {record_num} contains NaN data values")
                        if np.any(np.isinf(data)):
                            logger.warning(f"Sweep {record_num} contains infinite data values")
                    
                    # Add to dataset with 1-based indexing
                    dataset.add_sweep(sweep_index, time_ms, data)
                    
                except Exception as e:
                    logger.error(f"Failed to load sweep {record_num}: {e}")
                    if validate_data:
                        raise
                    else:
                        logger.warning(f"Skipped corrupted sweep {record_num}: {e}")
                        continue
            
            # Verify at least some sweeps were loaded
            if dataset.is_empty():
                raise ValueError("No valid sweeps could be loaded from WCP file")
            
            # Apply auto-detected channel configuration            
            logger.info(f"Successfully loaded {dataset.sweep_count()} sweeps from {file_path.name}")
            
            return dataset
            
    except Exception as e:
        logger.error(f"Failed to load WCP file: {e}")
        raise IOError(f"Failed to load WCP file: {e}")

# =============================================================================
# WCP Parser Classes (unchanged from original)
# =============================================================================

from dataclasses import dataclass
from typing import List


@dataclass
class WCPChannel:
    """Channel metadata"""
    name: str
    units: str
    calibration_factor: float
    amplifier_gain: float
    adc_zero: int
    adc_zero_at: int
    channel_offset: int


@dataclass
class WCPRecordHeader:
    """Record (sweep) metadata"""
    status: str
    rec_type: str
    number: float
    time: float  # Time in seconds - THIS IS THE KEY FIELD
    dt: float
    adc_voltage_range: List[float]
    ident: str


@dataclass
class WCPFileHeader:
    """WCP file metadata"""
    version: float
    num_channels: int
    num_samples: int
    num_records: int
    dt: float
    adc_voltage_range: float
    max_adc_value: int
    min_adc_value: int
    num_bytes_in_header: int
    num_analysis_bytes_per_record: int
    num_data_bytes_per_record: int
    num_bytes_per_record: int
    num_zero_avg: int
    channels: List[WCPChannel]


class WCPParser:
    """Parser for WCP electrophysiology data files"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.file_header: Optional[WCPFileHeader] = None
        self._file = None
        
    def __enter__(self):
        self._file = open(self.filepath, 'rb')
        self.file_header = self._parse_file_header()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            
    def _parse_key_value_header(self, header_bytes: bytes) -> Dict[str, str]:
        """Parse text-based key=value header"""
        header_text = header_bytes.decode('ascii', errors='ignore').rstrip('\x00')
        
        params = {}
        for line in header_text.split('\n'):
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                params[key.strip()] = value.strip()
        
        return params
    
    def _get_param_float(self, params: Dict[str, str], key: str, default: float = 0.0) -> float:
        """Extract float parameter"""
        try:
            return float(params.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def _get_param_int(self, params: Dict[str, str], key: str, default: int = 0) -> int:
        """Extract integer parameter"""
        try:
            return int(params.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def _parse_file_header(self) -> WCPFileHeader:
        """Parse the file header"""
        self._file.seek(0)
        initial_header = self._file.read(1024)
        params = self._parse_key_value_header(initial_header)
        
        num_bytes_in_header = self._get_param_int(params, 'NBH', 1024)
        
        if num_bytes_in_header > 1024:
            self._file.seek(0)
            initial_header = self._file.read(num_bytes_in_header)
            params = self._parse_key_value_header(initial_header)
        
        version = self._get_param_float(params, 'VER', 9.0)
        num_channels = self._get_param_int(params, 'NC', 1)
        max_adc_value = self._get_param_int(params, 'ADCMAX', 2047)
        min_adc_value = -max_adc_value - 1
        
        nba_sectors = self._get_param_int(params, 'NBA', 2)
        num_analysis_bytes_per_record = nba_sectors * 512
        
        nbd_sectors = self._get_param_int(params, 'NBD', 0)
        num_data_bytes_per_record = nbd_sectors * 512
        
        num_bytes_per_record = num_analysis_bytes_per_record + num_data_bytes_per_record
        num_samples = num_data_bytes_per_record // (2 * num_channels)
        
        num_records = self._get_param_int(params, 'NR', 0)
        dt = self._get_param_float(params, 'DT', 0.001)
        adc_voltage_range = self._get_param_float(params, 'AD', 5.0)

        num_zero_avg = self._get_param_int(params, 'NZ', 20)  # Default = 20
        num_zero_avg = max(num_zero_avg, 1)  # Ensure at least 1
        
        channels = []
        for ch in range(num_channels):
            name = params.get(f'YN{ch}', f'Ch.{ch}')
            units = params.get(f'YU{ch}', 'mV')
            calibration_factor = self._get_param_float(params, f'YG{ch}', 0.001)
            amplifier_gain = 1.0
            adc_zero = self._get_param_int(params, f'YZ{ch}', 0)
            adc_zero_at = self._get_param_int(params, f'YR{ch}', -1)
            channel_offset = self._get_param_int(params, f'YO{ch}', ch)
            
            channels.append(WCPChannel(
                name=name,
                units=units,
                calibration_factor=calibration_factor,
                amplifier_gain=amplifier_gain,
                adc_zero=adc_zero,
                adc_zero_at=adc_zero_at,
                channel_offset=channel_offset
            ))
        
        return WCPFileHeader(
            version=version,
            num_channels=num_channels,
            num_samples=num_samples,
            num_records=num_records,
            dt=dt,
            adc_voltage_range=adc_voltage_range,
            max_adc_value=max_adc_value,
            min_adc_value=min_adc_value,
            num_bytes_in_header=num_bytes_in_header,
            num_analysis_bytes_per_record=num_analysis_bytes_per_record,
            num_data_bytes_per_record=num_data_bytes_per_record,
            num_bytes_per_record=num_bytes_per_record,
            num_zero_avg=num_zero_avg,
            channels=channels
        )
    
    def _parse_record_header(self, record_num: int) -> WCPRecordHeader:
        """Parse record header for a specific record"""
        fh = self.file_header
        
        record_offset = fh.num_bytes_in_header + (record_num - 1) * fh.num_bytes_per_record
        self._file.seek(record_offset)
        
        status = self._file.read(8).decode('ascii', errors='ignore').strip('\x00').strip()
        rec_type = self._file.read(4).decode('ascii', errors='ignore').strip('\x00').strip()
        
        number = struct.unpack('<f', self._file.read(4))[0]
        time = struct.unpack('<f', self._file.read(4))[0]  # KEY: Actual sweep time in seconds
        dt = struct.unpack('<f', self._file.read(4))[0]
        
        adc_voltage_range = []
        for _ in range(fh.num_channels):
            voltage_range = struct.unpack('<f', self._file.read(4))[0]
            adc_voltage_range.append(voltage_range)
        
        ident = self._file.read(16).decode('ascii', errors='ignore').strip('\x00').strip()
        
        return WCPRecordHeader(
            status=status,
            rec_type=rec_type,
            number=number,
            time=time,
            dt=dt,
            adc_voltage_range=adc_voltage_range,
            ident=ident
        )
    
    def read_record(self, record_num: int, calibrated: bool = True) -> Tuple[WCPRecordHeader, np.ndarray]:
        """
        Read a single record (sweep)
        
        Parameters:
        -----------
        record_num : int
            Record number (1-indexed)
        calibrated : bool
            If True, return calibrated values; if False, return raw ADC values
            
        Returns:
        --------
        header : WCPRecordHeader
            Record metadata
        data : np.ndarray
            Shape (num_samples, num_channels) with data for each channel
        """
        if not (1 <= record_num <= self.file_header.num_records):
            raise ValueError(f"Record number must be between 1 and {self.file_header.num_records}")
        
        fh = self.file_header
        
        header = self._parse_record_header(record_num)
        
        # Read raw data
        data_offset = (fh.num_bytes_in_header + 
                    (record_num - 1) * fh.num_bytes_per_record + 
                    fh.num_analysis_bytes_per_record)
        self._file.seek(data_offset)
        
        num_values = fh.num_samples * fh.num_channels
        raw_data = np.frombuffer(
            self._file.read(num_values * 2),
            dtype=np.int16
        )
        
        data = raw_data.reshape((fh.num_samples, fh.num_channels)).copy()
        
        if calibrated:
            data = data.astype(np.float64)
            
            # Calculate dynamic zero levels (if applicable) and apply calibration
            for ch_idx, channel in enumerate(fh.channels):
                # Determine the zero level for this channel in this sweep
                if channel.adc_zero_at >= 0:
                    # Calculate zero from baseline region in THIS sweep
                    zero_level = self._calculate_dynamic_zero(
                        data[:, ch_idx], 
                        channel.adc_zero_at, 
                        fh.num_zero_avg, 
                        fh.num_samples
                    )
                else:
                    # Use fixed zero from file header
                    zero_level = channel.adc_zero
                
                # Calculate scale factor (same as before)
                adc_scale = (abs(header.adc_voltage_range[ch_idx]) / 
                        (channel.calibration_factor * (fh.max_adc_value + 1)))
                
                # Apply calibration: Physical = (Raw - Zero) * Scale
                data[:, ch_idx] = (data[:, ch_idx] - zero_level) * adc_scale
        
        return header, data
    
    def _calculate_dynamic_zero(
        self, 
        raw_channel_data: np.ndarray, 
        adc_zero_at: int, 
        num_zero_avg: int,
        num_samples: int
    ) -> float:
        """
        Calculate dynamic zero level from baseline region in sweep.
        
        This matches WinWCP's logic exactly:
        - Average num_zero_avg samples starting at adc_zero_at
        - Ensure indices are within valid bounds
        - Return the mean of raw ADC values
        
        Parameters:
        -----------
        raw_channel_data : np.ndarray
            Raw ADC values for single channel (BEFORE calibration)
        adc_zero_at : int
            Starting sample index for baseline region
        num_zero_avg : int
            Number of samples to average
        num_samples : int
            Total number of samples in sweep
            
        Returns:
        --------
        zero_level : float
            Calculated baseline (mean of raw ADC values)
        """
        # Bound the start index
        i0 = max(0, min(adc_zero_at, num_samples - 1))
        
        # Bound the end index
        i1 = i0 + num_zero_avg - 1
        i1 = max(0, min(i1, num_samples - 1))
        
        # Calculate mean of baseline region (raw ADC values)
        zero_level = np.mean(raw_channel_data[i0:i1+1])
        
        return zero_level
    
    def get_time_axis(self, header: WCPRecordHeader = None) -> np.ndarray:
        """Get time axis for a record in seconds"""
        dt = header.dt if header is not None else self.file_header.dt
        return np.arange(self.file_header.num_samples) * dt