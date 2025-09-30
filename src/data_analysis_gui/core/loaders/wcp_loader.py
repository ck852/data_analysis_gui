"""
WCP (WinWCP) File Loader for PatchBatch

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

WCP file loader for electrophysiology data.

This module provides functionality to load WCP files and convert them
to the standardized ElectrophysiologyDataset format.

Features:
    - Automatic channel detection and labeling
    - Extraction of actual sweep times from file headers
    - Unit conversion to mV and pA
    - Metadata preservation
"""

import struct
import logging
from pathlib import Path
from typing import Optional, Any, Union, Dict, Tuple, List
import numpy as np

# Set up module logger
logger = logging.getLogger(__name__)

from data_analysis_gui.core.dataset import ElectrophysiologyDataset


def load_wcp(
    file_path: Union[str, Path],
    channel_map: Optional[Any] = None,
    validate_data: bool = True,
) -> "ElectrophysiologyDataset":
    """
    Load a WCP (WinWCP) file into a standardized dataset.

    This function reads WCP files and converts them to the ElectrophysiologyDataset
    format used throughout the application. Unlike ABF/MAT files, WCP files contain
    actual sweep times and channel metadata that are extracted and used for automatic
    channel detection.

    Args:
        file_path: Path to the WCP file
        channel_map: Optional ChannelDefinitions instance for custom channel mapping
        validate_data: If True, check for NaN/Inf values and warn about anomalies

    Returns:
        ElectrophysiologyDataset containing all sweeps from the WCP file

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        IOError: If file cannot be read or is corrupted
        ValueError: If file structure is invalid or contains no data
    """
    file_path = Path(file_path)

    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"WCP file not found: {file_path}")

    # Load WCP file
    logger.info(f"Loading WCP file: {file_path.name}")
    
    try:
        with WCPParser(str(file_path)) as wcp:
            # Create dataset
            dataset = ElectrophysiologyDataset()
            
            # Extract and store basic metadata
            dataset.metadata["format"] = "wcp"
            dataset.metadata["source_file"] = str(file_path)
            dataset.metadata["sampling_rate_hz"] = (
                1000.0 / wcp.file_header.dt if wcp.file_header.dt > 0 else None
            )
            dataset.metadata["wcp_version"] = wcp.file_header.version
            dataset.metadata["channel_count"] = wcp.file_header.num_channels
            dataset.metadata["sweep_count"] = wcp.file_header.num_records
            
            # Store channel information (original from file)
            channel_labels = [ch.name for ch in wcp.file_header.channels]
            channel_units = [ch.units for ch in wcp.file_header.channels]
            dataset.metadata["channel_labels"] = channel_labels
            dataset.metadata["channel_units"] = channel_units
            
            # === NEW: Auto-detect channel assignments ===
            detection_results = _detect_channel_assignments(wcp.file_header.channels)
            dataset.metadata["wcp_channel_detection"] = detection_results
            dataset.metadata["auto_detected_channels"] = True
            
            logger.info(
                f"Auto-detected channels: "
                f"Voltage=Ch{detection_results['voltage_channel']}, "
                f"Current=Ch{detection_results['current_channel']}"
            )
            # === END NEW ===
            
            # Initialize sweep_times dictionary
            dataset.metadata["sweep_times"] = {}
            
            # Load all sweeps
            logger.debug(
                f"Loading {wcp.file_header.num_records} sweeps with "
                f"{wcp.file_header.num_channels} channel(s)"
            )
            
            for record_num in range(1, wcp.file_header.num_records + 1):
                try:
                    # Read sweep data and header
                    header, data = wcp.read_record(record_num, calibrated=True)
                    
                    # Get time axis in milliseconds
                    time_ms = wcp.get_time_axis() * 1000.0
                    
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
            
            # === NEW: Apply auto-detection to channel_map if provided ===
            if channel_map is not None:
                channel_map.set_from_wcp_detection(
                    voltage_channel=detection_results['voltage_channel'],
                    current_channel=detection_results['current_channel'],
                    voltage_units=detection_results['voltage_units'],
                    current_units=detection_results['current_units']
                )
            # === END NEW ===
            
            logger.info(
                f"Successfully loaded {dataset.sweep_count()} sweeps from {file_path.name}"
            )
            
            return dataset
            
    except Exception as e:
        logger.error(f"Failed to load WCP file: {e}")
        raise IOError(f"Failed to load WCP file: {e}")

def _detect_channel_assignments(wcp_channels: List['WCPChannel']) -> Dict[str, any]:
    """
    Automatically detect voltage and current channel assignments from WCP metadata.
    
    Detection strategy:
    1. Check channel names for patterns (Vm, Im1, Im, I)
    2. Check units as fallback/confirmation (mV/V, pA/µA/uA)
    3. Use default assignment if detection fails (Ch0=voltage, Ch1=current)
    
    Args:
        wcp_channels: List of WCPChannel objects from file header
        
    Returns:
        Dictionary containing:
            - voltage_channel: int (channel index)
            - current_channel: int (channel index)
            - voltage_units: str (detected units)
            - current_units: str (detected units)
    """
    voltage_ch = None
    current_ch = None
    
    # Expected patterns
    VOLTAGE_NAME_PATTERNS = ['vm', 'v_m']  # Case-insensitive
    CURRENT_NAME_PATTERNS = ['im1', 'im', 'i_m']  # Prioritize longer matches first
    
    VOLTAGE_UNITS = {'mV', 'V'}
    # Include both µ (micro sign) and u (letter u) variants
    CURRENT_UNITS = {'pA', 'µA', 'uA', 'Î¼A'}  # Î¼ is Greek mu
    
    # Strategy 1: Check channel names
    for i, ch in enumerate(wcp_channels):
        name_lower = ch.name.lower().strip()
        
        # Check for voltage patterns
        if any(pattern in name_lower for pattern in VOLTAGE_NAME_PATTERNS):
            voltage_ch = i
            logger.debug(f"Detected voltage channel {i} from name: {ch.name}")
        
        # Check for current patterns (prioritize longer matches)
        elif any(pattern in name_lower for pattern in CURRENT_NAME_PATTERNS):
            current_ch = i
            logger.debug(f"Detected current channel {i} from name: {ch.name}")
    
    # Strategy 2: Check units as fallback or confirmation
    if voltage_ch is None or current_ch is None:
        for i, ch in enumerate(wcp_channels):
            units = ch.units.strip()
            
            if voltage_ch is None and units in VOLTAGE_UNITS:
                voltage_ch = i
                logger.debug(f"Detected voltage channel {i} from units: {units}")
            
            if current_ch is None and units in CURRENT_UNITS:
                current_ch = i
                logger.debug(f"Detected current channel {i} from units: {units}")
    
    # Strategy 3: Default fallback
    if voltage_ch is None:
        voltage_ch = 0
        logger.warning(
            "Could not detect voltage channel from metadata, using default Ch0"
        )
    
    if current_ch is None:
        current_ch = 1
        logger.warning(
            "Could not detect current channel from metadata, using default Ch1"
        )
    
    # Validate that we have different channels
    if voltage_ch == current_ch:
        logger.error(
            f"Detection error: Both channels mapped to Ch{voltage_ch}. "
            "Using default Ch0=voltage, Ch1=current"
        )
        voltage_ch = 0
        current_ch = 1
    
    # Extract units from detected channels
    voltage_units = wcp_channels[voltage_ch].units.strip()
    current_units = wcp_channels[current_ch].units.strip()
    
    # Normalize current units to handle different representations
    current_units = _normalize_current_units(current_units)
    
    result = {
        'voltage_channel': voltage_ch,
        'current_channel': current_ch,
        'voltage_units': voltage_units,
        'current_units': current_units,
    }
    
    logger.info(
        f"WCP channel detection complete: "
        f"V=Ch{voltage_ch} ({voltage_units}), "
        f"I=Ch{current_ch} ({current_units})"
    )
    
    return result

def _normalize_current_units(units: str) -> str:
    """
    Normalize current units to handle different micro symbol representations.
    
    Args:
        units: Raw units string from WCP file
        
    Returns:
        Normalized units string (prefers µ over u)
    """
    # Map all variants to preferred representation
    if units in ['uA', 'Î¼A']:  # Î¼ is Greek mu
        return 'µA'  # Use micro sign
    elif units == 'pA':
        return 'pA'
    else:
        # Return as-is for any unexpected units
        return units

def _apply_channel_mapping_wcp(
    dataset: "ElectrophysiologyDataset", channel_map: Any
) -> None:
    """
    Apply custom channel definitions to dataset metadata.

    Args:
        dataset: Dataset to update
        channel_map: ChannelDefinitions instance
    """
    if not hasattr(channel_map, "get_channel_label"):
        logger.warning(
            "Channel map doesn't have get_channel_label method. Skipping mapping."
        )
        return

    num_channels = dataset.channel_count()
    labels = []
    units = []

    for ch_id in range(num_channels):
        # Try to get label from channel_map
        try:
            label = channel_map.get_channel_label(ch_id, include_units=False)

            # If channel_map returns a generic label, prefer WCP's label
            if label.startswith("Channel ") and ch_id < len(
                dataset.metadata["channel_labels"]
            ):
                original_label = dataset.metadata["channel_labels"][ch_id]
                if not original_label.startswith("Channel "):
                    label = original_label

            labels.append(label)
        except Exception as e:
            logger.warning(
                f"Failed to get label for channel {ch_id} from channel_map: {e}"
            )
            labels.append(
                dataset.metadata["channel_labels"][ch_id]
                if ch_id < len(dataset.metadata["channel_labels"])
                else f"Channel {ch_id}"
            )

        # Use the channel units from WCP file
        if ch_id < len(dataset.metadata["channel_units"]):
            units.append(dataset.metadata["channel_units"][ch_id])
        else:
            units.append("")

    dataset.metadata["channel_labels"] = labels
    dataset.metadata["channel_units"] = units


# =============================================================================
# WCP Parser Classes (from wcp_sweep.py)
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
        
        channels = []
        for ch in range(num_channels):
            name = params.get(f'YN{ch}', f'Ch.{ch}')
            units = params.get(f'YU{ch}', 'mV')
            calibration_factor = self._get_param_float(params, f'YG{ch}', 0.001)
            amplifier_gain = 1.0
            adc_zero = self._get_param_int(params, f'YZ{ch}', 0)
            channel_offset = self._get_param_int(params, f'YO{ch}', ch)
            
            channels.append(WCPChannel(
                name=name,
                units=units,
                calibration_factor=calibration_factor,
                amplifier_gain=amplifier_gain,
                adc_zero=adc_zero,
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
            
            for ch_idx, channel in enumerate(fh.channels):
                adc_scale = (abs(header.adc_voltage_range[ch_idx]) / 
                           (channel.calibration_factor * (fh.max_adc_value + 1)))
                data[:, ch_idx] = data[:, ch_idx] * adc_scale
        
        return header, data
    
    def get_time_axis(self) -> np.ndarray:
        """Get time axis for a record in seconds"""
        return np.arange(self.file_header.num_samples) * self.file_header.dt