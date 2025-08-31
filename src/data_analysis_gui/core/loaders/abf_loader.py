"""
ABF (Axon Binary Format) loader for electrophysiology data.

This module provides functionality to load ABF files and convert them
to the standardized ElectrophysiologyDataset format. Supports both
ABF1 and ABF2 file formats through the pyabf library.

Features:
    - Automatic channel detection and labeling
    - Unit conversion (V→mV, nA→pA, µA→pA)
    - Sweep indexing compatible with existing MAT file structure
    - Metadata preservation (protocols, comments, tags)
    - Performance optimizations for large files
    - Comprehensive error handling

Author: Data Analysis GUI Contributors
License: MIT
"""

import warnings
import logging
from pathlib import Path
from typing import Optional, Any, Union, Dict, Tuple
import numpy as np

# Set up module logger
logger = logging.getLogger(__name__)

from data_analysis_gui.core.dataset import ElectrophysiologyDataset

# Try to import pyabf
try:
    import pyabf
    PYABF_AVAILABLE = True
except ImportError:
    PYABF_AVAILABLE = False
    logger.warning("pyabf not installed. ABF file support will be unavailable.")

# Performance thresholds
LARGE_FILE_SWEEP_THRESHOLD = 100  # Warn if more than this many sweeps
LARGE_FILE_SAMPLE_THRESHOLD = 1e7  # Warn if more than this many total samples
MAX_REASONABLE_SWEEPS = 1000  # Error if more than this many sweeps


def load_abf(file_path: Union[str, Path], 
             channel_map: Optional[Any] = None,
             validate_data: bool = True,
             lazy_load: bool = False) -> 'ElectrophysiologyDataset':
    """
    Load an ABF (Axon Binary Format) file into a standardized dataset.
    
    This function reads ABF files (versions 1 and 2) and converts them to
    the ElectrophysiologyDataset format used throughout the application.
    
    Args:
        file_path: Path to the ABF file
        channel_map: Optional ChannelDefinitions instance for custom channel mapping
        validate_data: If True, check for NaN/Inf values and warn about anomalies
        lazy_load: If True, return a lazy-loading dataset (future feature)
    
    Returns:
        ElectrophysiologyDataset containing all sweeps from the ABF file
    
    Raises:
        ImportError: If pyabf is not installed
        FileNotFoundError: If the specified file doesn't exist
        IOError: If file cannot be read or is corrupted
        ValueError: If file structure is invalid or contains no data
        
    Example:
        >>> dataset = load_abf('recording.abf')
        >>> print(f"Loaded {dataset.sweep_count()} sweeps")
        >>> time_ms, data = dataset.get_sweep('1')
    
    Notes:
        - Sweep indices are converted to 1-based to match MAT file convention
        - Voltage units are automatically converted to mV
        - Current units are automatically converted to pA
        - Large files (>100 sweeps) will trigger a performance warning
    """
    
    
    # Check pyabf availability
    if not PYABF_AVAILABLE:
        raise ImportError(
            "pyabf is required for ABF file support. "
            "Install with: pip install pyabf"
        )
    
    # Lazy loading implementation (future feature)
    if lazy_load:
        logger.info("Lazy loading requested but not yet implemented. Using standard loading.")
        # return LazyABFDataset(file_path, channel_map)  # Future implementation
    
    file_path = Path(file_path)
    
    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"ABF file not found: {file_path}")
    
    # Check file size for performance warnings
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 100:
        logger.warning(f"Large ABF file ({file_size_mb:.1f} MB). Loading may take some time.")
    
    # Load ABF file
    logger.info(f"Loading ABF file: {file_path.name}")
    try:
        abf = pyabf.ABF(str(file_path), loadData=True)
    except ValueError as e:
        raise IOError(f"Failed to parse ABF file structure: {e}")
    except Exception as e:
        raise IOError(f"Failed to load ABF file: {e}")
    
    # Validate basic structure
    if abf.sweepCount == 0:
        raise ValueError("ABF file contains no sweeps")
    
    if abf.sweepCount > MAX_REASONABLE_SWEEPS:
        raise ValueError(
            f"ABF file contains {abf.sweepCount} sweeps, exceeding reasonable limit "
            f"of {MAX_REASONABLE_SWEEPS}. This may be a corrupted file or require "
            "special handling."
        )
    
    # Performance warning for large datasets
    total_samples = abf.sweepCount * abf.sweepPointCount * abf.channelCount
    if abf.sweepCount > LARGE_FILE_SWEEP_THRESHOLD:
        warnings.warn(
            f"ABF file contains {abf.sweepCount} sweeps. "
            "Consider processing in batches for better performance.",
            UserWarning
        )
    
    if total_samples > LARGE_FILE_SAMPLE_THRESHOLD:
        warnings.warn(
            f"ABF file contains {total_samples:.1e} total data points. "
            "Large memory usage expected.",
            UserWarning
        )
    
    # Create dataset
    dataset = ElectrophysiologyDataset()
    
    # Extract and store metadata
    dataset.metadata['format'] = 'abf'
    dataset.metadata['source_file'] = str(file_path)
    dataset.metadata['sampling_rate_hz'] = float(abf.sampleRate)
    dataset.metadata['abf_version'] = abf.abfVersion
    dataset.metadata['abf_version_string'] = abf.abfVersionString
    dataset.metadata['protocol'] = abf.protocol if abf.protocol else None
    dataset.metadata['creation_date'] = abf.abfDateTime.isoformat() if hasattr(abf, 'abfDateTime') else None
    dataset.metadata['data_point_count'] = abf.dataPointCount
    dataset.metadata['data_seconds_total'] = abf.dataSecPerPoint * abf.dataPointCount
    
    # Store acquisition mode
    #acquisition_modes = {0: 'gap-free', 1: 'event-driven', 2: 'oscilloscope', 
    #                    3: 'episodic', 4: 'episodic-stimulation', 5: 'event-driven-fixed-length'}
    #dataset.metadata['acquisition_mode'] = acquisition_modes.get(abf.abfFileHeader.nOperationMode, 'unknown')
    
    # Extract channel information
    channel_count = abf.channelCount
    dataset.metadata['channel_count'] = channel_count
    
    # Process channel labels and units
    channel_labels, channel_units = _process_channel_info(abf, channel_count)
    dataset.metadata['channel_labels'] = channel_labels
    dataset.metadata['channel_units'] = channel_units
    
    # Store original units for reference
    dataset.metadata['original_channel_units'] = abf.adcUnits[:channel_count] if hasattr(abf, 'adcUnits') else []
    
    # Load all sweeps
    logger.debug(f"Loading {abf.sweepCount} sweeps with {channel_count} channel(s)")
    
    for sweep_idx in range(abf.sweepCount):
        try:
            # Load sweep data
            sweep_data = _load_single_sweep(abf, sweep_idx, channel_count, 
                                           channel_labels, channel_units,
                                           validate_data)
            
            # Add to dataset with 1-based indexing (matching MAT convention)
            sweep_index = str(sweep_idx + 1)
            dataset.add_sweep(sweep_index, sweep_data['time_ms'], sweep_data['data_matrix'])
            
        except Exception as e:
            logger.error(f"Failed to load sweep {sweep_idx}: {e}")
            if validate_data:
                raise
            else:
                warnings.warn(f"Skipped corrupted sweep {sweep_idx}: {e}", UserWarning)
                continue
    
    # Verify at least some sweeps were loaded
    if dataset.is_empty():
        raise ValueError("No valid sweeps could be loaded from ABF file")
    
    # Apply channel mapping if provided
    if channel_map is not None:
        _apply_channel_mapping_abf(dataset, channel_map, abf)
    
    # Store additional ABF-specific metadata
    _store_extended_metadata(dataset, abf)
    
    logger.info(f"Successfully loaded {dataset.sweep_count()} sweeps from {file_path.name}")
    
    return dataset


def _process_channel_info(abf: 'pyabf.ABF', 
                          channel_count: int) -> Tuple[list, list]:
    """
    Process channel labels and units from ABF file.
    
    Args:
        abf: PyABF object
        channel_count: Number of channels
    
    Returns:
        Tuple of (channel_labels, channel_units) lists
    """
    channel_labels = []
    channel_units = []
    
    for ch_idx in range(channel_count):
        # Get channel name
        if hasattr(abf, 'adcNames') and ch_idx < len(abf.adcNames):
            label = abf.adcNames[ch_idx]
            # Clean up common label issues
            label = label.strip()
            if not label or label.lower() in ['none', 'n/a', '']:
                label = f'Channel {ch_idx}'
        else:
            label = f'Channel {ch_idx}'
        channel_labels.append(label)
        
        # Get and normalize units
        if hasattr(abf, 'adcUnits') and ch_idx < len(abf.adcUnits):
            unit = abf.adcUnits[ch_idx].strip()
            
            # Comprehensive unit normalization map
            unit_map = {
                # Voltage units
                'mV': 'mV',
                'V': 'V',
                'uV': 'µV',
                'µV': 'µV',
                'mv': 'mV',
                'v': 'V',
                
                # Current units
                'pA': 'pA',
                'nA': 'nA',
                'uA': 'µA',
                'µA': 'µA',
                'mA': 'mA',
                'A': 'A',
                'pa': 'pA',
                'na': 'nA',
                'ua': 'µA',
                'ma': 'mA',
                'a': 'A',
                
                # Other units
                '': '',
                'none': '',
                'None': '',
                'N/A': '',
            }
            
            normalized_unit = unit_map.get(unit, unit)
            channel_units.append(normalized_unit)
        else:
            channel_units.append('')
    
    return channel_labels, channel_units


def _load_single_sweep(abf: 'pyabf.ABF', 
                       sweep_idx: int,
                       channel_count: int,
                       channel_labels: list,
                       channel_units: list,
                       validate_data: bool) -> Dict[str, np.ndarray]:
    """
    Load a single sweep from the ABF file.
    
    Args:
        abf: PyABF object
        sweep_idx: Sweep index (0-based)
        channel_count: Number of channels
        channel_labels: List of channel labels
        channel_units: List of channel units (will be modified for conversions)
        validate_data: Whether to validate data for NaN/Inf
    
    Returns:
        Dictionary with 'time_ms' and 'data_matrix' arrays
    """
    # Set sweep for ABF object
    abf.setSweep(sweep_idx)
    
    # Get time vector in milliseconds
    time_s = abf.sweepX  # Time in seconds
    time_ms = time_s * 1000.0
    
    # Validate time vector
    if validate_data:
        if np.any(np.isnan(time_ms)) or np.any(np.isinf(time_ms)):
            raise ValueError(f"Sweep {sweep_idx} contains invalid time values")
    
    # Prepare data matrix for all channels
    if channel_count == 1:
        # Single channel - simple case
        data_matrix = abf.sweepY.reshape(-1, 1)
        
        # Apply unit conversion if needed
        data_matrix[:, 0], channel_units[0] = _convert_units(
            data_matrix[:, 0], 
            channel_units[0], 
            channel_labels[0]
        )
    else:
        # Multiple channels - collect data for each
        data_matrix = np.zeros((len(time_ms), channel_count), dtype=np.float32)
        
        for ch_idx in range(channel_count):
            abf.setSweep(sweep_idx, channel=ch_idx)
            channel_data = abf.sweepY.astype(np.float32)
            
            # Apply unit conversion
            channel_data, channel_units[ch_idx] = _convert_units(
                channel_data,
                channel_units[ch_idx],
                channel_labels[ch_idx]
            )
            
            data_matrix[:, ch_idx] = channel_data
    
    # Validate data if requested
    if validate_data:
        nan_count = np.sum(np.isnan(data_matrix))
        inf_count = np.sum(np.isinf(data_matrix))
        
        if nan_count > 0:
            warnings.warn(
                f"Sweep {sweep_idx} contains {nan_count} NaN values. "
                "This may affect analysis results.",
                UserWarning
            )
        
        if inf_count > 0:
            warnings.warn(
                f"Sweep {sweep_idx} contains {inf_count} infinite values. "
                "This may affect analysis results.",
                UserWarning
            )
    
    return {
        'time_ms': time_ms,
        'data_matrix': data_matrix
    }


def _convert_units(data: np.ndarray, 
                   unit: str, 
                   channel_label: str) -> Tuple[np.ndarray, str]:
    """
    Convert data to standard units based on channel type.
    
    Standard units:
        - Voltage: mV
        - Current: pA
    
    Args:
        data: Data array to convert
        unit: Current unit
        channel_label: Channel label for type detection
    
    Returns:
        Tuple of (converted_data, new_unit)
    """
    # Determine channel type from label
    label_lower = channel_label.lower()
    is_voltage = any(v in label_lower for v in ['volt', 'potential', 'vm', 'v_m', 'membrane'])
    is_current = any(c in label_lower for c in ['curr', 'i_', 'im', 'i_m', 'amp'])
    
    # Voltage conversions to mV
    if is_voltage or unit in ['V', 'v']:
        if unit in ['V', 'v']:
            return data * 1000.0, 'mV'  # V to mV
        elif unit in ['uV', 'µV', 'μV']:
            return data / 1000.0, 'mV'  # µV to mV
        elif unit in ['mV', 'mv']:
            return data, 'mV'  # Already in mV
    
    # Current conversions to pA
    elif is_current or unit in ['A', 'mA', 'µA', 'uA', 'nA', 'a', 'ma', 'ua', 'na']:
        if unit in ['A', 'a']:
            return data * 1e12, 'pA'  # A to pA
        elif unit in ['mA', 'ma']:
            return data * 1e9, 'pA'  # mA to pA
        elif unit in ['µA', 'uA', 'μA', 'ua']:
            return data * 1e6, 'pA'  # µA to pA
        elif unit in ['nA', 'na']:
            return data * 1000.0, 'pA'  # nA to pA
        elif unit in ['pA', 'pa']:
            return data, 'pA'  # Already in pA
    
    # No conversion needed or unknown unit
    return data, unit


def _apply_channel_mapping_abf(dataset: 'ElectrophysiologyDataset', 
                                channel_map: Any,
                                abf: 'pyabf.ABF') -> None:
    """
    Apply custom channel definitions to dataset metadata.
    
    Args:
        dataset: Dataset to update
        channel_map: ChannelDefinitions instance
        abf: PyABF object for additional metadata
    """
    if not hasattr(channel_map, 'get_channel_label'):
        logger.warning("Channel map doesn't have get_channel_label method. Skipping mapping.")
        return
    
    num_channels = dataset.channel_count()
    labels = []
    units = []
    
    for ch_id in range(num_channels):
        # Try to get label from channel_map
        try:
            label = channel_map.get_channel_label(ch_id, include_units=False)
            
            # If channel_map returns a generic label, prefer ABF's label
            if label.startswith('Channel ') and ch_id < len(dataset.metadata['channel_labels']):
                original_label = dataset.metadata['channel_labels'][ch_id]
                if not original_label.startswith('Channel '):
                    label = original_label
            
            labels.append(label)
        except Exception as e:
            logger.warning(f"Failed to get label for channel {ch_id} from channel_map: {e}")
            labels.append(dataset.metadata['channel_labels'][ch_id] 
                         if ch_id < len(dataset.metadata['channel_labels']) 
                         else f'Channel {ch_id}')
        
        # Determine units
        if hasattr(channel_map, 'get_type_for_channel'):
            try:
                ch_type = channel_map.get_type_for_channel(ch_id)
                if ch_type == 'voltage':
                    units.append('mV')
                elif ch_type == 'current':
                    units.append('pA')
                else:
                    # Keep original units
                    units.append(dataset.metadata['channel_units'][ch_id] 
                               if ch_id < len(dataset.metadata['channel_units']) 
                               else '')
            except Exception as e:
                logger.warning(f"Failed to get type for channel {ch_id} from channel_map: {e}")
                units.append(dataset.metadata['channel_units'][ch_id] 
                           if ch_id < len(dataset.metadata['channel_units']) 
                           else '')
        else:
            # Keep original units
            units.append(dataset.metadata['channel_units'][ch_id] 
                       if ch_id < len(dataset.metadata['channel_units']) 
                       else '')
    
    dataset.metadata['channel_labels'] = labels
    dataset.metadata['channel_units'] = units


def _store_extended_metadata(dataset: 'ElectrophysiologyDataset', 
                             abf: 'pyabf.ABF') -> None:
    """
    Store additional ABF-specific metadata that might be useful.
    
    Args:
        dataset: Dataset to update
        abf: PyABF object
    """
    # Stimulus/protocol information
    if hasattr(abf, 'stimulusByChannel') and abf.stimulusByChannel:
        dataset.metadata['stimulus_info'] = abf.stimulusByChannel
    
    # Comments and tags
    if hasattr(abf, 'tagComments') and abf.tagComments:
        dataset.metadata['comments'] = abf.tagComments
    
    if hasattr(abf, 'tagTimesMin') and len(abf.tagTimesMin) > 0:
        dataset.metadata['tag_times_min'] = abf.tagTimesMin.tolist()
    
    # Creator information
    if hasattr(abf, 'abfID'):
        dataset.metadata['abf_id'] = abf.abfID
    
    if hasattr(abf, 'creatorVersion'):
        dataset.metadata['creator_version'] = abf.creatorVersion
    
    # Holding values for voltage clamp
    if hasattr(abf, 'holdingCommand'):
        dataset.metadata['holding_command'] = abf.holdingCommand

    
    # Data quality metrics
    dataset.metadata['data_byte_start'] = abf.dataByteStart if hasattr(abf, 'dataByteStart') else None
    dataset.metadata['data_point_byte_size'] = abf.dataPointByteSize if hasattr(abf, 'dataPointByteSize') else None
    
    logger.debug(f"Stored extended metadata: {list(dataset.metadata.keys())}")


# Optional: Lazy loading implementation for future use
class LazyABFDataset:
    """
    Lazy-loading wrapper for ABF files (future implementation).
    
    This class would load sweeps on-demand rather than all at once,
    useful for very large files.
    """
    
    def __init__(self, file_path: Union[str, Path], channel_map: Optional[Any] = None):
        """Initialize lazy dataset (not yet implemented)."""
        raise NotImplementedError(
            "Lazy loading for ABF files is planned but not yet implemented. "
            "Use standard loading (lazy_load=False) for now."
        )


# Utility function for testing
def validate_abf_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate an ABF file and return basic information without full loading.
    
    Args:
        file_path: Path to ABF file
    
    Returns:
        Dictionary with file information
    
    Example:
        >>> info = validate_abf_file('recording.abf')
        >>> print(f"File has {info['sweep_count']} sweeps")
    """
    if not PYABF_AVAILABLE:
        raise ImportError("pyabf is required for ABF validation")
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {'valid': False, 'error': 'File not found'}
    
    try:
        abf = pyabf.ABF(str(file_path), loadData=False)  # Don't load data for validation
        
        return {
            'valid': True,
            'sweep_count': abf.sweepCount,
            'channel_count': abf.channelCount,
            'sampling_rate_hz': abf.sampleRate,
            'duration_seconds': abf.dataSecPerPoint * abf.dataPointCount,
            'abf_version': abf.abfVersionString,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'protocol': abf.protocol if abf.protocol else None,
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}