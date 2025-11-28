"""
Electrophysiology Dataset Abstraction module

This module is foundational for all data operations within the entire program! It provides a format-agnostic
container for electrophysiology data. Whether the input file is ABF, WCP, or another supported format, the 
ElectrophysiologyDataset object contains arrays of a single Voltage channel and a single Current channel, as well as the 
time array, for each sweep. This enables all downstream operations to proceed without concern for file format and 
facilitates expansion to additional formats in the future. Works in conjunction with DataExtractor to retrieve 
electrophysiology data from raw input files.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, Any, Union
import numpy as np

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ElectrophysiologyDataset:
    """
    Format-agnostic container for multi-sweep electrophysiology recordings.
    
    Each sweep contains a time vector (ms) and a 2D data matrix (samples × channels).
    Metadata tracks channel labels, units, sampling rate, and file provenance.
    """

    def __init__(self):

        self._sweeps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.metadata: Dict[str, Any] = {
            "channel_labels": [],
            "channel_units": [],
            "sampling_rate_hz": None,
            "format": None,
            "source_file": None,
            "channel_count": 0,
            "sweep_count": 0,
            "sweep_times": {},
        }
        logger.debug("Initialized empty ElectrophysiologyDataset")

    def add_sweep(
        self, sweep_index: str, time_ms: np.ndarray, data_matrix: np.ndarray
    ) -> None:
        """
        Add a sweep with its time vector and channel data.

        Args:
            sweep_index: Unique identifier for this sweep
            time_ms: 1D time array in milliseconds (length N)
            data_matrix: 2D array (N samples × C channels); 1D arrays auto-converted

        Raises:
            ValueError: If time and data sample counts don't match
        """
        # Validate inputs
        time_ms = np.asarray(time_ms)
        data_matrix = np.asarray(data_matrix)

        # Ensure data is 2D
        if data_matrix.ndim == 1:
            data_matrix = data_matrix.reshape(-1, 1)

        # Check dimensions match
        if len(time_ms) != data_matrix.shape[0]:
            error_msg = (
                f"Time vector length ({len(time_ms)}) doesn't match "
                f"data samples ({data_matrix.shape[0]})"
            )
            logger.error(f"Failed to add sweep {sweep_index}: {error_msg}")
            raise ValueError(error_msg)

        # Store the sweep
        self._sweeps[sweep_index] = (time_ms, data_matrix)

        # Update metadata
        self.metadata["sweep_count"] = len(self._sweeps)
        if data_matrix.shape[1] > self.metadata["channel_count"]:
            self.metadata["channel_count"] = data_matrix.shape[1]

        logger.debug(
            f"Added sweep {sweep_index}: {len(time_ms)} samples, "
            f"{data_matrix.shape[1]} channel(s)"
        )

    def sweeps(self) -> Iterable[str]:
        """Return iterable of sweep indices."""
        return self._sweeps.keys()

    def get_sweep(self, sweep_index: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve (time_ms, data_matrix) tuple for a sweep, or None if not found."""
        result = self._sweeps.get(sweep_index)
        if result is None:
            logger.warning(f"Sweep {sweep_index} not found in dataset")
        return result

    def get_channel_vector(
        self, sweep_index: str, channel_id: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract a single channel's time and data from a sweep.
        
        Returns (None, None) if sweep doesn't exist or channel_id is out of range.
        """
        sweep_data = self.get_sweep(sweep_index)
        if sweep_data is None:
            logger.warning(f"Cannot get channel {channel_id}: sweep {sweep_index} not found")
            return None, None

        time_ms, data_matrix = sweep_data

        # Check channel bounds
        if channel_id < 0 or channel_id >= data_matrix.shape[1]:
            logger.warning(
                f"Channel {channel_id} out of range for sweep {sweep_index} "
                f"(available channels: 0-{data_matrix.shape[1] - 1})"
            )
            return None, None

        # Extract specific channel
        channel_data = data_matrix[:, channel_id]

        return time_ms, channel_data

    def channel_count(self) -> int:
        """Maximum number of channels across all sweeps."""
        return self.metadata.get("channel_count", 0)

    def sweep_count(self) -> int:
        """Total number of sweeps in the dataset."""
        return len(self._sweeps)

    def is_empty(self) -> bool:
        """True if no sweeps loaded."""
        return len(self._sweeps) == 0

    def get_sweep_duration_ms(self, sweep_index: str) -> Optional[float]:

        sweep_data = self.get_sweep(sweep_index)
        if sweep_data is None:
            return None

        time_ms, _ = sweep_data
        if len(time_ms) < 2:
            return 0.0

        return float(time_ms[-1] - time_ms[0])

    def get_max_sweep_time(self) -> float:
        """Maximum sweep duration across all sweeps. Used for plot autoscaling."""
        if self.is_empty():
            return 0.0

        max_duration = 0.0
        for sweep_idx in self.sweeps():
            duration = self.get_sweep_duration_ms(sweep_idx)
            if duration is not None and duration > max_duration:
                max_duration = duration

        return max_duration

    def create_filtered_copy(
        self, keep_sweeps: Iterable[str], reset_time: bool = False
    ) -> "ElectrophysiologyDataset":
        """
        For removing rejected sweeps. Original dataset remains unchanged.
        If reset_time=True, sweep_times metadata is offset so the first kept sweep starts at t=0.
        Within-sweep time arrays are always preserved unchanged.
        """
        logger.info(f"Creating filtered dataset copy with {len(list(keep_sweeps))} sweeps")
        
        # Convert to list to allow multiple iterations
        keep_sweeps_list = list(keep_sweeps)
        
        if not keep_sweeps_list:
            raise ValueError("keep_sweeps cannot be empty")
        
        # Validate all sweeps exist
        for sweep_idx in keep_sweeps_list:
            if sweep_idx not in self._sweeps:
                raise ValueError(f"Sweep '{sweep_idx}' not found in dataset")
        
        # Create new dataset
        new_dataset = ElectrophysiologyDataset()
        
        # Calculate time offset for sweep_times if resetting
        time_offset_sec = 0.0
        if reset_time:
            sweep_times = self.metadata.get('sweep_times', {})
            if sweep_times and keep_sweeps_list:
                first_sweep_time = sweep_times.get(keep_sweeps_list[0], 0.0)
                time_offset_sec = first_sweep_time
                logger.info(f"Time reset enabled: offsetting by {time_offset_sec:.3f} seconds")
        
        # Copy sweeps to new dataset
        for sweep_idx in keep_sweeps_list:
            time_ms, data = self.get_sweep(sweep_idx)
            if time_ms is None or data is None:
                logger.warning(f"Skipping sweep {sweep_idx}: failed to retrieve data")
                continue
            # Add sweep with original time array (within-sweep timing unchanged)
            new_dataset.add_sweep(sweep_idx, time_ms.copy(), data.copy())
        
        # Copy metadata (deep copy for mutable nested structures)
        new_dataset.metadata = {
            'channel_labels': self.metadata.get('channel_labels', []).copy(),
            'channel_units': self.metadata.get('channel_units', []).copy(),
            'sampling_rate_hz': self.metadata.get('sampling_rate_hz'),
            'format': self.metadata.get('format'),
            'source_file': self.metadata.get('source_file'),
            'channel_count': self.metadata.get('channel_count', 0),
            'sweep_count': len(keep_sweeps_list),
        }
        
        # Copy channel_config if present
        if 'channel_config' in self.metadata:
            new_dataset.metadata['channel_config'] = self.metadata['channel_config'].copy()
        
        # Handle sweep_times with optional offset
        old_sweep_times = self.metadata.get('sweep_times', {})
        if old_sweep_times:
            new_sweep_times = {}
            for sweep_idx in keep_sweeps_list:
                if sweep_idx in old_sweep_times:
                    old_time = old_sweep_times[sweep_idx]
                    # Apply offset if resetting time
                    new_sweep_times[sweep_idx] = old_time - time_offset_sec
            new_dataset.metadata['sweep_times'] = new_sweep_times
        
        logger.info(
            f"Created filtered dataset: {new_dataset.sweep_count()} sweeps, "
            f"time_offset={time_offset_sec:.3f}s"
        )
        
        return new_dataset

    def get_sampling_rate(self, sweep_index: Optional[str] = None) -> Optional[float]:
        """
        Estimate sampling rate in Hz from sweep time vector or return metadata value.
        
        Calculates from mean time step if sweep data available, otherwise returns stored rate.
        """
        # Use provided sweep or get first one
        if sweep_index is None:
            if self.is_empty():
                return self.metadata.get("sampling_rate_hz")
            sweep_index = next(iter(self.sweeps()))

        sweep_data = self.get_sweep(sweep_index)
        if sweep_data is None:
            return self.metadata.get("sampling_rate_hz")

        time_ms, _ = sweep_data
        if len(time_ms) < 2:
            return self.metadata.get("sampling_rate_hz")

        # Calculate sampling rate from time vector
        dt_ms = np.mean(np.diff(time_ms))
        if dt_ms > 0:
            return 1000.0 / dt_ms  # Convert from ms to Hz

        return self.metadata.get("sampling_rate_hz")

    def clear(self) -> None:
        """Remove all sweeps and reset metadata to defaults."""
        sweep_count = len(self._sweeps)
        self._sweeps.clear()
        self.metadata = {
            "channel_labels": [],
            "channel_units": [],
            "sampling_rate_hz": None,
            "format": None,
            "source_file": None,
            "channel_count": 0,
            "sweep_count": 0,
        }
        logger.info(f"Cleared dataset: removed {sweep_count} sweep(s)")

    def __len__(self) -> int:
        """Number of sweeps (enables len(dataset))."""
        return len(self._sweeps)

    def __repr__(self) -> str:
        """String summary: sweeps, channels, and file format."""
        return (
            f"ElectrophysiologyDataset("
            f"sweeps={self.sweep_count()}, "
            f"channels={self.channel_count()}, "
            f"format={self.metadata.get('format', 'unknown')})"
        )

class DatasetLoader:
    """
    Loads electrophysiology data from ABF and WCP files into ElectrophysiologyDataset objects.
    
    Delegates to format-specific loaders in core/loaders/ based on file extension detection.
    """

    # Supported file extensions and their formats
    FORMAT_EXTENSIONS = {
        ".abf": "abf",  # Axon Binary Format
        ".wcp": "wcp",  # WinWCP format
    }

    @staticmethod
    def detect_format(file_path: Union[str, Path]) -> Optional[str]:
        """Detect file format from extension. Returns 'abf', 'wcp', or None if unknown."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        format_type = DatasetLoader.FORMAT_EXTENSIONS.get(extension)
        
        if format_type:
            logger.debug(f"Detected format '{format_type}' for file: {file_path.name}")
        else:
            logger.warning(f"Unknown file format for extension '{extension}': {file_path.name}")
        
        return format_type

    @staticmethod
    def load(filepath: str) -> "ElectrophysiologyDataset":
        """
        Load an electrophysiology file into a dataset object.
        
        Detects format from extension and delegates to the appropriate loader.
        """
        logger.info(f"Loading dataset from: {Path(filepath).name}")
        
        format_type = DatasetLoader.detect_format(filepath)

        try:
            if format_type == "wcp":
                from data_analysis_gui.core.loaders.wcp_loader import load_wcp
                dataset = load_wcp(filepath)
            elif format_type == "abf":
                from data_analysis_gui.core.loaders.abf_loader import load_abf
                dataset = load_abf(filepath)
            else:
                error_msg = f"Unsupported file format: {format_type}"
                logger.error(f"Load failed: {error_msg}")
                raise ValueError(error_msg)
            
            logger.info(
                f"Successfully loaded {dataset.sweep_count()} sweep(s), "
                f"{dataset.channel_count()} channel(s) from {format_type.upper()} file"
            )
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}", exc_info=True)
            raise