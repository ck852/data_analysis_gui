"""
Electrophysiology Dataset Abstraction for PatchBatch Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, Any, Union
import numpy as np
import scipy.io

from data_analysis_gui.config.logging import get_logger, log_performance

logger = get_logger(__name__)


class ElectrophysiologyDataset:
    """
    Container for electrophysiology data with multiple sweeps and channels.

    Provides a format-agnostic, unified interface for managing and accessing electrophysiology recordings.
    All time values are stored in milliseconds. Supports sweep and channel operations, metadata management,
    and compatibility with various file formats.

    Attributes:
        _sweeps (Dict[str, Tuple[np.ndarray, np.ndarray]]): Maps sweep indices to (time, data) tuples.
        metadata (Dict[str, Any]): Stores dataset metadata such as channel labels, units, sampling rate, format, etc.

    Example:
        >>> dataset = ElectrophysiologyDataset()
        >>> dataset.add_sweep("1", time_ms, data_matrix)
        >>> time, data = dataset.get_sweep("1")
        >>> time, voltage = dataset.get_channel_vector("1", 0)
    """

    def __init__(self):
        """
        Initialize an empty ElectrophysiologyDataset.

        All metadata fields are set to default values.
        """
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
        Add a sweep to the dataset.

        Args:
            sweep_index (str): Unique identifier for the sweep.
            time_ms (np.ndarray): 1D array of time values in milliseconds (length N).
            data_matrix (np.ndarray): 2D array of shape (N, C) where N=samples, C=channels.

        Raises:
            ValueError: If time and data dimensions do not match.
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
        """
        Get an iterable of all sweep indices in the dataset.

        Returns:
            Iterable[str]: Iterable of sweep index strings.

        Example:
            >>> for sweep_idx in dataset.sweeps():
            ...     time, data = dataset.get_sweep(sweep_idx)
        """
        return self._sweeps.keys()

    def get_sweep(self, sweep_index: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve time and data for a specific sweep.

        Args:
            sweep_index (str): The sweep identifier.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: Tuple of (time_ms, data_matrix) if sweep exists,
            otherwise None. time_ms has shape (N,), data_matrix has shape (N, C).
        """
        result = self._sweeps.get(sweep_index)
        if result is None:
            logger.warning(f"Sweep {sweep_index} not found in dataset")
        return result

    def get_channel_vector(
        self, sweep_index: str, channel_id: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get time and data for a specific channel in a sweep.

        Args:
            sweep_index (str): The sweep identifier.
            channel_id (int): The channel index (0-based).

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Tuple of (time_ms, channel_data) as 1D arrays,
            or (None, None) if sweep does not exist or channel is out of range.

        Example:
            >>> time, voltage = dataset.get_channel_vector("1", 0)
            >>> time, current = dataset.get_channel_vector("1", 1)
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
        """
        Get the maximum number of channels across all sweeps in the dataset.

        Returns:
            int: Maximum channel count in the dataset.
        """
        return self.metadata.get("channel_count", 0)

    def sweep_count(self) -> int:
        """
        Get the total number of sweeps in the dataset.

        Returns:
            int: Number of sweeps in the dataset.
        """
        return len(self._sweeps)

    def is_empty(self) -> bool:
        """
        Check if the dataset contains any sweeps.

        Returns:
            bool: True if no sweeps are loaded, False otherwise.
        """
        return len(self._sweeps) == 0

    def get_sweep_duration_ms(self, sweep_index: str) -> Optional[float]:
        """
        Get the duration of a specific sweep in milliseconds.

        Args:
            sweep_index (str): The sweep identifier.

        Returns:
            Optional[float]: Duration in milliseconds, or None if sweep doesn't exist.
            Returns 0.0 if the sweep exists but contains fewer than two samples.
        """
        sweep_data = self.get_sweep(sweep_index)
        if sweep_data is None:
            return None

        time_ms, _ = sweep_data
        if len(time_ms) < 2:
            return 0.0

        return float(time_ms[-1] - time_ms[0])

    def get_max_sweep_time(self) -> float:
        """
        Get the maximum sweep duration across all sweeps in the dataset.

        Returns:
            float: Maximum sweep time in milliseconds, or 0.0 if no sweeps.
        """
        if self.is_empty():
            return 0.0

        max_duration = 0.0
        for sweep_idx in self.sweeps():
            duration = self.get_sweep_duration_ms(sweep_idx)
            if duration is not None and duration > max_duration:
                max_duration = duration

        return max_duration

    def get_sampling_rate(self, sweep_index: Optional[str] = None) -> Optional[float]:
        """
        Estimate the sampling rate for a sweep or the dataset.

        Args:
            sweep_index (str, optional): Specific sweep to check, or None for first sweep.

        Returns:
            Optional[float]: Sampling rate in Hz, or None if cannot be determined.
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
        """
        Remove all sweeps and reset metadata to default values.
        """
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
        """
        Return the number of sweeps in the dataset.

        Returns:
            int: Number of sweeps.
        """
        return len(self._sweeps)

    def __repr__(self) -> str:
        """
        Return a string representation of the dataset summarizing sweeps, channels, and format.

        Returns:
            str: String summary of the dataset.
        """
        return (
            f"ElectrophysiologyDataset("
            f"sweeps={self.sweep_count()}, "
            f"channels={self.channel_count()}, "
            f"format={self.metadata.get('format', 'unknown')})"
        )

class DatasetLoader:
    """
    Static methods for loading electrophysiology data from various file formats.

    Provides format detection, loading, and channel mapping for supported file types used in electrophysiology recordings.
    """

    # Supported file extensions and their formats
    FORMAT_EXTENSIONS = {
        ".abf": "abf",  # Axon Binary Format
        ".wcp": "wcp",  # WinWCP format
    }

    @staticmethod
    def detect_format(file_path: Union[str, Path]) -> Optional[str]:
        """
        Detect the file format based on file extension.

        Args:
            file_path (Union[str, Path]): Path to the file.

        Returns:
            Optional[str]: Format string (e.g., 'matlab', 'axon'), or None if unknown.
        """
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
        Load a dataset from any supported file format.
        
        Channel configuration is automatically detected from file metadata.

        Args:
            filepath: Path to the data file

        Returns:
            ElectrophysiologyDataset with loaded data and auto-detected
            channel configuration stored in metadata['channel_config']

        Raises:
            ValueError: If file format is not supported
        """
        logger.info(f"Loading dataset from: {Path(filepath).name}")
        
        format_type = DatasetLoader.detect_format(filepath)

        try:
            with log_performance(logger, f"load {format_type} file"):
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

    @staticmethod
    def load_wcp(file_path: Union[str, Path], channel_map: Optional[Any] = None) -> ElectrophysiologyDataset:
        """
        Load a WCP (WinWCP) file containing electrophysiology data.
        
        Args:
            file_path (Union[str, Path]): Path to the WCP file.
            channel_map (Any, optional): ChannelDefinitions instance for channel mapping.
        
        Returns:
            ElectrophysiologyDataset: Loaded dataset with actual sweep times.
        """
        logger.info(f"Loading WCP file: {Path(file_path).name}")
        
        try:
            from data_analysis_gui.core.loaders.wcp_loader import load_wcp
        except ImportError as e:
            logger.error("WCP loader module not found", exc_info=True)
            raise ImportError(
                "WCP loader not found. Ensure wcp_loader.py is in core/loaders/"
            ) from e
        
        return load_wcp(file_path, channel_map)