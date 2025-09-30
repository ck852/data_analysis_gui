"""
Electrophysiology Dataset Abstraction for PatchBatch Data Analysis Tool

This module defines a unified, format-agnostic data structure for managing electrophysiology recordings,
supporting multiple sweeps and channels. It provides a consistent interface for accessing, manipulating,
and annotating time-series data from various acquisition formats (e.g., Axon ABF, MATLAB .mat).

Features:
    - Stores sweeps as (time, data) pairs, with all time values in milliseconds.
    - Supports arbitrary numbers of sweeps and channels per dataset.
    - Metadata management for channel labels, units, sampling rate, and source file information.
    - Methods for sweep/channel access, duration calculation, and sampling rate estimation.
    - Format detection and loading utilities for supported file types.
    - Channel mapping and metadata updating for flexible labeling and unit assignment.

Typical Usage:
    >>> dataset = ElectrophysiologyDataset()
    >>> dataset.add_sweep("1", time_ms, data_matrix)
    >>> time, data = dataset.get_sweep("1")
    >>> time, voltage = dataset.get_channel_vector("1", 0)
    >>> print(dataset.metadata)

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, Any, Union
import numpy as np
import scipy.io


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
            "channel_labels": [],  # List of channel names
            "channel_units": [],  # List of unit strings
            "sampling_rate_hz": None,  # Sampling rate in Hz
            "format": None,  # Original file format
            "source_file": None,  # Path to source file
            "channel_count": 0,  # Number of channels
            "sweep_count": 0,  # Number of sweeps
            "sweep_times": {},  # Optional dict of sweep times (if available)
        }

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
            raise ValueError(
                f"Time vector length ({len(time_ms)}) doesn't match "
                f"data samples ({data_matrix.shape[0]})"
            )

        # Store the sweep
        self._sweeps[sweep_index] = (time_ms, data_matrix)

        # Update metadata
        self.metadata["sweep_count"] = len(self._sweeps)
        if data_matrix.shape[1] > self.metadata["channel_count"]:
            self.metadata["channel_count"] = data_matrix.shape[1]

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
        return self._sweeps.get(sweep_index)

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
            return None, None

        time_ms, data_matrix = sweep_data

        # Check channel bounds
        if channel_id < 0 or channel_id >= data_matrix.shape[1]:
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

    def update_channel_definitions(self, channel_map: Optional[Any]) -> None:
        """
        Update the dataset's channel metadata based on new channel definitions.

        This is used when channels are swapped or remapped after the dataset is loaded.

        Args:
            channel_map (Any): ChannelDefinitions instance with updated mapping.
        """
        if channel_map is None:
            return

        # Update channel labels based on new mapping
        num_channels = self.channel_count()
        labels = []
        units = []

        for ch_id in range(num_channels):
            # Try to get label from channel_map
            if hasattr(channel_map, "get_channel_label"):
                label = channel_map.get_channel_label(ch_id, include_units=False)
                labels.append(label)

                # Determine units based on type
                if hasattr(channel_map, "get_type_for_channel"):
                    ch_type = channel_map.get_type_for_channel(ch_id)
                    if ch_type == "voltage":
                        units.append("mV")
                    elif ch_type == "current":
                        units.append("pA")
                    else:
                        units.append("")
                else:
                    # Fallback: guess units from label
                    if "voltage" in label.lower():
                        units.append("mV")
                    elif "current" in label.lower():
                        units.append("pA")
                    else:
                        units.append("")
            else:
                # No channel map available
                labels.append(f"Channel {ch_id}")
                units.append("")

        # Update metadata
        self.metadata["channel_labels"] = labels
        self.metadata["channel_units"] = units


class DatasetLoader:
    """
    Static methods for loading electrophysiology data from various file formats.

    Provides format detection, loading, and channel mapping for supported file types used in electrophysiology recordings.
    """

    # Supported file extensions and their formats
    FORMAT_EXTENSIONS = {
        ".mat": "matlab",
        ".abf": "axon",  # Axon Binary Format
        ".wcp": "wcp",  # WinWCP format
        ".h5": "hdf5",  # HDF5 format (future)
        ".csv": "csv",  # CSV export (future)
        ".txt": "text",  # Text export (future)
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
        return DatasetLoader.FORMAT_EXTENSIONS.get(extension)

    @staticmethod
    def load(
        file_path: Union[str, Path], channel_map: Optional[Any] = None
    ) -> ElectrophysiologyDataset:
        """
        Load a dataset from file with automatic format detection and channel mapping.

        Args:
            file_path (Union[str, Path]): Path to the data file.
            channel_map (Any, optional): ChannelDefinitions instance for channel mapping.

        Returns:
            ElectrophysiologyDataset: Loaded dataset.

        Raises:
            ValueError: If file format is not supported.
            FileNotFoundError: If file does not exist.
            NotImplementedError: For recognized but unimplemented formats (e.g., '.csv', '.h5').
        """
        file_path = Path(file_path)

        # Check file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect format
        format_type = DatasetLoader.detect_format(file_path)

        if format_type == "matlab":
            return DatasetLoader.load_mat(file_path, channel_map)
        elif format_type == "axon":
            return DatasetLoader.load_abf(file_path, channel_map)
        elif format_type == "wcp":
            return DatasetLoader.load_wcp(file_path, channel_map)
        # elif format_type == "csv":
        #     # Future implementation
        #     raise NotImplementedError("CSV loading not yet implemented")
        # elif format_type == "hdf5":
        #     # Future implementation
        #     raise NotImplementedError("HDF5 loading not yet implemented")
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {list(DatasetLoader.FORMAT_EXTENSIONS.keys())}"
            )

    @staticmethod
    def load_abf(
        file_path: Union[str, Path], channel_map: Optional[Any] = None
    ) -> ElectrophysiologyDataset:
        """
        Load an Axon Binary Format (ABF) file containing electrophysiology data.

        Args:
            file_path (Union[str, Path]): Path to the ABF file.
            channel_map (Any, optional): ChannelDefinitions instance for channel mapping.

        Returns:
            ElectrophysiologyDataset: Loaded dataset.

        Raises:
            ImportError: If pyabf is not installed.
        """
        try:
            from data_analysis_gui.core.loaders.abf_loader import load_abf
        except ImportError as e:
            raise ImportError(
                "ABF support requires pyabf. Install with: pip install pyabf"
            ) from e

        return load_abf(file_path, channel_map)

    @staticmethod
    def load_mat(
        file_path: Union[str, Path], channel_map: Optional[Any] = None
    ) -> ElectrophysiologyDataset:
        """
        Load a MATLAB (.mat) file containing electrophysiology data.

        Expects MAT files with the structure:
            - T{n}: Time vectors for sweep n
            - Y{n}: Data matrices for sweep n

        Args:
            file_path (Union[str, Path]): Path to the MAT file.
            channel_map (Any, optional): ChannelDefinitions instance for channel labeling.

        Returns:
            ElectrophysiologyDataset: Dataset containing all sweeps from the MAT file.

        Raises:
            IOError: If file cannot be read.
            ValueError: If file structure is invalid.
        """
        file_path = Path(file_path)

        try:
            mat_data = scipy.io.loadmat(str(file_path))
        except Exception as e:
            raise IOError(f"Failed to load MAT file: {e}")

        # Create dataset
        dataset = ElectrophysiologyDataset()

        # Find all sweep pairs (T{n}, Y{n})
        sweep_indices = []
        for key in mat_data.keys():
            if key.startswith("T") and not key.startswith("__"):
                index = key[1:]
                if f"Y{index}" in mat_data:
                    sweep_indices.append(index)

        if not sweep_indices:
            raise ValueError(
                "No valid sweep data found in MAT file. "
                "Expected T{n} and Y{n} variable pairs."
            )

        # Sort sweep indices numerically if possible
        try:
            sweep_indices.sort(key=int)
        except ValueError:
            sweep_indices.sort()

        # Load each sweep
        for index in sweep_indices:
            time_key = f"T{index}"
            data_key = f"Y{index}"

            # Extract time vector (convert to milliseconds)
            time_s = mat_data[time_key].squeeze()
            time_ms = time_s * 1000.0

            # Extract data matrix
            data = mat_data[data_key]

            # Ensure data is 2D
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim > 2:
                # Squeeze out singleton dimensions
                data = np.squeeze(data)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)

            # Add sweep to dataset
            dataset.add_sweep(index, time_ms, data)

        # Set metadata
        dataset.metadata["format"] = "matlab"
        dataset.metadata["source_file"] = str(file_path)

        # Estimate sampling rate from first sweep
        first_index = sweep_indices[0]
        time_ms, _ = dataset.get_sweep(first_index)
        if len(time_ms) >= 2:
            dt_ms = np.mean(np.diff(time_ms))
            if dt_ms > 0:
                dataset.metadata["sampling_rate_hz"] = 1000.0 / dt_ms

        # Apply channel mapping if provided
        if channel_map is not None:
            DatasetLoader._apply_channel_mapping(dataset, channel_map)

        return dataset

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
        try:
            from data_analysis_gui.core.loaders.wcp_loader import load_wcp
        except ImportError as e:
            raise ImportError(
                "WCP loader not found. Ensure wcp_loader.py is in core/loaders/"
            ) from e
        
        return load_wcp(file_path, channel_map)

    @staticmethod
    def _apply_channel_mapping(
        dataset: ElectrophysiologyDataset, channel_map: Any
    ) -> None:
        """
        Apply channel definitions to dataset metadata, updating channel labels and units.

        Args:
            dataset (ElectrophysiologyDataset): Dataset to update.
            channel_map (Any): ChannelDefinitions instance.
        """
        # Update channel labels based on mapping
        num_channels = dataset.channel_count()
        labels = []
        units = []

        for ch_id in range(num_channels):
            # Try to get label from channel_map
            if hasattr(channel_map, "get_channel_label"):
                label = channel_map.get_channel_label(ch_id, include_units=False)
                labels.append(label)

                # Determine units based on type
                if hasattr(channel_map, "get_type_for_channel"):
                    ch_type = channel_map.get_type_for_channel(ch_id)
                    if ch_type == "voltage":
                        units.append("mV")
                    elif ch_type == "current":
                        units.append("pA")
                    else:
                        units.append("")
                else:
                    # Fallback: guess units from label
                    if "voltage" in label.lower():
                        units.append("mV")
                    elif "current" in label.lower():
                        units.append("pA")
                    else:
                        units.append("")
            else:
                # No channel map available
                labels.append(f"Channel {ch_id}")
                units.append("")

        dataset.metadata["channel_labels"] = labels
        dataset.metadata["channel_units"] = units
