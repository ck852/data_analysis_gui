"""
Electrophysiology Data Extraction Utilities for PatchBatch

This module provides functions and classes for extracting, validating, and formatting time-series data
from electrophysiology datasets. It ensures correct channel mapping, data integrity, and compatibility
with downstream analysis and visualization tools.

Features:
    - Extraction of sweep and channel data (voltage/current) from unified dataset objects.
    - Validation of input arguments and data arrays, with informative error handling.
    - Logging of data quality issues (e.g., NaN values) for traceability.
    - Formatting of extracted data for plotting and analysis workflows.

Typical Usage:
    >>> extractor = DataExtractor()
    >>> sweep_data = extractor.extract_sweep_data(dataset, "1")
    >>> time_ms, data_matrix, channel_id = extractor.extract_channel_for_plot(dataset, "1", "Voltage")

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import Dict, Tuple
import numpy as np

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.exceptions import (
    DataError,
    ValidationError,
    validate_not_none,
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class DataExtractor:
    """
    Extracts and validates time series data from electrophysiology datasets.

    This class provides methods to extract sweep and channel data, ensuring proper channel mapping,
    data integrity, and compatibility with downstream analysis and plotting tools.
    """

    def __init__(self):
        """
        Initialize the DataExtractor.
        
        Channel configuration is read from dataset metadata during extraction.
        """
        pass

    def extract_sweep_data(
            self, dataset: ElectrophysiologyDataset, sweep_index: str
        ) -> Dict[str, np.ndarray]:
            """
            Extract time series data for a specific sweep, including voltage and current channels.

            Args:
                dataset (ElectrophysiologyDataset): Dataset object containing sweeps and channel data.
                sweep_index (str): Identifier for the sweep to extract.

            Returns:
                Dict[str, np.ndarray]: Dictionary with keys 'time_ms', 'voltage', and 'current', each containing a numpy array.

            Raises:
                ValidationError: If required arguments are None or invalid.
                DataError: If the sweep is not found, data is missing, channel config is missing, or time array contains NaN values.
            """
            validate_not_none(dataset, "dataset")
            validate_not_none(sweep_index, "sweep_index")

            if sweep_index not in dataset.sweeps():
                raise DataError(
                    f"Sweep '{sweep_index}' not found",
                    details={"available_sweeps": dataset.sweeps()[:10]},
                )

            # Get channel configuration from dataset metadata (set by loader)
            channel_config = dataset.metadata.get('channel_config')
            if not channel_config:
                raise DataError(
                    "No channel configuration found in dataset. "
                    "File may be corrupted or incompletely loaded."
                )
            
            voltage_ch = channel_config['voltage_channel']
            current_ch = channel_config['current_channel']

            # Extract data
            time_ms, voltage = dataset.get_channel_vector(sweep_index, voltage_ch)
            _, current = dataset.get_channel_vector(sweep_index, current_ch)

            if time_ms is None or voltage is None or current is None:
                raise DataError(
                    f"Failed to extract data for sweep '{sweep_index}'",
                    details={
                        "sweep": sweep_index,
                        "voltage_channel": voltage_ch,
                        "current_channel": current_ch,
                    },
                )

            # Log warnings for NaN but don't fail for voltage/current
            if np.any(np.isnan(time_ms)):
                raise DataError(f"Time array contains NaN for sweep {sweep_index}")

            if np.any(np.isnan(voltage)):
                logger.warning(f"Voltage contains NaN for sweep {sweep_index}")

            if np.any(np.isnan(current)):
                logger.warning(f"Current contains NaN for sweep {sweep_index}")

            return {"time_ms": time_ms, "voltage": voltage, "current": current}

    def extract_channel_for_plot(
            self, dataset: ElectrophysiologyDataset, sweep_index: str, channel_type: str
        ) -> Tuple[np.ndarray, np.ndarray, int]:
            """
            Extract data for a single channel (voltage or current) and format for plotting.

            Args:
                dataset (ElectrophysiologyDataset): Dataset object containing sweeps and channel data.
                sweep_index (str): Identifier for the sweep to extract.
                channel_type (str): Type of channel to extract; must be "Voltage" or "Current".

            Returns:
                Tuple[np.ndarray, np.ndarray, int]:
                    - time_ms: 1D numpy array of time values in milliseconds
                    - data_matrix: 2D numpy array (shape: [time, channels]) with extracted channel data populated in the correct column
                    - channel_id: Integer channel index used for extraction

            Raises:
                ValidationError: If channel_type is not "Voltage" or "Current".
                DataError: If extraction fails, channel config is missing, or channel_id is out of bounds.
            """
            if channel_type not in ["Voltage", "Current"]:
                raise ValidationError(
                    f"Invalid channel_type: '{channel_type}'",
                    details={"valid_types": ["Voltage", "Current"]},
                )

            # Get channel configuration from dataset metadata (set by loader)
            channel_config = dataset.metadata.get('channel_config')
            if not channel_config:
                raise DataError(
                    "No channel configuration found in dataset. "
                    "File may be corrupted or incompletely loaded."
                )
            
            if channel_type == "Voltage":
                channel_id = channel_config['voltage_channel']
            else:
                channel_id = channel_config['current_channel']

            # Get raw data
            time_ms, channel_data = dataset.get_channel_vector(sweep_index, channel_id)

            if time_ms is None or channel_data is None:
                raise DataError(
                    f"No data for sweep '{sweep_index}' channel '{channel_type}'"
                )

            # Create 2D matrix for plot manager compatibility
            num_channels = dataset.channel_count()
            data_matrix = np.zeros((len(time_ms), num_channels))

            if channel_id >= num_channels:
                raise DataError(f"Channel ID {channel_id} out of bounds")

            data_matrix[:, channel_id] = channel_data

            return time_ms, data_matrix, channel_id
