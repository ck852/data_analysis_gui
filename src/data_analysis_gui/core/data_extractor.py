"""
Electrophysiology Data Extraction Utilities

This module provides the DataExtractor class, which is responsible for extracting and validating the numbered channels from ElectrophysiologyDataset
and adding semantic meaning (voltage, current) based on channel configuration metadata. It also formats data outputs for analysis (MetricsCalculator)
and plotting (PlotManager).

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import Dict, Tuple
import numpy as np

from data_analysis_gui.core.dataset import ElectrophysiologyDataset

from data_analysis_gui.config.exceptions import DataError, ValidationError

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class DataExtractor:
    """
    Extracts and validates time series data from electrophysiology datasets.

    Stateless class that provides methods to extract sweep and channel data, ensuring proper channel mapping,
    data integrity, and compatibility with downstream analysis and plotting tools.
    """

    def extract_sweep_data(
            self, dataset: ElectrophysiologyDataset, sweep_index: str
        ) -> Dict[str, np.ndarray]:
            
            if dataset is None:
                raise ValidationError("Dataset cannot be None")
            if sweep_index is None:
                raise ValidationError("Sweep index cannot be None")

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
