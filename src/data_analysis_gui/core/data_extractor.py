"""
Extracts and validates data from electrophysiology datasets.
PHASE 5: Fail-fast validation with proper error messages.
"""

from typing import Dict, Tuple, Optional
import numpy as np

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.exceptions import DataError, ValidationError, validate_not_none
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class DataExtractor:
    """
    Extracts time series data from datasets with validation.
    Handles channel mapping and data integrity checks.
    """
    
    def __init__(self, channel_definitions: ChannelDefinitions):
        """
        Initialize with channel configuration.
        
        Args:
            channel_definitions: Channel mapping configuration
        """
        self.channel_definitions = channel_definitions
    
    def extract_sweep_data(
        self, 
        dataset: ElectrophysiologyDataset,
        sweep_index: str
    ) -> Dict[str, np.ndarray]:
        """
        Extract time series data for a sweep.
        
        Args:
            dataset: Dataset to extract from
            sweep_index: Sweep identifier
            
        Returns:
            Dict with 'time_ms', 'voltage', 'current' arrays
            
        Raises:
            ValidationError: If inputs invalid
            DataError: If sweep not found or data corrupted
        """
        validate_not_none(dataset, "dataset")
        validate_not_none(sweep_index, "sweep_index")
        
        if sweep_index not in dataset.sweeps():
            raise DataError(
                f"Sweep '{sweep_index}' not found",
                details={'available_sweeps': dataset.sweeps()[:10]}
            )
        
        # Get channel IDs
        voltage_ch = self.channel_definitions.get_voltage_channel()
        current_ch = self.channel_definitions.get_current_channel()
        
        # Extract data
        time_ms, voltage = dataset.get_channel_vector(sweep_index, voltage_ch)
        _, current = dataset.get_channel_vector(sweep_index, current_ch)
        
        if time_ms is None or voltage is None or current is None:
            raise DataError(
                f"Failed to extract data for sweep '{sweep_index}'",
                details={
                    'sweep': sweep_index,
                    'voltage_channel': voltage_ch,
                    'current_channel': current_ch
                }
            )
        
        # Log warnings for NaN but don't fail
        if np.any(np.isnan(time_ms)):
            raise DataError(f"Time array contains NaN for sweep {sweep_index}")
        
        if np.any(np.isnan(voltage)):
            logger.warning(f"Voltage contains NaN for sweep {sweep_index}")
        
        if np.any(np.isnan(current)):
            logger.warning(f"Current contains NaN for sweep {sweep_index}")
        
        return {
            'time_ms': time_ms,
            'voltage': voltage,
            'current': current
        }
    
    def extract_channel_for_plot(
        self,
        dataset: ElectrophysiologyDataset,
        sweep_index: str,
        channel_type: str
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Extract single channel data formatted for plotting.
        
        Args:
            dataset: Dataset to extract from
            sweep_index: Sweep identifier
            channel_type: "Voltage" or "Current"
            
        Returns:
            Tuple of (time_ms, data_matrix, channel_id)
            
        Raises:
            ValidationError: If channel_type invalid
            DataError: If extraction fails
        """
        if channel_type not in ["Voltage", "Current"]:
            raise ValidationError(
                f"Invalid channel_type: '{channel_type}'",
                details={'valid_types': ["Voltage", "Current"]}
            )
        
        # Get channel ID
        if channel_type == "Voltage":
            channel_id = self.channel_definitions.get_voltage_channel()
        else:
            channel_id = self.channel_definitions.get_current_channel()
        
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