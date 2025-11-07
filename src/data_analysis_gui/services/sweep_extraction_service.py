"""
Sweep Extraction Service for PatchBatch Electrophysiology Data Analysis Tool

This service provides centralized logic for extracting sweep data from datasets
and formatting it for export or clipboard operations.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import List, Tuple, Optional, Dict
import numpy as np

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.data_extractor import DataExtractor
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class SweepExtractionService:
    """
    Service for extracting sweep data from datasets.
    
    Provides methods to extract single or multiple sweeps with configurable
    channel selection and time ranges, formatted for export or clipboard.
    """
    
    def __init__(self):
        """Initialize the sweep extraction service."""
        self.data_extractor = DataExtractor()
    
    def extract_sweeps(
        self,
        dataset: ElectrophysiologyDataset,
        sweep_indices: List[str],
        channel_mode: str = 'both',
        time_range: Optional[Tuple[float, float]] = None,
        file_label: str = ""
    ) -> Dict:
        """
        Extract sweep data from a dataset.
        
        Args:
            dataset: Dataset to extract from
            sweep_indices: List of sweep indices to extract (e.g., ['0', '1', '2'])
            channel_mode: 'voltage', 'current', or 'both'
            time_range: Optional (start_ms, end_ms) tuple. If None, uses full trace.
            file_label: Optional label for column headers (useful in batch mode)
        
        Returns:
            Dict with keys:
                - 'headers': List of column headers
                - 'data': 2D numpy array with time + channel data
                - 'units': Dict with voltage_units and current_units
        
        Raises:
            ValueError: If no valid data can be extracted
        """
        # Get channel configuration for units
        channel_config = dataset.metadata.get('channel_config', {})
        voltage_units = channel_config.get('voltage_units', 'mV')
        current_units = channel_config.get('current_units', 'pA')
        
        # Determine time range
        if time_range is None:
            start_ms = 0.0
            end_ms = dataset.get_max_sweep_time()
        else:
            start_ms, end_ms = time_range
        
        # Extract data for all sweeps
        all_data = {}
        reference_time = None
        
        for sweep_idx in sweep_indices:
            try:
                # Extract sweep data
                sweep_data = self.data_extractor.extract_sweep_data(dataset, sweep_idx)
                time_ms = sweep_data['time_ms']
                voltage = sweep_data['voltage']
                current = sweep_data['current']
                
                # Apply time range filter
                mask = (time_ms >= start_ms) & (time_ms <= end_ms)
                filtered_time = time_ms[mask]
                filtered_voltage = voltage[mask]
                filtered_current = current[mask]
                
                # Use first sweep's time array as reference
                if reference_time is None:
                    reference_time = filtered_time
                
                # Store filtered data
                all_data[sweep_idx] = {
                    'time': filtered_time,
                    'voltage': filtered_voltage,
                    'current': filtered_current
                }
                
            except Exception as e:
                logger.warning(f"Could not extract sweep {sweep_idx}: {e}")
                # Store NaN arrays for failed sweeps
                if reference_time is not None:
                    all_data[sweep_idx] = {
                        'time': reference_time,
                        'voltage': np.full_like(reference_time, np.nan),
                        'current': np.full_like(reference_time, np.nan)
                    }
        
        if reference_time is None or len(reference_time) == 0:
            raise ValueError("No valid data extracted from selected sweeps")
        
        # Build output structure
        headers, data_array = self._build_arrays(
            all_data, sweep_indices, channel_mode, 
            reference_time, voltage_units, current_units, file_label
        )
        
        return {
            'headers': headers,
            'data': data_array,
            'units': {
                'voltage_units': voltage_units,
                'current_units': current_units
            }
        }
    
    def _build_arrays(
        self,
        all_data: Dict,
        sweep_indices: List[str],
        channel_mode: str,
        reference_time: np.ndarray,
        voltage_units: str,
        current_units: str,
        file_label: str = ""
    ) -> Tuple[List[str], np.ndarray]:
        """
        Build headers and data array for output.
        
        Args:
            all_data: Dictionary of extracted sweep data
            sweep_indices: List of sweep indices
            channel_mode: 'voltage', 'current', or 'both'
            reference_time: Time array to use
            voltage_units: Units for voltage channel
            current_units: Units for current channel
            file_label: Optional prefix for column names
        
        Returns:
            Tuple of (headers list, data array)
        """
        # Add file label prefix if provided
        prefix = f"{file_label} " if file_label else ""
        
        # Start with time column
        headers = ["Time (ms)"]
        columns = [reference_time]
        
        # Add data columns based on channel mode
        if channel_mode == 'voltage':
            for sweep_idx in sweep_indices:
                sweep_data = all_data.get(sweep_idx, {})
                voltage = sweep_data.get('voltage', np.full_like(reference_time, np.nan))
                headers.append(f"{prefix}Sweep {sweep_idx} Voltage ({voltage_units})")
                columns.append(voltage)
                
        elif channel_mode == 'current':
            for sweep_idx in sweep_indices:
                sweep_data = all_data.get(sweep_idx, {})
                current = sweep_data.get('current', np.full_like(reference_time, np.nan))
                headers.append(f"{prefix}Sweep {sweep_idx} Current ({current_units})")
                columns.append(current)
                
        else:  # both - group all voltage columns, then all current columns
            # First add all voltage columns
            for sweep_idx in sweep_indices:
                sweep_data = all_data.get(sweep_idx, {})
                voltage = sweep_data.get('voltage', np.full_like(reference_time, np.nan))
                headers.append(f"{prefix}Sweep {sweep_idx} Voltage ({voltage_units})")
                columns.append(voltage)
            
            # Then add all current columns
            for sweep_idx in sweep_indices:
                sweep_data = all_data.get(sweep_idx, {})
                current = sweep_data.get('current', np.full_like(reference_time, np.nan))
                headers.append(f"{prefix}Sweep {sweep_idx} Current ({current_units})")
                columns.append(current)
        
        # Combine into single array
        data_array = np.column_stack(columns)
        
        return headers, data_array