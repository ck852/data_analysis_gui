"""
Service for current density calculations and data preparation.

This module provides utility functions for current density analysis,
including summary export preparation and data transformations.

Author: Data Analysis GUI Contributors  
License: MIT
"""

from typing import Dict, List, Set, Any, Optional
import numpy as np

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class CurrentDensityService:
    """Service class for current density calculations and data preparation."""
    
    @staticmethod
    def prepare_summary_export(
        voltage_data: Dict[float, List[float]],
        file_mapping: Dict[str, str],
        cslow_mapping: Dict[str, float],
        selected_files: Set[str],
        y_unit: str = "pA/pF"
    ) -> Dict[str, Any]:
        """
        Prepare current density summary data for export.
        
        Args:
            voltage_data: Dictionary mapping voltages to lists of current density values
            file_mapping: Dictionary mapping recording IDs to file names
            cslow_mapping: Dictionary mapping file names to Cslow values
            selected_files: Set of selected file names to include
            y_unit: Unit for current density values
            
        Returns:
            Dictionary with 'headers', 'data', and 'format_spec' for CSV export
        """
        # Get sorted voltages
        voltages = sorted(voltage_data.keys())
        
        # Build headers
        headers = [f"Voltage (mV)"]
        data_columns = [voltages]
        
        # Sort recordings
        sorted_recordings = sorted(file_mapping.keys(), 
                                 key=lambda x: int(x.split()[-1]))
        
        # Add data for each file
        included_count = 0
        for recording_id in sorted_recordings:
            file_name = file_mapping.get(recording_id, recording_id)
            
            # Skip if not selected
            if selected_files and file_name not in selected_files:
                continue
            
            # Get Cslow value
            cslow = cslow_mapping.get(file_name, 0.0)
            if cslow <= 0:
                logger.warning(f"Skipping {file_name} - invalid Cslow value")
                continue
            
            # Add header with file name and Cslow
            headers.append(f"{file_name} ({cslow:.2f} pF)")
            
            # Extract current density values for this file
            cd_values = []
            recording_index = int(recording_id.split()[-1]) - 1
            
            for voltage in voltages:
                if recording_index < len(voltage_data[voltage]):
                    cd_values.append(voltage_data[voltage][recording_index])
                else:
                    cd_values.append(np.nan)
            
            data_columns.append(cd_values)
            included_count += 1
        
        # Convert to array format
        if included_count > 0:
            data_array = np.column_stack(data_columns)
        else:
            data_array = np.array([[]])
        
        logger.info(f"Prepared current density summary for {included_count} files")
        
        return {
            'headers': headers,
            'data': data_array,
            'format_spec': '%.6f'
        }
    
    @staticmethod
    def calculate_current_density(
        current_values: np.ndarray,
        cslow: float
    ) -> np.ndarray:
        """
        Calculate current density from current values and slow capacitance.
        
        Args:
            current_values: Array of current values in pA
            cslow: Slow capacitance in pF
            
        Returns:
            Array of current density values in pA/pF
            
        Raises:
            ValueError: If cslow is <= 0
        """
        if cslow <= 0:
            raise ValueError(f"Cslow must be positive, got {cslow}")
        
        return current_values / cslow
    
    @staticmethod
    def validate_cslow_values(
        cslow_mapping: Dict[str, float],
        file_names: Set[str]
    ) -> Dict[str, str]:
        """
        Validate Cslow values for a set of files.
        
        Args:
            cslow_mapping: Dictionary mapping file names to Cslow values
            file_names: Set of file names to validate
            
        Returns:
            Dictionary mapping file names to error messages (empty if all valid)
        """
        errors = {}
        
        for file_name in file_names:
            if file_name not in cslow_mapping:
                errors[file_name] = "Missing Cslow value"
                continue
            
            cslow = cslow_mapping[file_name]
            if not isinstance(cslow, (int, float)):
                errors[file_name] = "Cslow must be numeric"
            elif cslow <= 0:
                errors[file_name] = f"Cslow must be positive (got {cslow})"
            elif cslow > 10000:
                errors[file_name] = f"Cslow seems unreasonably large ({cslow} pF)"
        
        return errors