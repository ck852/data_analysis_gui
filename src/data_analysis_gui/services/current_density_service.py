"""
Service for current density calculations.

This module provides simple calculations to convert current measurements
to current density using slow capacitance values.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, List, Any, Optional
import numpy as np

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class CurrentDensityService:
    """
    Simple service for current density calculations.
    
    Current density = Current (pA) / Slow Capacitance (pF)
    Result is in pA/pF
    """
    
    @staticmethod
    def calculate_current_density(current_pa: np.ndarray, 
                                cslow_pf: float) -> np.ndarray:
        """
        Calculate current density from current and slow capacitance.
        
        Args:
            current_pa: Current values in picoamperes
            cslow_pf: Slow capacitance in picofarads
            
        Returns:
            Current density in pA/pF
            
        Raises:
            ValueError: If cslow_pf is not positive
        """
        if cslow_pf <= 0:
            raise ValueError(f"Slow capacitance must be positive, got {cslow_pf}")
        
        return current_pa / cslow_pf
    
    @staticmethod
    def prepare_export_data(voltages: np.ndarray,
                          current_density: np.ndarray,
                          cslow_pf: float,
                          y_unit: str = "pA/pF") -> Dict[str, Any]:
        """
        Prepare data for CSV export.
        
        Args:
            voltages: Voltage values in mV
            current_density: Current density values
            cslow_pf: Slow capacitance used
            y_unit: Unit for current density
            
        Returns:
            Dictionary with headers, data, and format spec for export
        """
        headers = [
            "Voltage (mV)",
            "Cslow (pF)",
            f"Current Density ({y_unit})"
        ]
        
        # Create constant cslow column
        cslow_column = np.full_like(voltages, cslow_pf)
        
        # Stack columns
        data = np.column_stack([voltages, cslow_column, current_density])
        
        return {
            'headers': headers,
            'data': data,
            'format_spec': '%.6f'
        }
    
    @staticmethod
    def prepare_summary_export(voltage_data: Dict[float, List[float]],
                             file_mapping: Dict[str, str],
                             cslow_mapping: Dict[str, float],
                             included_files: set,
                             y_unit: str = "pA/pF") -> Dict[str, Any]:
        """
        Prepare summary data for CSV export.
        
        Args:
            voltage_data: Dict mapping voltages to lists of current density values
            file_mapping: Recording ID to filename mapping
            cslow_mapping: Filename to Cslow mapping
            included_files: Set of filenames to include
            y_unit: Unit for current density
            
        Returns:
            Dictionary with headers, data, and format spec for export
        """
        # Get sorted voltages
        voltages = sorted(voltage_data.keys())
        
        # Build headers
        headers = ["Voltage (mV)"]
        data_columns = [voltages]
        
        # Sort recordings
        sorted_recordings = sorted(file_mapping.keys(), 
                                 key=lambda x: int(x.split()[-1]))
        
        for recording_id in sorted_recordings:
            base_name = file_mapping.get(recording_id, recording_id)
            
            # Skip if not included
            if base_name not in included_files:
                continue
            
            # Get Cslow value
            cslow = cslow_mapping.get(base_name, np.nan)
            if np.isnan(cslow):
                continue
            
            # Add header with Cslow info
            headers.append(f"{base_name} Cslow={cslow:.1f}pF")
            
            # Extract current density values
            recording_index = int(recording_id.split()[-1]) - 1
            cd_values = []
            
            for voltage in voltages:
                if recording_index < len(voltage_data[voltage]):
                    cd_values.append(voltage_data[voltage][recording_index])
                else:
                    cd_values.append(np.nan)
            
            data_columns.append(cd_values)
        
        # Convert to array
        data_array = np.column_stack(data_columns)
        
        return {
            'headers': headers,
            'data': data_array,
            'format_spec': '%.6f'
        }