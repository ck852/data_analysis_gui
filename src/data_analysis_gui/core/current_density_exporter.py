# src/data_analysis_gui/core/current_density_exporter.py
"""
Core exporter for Current Density analysis data.
This module provides a GUI-independent exporter for preparing
current density data for various export formats.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

from data_analysis_gui.utils.data_processing import calculate_current_density


class CurrentDensityExporter:
    """Handles data preparation for exporting current density results."""

    def __init__(self, file_data: Dict[str, Any], iv_file_mapping: Dict[str, str], included_files: List[str]):
        """
        Initializes the exporter with the necessary data.

        Args:
            file_data: Dictionary containing the raw data for each file.
            iv_file_mapping: Mapping from file ID (e.g., "Recording 1") to the actual filename.
            included_files: A list of file IDs that are selected for export.
        """
        self.file_data = file_data
        self.iv_file_mapping = iv_file_mapping
        self.included_files = included_files

    def prepare_individual_files_data(self) -> List[Dict[str, Any]]:
        """
        Prepares data for exporting each included file to a separate CSV.

        Returns:
            A list of dictionaries, where each dictionary contains the
            'filename', 'data', and 'headers' for a single file.
        """
        files_data = []
        for file_id in self.included_files:
            file_info = self.file_data[file_id]
            cslow = file_info['cslow']
            if cslow <= 0:
                continue

            file_basename = self.iv_file_mapping.get(file_id, file_id.replace(" ", "_"))
            
            voltages = []
            current_densities = []
            for voltage, current in file_info['data'].items():
                voltages.append(voltage)
                current_densities.append(calculate_current_density(current, cslow))
            
            sorted_indices = np.argsort(voltages)
            sorted_voltages = np.array(voltages)[sorted_indices]
            sorted_currents = np.array(current_densities)[sorted_indices]
            
            files_data.append({
                'filename': f"{file_basename}_CD.csv",
                'data': np.column_stack((sorted_voltages, sorted_currents)),
                'headers': ['Voltage (mV)', 'Current Density (pA/pF)', f'Cslow = {cslow:.2f} pF']
            })
        return files_data

    def prepare_summary_data(self) -> Dict[str, Any]:
        """
        Prepares a summary of all included files for export to a single CSV.

        Returns:
            A dictionary containing the combined 'data' and 'headers' for the summary file.
        """
        if not self.included_files:
            return {}

        first_file_id = self.included_files[0]
        voltages = sorted(self.file_data[first_file_id]['data'].keys())
        
        headers = ["Voltage (mV)"]
        data_to_export = [voltages]
        
        for file_id in self.included_files:
            file_info = self.file_data[file_id]
            cslow = file_info['cslow']
            raw_data = file_info['data']
            
            headers.append(self.iv_file_mapping.get(file_id, file_id))
            
            current_densities = [
                calculate_current_density(raw_data.get(v, np.nan), cslow) 
                for v in voltages
            ]
            data_to_export.append(current_densities)
        
        return {
            'data': np.array(data_to_export).T,
            'headers': headers
        }