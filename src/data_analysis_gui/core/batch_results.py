# src/data_analysis_gui/core/batch_results.py
"""
Core batch results functionality that can be used independently of GUI.
This module provides all the data processing and export logic for batch analysis results
without any GUI dependencies.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BatchResultsData:
    """Data structure for batch analysis results"""
    batch_data: Dict[str, Dict[str, Any]]  # File name -> data dict
    iv_data: Optional[Dict[float, List[float]]] = None  # Note the type hint fix
    iv_file_mapping: Optional[Dict[str, str]] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    destination_folder: Optional[str] = None
    
    # Track which files are included/visible
    included_files: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize included files if not provided"""
        if not self.included_files:
            self.included_files = set(self.batch_data.keys())
    
    def get_included_data(self) -> Dict[str, Dict[str, Any]]:
        """Get only the data for included files"""
        return {
            file_name: data 
            for file_name, data in self.batch_data.items()
            if file_name in self.included_files
        }
    
    def toggle_file_inclusion(self, file_name: str, include: bool) -> None:
        """Toggle whether a file is included in the results"""
        if include:
            self.included_files.add(file_name)
        else:
            self.included_files.discard(file_name)


class BatchResultsExporter:
    """Handles export operations for batch results"""
    
    def __init__(self, results_data: BatchResultsData):
        self.results_data = results_data
    
    def prepare_current_density_data(self) -> Dict[str, Any]:
        """
        Prepare data for current density analysis.
        
        Returns:
            Dictionary containing IV data and file mappings (all files, not just included).
        """
        if not self.results_data.iv_data:
            return {}
        
        # Return ALL data, not filtered - let the dialog handle inclusion state
        return {
            'iv_data': self.results_data.iv_data,
            'iv_file_mapping': self.results_data.iv_file_mapping,
            'destination_folder': self.results_data.destination_folder
        }
    
    def export_all_data_to_csv(self, output_path: str, format_spec: str = '%.6f') -> None:
        """
        Export all included batch data to a single CSV file.
        
        Args:
            output_path: Path for the output CSV file
            format_spec: Format specification for numbers
            
        Raises:
            ValueError: If no data to export or data inconsistency
        """
        included_data = self.results_data.get_included_data()
        
        if not included_data:
            raise ValueError("No data to export")
        
        # Determine common x values (assuming all files have same x values)
        first_file = next(iter(included_data.values()))
        x_values = np.array(first_file['x_values'])
        
        # Build output array
        output_columns = [x_values]
        
        for file_name in sorted(included_data.keys()):
            data = included_data[file_name]
            y_values = np.array(data['y_values'])
            
            # Validate that x values match
            if not np.array_equal(np.array(data['x_values']), x_values):
                raise ValueError(f"X values mismatch for file {file_name}")
            
            output_columns.append(y_values)
            
            # Add second y values if present (dual range)
            if 'y_values2' in data and len(data['y_values2']) > 0:
                output_columns.append(np.array(data['y_values2']))
        
        # Stack columns
        output_data = np.column_stack(output_columns)
        
        # Create header
        header_parts = [self.results_data.x_label or "X"]
        for file_name in sorted(included_data.keys()):
            header_parts.append(file_name)
            data = included_data[file_name]
            if 'y_values2' in data and len(data['y_values2']) > 0:
                header_parts.append(f"{file_name}_R2")
        
        header = ",".join(header_parts)
        
        # Export
        np.savetxt(output_path, output_data, delimiter=',', fmt=format_spec,
                   header=header, comments='')
    
    def export_individual_files(self, output_folder: str) -> List[Tuple[str, bool, str]]:
        """
        Export each included file to a separate CSV.
        
        Args:
            output_folder: Folder to save individual files
            
        Returns:
            List of tuples (filename, success, message)
        """
        results = []
        included_data = self.results_data.get_included_data()
        
        for file_name in sorted(included_data.keys()):
            try:
                data = included_data[file_name]
                output_path = os.path.join(output_folder, f"{file_name}.csv")
                
                # Prepare data
                x_values = np.array(data['x_values'])
                y_values = np.array(data['y_values'])
                
                if 'y_values2' in data and len(data['y_values2']) > 0:
                    y_values2 = np.array(data['y_values2'])
                    output_data = np.column_stack((x_values, y_values, y_values2))
                    header = f"{self.results_data.x_label},{self.results_data.y_label},Range2"
                else:
                    output_data = np.column_stack((x_values, y_values))
                    header = f"{self.results_data.x_label},{self.results_data.y_label}"
                
                # Export
                np.savetxt(output_path, output_data, delimiter=',', fmt='%.6f',
                           header=header, comments='')
                
                results.append((file_name, True, f"Saved to {output_path}"))
                
            except Exception as e:
                results.append((file_name, False, str(e)))
        
        return results


class BatchResultsAnalyzer:
    """Analyzes batch results for statistics and insights"""
    
    def calculate_statistics(self, results_data: BatchResultsData) -> Dict[str, float]:
        """
        Calculate statistics for included files.
        
        Args:
            results_data: Batch results data
            
        Returns:
            Dictionary with statistics (mean, std, min, max, etc.)
        """
        included_data = results_data.get_included_data()
        
        if not included_data:
            return {}
        
        # Collect all y values
        all_y_values = []
        all_x_values = []
        
        for data in included_data.values():
            all_y_values.extend(data['y_values'])
            all_x_values.extend(data['x_values'])
            
            # Include second range if present
            if 'y_values2' in data and len(data['y_values2']) > 0:
                all_y_values.extend(data['y_values2'])
        
        if not all_y_values:
            return {}
        
        y_array = np.array(all_y_values)
        x_array = np.array(all_x_values)
        
        return {
            'num_files': len(included_data),
            'y_mean': np.mean(y_array),
            'y_std': np.std(y_array),
            'y_min': np.min(y_array),
            'y_max': np.max(y_array),
            'y_median': np.median(y_array),
            'x_range': np.ptp(x_array) if len(x_array) > 0 else 0
        }


# Convenience functions for non-GUI usage
def export_batch_results(batch_data: Dict, output_path: str, 
                        included_files: Optional[Set[str]] = None) -> None:
    """
    Export batch results to CSV.
    
    Args:
        batch_data: Dictionary of file data
        output_path: Output CSV path
        included_files: Optional set of files to include
    """
    results_data = BatchResultsData(
        batch_data=batch_data,
        included_files=included_files
    )
    exporter = BatchResultsExporter(results_data)
    exporter.export_all_data_to_csv(output_path)


def analyze_batch_results(batch_data: Dict, 
                         included_files: Optional[Set[str]] = None) -> Dict[str, float]:
    """
    Get statistics for batch results.
    
    Args:
        batch_data: Dictionary of file data
        included_files: Optional set of files to include
        
    Returns:
        Dictionary of statistics
    """
    results_data = BatchResultsData(
        batch_data=batch_data,
        included_files=included_files
    )
    analyzer = BatchResultsAnalyzer()
    return analyzer.calculate_statistics(results_data)