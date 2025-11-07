"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Generalized summary export service for batch analysis results.

This module provides export functionality for any analysis parameter combination,
using a two-column-per-file format (X, Y) with blank spacing between files.
Each file's data is independent, allowing for different numbers of data points
across files without requiring NaN padding.

Classes:
    - GeneralizedSummaryExporter: Prepares summary tables for any X vs Y analysis

The format supports time-course analyses, custom parameter combinations, and any
analysis where aggregating results across multiple files is meaningful.
"""

from typing import Dict, Any, Optional
import numpy as np

from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class GeneralizedSummaryExporter:
    """
    Handles exporting generalized summary data for any analysis parameter combination.
    
    Creates a two-column-per-file format: [File1_X] [File1_Y] [blank] [File2_X] [File2_Y] ...
    Each file's columns are independent, naturally handling different data lengths.
    """
    
    @staticmethod
    def _get_axis_label(axis_config: AxisConfig, params: AnalysisParameters, current_units: str) -> str:
        """
        Generate axis label from AxisConfig.
        
        Args:
            axis_config: Configuration for the axis (measure, channel)
            params: Full analysis parameters (needed for conductance units)
            current_units: Units for current measurements ('pA', 'nA', or 'μA')
            
        Returns:
            Formatted axis label (e.g., "Time (s)", "Peak Current (pA)", "Conductance (nS)")
        """
        measure = axis_config.measure
        channel = axis_config.channel
        
        # Special case: Time measure
        if measure == "Time":
            return "Time (s)"
        
        # Special case: Conductance measure
        if measure == "Conductance":
            units = params.conductance_config.units if params.conductance_config else "None"
            return f"Conductance ({units})"
        
        # Determine units based on channel for standard measures
        if channel == "Voltage":
            units = "mV"
        elif channel == "Current":
            units = current_units
        else:
            units = ""
        
        # Build label: "Measure Channel (units)"
        label = f"{measure} {channel}"
        
        # Add units
        if units:
            label = f"{label} ({units})"
        
        return label


    @staticmethod
    def prepare_summary_table(
        batch_results: Dict[str, Dict[str, Any]],
        params: AnalysisParameters,
        included_files: Optional[set] = None,
        current_units: str = "pA"
    ) -> Dict[str, Any]:
        """
        Prepare generalized summary data with two columns per file (X, Y).
        
        Creates a table where each file contributes two columns (X and Y values),
        separated by blank columns. Files can have different numbers of data points,
        with empty cells where data is absent.
        
        Args:
            batch_results: Dictionary mapping filenames to data dictionaries
                        (containing 'x_values' and 'y_values' lists)
            params: Analysis parameters defining X and Y axes
            included_files: Optional set of filenames to include (None = all files)
            current_units: Units for current measurements ('pA', 'nA', or 'μA')
            
        Returns:
            Dictionary with keys:
                - 'headers': List of column headers
                - 'data': 2D numpy array (object dtype to handle mixed types)
                - 'format_spec': Format specification for numeric values
        """
        logger.info(f"Preparing generalized summary from {len(batch_results)} batch results")
        logger.debug(f"Current units: {current_units}")
        
        # Filter results to included files
        if included_files:
            filtered_results = {
                name: data for name, data in batch_results.items()
                if name in included_files
            }
            logger.debug(f"Filtered to {len(filtered_results)} included files")
        else:
            filtered_results = batch_results
            logger.debug("Including all files (no filter)")
        
        if not filtered_results:
            logger.warning("No results to export after filtering")
            return {"headers": [], "data": np.array([]), "format_spec": "%.6f"}
        
        # Sort filenames for consistent ordering
        sorted_files = sorted(filtered_results.keys())
        logger.debug(f"Processing {len(sorted_files)} files in sorted order")
        
        # Generate axis labels - NOW PASSING params
        x_label = GeneralizedSummaryExporter._get_axis_label(params.x_axis, params, current_units)
        y_label = GeneralizedSummaryExporter._get_axis_label(params.y_axis, params, current_units)
        logger.debug(f"Axis labels: X='{x_label}', Y='{y_label}'")
        
        # Build headers: [File1 X_label] [File1 Y_label] [blank] [File2 X_label] ...
        headers = []
        for filename in sorted_files:
            headers.extend([
                f"{filename} {x_label}",
                f"{filename} {y_label}",
                ""  # Blank column separator
            ])
        
        # Remove the trailing blank column
        if headers and headers[-1] == "":
            headers.pop()
        
        logger.debug(f"Created {len(headers)} column headers")
        
        # Find maximum data length across all files
        max_length = 0
        for filename in sorted_files:
            data = filtered_results[filename]
            x_len = len(data.get("x_values", []))
            y_len = len(data.get("y_values", []))
            file_max = max(x_len, y_len)
            if file_max > max_length:
                max_length = file_max
        
        logger.debug(f"Maximum data length across all files: {max_length}")
        
        # Build data array as list of lists (easier to work with mixed empty strings and floats)
        data_rows = []
        for row_idx in range(max_length):
            row = []
            
            for file_idx, filename in enumerate(sorted_files):
                data = filtered_results[filename]
                x_values = data.get("x_values", [])
                y_values = data.get("y_values", [])
                
                # Add X value or empty string
                if row_idx < len(x_values):
                    row.append(x_values[row_idx])
                else:
                    row.append("")
                
                # Add Y value or empty string
                if row_idx < len(y_values):
                    row.append(y_values[row_idx])
                else:
                    row.append("")
                
                # Add blank column separator (except after last file)
                if file_idx < len(sorted_files) - 1:
                    row.append("")
            
            data_rows.append(row)
        
        # Convert to numpy array with object dtype to handle mixed types
        data_array = np.array(data_rows, dtype=object)
        
        logger.info(f"Summary table prepared: {len(sorted_files)} files × {max_length} rows × {len(headers)} columns")
        
        return {
            "headers": headers,
            "data": data_array,
            "format_spec": "%.6f"
        }