"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Manages generalized summary export for batch analyses. Similar in concept to Summary IV, but generalized
for any analysis parameter combination. Best suited for time-course analyses. Each file's data is independent,
allowing for different data lengths (ie from files with different recording durations). Output CSV has two columns
per file (three if dual range is enabled) plus blank columns between files for readability.
"""

from typing import Dict, Any, Optional
import numpy as np

from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class GeneralizedSummaryExporter:
    """
    Handles exporting generalized summary data for any analysis parameter combination.
    
    Creates a format with columns per file: 
    - Single range: [File1_X] [File1_Y] [blank] [File2_X] [File2_Y] ...
    - Dual range: [File1_X] [File1_Y_R1] [File1_Y_R2] [blank] [File2_X] [File2_Y_R1] [File2_Y_R2] ...
    
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
    def _extract_y_labels_from_headers(headers: list, use_dual_range: bool) -> tuple:
        """
        Extract Y-axis label(s) from pre-calculated export headers.
        
        Args:
            headers: List of header strings from export_table (e.g., ["Time (s)", "Average Current (pA) (-100mV)", ...])
            use_dual_range: Whether dual range analysis is enabled
            
        Returns:
            tuple: (y_label_r1, y_label_r2) where y_label_r2 is None for single range
        """
        if not headers or len(headers) < 2:
            logger.warning("No headers available, using default labels")
            return ("Y", None)
        
        # Headers format:
        # Single range: [X_label, Y_label]
        # Dual range: [X_label, Y_label_R1, Y_label_R2] (when X is Time)
        #         or: [X_label_R1, Y_label_R1, X_label_R2, Y_label_R2] (when X varies)
        
        if use_dual_range:
            if len(headers) >= 3:
                # Check if this is Time-based (3 columns) or variable X (4 columns)
                if len(headers) == 3:
                    # Time-based: [Time, Y_R1, Y_R2]
                    return (headers[1], headers[2])
                else:
                    # Variable X: [X_R1, Y_R1, X_R2, Y_R2]
                    return (headers[1], headers[3])
            else:
                logger.warning(f"Expected 3+ headers for dual range, got {len(headers)}")
                return (headers[1] if len(headers) > 1 else "Y Range 1", "Y Range 2")
        else:
            # Single range: just use second header
            return (headers[1], None)


    @staticmethod
    def prepare_summary_table(
        batch_results: Dict[str, Dict[str, Any]],
        params: AnalysisParameters,
        included_files: Optional[set] = None,
        current_units: str = "pA"
    ) -> Dict[str, Any]:

        logger.info(f"Preparing generalized summary from {len(batch_results)} batch results")
        logger.debug(f"Current units: {current_units}, Dual range: {params.use_dual_range}")
        
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
            return {"headers": [], "data": np.array([]), "format_spec": "%s"}
        
        # Sort filenames for consistent ordering
        sorted_files = sorted(filtered_results.keys())
        logger.debug(f"Processing {len(sorted_files)} files in sorted order")
        
        # Generate X-axis label
        x_label = GeneralizedSummaryExporter._get_axis_label(params.x_axis, params, current_units)
        
        # Get Y-axis labels from the first file's pre-calculated headers
        first_file = sorted_files[0]
        first_headers = filtered_results[first_file].get("headers", [])
        y_label_r1, y_label_r2 = GeneralizedSummaryExporter._extract_y_labels_from_headers(
            first_headers, 
            params.use_dual_range
        )
        
        logger.debug(f"Axis labels: X='{x_label}', Y_R1='{y_label_r1}', Y_R2='{y_label_r2}'")
        
        # Build headers based on whether dual range is enabled
        headers = []
        for filename in sorted_files:
            if params.use_dual_range:
                # Three columns per file: X, Y_R1, Y_R2
                headers.extend([
                    f"{filename} {x_label}",
                    f"{filename} {y_label_r1}",
                    f"{filename} {y_label_r2}",
                    ""  # Blank column separator
                ])
            else:
                # Two columns per file: X, Y
                headers.extend([
                    f"{filename} {x_label}",
                    f"{filename} {y_label_r1}",
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
            
            if params.use_dual_range:
                y2_len = len(data.get("y_values2", []))
                file_max = max(x_len, y_len, y2_len)
            else:
                file_max = max(x_len, y_len)
            
            if file_max > max_length:
                max_length = file_max
        
        logger.debug(f"Maximum data length across all files: {max_length}")
        
        # Build data array as list of lists
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
                
                # Add Y value (Range 1) or empty string
                if row_idx < len(y_values):
                    row.append(y_values[row_idx])
                else:
                    row.append("")
                
                # If dual range, add Y value (Range 2) or empty string
                if params.use_dual_range:
                    y_values2 = data.get("y_values2", [])
                    if row_idx < len(y_values2):
                        row.append(y_values2[row_idx])
                    else:
                        row.append("")
                
                # Add blank column separator (except after last file)
                if file_idx < len(sorted_files) - 1:
                    row.append("")
            
            data_rows.append(row)
        
        # Convert to numpy array with object dtype to handle mixed types
        data_array = np.array(data_rows, dtype=object)
        
        cols_per_file = 3 if params.use_dual_range else 2
        logger.info(
            f"Summary table prepared: {len(sorted_files)} files × {max_length} rows × "
            f"{len(headers)} columns ({cols_per_file} data columns per file)"
        )
        
        return {
            "headers": headers,
            "data": data_array,
            "format_spec": "%s"  # Changed from "%.6f" to handle object array with mixed types
        }