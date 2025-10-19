"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module provides services for I-V (Current-Voltage) analysis and data transformation for patch clamp electrophysiology experiments.
It includes utilities for preparing I-V curve data from batch results, supporting both single and dual-range analysis, and for exporting summary tables with unit-aware formatting.

Classes:
    - IVAnalysisService: Transforms batch results into I-V curve data structures suitable for analysis and plotting.
    - IVSummaryExporter: Prepares summary tables for export, including headers and formatted data for CSV or spreadsheet output.

Intended for use in automated analysis pipelines and GUI applications for electrophysiology data.
"""

from typing import Dict, Any, Tuple, Optional
from data_analysis_gui.core.params import AnalysisParameters


class IVAnalysisService:
    """
    Provides services for preparing and analyzing I-V (Current-Voltage) data.

    This service transforms raw batch results into formats suitable for I-V curve analysis,
    supporting both single and dual-range data.
    """

    @staticmethod
    def prepare_iv_data(
        batch_results: Dict[str, Dict[str, Any]], params: AnalysisParameters
    ) -> Tuple[Dict[float, list], Dict[str, str], Optional[Dict[float, list]]]:
        """
        Transform raw batch results into a format suitable for I-V curve analysis.

        Processes batch results and organizes them by rounded voltage values for each recording.
        Supports dual-range analysis by returning a third element for range 2 data if enabled.

        Args:
            batch_results (Dict[str, Dict[str, Any]]): Raw batch results keyed by base filename.
            params (AnalysisParameters): Analysis parameters, including axis configuration and dual-range flag.

        Returns:
            Tuple[Dict[float, list], Dict[str, str], Optional[Dict[float, list]]]:
                - Dictionary mapping rounded voltages to lists of current values (range 1).
                - Dictionary mapping recording IDs to base filenames.
                - Dictionary mapping rounded voltages to lists of current values (range 2), or None if not used.
        """
        iv_data_range1: Dict[float, list] = {}
        iv_data_range2: Optional[Dict[float, list]] = (
            {} if params.use_dual_range else None
        )
        iv_file_mapping: Dict[str, str] = {}

        # Condition check
        is_iv_analysis = (
            params.x_axis.measure in ["Average", "Peak"]
            and params.x_axis.channel == "Voltage"
            and params.y_axis.measure in ["Average", "Peak"]
            and params.y_axis.channel == "Current"
        )

        if not is_iv_analysis:
            return iv_data_range1, iv_file_mapping, iv_data_range2

        # Process sorted batch results
        for idx, (base_name, data) in enumerate(sorted(batch_results.items())):
            # Process Range 1 data with its own x_values
            if "x_values" in data and "y_values" in data:
                for x_val, y_val in zip(data["x_values"], data["y_values"]):
                    rounded_voltage = round(x_val, 1)
                    if rounded_voltage not in iv_data_range1:
                        iv_data_range1[rounded_voltage] = []
                    iv_data_range1[rounded_voltage].append(y_val)

            # Process Range 2 data with ITS OWN x_values (not Range 1's)
            if params.use_dual_range:
                # Range 2 should have its own x_values!
                # Check if x_values2 exists (for separate voltage measurements in range 2)
                if "x_values2" in data and "y_values2" in data:
                    # Use Range 2's own x_values
                    for x_val2, y_val2 in zip(data["x_values2"], data["y_values2"]):
                        rounded_voltage = round(x_val2, 1)
                        if rounded_voltage not in iv_data_range2:
                            iv_data_range2[rounded_voltage] = []
                        iv_data_range2[rounded_voltage].append(y_val2)
                elif "y_values2" in data:
                    # Fallback - this shouldn't happen if batch processor is fixed
                    # but keeping for compatibility
                    print(
                        "Warning: Range 2 missing separate x_values, using Range 1 x_values"
                    )
                    for x_val, y_val2 in zip(data["x_values"], data["y_values2"]):
                        rounded_voltage = round(x_val, 1)
                        if rounded_voltage not in iv_data_range2:
                            iv_data_range2[rounded_voltage] = []
                        iv_data_range2[rounded_voltage].append(y_val2)

            recording_id = f"Recording {idx + 1}"
            iv_file_mapping[recording_id] = base_name

        return iv_data_range1, iv_file_mapping, iv_data_range2


class IVSummaryExporter:
    """
    Handles exporting IV summary data without current density calculations.

    Prepares summary tables for export, including unit-aware headers and data formatting.
    """

    @staticmethod
    def prepare_summary_table(
        iv_data: Dict[float, list],
        iv_file_mapping: Dict[str, str],
        included_files: set = None,
        current_units: str = "pA",
    ) -> Dict[str, Any]:
        """
        Prepare IV summary data for export with unit-aware headers.

        Organizes IV data for CSV export, including voltage and current columns for each recording.
        Handles missing data by inserting NaN values.

        Args:
            iv_data (Dict[float, list]): Dictionary mapping voltages (float) to lists of current values.
            iv_file_mapping (Dict[str, str]): Dictionary mapping recording IDs to file names.
            included_files (set, optional): Set of file names to include (None = all files).
            current_units (str, optional): Units for current measurements ('pA', 'nA', or 'μA'). Defaults to 'pA'.

        Returns:
            Dict[str, Any]: Dictionary with keys:
                - 'headers': List of column headers for export.
                - 'data': 2D numpy array of export data.
                - 'format_spec': String format specification for numeric values.
        """
        import numpy as np

        # Get sorted voltages
        voltages = sorted(iv_data.keys())

        # Build headers with units
        headers = ["Voltage (mV)"]  # Voltage header already includes units
        data_columns = [voltages]

        # Sort recordings
        sorted_recordings = sorted(
            iv_file_mapping.keys(), key=lambda x: int(x.split()[-1])
        )

        for recording_id in sorted_recordings:
            base_name = iv_file_mapping.get(recording_id, recording_id)

            # Skip if not included
            if included_files and base_name not in included_files:
                continue

            # Add header with current units
            headers.append(f"{base_name} ({current_units})")
            recording_index = int(recording_id.split()[-1]) - 1

            # Extract current values
            current_values = []
            for voltage in voltages:
                if recording_index < len(iv_data[voltage]):
                    current_values.append(iv_data[voltage][recording_index])
                else:
                    current_values.append(np.nan)

            data_columns.append(current_values)

        # Convert to array format expected by exporter
        data_array = np.column_stack(data_columns)

        return {"headers": headers, "data": data_array, "format_spec": "%.6f"}