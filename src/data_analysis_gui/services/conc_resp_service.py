"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Service for concentration-response analysis business logic.

This module provides all business logic for loading CSV time-series data,
calculating range-based metrics with optional background subtraction, and
exporting results in pivoted format. Completely UI-agnostic for testability.

Classes:
    - ConcentrationResponseService: Stateless service for concentration-response analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import replace
import re

from data_analysis_gui.core.conc_resp_models import (
    ConcentrationRange,
    AnalysisType,
    PeakType,
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ConcentrationResponseService:
    """
    Stateless service for concentration-response analysis operations.
    
    Handles CSV loading, validation, metric calculation, background subtraction,
    and export formatting. All methods are static for easy testing without state.
    """
    
    @staticmethod
    def load_and_validate_csv(filepath: str) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
        """
        Load and validate a CSV file for concentration-response analysis.
        
        Transforms data column headers to extract only voltage information
        (e.g., "Average Current (pA) (+100mV)" -> "+100mV") for analysis,
        but preserves original headers for display purposes.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Tuple containing:
                - DataFrame with loaded data (columns renamed to voltage only)
                - Name of the time column (first column, unchanged)
                - List of simplified data column names (voltage only)
                - List of original full data column names
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV has fewer than 2 columns or is empty
            pd.errors.ParserError: If CSV parsing fails
        
        Example:
            >>> df, time_col, data_cols, orig_cols = service.load_and_validate_csv("data.csv")
            >>> print(f"Simplified: {data_cols}")
            >>> print(f"Original: {orig_cols}")
            Simplified: ['+100mV', '-60mV']
            Original: ['Average Current (pA) (+100mV)', 'Average Current (pA) (-60mV)']
        """
        filepath_obj = Path(filepath)
        
        if not filepath_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Failed to parse CSV {filepath}: {e}")
            raise ValueError(f"Failed to parse CSV file: {e}")
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        if df.shape[1] < 2:
            raise ValueError(
                f"CSV must have at least 2 columns (time + data), got {df.shape[1]}"
            )
        
        time_col = df.columns[0]
        original_data_cols = df.columns[1:].tolist()
        
        # Transform data column names to extract voltage only
        simplified_data_cols = [
            ConcentrationResponseService._extract_voltage_from_header(col)
            for col in original_data_cols
        ]
        
        # Rename DataFrame columns to simplified names
        rename_dict = {orig: new for orig, new in zip(original_data_cols, simplified_data_cols)}
        df = df.rename(columns=rename_dict)
        
        logger.info(
            f"Loaded CSV: {filepath_obj.name} - "
            f"{len(df)} rows, time column: '{time_col}', "
            f"{len(simplified_data_cols)} data column(s): {simplified_data_cols}"
        )
        
        return df, time_col, simplified_data_cols, original_data_cols
    
    @staticmethod
    def calculate_range_value(
        df: pd.DataFrame,
        time_col: str,
        data_col: str,
        start_time: float,
        end_time: float,
        analysis_type: AnalysisType,
        peak_type: Optional[PeakType] = None,
    ) -> float:
        """
        Calculate the metric value for a specific range.
        
        Args:
            df: DataFrame containing the time-series data
            time_col: Name of the time column
            data_col: Name of the data column to analyze
            start_time: Start time for the range
            end_time: End time for the range
            analysis_type: Type of analysis (Average or Peak)
            peak_type: Type of peak detection (required if analysis_type is PEAK)
            
        Returns:
            Calculated value for the range (average or peak)
            Returns np.nan if no data points in range
        
        Raises:
            ValueError: If peak_type not provided for Peak analysis
        """
        # Extract data within time range
        mask = (df[time_col] >= start_time) & (df[time_col] <= end_time)
        subset = df.loc[mask, data_col]
        
        if subset.empty:
            logger.warning(
                f"No data points in range [{start_time}, {end_time}] "
                f"for column '{data_col}'"
            )
            return np.nan
        
        # Calculate based on analysis type
        if analysis_type == AnalysisType.AVERAGE:
            return float(subset.mean())
        
        elif analysis_type == AnalysisType.PEAK:
            if peak_type is None:
                raise ValueError("peak_type must be specified for Peak analysis")
            
            if peak_type == PeakType.MAX:
                return float(subset.max())
            elif peak_type == PeakType.MIN:
                return float(subset.min())
            elif peak_type == PeakType.ABSOLUTE_MAX:
                # Value with maximum absolute magnitude
                return float(subset.loc[subset.abs().idxmax()])
            else:
                raise ValueError(f"Unknown peak_type: {peak_type}")
        
        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")
    
    @staticmethod
    def calculate_background_values(
        df: pd.DataFrame,
        time_col: str,
        data_cols: List[str],
        bg_ranges: List[ConcentrationRange],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate background values for all background ranges and data columns.
        
        Args:
            df: DataFrame containing the time-series data
            time_col: Name of the time column
            data_cols: List of data column names
            bg_ranges: List of background ConcentrationRange objects
            
        Returns:
            Nested dictionary: {bg_range_id: {data_col_name: value}}
            
        Example:
            >>> bg_values = service.calculate_background_values(
            ...     df, "Time (s)", ["Current (pA)"], [bg_range]
            ... )
            >>> print(bg_values["Background_1"]["Current (pA)"])
            -12.5
        """
        bg_values = {}
        
        for bg_range in bg_ranges:
            bg_values[bg_range.range_id] = {}
            
            for data_col in data_cols:
                value = ConcentrationResponseService.calculate_range_value(
                    df=df,
                    time_col=time_col,
                    data_col=data_col,
                    start_time=bg_range.start_time,
                    end_time=bg_range.end_time,
                    analysis_type=bg_range.analysis_type,
                    peak_type=bg_range.peak_type,
                )
                bg_values[bg_range.range_id][data_col] = value
        
        logger.debug(
            f"Calculated background values for {len(bg_ranges)} range(s), "
            f"{len(data_cols)} data column(s)"
        )
        
        return bg_values
    
    @staticmethod
    def apply_auto_pairing(
        ranges: List[ConcentrationRange],
    ) -> Tuple[List[ConcentrationRange], bool]:
        """
        Automatically pair all non-background ranges to a single background range.
        
        Auto-pairing occurs when:
        - Exactly one background range exists
        - All non-background ranges have no paired background (None)
        
        Args:
            ranges: List of ConcentrationRange objects
            
        Returns:
            Tuple containing:
                - List of ranges (with auto-pairing applied if applicable)
                - Boolean indicating whether auto-pairing was applied
        
        Example:
            >>> ranges = [bg_range, range1, range2]
            >>> modified_ranges, was_paired = service.apply_auto_pairing(ranges)
            >>> print(f"Auto-paired: {was_paired}")
            Auto-paired: True
        """
        bg_ranges = [r for r in ranges if r.is_background]
        non_bg_ranges = [r for r in ranges if not r.is_background]
        
        # Check if auto-pairing conditions are met
        if len(bg_ranges) != 1:
            return ranges, False
        
        if not all(r.paired_background is None for r in non_bg_ranges):
            return ranges, False
        
        # Apply auto-pairing
        single_bg_id = bg_ranges[0].range_id
        modified_ranges = []
        
        for r in ranges:
            if r.is_background:
                modified_ranges.append(r)
            else:
                # Create new range with paired_background set
                modified_ranges.append(
                    replace(r, paired_background=single_bg_id)
                )
        
        logger.info(
            f"Auto-paired {len(non_bg_ranges)} range(s) to "
            f"background '{single_bg_id}'"
        )
        
        return modified_ranges, True
    
    @staticmethod
    def run_analysis(
        df: pd.DataFrame,
        time_col: str,
        data_cols: List[str],
        ranges: List[ConcentrationRange],
        filename: str = "data",
    ) -> Dict[str, pd.DataFrame]:
        """
        Run complete concentration-response analysis for all data columns.
        
        Args:
            df: DataFrame containing the time-series data
            time_col: Name of the time column
            data_cols: List of data column names to analyze
            ranges: List of ConcentrationRange objects (after auto-pairing if applicable)
            filename: Name of the source file (for results table)
            
        Returns:
            Dictionary mapping data column names to results DataFrames.
            Each DataFrame has columns: File, Data Trace, Concentration (µM), 
            Raw Value, Background, Corrected Value
        
        Raises:
            ValueError: If ranges configuration is invalid
        """
        if not ranges:
            raise ValueError("No ranges provided for analysis")
        
        # Separate background and analysis ranges
        bg_ranges = [r for r in ranges if r.is_background]
        analysis_ranges = [r for r in ranges if not r.is_background]
        
        if not analysis_ranges:
            raise ValueError("No analysis ranges defined")
        
        # Calculate all background values upfront
        bg_values = ConcentrationResponseService.calculate_background_values(
            df, time_col, data_cols, bg_ranges
        )
        
        # Run analysis for each data column
        results_dfs = {}
        
        for data_col in data_cols:
            results_rows = []
            
            for analysis_range in analysis_ranges:
                # Calculate raw value
                raw_value = ConcentrationResponseService.calculate_range_value(
                    df=df,
                    time_col=time_col,
                    data_col=data_col,
                    start_time=analysis_range.start_time,
                    end_time=analysis_range.end_time,
                    analysis_type=analysis_range.analysis_type,
                    peak_type=analysis_range.peak_type,
                )
                
                # Get background value if paired
                bg_value = 0.0
                if analysis_range.paired_background:
                    bg_id = analysis_range.paired_background
                    if bg_id in bg_values:
                        bg_value = bg_values[bg_id].get(data_col, 0.0)
                    else:
                        logger.warning(
                            f"Paired background '{bg_id}' not found for "
                            f"range '{analysis_range.range_id}'"
                        )
                
                # Calculate corrected value
                corrected_value = raw_value - bg_value
                
                results_rows.append({
                    "File": filename,
                    "Data Trace": data_col,
                    "Concentration (µM)": analysis_range.concentration,
                    "Raw Value": raw_value,
                    "Background": bg_value,
                    "Corrected Value": corrected_value,
                })
            
            if results_rows:
                results_dfs[data_col] = pd.DataFrame(results_rows)
        
        logger.info(
            f"Analysis complete: {len(analysis_ranges)} range(s), "
            f"{len(data_cols)} trace(s), {sum(len(df) for df in results_dfs.values())} total results"
        )
        
        return results_dfs
    
    @staticmethod
    def pivot_for_export(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot results DataFrame to export format (concentrations as rows).
        
        Args:
            results_df: Results DataFrame with columns:
                File, Data Trace, Concentration (µM), Raw Value, Background, Corrected Value
                
        Returns:
            Pivoted DataFrame with:
                - First column containing concentration values (numeric)
                - Second column containing corrected values
                - Empty column headers
        
        Example:
            Input:
                | Concentration (µM) | Corrected Value |
                |--------------------|-----------------|
                | 0.1                | -50.2           |
                | 1.0                | -75.8           |
            
            Output CSV:
                ,
                0.1,-50.2
                1.0,-75.8
        """
        if results_df.empty:
            logger.warning("Attempting to pivot empty results DataFrame")
            return pd.DataFrame()
        
        # Extract concentrations and corrected values
        export_df = pd.DataFrame({
            "": results_df["Concentration (µM)"].tolist(),
            " ": results_df["Corrected Value"].tolist()
        })
        
        logger.debug(f"Pivoted results: {len(results_df)} rows")
        
        return export_df
    
    def _extract_voltage_from_header(header: str) -> str:
        """
        Extract voltage portion from CSV header.
        
        Expected format: "{Average | Peak} Current ({units}) ({voltage})"
        Example: "Average Current (pA) (+100mV)" -> "+100mV"
        
        Args:
            header: Full CSV column header
            
        Returns:
            Extracted voltage string, or original header if pattern not matched
        """
        # Match last parenthetical containing voltage info
        # Pattern: (±digits[.digits] [m]V) at end of string
        pattern = r'\(([+-]?\d+\.?\d*\s*m?V)\)\s*$'
        match = re.search(pattern, header)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: return original header if pattern doesn't match
        logger.warning(f"Could not extract voltage from header: '{header}'")
        return header