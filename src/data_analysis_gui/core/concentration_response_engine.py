"""
Core analysis engine for concentration-response analysis.

This module provides a GUI-independent engine for analyzing time-series
patch-clamp data with user-defined concentration ranges and background
correction.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import os


class AnalysisType(Enum):
    """Types of analysis that can be performed on a range."""
    AVERAGE = "Average"
    PEAK = "Peak"


class PeakType(Enum):
    """Types of peak detection for peak analysis."""
    MAX = "Max"
    MIN = "Min"
    ABSOLUTE_MAX = "Absolute Max"


@dataclass
class ConcentrationRange:
    """Data model for a concentration analysis range."""
    name: str
    start_time: float
    end_time: float
    analysis_type: AnalysisType
    peak_type: Optional[PeakType] = None
    is_background: bool = False
    paired_background_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate and convert string inputs if needed."""
        if isinstance(self.analysis_type, str):
            self.analysis_type = AnalysisType(self.analysis_type)
        if self.peak_type and isinstance(self.peak_type, str):
            self.peak_type = PeakType(self.peak_type)
    
    def overlaps_with(self, other: 'ConcentrationRange') -> bool:
        """Check if this range overlaps with another range."""
        return not (self.end_time < other.start_time or 
                   self.start_time > other.end_time)


@dataclass
class RangeAnalysisResult:
    """Results from analyzing a single range for a single data trace."""
    range_name: str
    trace_name: str
    raw_value: float
    background_value: float = 0.0
    corrected_value: float = field(init=False)
    
    def __post_init__(self):
        """Calculate corrected value."""
        self.corrected_value = self.raw_value - self.background_value


@dataclass
class DatasetInfo:
    """Metadata about the loaded dataset."""
    filepath: str
    filename: str
    num_points: int
    num_traces: int
    time_column: str
    data_columns: List[str]
    time_range: Tuple[float, float]


class ConcentrationResponseEngine:
    """
    Core engine for concentration-response analysis.
    
    This class provides a framework-independent interface for analyzing
    time-series data with user-defined concentration ranges and optional
    background correction.
    
    Example:
        >>> engine = ConcentrationResponseEngine()
        >>> engine.load_csv("data.csv")
        >>> engine.add_range("Range 1", 10.0, 20.0, AnalysisType.AVERAGE)
        >>> engine.add_range("Background", 0.0, 5.0, AnalysisType.AVERAGE, is_background=True)
        >>> results = engine.analyze()
    """
    
    def __init__(self):
        """Initialize the engine."""
        self._data_df: Optional[pd.DataFrame] = None
        self._dataset_info: Optional[DatasetInfo] = None
        self._ranges: List[ConcentrationRange] = []
        self._analysis_results: Dict[str, List[RangeAnalysisResult]] = {}
        
        # Cache for filtered data
        self._filtered_data_cache: Dict[str, pd.DataFrame] = {}
    
    # =========================================================================
    # Data Loading and Validation
    # =========================================================================
    
    def load_csv(self, filepath: str) -> Tuple[bool, str]:
        """
        Load and validate a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Tuple of (success, message)
        """
        try:
            df = pd.read_csv(filepath)
            
            # Validate structure
            if df.shape[1] < 2:
                return False, "CSV must have at least 2 columns (time and data)"
            
            if df.empty:
                return False, "CSV file is empty"
            
            # Check for non-numeric columns (except possibly the first)
            time_col = df.columns[0]
            data_cols = df.columns[1:].tolist()
            
            # Ensure time column is numeric
            if not pd.api.types.is_numeric_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_numeric(df[time_col])
                except:
                    return False, f"Time column '{time_col}' contains non-numeric values"
            
            # Store the data
            self._data_df = df
            self._dataset_info = DatasetInfo(
                filepath=filepath,
                filename=os.path.basename(filepath),
                num_points=len(df),
                num_traces=len(data_cols),
                time_column=time_col,
                data_columns=data_cols,
                time_range=(df[time_col].min(), df[time_col].max())
            )
            
            # Clear caches
            self._filtered_data_cache.clear()
            self._analysis_results.clear()
            
            return True, f"Loaded {self._dataset_info.num_points} points, {self._dataset_info.num_traces} traces"
            
        except Exception as e:
            return False, str(e)
    
    def get_dataset_info(self) -> Optional[DatasetInfo]:
        """Get information about the loaded dataset."""
        return self._dataset_info
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the raw data DataFrame."""
        return self._data_df
    
    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self._data_df is not None
    
    # =========================================================================
    # Range Management
    # =========================================================================
    
    def add_range(self, name: str, start_time: float, end_time: float,
                  analysis_type: AnalysisType = AnalysisType.AVERAGE,
                  peak_type: Optional[PeakType] = None,
                  is_background: bool = False,
                  paired_background_name: Optional[str] = None) -> ConcentrationRange:
        """
        Add a new analysis range.
        
        Args:
            name: Name for the range
            start_time: Start time
            end_time: End time
            analysis_type: Type of analysis
            peak_type: Peak type if analysis is PEAK
            is_background: Whether this is a background range
            paired_background_name: Name of paired background range
            
        Returns:
            The created ConcentrationRange object
        """
        # Validate peak_type requirement
        if analysis_type == AnalysisType.PEAK and peak_type is None:
            peak_type = PeakType.MAX
        elif analysis_type != AnalysisType.PEAK:
            peak_type = None
        
        # Create the range
        new_range = ConcentrationRange(
            name=name,
            start_time=start_time,
            end_time=end_time,
            analysis_type=analysis_type,
            peak_type=peak_type,
            is_background=is_background,
            paired_background_name=paired_background_name if not is_background else None
        )
        
        self._ranges.append(new_range)
        self._invalidate_results_cache()
        
        return new_range
    
    def update_range(self, name: str, **kwargs) -> bool:
        """
        Update an existing range.
        
        Args:
            name: Name of the range to update
            **kwargs: Fields to update
            
        Returns:
            True if range was found and updated
        """
        for range_obj in self._ranges:
            if range_obj.name == name:
                for key, value in kwargs.items():
                    if hasattr(range_obj, key):
                        setattr(range_obj, key, value)
                self._invalidate_results_cache()
                return True
        return False
    
    def remove_range(self, name: str) -> bool:
        """
        Remove a range by name.
        
        Args:
            name: Name of the range to remove
            
        Returns:
            True if range was found and removed
        """
        initial_count = len(self._ranges)
        self._ranges = [r for r in self._ranges if r.name != name]
        
        if len(self._ranges) < initial_count:
            self._invalidate_results_cache()
            return True
        return False
    
    def get_ranges(self) -> List[ConcentrationRange]:
        """Get all defined ranges."""
        return self._ranges.copy()
    
    def get_background_ranges(self) -> List[ConcentrationRange]:
        """Get only background ranges."""
        return [r for r in self._ranges if r.is_background]
    
    def get_analysis_ranges(self) -> List[ConcentrationRange]:
        """Get only non-background ranges."""
        return [r for r in self._ranges if not r.is_background]
    
    def clear_ranges(self) -> None:
        """Remove all ranges."""
        self._ranges.clear()
        self._invalidate_results_cache()
    
    def generate_unique_range_name(self, is_background: bool = False) -> str:
        """
        Generate a unique range name.
        
        Args:
            is_background: Whether this is for a background range
            
        Returns:
            A unique range name
        """
        existing_names = {r.name for r in self._ranges}
        
        if is_background:
            if "Background" not in existing_names:
                return "Background"
            i = 2
            while f"Background_{i}" in existing_names:
                i += 1
            return f"Background_{i}"
        else:
            i = 1
            while f"Range {i}" in existing_names:
                i += 1
            return f"Range {i}"
    
    def suggest_next_range_times(self, buffer: float = 5.0) -> Tuple[float, float]:
        """
        Suggest start and end times for a new range.
        
        Args:
            buffer: Time buffer between ranges
            
        Returns:
            Tuple of (suggested_start, suggested_end)
        """
        if not self._ranges:
            # First range - start at beginning of data
            if self._dataset_info:
                return (self._dataset_info.time_range[0], 
                       min(self._dataset_info.time_range[0] + 10.0,
                           self._dataset_info.time_range[1]))
            else:
                return (0.0, 10.0)
        
        # Find the latest end time
        latest_end = max(r.end_time for r in self._ranges)
        new_start = latest_end + buffer
        new_end = new_start + 10.0
        
        # Clamp to data range if available
        if self._dataset_info:
            max_time = self._dataset_info.time_range[1]
            if new_start > max_time:
                new_start = max(0, max_time - 10.0)
            new_end = min(new_end, max_time)
        
        return (new_start, new_end)
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    def analyze(self, auto_pair_single_background: bool = True) -> Dict[str, List[RangeAnalysisResult]]:
        """
        Analyze all defined ranges.
        
        Args:
            auto_pair_single_background: If True and there's only one background,
                                        automatically pair it with all analysis ranges
                                        that don't have a pairing
        
        Returns:
            Dictionary mapping trace names to lists of results
        """
        if self._data_df is None or not self._ranges:
            return {}
        
        # Handle auto-pairing
        if auto_pair_single_background:
            self._auto_pair_background()
        
        results = {}
        bg_ranges = self.get_background_ranges()
        analysis_ranges = self.get_analysis_ranges()
        
        # Process each data trace
        for trace_name in self._dataset_info.data_columns:
            trace_results = []
            
            # Calculate background values
            bg_values = {}
            for bg_range in bg_ranges:
                bg_value = self._calculate_range_value(bg_range, trace_name)
                bg_values[bg_range.name] = bg_value
            
            # Process analysis ranges
            for analysis_range in analysis_ranges:
                raw_value = self._calculate_range_value(analysis_range, trace_name)
                
                # Get background value
                bg_value = 0.0
                if analysis_range.paired_background_name:
                    bg_value = bg_values.get(analysis_range.paired_background_name, 0.0)
                
                result = RangeAnalysisResult(
                    range_name=analysis_range.name,
                    trace_name=trace_name,
                    raw_value=raw_value,
                    background_value=bg_value
                )
                trace_results.append(result)
            
            if trace_results:
                results[trace_name] = trace_results
        
        self._analysis_results = results
        return results
    
    def get_last_results(self) -> Dict[str, List[RangeAnalysisResult]]:
        """Get the most recent analysis results."""
        return self._analysis_results.copy()
    
    def get_filtered_data(self, start_time: float, end_time: float) -> Optional[pd.DataFrame]:
        """
        Get data filtered by time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Filtered DataFrame or None if no data
        """
        if self._data_df is None:
            return None
        
        # Check cache
        cache_key = f"{start_time:.6f}_{end_time:.6f}"
        if cache_key in self._filtered_data_cache:
            return self._filtered_data_cache[cache_key]
        
        # Filter data
        time_col = self._dataset_info.time_column
        mask = (self._data_df[time_col] >= start_time) & (self._data_df[time_col] <= end_time)
        filtered = self._data_df.loc[mask].copy()
        
        # Cache and return
        self._filtered_data_cache[cache_key] = filtered
        return filtered
    
    def export_results_to_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Export results as a dictionary suitable for CSV export.
        
        Returns:
            Dictionary with trace names as keys, and dictionaries of
            range_name: corrected_value as values
        """
        export_dict = {}
        
        for trace_name, results in self._analysis_results.items():
            trace_dict = {}
            for result in results:
                trace_dict[result.range_name] = result.corrected_value
            export_dict[trace_name] = trace_dict
        
        return export_dict
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _calculate_range_value(self, range_obj: ConcentrationRange, trace_name: str) -> float:
        """Calculate a value for a specific range and trace."""
        filtered_data = self.get_filtered_data(range_obj.start_time, range_obj.end_time)
        
        if filtered_data is None or filtered_data.empty or trace_name not in filtered_data.columns:
            return np.nan
        
        data_series = filtered_data[trace_name]
        
        if range_obj.analysis_type == AnalysisType.AVERAGE:
            return data_series.mean()
        
        elif range_obj.analysis_type == AnalysisType.PEAK:
            if range_obj.peak_type == PeakType.MAX:
                return data_series.max()
            elif range_obj.peak_type == PeakType.MIN:
                return data_series.min()
            else:  # ABSOLUTE_MAX
                return data_series.loc[data_series.abs().idxmax()]
        
        return np.nan
    
    def _auto_pair_background(self) -> None:
        """Automatically pair single background with unpaired ranges."""
        bg_ranges = self.get_background_ranges()
        
        if len(bg_ranges) == 1:
            bg_name = bg_ranges[0].name
            for range_obj in self._ranges:
                if not range_obj.is_background and not range_obj.paired_background_name:
                    range_obj.paired_background_name = bg_name
    
    def _invalidate_results_cache(self) -> None:
        """Clear cached results when ranges change."""
        self._analysis_results.clear()
        # Note: We keep filtered_data_cache as it's still valid