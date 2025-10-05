"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Data models for concentration-response analysis.

This module defines immutable data structures for representing analysis ranges
in concentration-response experiments. Each range defines a time window over which
measurements (average or peak) are calculated, with optional background subtraction.

Classes:
    - AnalysisType: Enum for types of analysis (Average or Peak)
    - PeakType: Enum for types of peak detection
    - ConcentrationRange: Immutable configuration for a single analysis range
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AnalysisType(Enum):
    """
    Type of analysis to perform on a range.
    
    AVERAGE: Calculate mean value over the range
    PEAK: Find peak value over the range (requires peak_type specification)
    """
    AVERAGE = "Average"
    PEAK = "Peak"


class PeakType(Enum):
    """
    Type of peak detection for Peak analysis.
    
    MAX: Maximum value in the range (most positive)
    MIN: Minimum value in the range (most negative)
    ABSOLUTE_MAX: Value with maximum absolute magnitude
    """
    MAX = "Max"
    MIN = "Min"
    ABSOLUTE_MAX = "Absolute Max"


@dataclass(frozen=True)
class ConcentrationRange:
    """
    Immutable configuration for a concentration-response analysis range.
    
    Represents a time window over which measurements are taken from time-series data.
    Supports both direct measurements and background-subtracted measurements when
    paired with a background range.
    
    Args:
        range_id: Internal identifier (e.g., "Range_1", "Background_1")
        concentration: Concentration value in µM
        start_time: Start time in seconds for the analysis window
        end_time: End time in seconds for the analysis window
        analysis_type: Type of analysis to perform (Average or Peak)
        peak_type: Type of peak detection (required if analysis_type is PEAK)
        is_background: Whether this range is a background range
        paired_background: Internal ID of background range to subtract from this range's values
            (None means no background subtraction)
    
    Raises:
        ValueError: If end_time <= start_time or analysis_type is invalid
    
    Example:
        >>> # Analysis range with background subtraction
        >>> analysis_range = ConcentrationRange(
        ...     range_id="Range_1",
        ...     concentration=10.0,
        ...     start_time=100.0,
        ...     end_time=150.0,
        ...     analysis_type=AnalysisType.AVERAGE,
        ...     is_background=False,
        ...     paired_background="Background_1"
        ... )
        
        >>> # Background range
        >>> bg_range = ConcentrationRange(
        ...     range_id="Background_1",
        ...     concentration=0.0,
        ...     start_time=10.0,
        ...     end_time=50.0,
        ...     analysis_type=AnalysisType.AVERAGE,
        ...     is_background=True,
        ...     paired_background=None
        ... )
    """
    
    range_id: str
    concentration: float
    start_time: float
    end_time: float
    analysis_type: AnalysisType
    peak_type: Optional[PeakType] = PeakType.ABSOLUTE_MAX
    is_background: bool = False
    paired_background: Optional[str] = None
    
    def __post_init__(self):
        """
        Validate range configuration after initialization.
        
        Ensures:
            - end_time is after start_time
            - analysis_type is valid
        
        Raises:
            ValueError: If validation fails
        """
        # Validate time ordering
        if self.end_time <= self.start_time:
            raise ValueError(
                f"Range '{self.range_id}': end_time ({self.end_time}) must be "
                f"greater than start_time ({self.start_time})"
            )
        
        # Validate analysis_type is an AnalysisType enum member
        if not isinstance(self.analysis_type, AnalysisType):
            raise ValueError(
                f"Range '{self.range_id}': analysis_type must be an AnalysisType enum, "
                f"got {type(self.analysis_type)}"
            )
    
    @property
    def duration(self) -> float:
        """
        Duration of the range in seconds.
        
        Returns:
            float: end_time - start_time
        """
        return self.end_time - self.start_time
    
    @property
    def has_background_subtraction(self) -> bool:
        """
        Whether this range uses background subtraction.
        
        Returns:
            bool: True if paired_background is specified
        """
        return self.paired_background is not None
    
    def describe(self) -> str:
        """
        Generate a human-readable description of the range.
        
        Returns:
            str: Description including time window, analysis type, and background pairing
        """
        desc = f"{self.range_id}: {self.concentration}µM, {self.start_time:.1f}-{self.end_time:.1f}s"
        
        if self.analysis_type == AnalysisType.PEAK and self.peak_type:
            desc += f", {self.analysis_type.value} ({self.peak_type.value})"
        else:
            desc += f", {self.analysis_type.value}"
        
        if self.is_background:
            desc += " [Background]"
        elif self.has_background_subtraction:
            desc += f" - BG: {self.paired_background}"
        
        return desc