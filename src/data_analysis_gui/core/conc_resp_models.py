"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Immutable data structures for concentration-response analysis ranges.

Each ConcentrationRange defines a time window for measurement extraction,
with optional background subtraction via paired background ranges.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AnalysisType(Enum):
    """Whether to calculate the mean or find a peak within the time window."""
    AVERAGE = "Average"
    PEAK = "Peak"


class PeakType(Enum):
    """Direction of peak detection: most positive, most negative, or largest magnitude."""
    MAX = "Max"
    MIN = "Min"
    ABSOLUTE_MAX = "Absolute Max"


@dataclass(frozen=True)
class ConcentrationRange:
    """
    Configuration for measuring a response within a specific time window.
    
    Represents a time window over which measurements are taken from time-series data.
    Supports both direct measurements and background-subtracted measurements when
    paired with a background range.
    
    Args:
        range_id: Internal identifier like "Range_1" or "Background_1"
        concentration: Concentration value in µM
        start_time: Beginning of measurement window in seconds
        end_time: End of measurement window in seconds
        analysis_type: AVERAGE takes the mean, PEAK finds max/min/abs_max
        peak_type: Required if analysis_type is PEAK
        is_background: True if this range is used for background subtraction
        paired_background: range_id of background to subtract from this measurement
    
    Example:
        Create an analysis range that subtracts baseline from response:
        
        >>> bg = ConcentrationRange("BG_1", 0.0, 10.0, 50.0, AnalysisType.AVERAGE, 
        ...                         is_background=True)
        >>> measurement = ConcentrationRange("Range_1", 10.0, 100.0, 150.0,
        ...                                   AnalysisType.AVERAGE, paired_background="BG_1")
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
        """Validates that end_time > start_time and analysis_type is valid."""
        if self.end_time <= self.start_time:
            raise ValueError(
                f"Range '{self.range_id}': end_time ({self.end_time}) must be "
                f"greater than start_time ({self.start_time})"
            )
        
        if not isinstance(self.analysis_type, AnalysisType):
            raise ValueError(
                f"Range '{self.range_id}': analysis_type must be an AnalysisType enum, "
                f"got {type(self.analysis_type)}"
            )
    
    @property
    def duration(self) -> float:
        """Duration of the range in seconds."""
        return self.end_time - self.start_time
    
    @property
    def has_background_subtraction(self) -> bool:
        """True if this range will have a background measurement subtracted."""
        return self.paired_background is not None
    
    def describe(self) -> str:
        """Human-readable summary for display purposes."""
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