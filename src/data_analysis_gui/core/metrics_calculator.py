"""
Pure metrics computation without any caching or I/O concerns.
PHASE 5: Fail-fast computation with proper error handling.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from data_analysis_gui.core.exceptions import DataError, ProcessingError, validate_no_nan
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SweepMetrics:
    """Computed metrics for a single sweep."""
    sweep_index: str
    time_s: float
    
    # Range 1 metrics
    voltage_mean_r1: float
    voltage_absolute_r1: float
    voltage_positive_r1: float
    voltage_negative_r1: float
    voltage_peakpeak_r1: float
    
    current_mean_r1: float
    current_absolute_r1: float
    current_positive_r1: float
    current_negative_r1: float
    current_peakpeak_r1: float
    
    # Range 2 metrics (optional)
    voltage_mean_r2: Optional[float] = None
    voltage_absolute_r2: Optional[float] = None
    voltage_positive_r2: Optional[float] = None
    voltage_negative_r2: Optional[float] = None
    voltage_peakpeak_r2: Optional[float] = None
    
    current_mean_r2: Optional[float] = None
    current_absolute_r2: Optional[float] = None
    current_positive_r2: Optional[float] = None
    current_negative_r2: Optional[float] = None
    current_peakpeak_r2: Optional[float] = None
    
    # Deprecated fields for compatibility
    @property
    def voltage_peak_r1(self): return self.voltage_absolute_r1
    @property
    def current_peak_r1(self): return self.current_absolute_r1
    @property
    def voltage_min_r1(self): return self.voltage_negative_r1
    @property
    def voltage_max_r1(self): return self.voltage_positive_r1
    @property
    def current_min_r1(self): return self.current_negative_r1
    @property
    def current_max_r1(self): return self.current_positive_r1


class MetricsCalculator:
    """
    Pure calculation of metrics from time series data.
    Stateless - all methods are essentially static.
    """
    
    @staticmethod
    def compute_sweep_metrics(
        time_ms: np.ndarray,
        voltage: np.ndarray,
        current: np.ndarray,
        sweep_index: str,
        sweep_number: int,
        range1_start: float,
        range1_end: float,
        stimulus_period: float,
        range2_start: Optional[float] = None,
        range2_end: Optional[float] = None
    ) -> SweepMetrics:
        """
        Compute metrics for a single sweep.
        
        Raises:
            DataError: If no data in specified ranges
            ProcessingError: If computation fails
        """
        # Validate inputs
        if len(time_ms) == 0:
            raise DataError(f"Empty time array for sweep {sweep_index}")
        
        # Extract range 1 data
        mask1 = (time_ms >= range1_start) & (time_ms <= range1_end)
        if not np.any(mask1):
            raise DataError(
                f"No data in range [{range1_start}, {range1_end}]",
                details={'sweep': sweep_index, 'time_range': (time_ms.min(), time_ms.max())}
            )
        
        v1, i1 = voltage[mask1], current[mask1]
        
        # Compute range 1 metrics
        metrics = SweepMetrics(
            sweep_index=sweep_index,
            time_s=sweep_number * (stimulus_period / 1000.0),
            voltage_mean_r1=MetricsCalculator._safe_mean(v1),
            voltage_absolute_r1=MetricsCalculator._absolute_peak(v1),
            voltage_positive_r1=MetricsCalculator._safe_max(v1),
            voltage_negative_r1=MetricsCalculator._safe_min(v1),
            voltage_peakpeak_r1=MetricsCalculator._peak_to_peak(v1),
            current_mean_r1=MetricsCalculator._safe_mean(i1),
            current_absolute_r1=MetricsCalculator._absolute_peak(i1),
            current_positive_r1=MetricsCalculator._safe_max(i1),
            current_negative_r1=MetricsCalculator._safe_min(i1),
            current_peakpeak_r1=MetricsCalculator._peak_to_peak(i1)
        )
        
        # Compute range 2 if specified
        if range2_start is not None and range2_end is not None:
            mask2 = (time_ms >= range2_start) & (time_ms <= range2_end)
            if np.any(mask2):
                v2, i2 = voltage[mask2], current[mask2]
                
                metrics.voltage_mean_r2 = MetricsCalculator._safe_mean(v2)
                metrics.voltage_absolute_r2 = MetricsCalculator._absolute_peak(v2)
                metrics.voltage_positive_r2 = MetricsCalculator._safe_max(v2)
                metrics.voltage_negative_r2 = MetricsCalculator._safe_min(v2)
                metrics.voltage_peakpeak_r2 = MetricsCalculator._peak_to_peak(v2)
                
                metrics.current_mean_r2 = MetricsCalculator._safe_mean(i2)
                metrics.current_absolute_r2 = MetricsCalculator._absolute_peak(i2)
                metrics.current_positive_r2 = MetricsCalculator._safe_max(i2)
                metrics.current_negative_r2 = MetricsCalculator._safe_min(i2)
                metrics.current_peakpeak_r2 = MetricsCalculator._peak_to_peak(i2)
        
        return metrics
    
    @staticmethod
    def _safe_mean(data: np.ndarray) -> float:
        """Calculate mean, returning NaN for empty arrays."""
        return np.mean(data) if len(data) > 0 else np.nan
    
    @staticmethod
    def _safe_max(data: np.ndarray) -> float:
        """Calculate max, returning NaN for empty arrays."""
        return np.max(data) if len(data) > 0 else np.nan
    
    @staticmethod
    def _safe_min(data: np.ndarray) -> float:
        """Calculate min, returning NaN for empty arrays."""
        return np.min(data) if len(data) > 0 else np.nan
    
    @staticmethod
    def _absolute_peak(data: np.ndarray) -> float:
        """Find value with maximum absolute magnitude."""
        if len(data) == 0:
            return np.nan
        return data[np.abs(data).argmax()]
    
    @staticmethod
    def _peak_to_peak(data: np.ndarray) -> float:
        """Calculate peak-to-peak amplitude."""
        if len(data) == 0:
            return np.nan
        return np.max(data) - np.min(data)