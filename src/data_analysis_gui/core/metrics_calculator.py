"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Calculates ALL metrics after being called by AnalysisEngine. Stateless class with only static methods. Passes to PlotFormatter which selects 
the desired results and formats them for plotting/exporting in GUI.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from data_analysis_gui.config.exceptions import DataError
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SweepMetrics:
    """
    Computed voltage and current metrics for a single sweep.

    Contains mean, absolute peak, positive/negative peaks, and peak-to-peak values 
    for both voltage and current. Supports optional second range metrics (r2).
    """

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

class MetricsCalculator:
    """Stateless calculator for voltage/current metrics from time series data."""

    @staticmethod
    def compute_sweep_metrics(
        time_ms: np.ndarray,
        voltage: np.ndarray,
        current: np.ndarray,
        sweep_index: str,
        sweep_number: int,  # deprecated, keep in case useful later
        range1_start: float,
        range1_end: float,
        actual_sweep_time: float,
        range2_start: Optional[float] = None,
        range2_end: Optional[float] = None,
    ) -> SweepMetrics:
        """
        Compute voltage and current metrics for a single sweep.

        Extracts data within specified time ranges and calculates mean, peak, and 
        peak-to-peak values. Uses actual_sweep_time from file metadata rather than 
        inferring from the time array. Raises DataError if time array is empty or 
        no data exists in range1.
        """
        # Validate inputs
        if len(time_ms) == 0:
            raise DataError(f"Empty time array for sweep {sweep_index}")

        # Use actual sweep time from file metadata
        time_s = actual_sweep_time

        # Extract range 1 data
        mask1 = (time_ms >= range1_start) & (time_ms <= range1_end)
        if not np.any(mask1):
            raise DataError(
                f"No data in range [{range1_start}, {range1_end}]",
                details={
                    "sweep": sweep_index,
                    "time_range": (time_ms.min(), time_ms.max()),
                },
            )

        v1, i1 = voltage[mask1], current[mask1]

        # Compute range 1 metrics
        metrics = SweepMetrics(
            sweep_index=sweep_index,
            time_s=time_s,
            voltage_mean_r1=MetricsCalculator._safe_mean(v1),
            voltage_absolute_r1=MetricsCalculator._absolute_peak(v1),
            voltage_positive_r1=MetricsCalculator._safe_max(v1),
            voltage_negative_r1=MetricsCalculator._safe_min(v1),
            voltage_peakpeak_r1=MetricsCalculator._peak_to_peak(v1),
            current_mean_r1=MetricsCalculator._safe_mean(i1),
            current_absolute_r1=MetricsCalculator._absolute_peak(i1),
            current_positive_r1=MetricsCalculator._safe_max(i1),
            current_negative_r1=MetricsCalculator._safe_min(i1),
            current_peakpeak_r1=MetricsCalculator._peak_to_peak(i1),
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
        """Return mean of data, or nan if empty."""
        return np.mean(data) if len(data) > 0 else np.nan

# ================= Peak Modes =================

    # Absolute Peak
    @staticmethod
    def _absolute_peak(data: np.ndarray) -> float:

        if len(data) == 0:
            return np.nan
        return data[np.abs(data).argmax()]

    # Positive Peak
    @staticmethod
    def _safe_max(data: np.ndarray) -> float:
        """Return maximum value, or nan if empty."""
        return np.max(data) if len(data) > 0 else np.nan

    # Negative Peak
    @staticmethod
    def _safe_min(data: np.ndarray) -> float:
        """Return minimum value, or nan if empty."""
        return np.min(data) if len(data) > 0 else np.nan

    # Peak-Peak
    @staticmethod
    def _peak_to_peak(data: np.ndarray) -> float:
        """Return difference between max and min, or nan if empty."""
        if len(data) == 0:
            return np.nan
        return np.max(data) - np.min(data)
