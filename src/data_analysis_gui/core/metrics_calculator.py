"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module provides core functionality for calculating quantitative metrics from electrophysiology time series data.
It defines the SweepMetrics dataclass for storing computed metrics for individual sweeps, including mean, peak, and peak-to-peak values for voltage and current over specified time ranges.

The MetricsCalculator class offers static methods for extracting and computing these metrics from numpy arrays of time, voltage, and current data.
Typical usage involves calling MetricsCalculator.compute_sweep_metrics() with the relevant data and time ranges to obtain a SweepMetrics object.

Classes:
    - SweepMetrics: Stores computed metrics for a single sweep, supporting up to two analysis ranges.
    - MetricsCalculator: Stateless class with static methods for calculating metrics from time series data.

Intended for use in automated analysis pipelines and GUI applications for patch clamp electrophysiology data.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from data_analysis_gui.core.exceptions import (
    DataError,
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SweepMetrics:
    """
    Computed metrics for a single sweep.

    Stores calculated voltage and current metrics for one sweep, including mean, absolute, positive, negative, and peak-to-peak values for up to two ranges.

    Args:
        sweep_index (str): Identifier for the sweep.
        time_s (float): Time in seconds for the sweep.
        voltage_mean_r1 (float): Mean voltage for range 1.
        voltage_absolute_r1 (float): Absolute peak voltage for range 1.
        voltage_positive_r1 (float): Maximum positive voltage for range 1.
        voltage_negative_r1 (float): Minimum negative voltage for range 1.
        voltage_peakpeak_r1 (float): Peak-to-peak voltage for range 1.
        current_mean_r1 (float): Mean current for range 1.
        current_absolute_r1 (float): Absolute peak current for range 1.
        current_positive_r1 (float): Maximum positive current for range 1.
        current_negative_r1 (float): Minimum negative current for range 1.
        current_peakpeak_r1 (float): Peak-to-peak current for range 1.
        voltage_mean_r2 (Optional[float]): Mean voltage for range 2.
        voltage_absolute_r2 (Optional[float]): Absolute peak voltage for range 2.
        voltage_positive_r2 (Optional[float]): Maximum positive voltage for range 2.
        voltage_negative_r2 (Optional[float]): Minimum negative voltage for range 2.
        voltage_peakpeak_r2 (Optional[float]): Peak-to-peak voltage for range 2.
        current_mean_r2 (Optional[float]): Mean current for range 2.
        current_absolute_r2 (Optional[float]): Absolute peak current for range 2.
        current_positive_r2 (Optional[float]): Maximum positive current for range 2.
        current_negative_r2 (Optional[float]): Minimum negative current for range 2.
        current_peakpeak_r2 (Optional[float]): Peak-to-peak current for range 2.

    Deprecated Properties:
        voltage_peak_r1: Use voltage_absolute_r1 instead.
        current_peak_r1: Use current_absolute_r1 instead.
        voltage_min_r1: Use voltage_negative_r1 instead.
        voltage_max_r1: Use voltage_positive_r1 instead.
        current_min_r1: Use current_negative_r1 instead.
        current_max_r1: Use current_positive_r1 instead.
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

    # Deprecated fields for compatibility
    @property
    def voltage_peak_r1(self):
        return self.voltage_absolute_r1

    @property
    def current_peak_r1(self):
        return self.current_absolute_r1

    @property
    def voltage_min_r1(self):
        return self.voltage_negative_r1

    @property
    def voltage_max_r1(self):
        return self.voltage_positive_r1

    @property
    def current_min_r1(self):
        return self.current_negative_r1

    @property
    def current_max_r1(self):
        return self.current_positive_r1


class MetricsCalculator:
    """
    Pure calculation of metrics from time series data.

    Stateless class for computing voltage and current metrics from time series data arrays.
    All methods are static and do not maintain state.
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
        actual_sweep_time: float,
        range2_start: Optional[float] = None,
        range2_end: Optional[float] = None,
    ) -> SweepMetrics:
        """
        Compute metrics for a single sweep.

        Args:
            time_ms (np.ndarray): Time values in milliseconds.
            voltage (np.ndarray): Voltage data array.
            current (np.ndarray): Current data array.
            sweep_index (str): Identifier for the sweep.
            sweep_number (int): Sweep number (0-based index in sweep list).
            range1_start (float): Start time for range 1 (ms).
            range1_end (float): End time for range 1 (ms).
            actual_sweep_time (float): Actual sweep time from file metadata (seconds).
                Must be provided from ABF/WCP file metadata.
            range2_start (Optional[float]): Start time for range 2 (ms).
            range2_end (Optional[float]): End time for range 2 (ms).

        Returns:
            SweepMetrics: Computed metrics for the sweep.

        Raises:
            DataError: If the requested ranges contain no samples.
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
        """
        Calculate the mean of the data array.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            float: Mean value, or NaN if array is empty.
        """
        return np.mean(data) if len(data) > 0 else np.nan

    @staticmethod
    def _safe_max(data: np.ndarray) -> float:
        """
        Calculate the maximum value of the data array.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            float: Maximum value, or NaN if array is empty.
        """
        return np.max(data) if len(data) > 0 else np.nan

    @staticmethod
    def _safe_min(data: np.ndarray) -> float:
        """
        Calculate the minimum value of the data array.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            float: Minimum value, or NaN if array is empty.
        """
        return np.min(data) if len(data) > 0 else np.nan

    @staticmethod
    def _absolute_peak(data: np.ndarray) -> float:
        """
        Find the value with maximum absolute magnitude in the data array.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            float: Value with maximum absolute magnitude, or NaN if array is empty.
        """
        if len(data) == 0:
            return np.nan
        return data[np.abs(data).argmax()]

    @staticmethod
    def _peak_to_peak(data: np.ndarray) -> float:
        """
        Calculate peak-to-peak amplitude of the data array.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            float: Peak-to-peak amplitude, or NaN if array is empty.
        """
        if len(data) == 0:
            return np.nan
        return np.max(data) - np.min(data)
