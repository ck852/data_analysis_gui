"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Conductance calculation service for electrophysiology data analysis.

This module provides stateless functions for calculating conductance (G = I / (V - Vrev))
from existing voltage and current metrics using SI units internally.

All calculations are performed in SI units (Amperes, Volts, Siemens) before
converting to user-specified output units.
"""

import numpy as np
from typing import Optional

from data_analysis_gui.core.metrics_calculator import SweepMetrics
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# SI Unit Conversion Factors
# =============================================================================

# Voltage conversion factors to Volts (V)
# Formula: voltage_V = voltage_value * factor
VOLTAGE_TO_V = {
    "V": 1.0,           # Base unit
    "mV": 1e-3,         # 1 mV = 0.001 V
    "μV": 1e-6,         # 1 μV = 0.000001 V
    "uV": 1e-6,         # Alternate spelling
    "kV": 1e3,          # 1 kV = 1000 V
}

# Current conversion factors to Amperes (A)
# Formula: current_A = current_value * factor
CURRENT_TO_A = {
    "A": 1.0,           # Base unit
    "mA": 1e-3,         # 1 mA = 0.001 A
    "μA": 1e-6,         # 1 μA = 0.000001 A
    "uA": 1e-6,         # Alternate spelling
    "nA": 1e-9,         # 1 nA = 0.000000001 A
    "pA": 1e-12,        # 1 pA = 0.000000000001 A
}

# Conductance conversion factors FROM Siemens (S) to target units
# Formula: conductance_target = conductance_S / factor
SIEMENS_TO_TARGET = {
    "S": 1.0,           # Base unit (no conversion)
    "mS": 1e-3,         # 1 mS = 0.001 S, so divide by 0.001
    "μS": 1e-6,         # 1 μS = 0.000001 S, so divide by 1e-6
    "nS": 1e-9,         # 1 nS = 0.000000001 S, so divide by 1e-9
    "pS": 1e-12,        # 1 pS = 0.000000000001 S, so divide by 1e-12
}


def calculate_conductance(
    metrics: SweepMetrics,
    params: AnalysisParameters,
    current_units: str,
    voltage_units: str,
    range_num: int = 1
) -> float:
    """
    Calculate conductance for a single sweep using G = I / (V - Vrev).
    
    All calculations performed in SI units (Amperes, Volts, Siemens) before
    converting to user-specified output units.
    
    Args:
        metrics: SweepMetrics object containing voltage and current data.
        params: AnalysisParameters with conductance_config.
        current_units: Current measurement units from file metadata (e.g., "pA", "nA", "μA").
        voltage_units: Voltage measurement units from file metadata (e.g., "mV", "V", "μV").
        range_num: Range number to use (1 or 2, default 1).
    
    Returns:
        Conductance value in specified units, or np.nan if calculation fails.
    
    Notes:
        - Returns np.nan if |V - Vrev| < tolerance (avoids division by zero)
        - Returns np.nan if required metrics are missing
        - All values converted to SI units (A, V) before calculation
        - Result (Siemens) converted to user-specified units
    """
    try:
        # Validate conductance config
        if params.conductance_config is None:
            logger.error("calculate_conductance called without conductance_config")
            return np.nan
        
        config = params.conductance_config
        
        # Determine peak type for measurements that use Peak
        peak_type = params.y_axis.peak_type if params.y_axis.peak_type else "Absolute"
        
        # Get current value (in file's native units)
        i_value = _get_measure_value(
            metrics=metrics,
            channel="current",
            measure=config.i_measure,
            peak_type=peak_type if config.i_measure == "Peak" else None,
            range_num=range_num
        )
        
        if i_value is None:
            logger.error(f"Failed to extract current value for sweep {metrics.sweep_index}")
            return np.nan
        
        # Convert current to Amperes (A)
        i_conversion_factor = CURRENT_TO_A.get(current_units)
        if i_conversion_factor is None:
            logger.error(f"Unknown current units: {current_units} - cannot calculate conductance")
            return np.nan
        
        i_value_A = i_value * i_conversion_factor
        logger.debug(
            f"Converting current to SI: {i_value:.2f}{current_units} × {i_conversion_factor} = {i_value_A:.6e}A"
        )
        
        # Get voltage value (in file's native units)
        v_value = _get_measure_value(
            metrics=metrics,
            channel="voltage",
            measure=config.v_measure,
            peak_type=peak_type if config.v_measure == "Peak" else None,
            range_num=range_num
        )
        
        if v_value is None:
            logger.error(f"Failed to extract voltage value for sweep {metrics.sweep_index}")
            return np.nan
        
        # Convert voltage to Volts (V)
        v_conversion_factor = VOLTAGE_TO_V.get(voltage_units)
        if v_conversion_factor is None:
            logger.error(f"Unknown voltage units: {voltage_units} - cannot calculate conductance")
            return np.nan
        
        v_value_V = v_value * v_conversion_factor
        logger.debug(
            f"Converting voltage to SI: {v_value:.2f}{voltage_units} × {v_conversion_factor} = {v_value_V:.6f}V"
        )
        
        # Convert reversal potential from mV to Volts
        vrev_V = config.vrev * VOLTAGE_TO_V["mV"]
        logger.debug(f"Converting Vrev to SI: {config.vrev:.2f}mV × {VOLTAGE_TO_V['mV']} = {vrev_V:.6f}V")
        
        # Calculate voltage difference from reversal potential (in Volts)
        v_diff_V = v_value_V - vrev_V
        
        # Convert tolerance from mV to V for comparison
        tolerance_V = config.tolerance * VOLTAGE_TO_V["mV"]
        
        # Check if voltage is too close to reversal potential
        if abs(v_diff_V) < tolerance_V:
            logger.debug(
                f"Skipping sweep {metrics.sweep_index}: V ({v_value_V*1e3:.2f}mV) "
                f"too close to Vrev ({vrev_V*1e3:.2f}mV), |diff|={abs(v_diff_V)*1e3:.3f}mV"
            )
            return np.nan
        
        # Calculate conductance in Siemens (S)
        # G = I / V, where I is in Amperes and V is in Volts
        conductance_S = i_value_A / v_diff_V
        
        logger.debug(
            f"SI calculation: G = {i_value_A:.6e}A / {v_diff_V:.6f}V = {conductance_S:.6e}S"
        )
        
        # Convert from Siemens to target units
        unit_factor = SIEMENS_TO_TARGET.get(config.units)
        if unit_factor is None:
            logger.error(f"Unknown conductance units: {config.units} - cannot calculate conductance")
            return np.nan
        
        conductance_target = conductance_S / unit_factor
        
        logger.debug(
            f"Sweep {metrics.sweep_index}: G={conductance_target:.3f}{config.units} "
            f"(I={i_value:.2f}{current_units}, V={v_value:.2f}{voltage_units}, Vrev={config.vrev:.2f}mV)"
        )
        
        return conductance_target
    
    except Exception as e:
        logger.error(
            f"Error calculating conductance for sweep {metrics.sweep_index}: {e}",
            exc_info=True
        )
        return np.nan


def _get_measure_value(
    metrics: SweepMetrics,
    channel: str,
    measure: str,
    peak_type: Optional[str],
    range_num: int
) -> Optional[float]:
    """
    Extract a measurement value from SweepMetrics based on channel, measure type, and range.
    
    Args:
        metrics: SweepMetrics object.
        channel: Channel name ("voltage" or "current").
        measure: Measurement type ("Average" or "Peak").
        peak_type: Peak type if measure is "Peak" ("Absolute", "Positive", "Negative", "Peak-Peak").
        range_num: Range number (1 or 2).
    
    Returns:
        Extracted measurement value, or None if metric not found.
    """
    try:
        # Build metric attribute name
        if measure == "Average":
            metric_name = f"{channel}_mean_r{range_num}"
        
        elif measure == "Peak":
            # Normalize peak type for attribute lookup
            if peak_type is None:
                peak_type = "Absolute"
            
            # Normalize peak type string (case-insensitive, handle variations)
            normalized_peak = (
                peak_type.lower()
                .replace(" ", "")
                .replace("-", "")
                .replace("_", "")
            )
            
            # Map to attribute suffix
            peak_map = {
                "absolute": "absolute",
                "positive": "positive",
                "negative": "negative",
                "peakpeak": "peakpeak",
                "peaktopeak": "peakpeak",
                "p2p": "peakpeak",
                "pp": "peakpeak",
            }
            
            peak_suffix = peak_map.get(normalized_peak, "absolute")
            metric_name = f"{channel}_{peak_suffix}_r{range_num}"
        
        else:
            logger.error(f"Unknown measure type: {measure}")
            return None
        
        # Extract value from metrics
        value = getattr(metrics, metric_name, None)
        
        if value is None:
            logger.warning(
                f"Metric '{metric_name}' is None for sweep {metrics.sweep_index}"
            )
            return None
        
        return value
    
    except AttributeError as e:
        logger.error(
            f"Metric '{metric_name}' not found in SweepMetrics for sweep {metrics.sweep_index}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Error extracting measure value for sweep {metrics.sweep_index}: {e}",
            exc_info=True
        )
        return None