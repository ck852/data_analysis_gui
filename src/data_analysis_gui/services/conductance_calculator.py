"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Conductance calculation service for electrophysiology data analysis.

This module provides stateless functions for calculating conductance (G = I / (V - Vrev))
from existing voltage and current metrics. Supports configurable measurement types
(Average or Peak), reversal potentials, and output units (nS, μS, pS).
"""

import numpy as np
from typing import Optional

from data_analysis_gui.core.metrics_calculator import SweepMetrics
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)

# Unit conversion factors for conductance output
# Base unit is nS (nanoSiemens)
# Formula: result = conductance_nS / factor
CONDUCTANCE_UNITS = {
    "pS": 0.001,    # 1 nS = 1000 pS, so divide by 0.001 (multiply by 1000)
    "nS": 1.0,      # Base unit (no conversion)
    "μS": 1000.0,   # 1000 nS = 1 μS, so divide by 1000
    "mS": 1e6,      # 1,000,000 nS = 1 mS, so divide by 1e6
    "S": 1e9,       # 1,000,000,000 nS = 1 S, so divide by 1e9
}

# Current unit conversion factors to pA (picoAmperes)
# Formula: current_pA = current_value * factor
CURRENT_TO_PA = {
    "pA": 1.0,           # Base unit
    "nA": 1e3,           # 1 nA = 1000 pA
    "μA": 1e6,           # 1 μA = 1,000,000 pA
    "uA": 1e6,           # Alternate spelling
    "mA": 1e9,           # 1 mA = 1,000,000,000 pA
    "A": 1e12,           # 1 A = 1,000,000,000,000 pA
}


def calculate_conductance(
    metrics: SweepMetrics,
    params: AnalysisParameters,
    current_units: str,
    range_num: int = 1
) -> float:
    """
    Calculate conductance for a single sweep using G = I / (V - Vrev).
    
    Args:
        metrics: SweepMetrics object containing voltage and current data.
        params: AnalysisParameters with conductance_config.
        current_units: Current measurement units from file metadata (e.g., "pA", "nA", "μA").
        range_num: Range number to use (1 or 2, default 1).
    
    Returns:
        Conductance value in specified units, or np.nan if calculation fails.
    
    Notes:
        - Returns np.nan if |V - Vrev| < tolerance (avoids division by zero)
        - Returns np.nan if required metrics are missing
        - Normalizes current to pA and voltage to mV before calculation
        - Base calculation: pA/mV = nS (by unit analysis)
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
        
        # Convert current to pA if needed
        i_conversion_factor = CURRENT_TO_PA.get(current_units, 1.0)
        if i_conversion_factor != 1.0:
            logger.debug(
                f"Converting current from {current_units} to pA: "
                f"{i_value:.2f}{current_units} × {i_conversion_factor} = {i_value * i_conversion_factor:.2f}pA"
            )
        i_value_pA = i_value * i_conversion_factor
        
        # Get voltage value (assumed to be in mV)
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
        
        # Calculate voltage difference from reversal potential
        v_diff = v_value - config.vrev
        
        # Check if voltage is too close to reversal potential
        if abs(v_diff) < config.tolerance:
            logger.debug(
                f"Skipping sweep {metrics.sweep_index}: V ({v_value:.2f}mV) "
                f"too close to Vrev ({config.vrev:.2f}mV), |diff|={abs(v_diff):.3f}mV"
            )
            return np.nan
        
        # Calculate conductance in nS (pA/mV = nS by unit analysis)
        conductance_nS = i_value_pA / v_diff
        
        # Convert to target units
        unit_factor = CONDUCTANCE_UNITS.get(config.units, 1.0)
        conductance_target = conductance_nS / unit_factor
        
        logger.debug(
            f"Sweep {metrics.sweep_index}: I={i_value_pA:.2f}pA, V={v_value:.2f}mV, "
            f"Vrev={config.vrev:.2f}mV, G={conductance_target:.3f}{config.units}"
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