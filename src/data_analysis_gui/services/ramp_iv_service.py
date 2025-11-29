"""
Ramp IV Service

This module provides ramp IV analysis functionality as a stateless service.
Expands usability for users with ramp protocols.

Finds closest voltage points to those specified by user and extracts the 
current (I) values at those points. Can be used on one or multiple sweeps.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.data_extractor import DataExtractor
from data_analysis_gui.config.exceptions import DataError, ValidationError
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)

DEFAULT_VOLTAGE_TOLERANCE_MV = 3.0
# Defines maximum acceptable difference between target voltage and actual voltage point in data file.

@dataclass
class RampIVResult:

    success: bool
    voltage_targets: List[float] = field(default_factory=list)
    extracted_data: Dict[str, Dict[float, float]] = field(default_factory=dict)
    closest_voltages: Dict[str, Dict[float, float]] = field(default_factory=dict)
    processed_sweeps: List[str] = field(default_factory=list)
    failed_sweeps: List[str] = field(default_factory=list)
    analysis_range_ms: Tuple[float, float] = None
    current_units: str = "pA"
    error_message: str = ""


class RampIVService:

    @staticmethod
    def find_closest_voltage_point(
        time_ms: np.ndarray,
        voltage_data: np.ndarray, 
        target_voltage: float,
        start_ms: float,
        end_ms: float,
        tolerance_mv: float = DEFAULT_VOLTAGE_TOLERANCE_MV
    ) -> Tuple[Optional[int], Optional[float]]:

        # Validate inputs
        if time_ms is None or voltage_data is None:
            raise ValidationError("Time and voltage data cannot be None")
        if len(time_ms) != len(voltage_data):
            raise ValidationError("Time and voltage arrays must have same length")
        if start_ms >= end_ms:
            raise ValidationError(f"Invalid range: start ({start_ms}) >= end ({end_ms})")
            
        time_ms = np.asarray(time_ms)
        voltage_data = np.asarray(voltage_data)
        
        # Find indices within analysis range
        range_mask = (time_ms >= start_ms) & (time_ms <= end_ms)
        if not np.any(range_mask):
            raise DataError(f"No data points in range [{start_ms}, {end_ms}] ms")
            
        # Get valid (non-NaN) voltages within range
        range_voltages = voltage_data[range_mask]
        range_indices = np.where(range_mask)[0]
        valid_mask = ~np.isnan(range_voltages)
        
        if not np.any(valid_mask):
            raise DataError("Analysis range contains only NaN voltage values")
            
        valid_voltages = range_voltages[valid_mask]
        valid_indices = range_indices[valid_mask]
        
        # Find closest voltage
        voltage_differences = np.abs(valid_voltages - target_voltage)
        closest_idx = np.argmin(voltage_differences)
        closest_difference = voltage_differences[closest_idx]
        
        # Check tolerance
        if closest_difference > tolerance_mv:
            logger.warning(f"Voltage difference {closest_difference:.1f} mV exceeds tolerance {tolerance_mv} mV")
            return None, None
            
        original_index = valid_indices[closest_idx]
        actual_voltage = valid_voltages[closest_idx]
        
        return int(original_index), float(actual_voltage)

    @staticmethod
    def extract_ramp_iv_data(
        dataset: ElectrophysiologyDataset,
        selected_sweeps: List[str],
        target_voltages: List[float],
        start_ms: float,
        end_ms: float,
        current_units: str = "pA",
        voltage_tolerance_mv: float = DEFAULT_VOLTAGE_TOLERANCE_MV
    ) -> RampIVResult:

        # Basic validation
        if not dataset or dataset.is_empty():
            return RampIVResult(success=False, error_message="Dataset is empty or None")
        if not selected_sweeps:
            return RampIVResult(success=False, error_message="No sweeps selected")
        if not target_voltages:
            return RampIVResult(success=False, error_message="No target voltages specified")
        if start_ms >= end_ms:
            return RampIVResult(success=False, error_message="Invalid time range")
        
        # Create data extractor - no arguments needed, reads from dataset metadata
        data_extractor = DataExtractor()
        extracted_data = {}
        closest_voltages = {}
        processed_sweeps = []
        failed_sweeps = []
        
        logger.info(f"Processing {len(selected_sweeps)} sweeps with {len(target_voltages)} voltage targets")
        
        # Process each sweep
        for sweep_idx in selected_sweeps:
            try:
                # Extract sweep data
                sweep_data = data_extractor.extract_sweep_data(dataset, sweep_idx)
                time_ms = sweep_data["time_ms"]
                voltage = sweep_data["voltage"]
                current = sweep_data["current"]
                
                sweep_iv_data = {}
                sweep_closest_voltages = {}
                successful_extractions = 0
                
                # Extract current for each target voltage
                for target_voltage in target_voltages:
                    try:
                        closest_idx, actual_voltage = RampIVService.find_closest_voltage_point(
                            time_ms, voltage, target_voltage, start_ms, end_ms, voltage_tolerance_mv
                        )
                        
                        if closest_idx is not None:
                            extracted_current = current[closest_idx]
                            if not np.isnan(extracted_current):
                                successful_extractions += 1
                            sweep_iv_data[target_voltage] = float(extracted_current)
                            sweep_closest_voltages[target_voltage] = actual_voltage
                        else:
                            sweep_iv_data[target_voltage] = np.nan
                            sweep_closest_voltages[target_voltage] = np.nan
                            
                    except (DataError, ValidationError):
                        sweep_iv_data[target_voltage] = np.nan
                        sweep_closest_voltages[target_voltage] = np.nan
                
                # Store results if at least some targets were found
                if successful_extractions > 0:
                    extracted_data[sweep_idx] = sweep_iv_data
                    closest_voltages[sweep_idx] = sweep_closest_voltages
                    processed_sweeps.append(sweep_idx)
                else:
                    failed_sweeps.append(sweep_idx)
                
            except Exception as e:
                logger.error(f"Error processing sweep {sweep_idx}: {e}")
                failed_sweeps.append(sweep_idx)
        
        success = len(processed_sweeps) > 0
        logger.info(f"Processed {len(processed_sweeps)}/{len(selected_sweeps)} sweeps successfully")
        
        return RampIVResult(
            success=success,
            voltage_targets=target_voltages.copy(),
            extracted_data=extracted_data,
            closest_voltages=closest_voltages,
            processed_sweeps=processed_sweeps,
            failed_sweeps=failed_sweeps,
            analysis_range_ms=(start_ms, end_ms),
            current_units=current_units,
            error_message="" if success else "No sweeps processed successfully"
        )

    @staticmethod
    def prepare_export_table(
        result: RampIVResult,
        sweep_order: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        if not result.success or not result.extracted_data:
            return {
                "headers": ["Voltage (mV)"],
                "data": np.array([result.voltage_targets]).T if result.voltage_targets else np.array([[]]),
                "format_spec": "%.6f"
            }
        
        # Determine sweep order
        if sweep_order is None:
            sweeps_to_include = result.processed_sweeps.copy()
            try:
                sweeps_to_include.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
            except (ValueError, TypeError):
                sweeps_to_include.sort()
        else:
            sweeps_to_include = [s for s in sweep_order if s in result.extracted_data]
        
        if not sweeps_to_include:
            return {"headers": ["Voltage (mV)"], "data": np.array([result.voltage_targets]).T, "format_spec": "%.6f"}
        
        # Build headers and data matrix
        headers = ["Voltage (mV)"]
        for sweep_idx in sweeps_to_include:
            headers.append(f"Current ({result.current_units})_sweep_{sweep_idx}")
        
        n_voltages = len(result.voltage_targets)
        n_sweeps = len(sweeps_to_include)
        data_matrix = np.zeros((n_voltages, 1 + n_sweeps))
        
        # Fill data
        data_matrix[:, 0] = result.voltage_targets
        for col_idx, sweep_idx in enumerate(sweeps_to_include):
            sweep_data = result.extracted_data[sweep_idx]
            for row_idx, target_voltage in enumerate(result.voltage_targets):
                data_matrix[row_idx, col_idx + 1] = sweep_data.get(target_voltage, np.nan)
        
        logger.info(f"Export table: {n_voltages} voltages × {n_sweeps} sweeps")
        
        return {
            "headers": headers,
            "data": data_matrix,
            "format_spec": "%.6f"
        }

    @staticmethod
    def get_plot_data_arrays(
        result: RampIVResult,
        sweep_indices: Optional[List[str]] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract (voltage, current) arrays for plotting, excluding NaN values.
        """
        if not result.success or not result.extracted_data:
            return {}
        
        sweep_indices = sweep_indices or result.processed_sweeps
        sweep_indices = [s for s in sweep_indices if s in result.extracted_data]
        
        plot_data = {}
        for sweep_idx in sweep_indices:
            sweep_data = result.extracted_data[sweep_idx]
            
            # Extract non-NaN voltage-current pairs
            voltages = []
            currents = []
            for voltage_target in result.voltage_targets:
                if voltage_target in sweep_data:
                    current_value = sweep_data[voltage_target]
                    if not np.isnan(current_value):
                        voltages.append(voltage_target)
                        currents.append(current_value)
            
            if voltages:
                plot_data[sweep_idx] = (np.array(voltages), np.array(currents))
        
        return plot_data

    @staticmethod
    def validate_voltage_targets(voltage_targets_str: str) -> Tuple[bool, List[float], str]:
        """
        Parse and validate comma-separated voltage string (e.g., "-80, -60, -40, 0, 40").
        """
        try:
            parts = [p.strip() for p in voltage_targets_str.split(",") if p.strip()]
            
            if not parts:
                return False, [], "No voltage values provided"
            
            voltages = []
            for part in parts:
                try:
                    value = float(part)
                    if abs(value) > 500:
                        return False, [], f"Voltage {value} mV out of reasonable range (±500 mV)"
                    voltages.append(value)
                except ValueError:
                    return False, [], f"Invalid voltage value: '{part}'"
            
            # Remove duplicates and sort
            voltages = sorted(list(set(voltages)))
            
            if len(voltages) < 2:
                return False, [], "Need at least 2 different voltage values"
            
            return True, voltages, ""
            
        except Exception as e:
            return False, [], f"Error parsing voltages: {str(e)}"