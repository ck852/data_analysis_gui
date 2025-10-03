"""
Background Subtraction Service for PatchBatch Electrophysiology Data Analysis Tool

This module provides background subtraction functionality as a stateless service.
It handles the calculation of background averages and application of background 
subtraction to entire electrophysiology datasets.

Features:
    - Calculate background average from specified time range
    - Apply background subtraction to all sweeps in a dataset
    - Robust error handling for missing data or invalid ranges
    - Comprehensive logging for debugging and traceability
    - Stateless operations for thread safety and simplicity

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.data_extractor import DataExtractor
from data_analysis_gui.core.exceptions import DataError, ValidationError
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BackgroundSubtractionResult:
    """
    Result of background subtraction operation.
    
    Attributes:
        success (bool): Whether the operation was successful
        processed_sweeps (int): Number of sweeps successfully processed
        total_sweeps (int): Total number of sweeps in the dataset
        failed_sweeps (List[str]): List of sweep indices that failed processing
        background_range_ms (Tuple[float, float]): The background range used (start, end)
        error_message (str): Error message if operation failed
    """
    success: bool
    processed_sweeps: int = 0
    total_sweeps: int = 0
    failed_sweeps: List[str] = None
    background_range_ms: Tuple[float, float] = None
    error_message: str = ""
    
    def __post_init__(self):
        if self.failed_sweeps is None:
            self.failed_sweeps = []


class BackgroundSubtractionService:
    """
    Stateless service for background subtraction operations on electrophysiology data.
    
    This service provides static methods for calculating background averages and applying
    background subtraction to entire datasets. All operations are stateless and can be
    called without instantiating the class.
    """

    @staticmethod
    def calculate_background_average(
        time_ms: np.ndarray, 
        current_data: np.ndarray, 
        start_ms: float, 
        end_ms: float
    ) -> float:
        """
        Calculate the average current value within a specified time range.
        
        Args:
            time_ms (np.ndarray): Time values in milliseconds
            current_data (np.ndarray): Current data values (pA)
            start_ms (float): Start time of background range (ms)
            end_ms (float): End time of background range (ms)
            
        Returns:
            float: Average current value in the specified range
            
        Raises:
            ValidationError: If input parameters are invalid
            DataError: If no data points exist in the specified range
        """
        # Validate inputs
        if time_ms is None or current_data is None:
            raise ValidationError("Time and current data cannot be None")
            
        if len(time_ms) != len(current_data):
            raise ValidationError(
                f"Time and current arrays must have same length: "
                f"time={len(time_ms)}, current={len(current_data)}"
            )
            
        if start_ms >= end_ms:
            raise ValidationError(
                f"Start time ({start_ms}) must be less than end time ({end_ms})"
            )
            
        # Convert to numpy arrays if needed
        time_ms = np.asarray(time_ms)
        current_data = np.asarray(current_data)
        
        # Find indices within the background range
        mask = (time_ms >= start_ms) & (time_ms <= end_ms)
        
        if not np.any(mask):
            raise DataError(
                f"No data points found in range [{start_ms}, {end_ms}] ms. "
                f"Available time range: [{time_ms.min():.1f}, {time_ms.max():.1f}] ms"
            )
            
        # Extract background data and calculate average
        background_data = current_data[mask]
        
        if len(background_data) == 0:
            raise DataError("Background range contains no valid data points")
            
        # Handle NaN values
        valid_data = background_data[~np.isnan(background_data)]
        if len(valid_data) == 0:
            raise DataError("Background range contains only NaN values")
            
        if len(valid_data) < len(background_data):
            logger.warning(
                f"Background range contains {len(background_data) - len(valid_data)} NaN values, "
                f"using {len(valid_data)} valid points for average calculation"
            )
            
        background_avg = np.mean(valid_data)
        
        logger.debug(
            f"Calculated background average: {background_avg:.3f} pA "
            f"from {len(valid_data)} data points in range [{start_ms}, {end_ms}] ms"
        )
        
        return float(background_avg)

    @staticmethod
    def apply_background_subtraction(
        dataset: ElectrophysiologyDataset,
        start_ms: float,
        end_ms: float
    ) -> BackgroundSubtractionResult:
        """
        Apply background subtraction to all sweeps in the dataset.
        
        This method processes each sweep in the dataset by:
        1. Extracting time and current data for the sweep
        2. Calculating the background average in the specified range
        3. Subtracting the background average from all current values
        4. Updating the dataset with the corrected current data
        
        Args:
            dataset (ElectrophysiologyDataset): Dataset to process 
            start_ms (float): Start time of background range (ms)
            end_ms (float): End time of background range (ms)
            
        Returns:
            BackgroundSubtractionResult: Result object containing success status,
                number of processed sweeps, and any error information
        """
        # Validate inputs
        if dataset is None:
            return BackgroundSubtractionResult(
                success=False, 
                error_message="Dataset cannot be None"
            )
            
        if dataset.is_empty():
            return BackgroundSubtractionResult(
                success=False, 
                error_message="Dataset is empty"
            )
            
        if start_ms >= end_ms:
            return BackgroundSubtractionResult(
                success=False, 
                error_message=f"Invalid range: start ({start_ms}) must be less than end ({end_ms})"
            )
        
        # Initialize data extractor and result tracking
        channel_config = dataset.metadata.get('channel_config')
        if not channel_config:
            return BackgroundSubtractionResult(
                success=False,
                error_message="No channel configuration in dataset"
            )
        
        current_channel = channel_config['current_channel']
        data_extractor = DataExtractor()
        
        sweep_indices = list(dataset.sweeps())
        total_sweeps = len(sweep_indices)
        processed_sweeps = 0
        failed_sweeps = []
        
        logger.info(
            f"Starting background subtraction on {total_sweeps} sweeps, "
            f"range: [{start_ms}, {end_ms}] ms, current channel: {current_channel}"
        )
        
        # Process each sweep
        for sweep_idx in sweep_indices:
            try:
                # Extract sweep data
                sweep_data = data_extractor.extract_sweep_data(dataset, sweep_idx)
                time_ms = sweep_data["time_ms"]
                current = sweep_data["current"]
                voltage = sweep_data["voltage"]  # Preserve voltage data
                
                # Calculate background average for this sweep
                background_avg = BackgroundSubtractionService.calculate_background_average(
                    time_ms, current, start_ms, end_ms
                )
                
                # Apply background subtraction
                corrected_current = current - background_avg
                
                # Get original sweep structure to preserve all data
                original_sweep = dataset.get_sweep(sweep_idx)
                if original_sweep is None:
                    logger.warning(f"Could not retrieve original sweep {sweep_idx}")
                    failed_sweeps.append(sweep_idx)
                    continue
                    
                time_orig, data_matrix_orig = original_sweep
                
                # Create new data matrix with corrected current
                data_matrix_new = data_matrix_orig.copy()
                data_matrix_new[:, current_channel] = corrected_current
                
                # Update the dataset with corrected data
                dataset.add_sweep(sweep_idx, time_orig, data_matrix_new)
                
                processed_sweeps += 1
                logger.debug(
                    f"Applied background subtraction to sweep {sweep_idx}: "
                    f"background avg = {background_avg:.3f} pA, "
                    f"corrected {len(corrected_current)} data points"
                )
                
            except (DataError, ValidationError) as e:
                logger.error(f"Failed to process sweep {sweep_idx}: {e}")
                failed_sweeps.append(sweep_idx)
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error processing sweep {sweep_idx}: {e}")
                failed_sweeps.append(sweep_idx)
                continue
        
        # Determine overall success
        success = processed_sweeps > 0
        
        if success:
            success_rate = (processed_sweeps / total_sweeps) * 100
            logger.info(
                f"Background subtraction completed: {processed_sweeps}/{total_sweeps} "
                f"sweeps processed successfully ({success_rate:.1f}%)"
            )
            
            if failed_sweeps:
                logger.warning(
                    f"Failed to process {len(failed_sweeps)} sweeps: {failed_sweeps}"
                )
        else:
            logger.error("Background subtraction failed: no sweeps were processed successfully")
        
        return BackgroundSubtractionResult(
            success=success,
            processed_sweeps=processed_sweeps,
            total_sweeps=total_sweeps,
            failed_sweeps=failed_sweeps,
            background_range_ms=(start_ms, end_ms),
            error_message="" if success else "No sweeps were processed successfully"
        )

    @staticmethod
    def validate_background_range(
        dataset: ElectrophysiologyDataset,
        start_ms: float,
        end_ms: float
    ) -> Tuple[bool, str]:
        """
        Validate that a background range is suitable for the given dataset.
        
        Args:
            dataset (ElectrophysiologyDataset): Dataset to validate against
            start_ms (float): Start time of background range (ms)
            end_ms (float): End time of background range (ms)
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if dataset is None or dataset.is_empty():
            return False, "Dataset is empty or None"
            
        if start_ms >= end_ms:
            return False, f"Start time ({start_ms}) must be less than end time ({end_ms})"
        
        # Check against dataset time bounds
        max_sweep_time = dataset.get_max_sweep_time()
        if max_sweep_time > 0:
            if start_ms < 0:
                return False, f"Start time ({start_ms}) cannot be negative"
                
            if end_ms > max_sweep_time:
                return False, (
                    f"End time ({end_ms}) exceeds maximum sweep time ({max_sweep_time:.1f})"
                )
        
        # Check that at least one sweep has data in the range
        has_data_in_range = False
        for sweep_idx in list(dataset.sweeps())[:5]:  # Check first 5 sweeps for efficiency
            sweep = dataset.get_sweep(sweep_idx)
            if sweep is not None:
                time_ms, _ = sweep
                if len(time_ms) > 0:
                    sweep_min, sweep_max = time_ms.min(), time_ms.max()
                    if start_ms <= sweep_max and end_ms >= sweep_min:
                        has_data_in_range = True
                        break
        
        if not has_data_in_range:
            return False, f"Background range [{start_ms}, {end_ms}] contains no data"
            
        return True, ""