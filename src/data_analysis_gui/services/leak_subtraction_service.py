"""
Leak Subtraction Service

Performs P/N leak subtraction on WCP files using voltage-scaled subtraction.

Core Algorithm:
1. Baseline Correction: All voltage and current traces are zeroed by subtracting 
   the value at the VHold cursor position (averaged over NAVG=20 samples). VHold 
   marks the holding potential before the voltage step.

2. Sweep Averaging: If multiple LEAK or TEST sweeps exist per group, they are 
   averaged after baseline correction to improve signal-to-noise ratio (uncommon 
   for most WinWCP outputs).

3. Voltage-Based Scaling: The LEAK current is scaled by the ratio of voltage steps:
   - VTest cursor marks the test pulse plateau
   - Measure voltage step in TEST sweep: ΔV_test = V(VTest) - V(VHold)
   - Measure voltage step in LEAK sweep: ΔV_leak = V(VTest) - V(VHold)
   - Calculate scaling factor: scale = ΔV_test / ΔV_leak

4. Subtraction: The final leak-subtracted current is calculated as:
   I_subtracted = I_test - (scale * I_leak)
   
   Both currents are baseline-corrected, so this isolates active (non-leak) currents.

Author: Charles Kissell, Northeastern University
License: MIT
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)


class LeakSubtractionService:
    """
    Performs leak current subtraction using proper baseline correction
    and voltage-based scaling without file format dependencies.
    """
    
    # Algorithm constants
    VLIMIT = 0.001  # Minimum voltage step for valid scaling (V)
    NAVG = 20       # Number of samples for cursor averaging (WinWCP uses 20)
    
    def __init__(self):
        """Initialize the service."""
        pass
    
    def validate_dataset(self, dataset) -> None:
        """
        Validate dataset for leak subtraction.
        
        Args:
            dataset: ElectrophysiologyDataset
            
        Raises:
            ValueError: If dataset is invalid
        """
        # Check format
        if dataset.metadata.get('format') != 'wcp':
            raise ValueError(
                f"Leak subtraction requires WCP format, got: "
                f"{dataset.metadata.get('format', 'unknown')}"
            )
        
        # Check for sweep classification
        if 'sweep_info' not in dataset.metadata:
            raise ValueError(
                "Dataset missing sweep_info metadata. "
                "Cannot classify LEAK/TEST sweeps."
            )
        
        # Verify at least some sweeps are classified
        sweep_info = dataset.metadata['sweep_info']
        classified = sum(
            1 for info in sweep_info.values()
            if info.get('rec_type') in ['LEAK', 'TEST']
        )
        
        if classified == 0:
            raise ValueError(
                "No LEAK or TEST sweeps found. "
                "Please classify sweeps in WinWCP before leak subtraction."
            )
        
        logger.info(f"Validation passed: {classified} classified sweeps")
    
    def group_sweeps(
        self,
        dataset,
        rejected_sweeps: Optional[Set[int]] = None
    ) -> Dict[int, Dict[str, List[str]]]:
        """
        Group sweeps by group number.
        
        Each group can have multiple LEAK and/or TEST sweeps that will be averaged.
        Only groups with at least 1 LEAK and 1 TEST are included.
        
        Args:
            dataset: ElectrophysiologyDataset
            rejected_sweeps: Set of sweep indices to exclude (1-based)
            
        Returns:
            Dict mapping group_number -> {'leak': [sweep_ids], 'test': [sweep_ids]}
        """
        if rejected_sweeps is None:
            rejected_sweeps = set()
        
        sweep_info = dataset.metadata['sweep_info']
        groups = {}
        
        # Group sweeps by group number
        for sweep_idx, info in sweep_info.items():
            sweep_num = int(sweep_idx)
            
            # Skip rejected sweeps
            if sweep_num in rejected_sweeps:
                logger.debug(f"Skipping rejected sweep {sweep_idx}")
                continue
            
            # Skip non-accepted sweeps
            if info.get('status') != 'ACCEPTED':
                logger.debug(
                    f"Skipping sweep {sweep_idx} with status: {info.get('status')}"
                )
                continue
            
            rec_type = info.get('rec_type', '')
            group_num = info.get('group')
            
            # Only process LEAK and TEST records
            if rec_type not in ['LEAK', 'TEST']:
                continue
            
            # Initialize group if needed
            if group_num not in groups:
                groups[group_num] = {'leak': [], 'test': []}
            
            # Add sweep to appropriate list
            if rec_type == 'LEAK':
                groups[group_num]['leak'].append(sweep_idx)
            elif rec_type == 'TEST':
                groups[group_num]['test'].append(sweep_idx)
        
        # Filter to valid groups (must have at least 1 of each type)
        valid_groups = {}
        for group_num, sweeps in groups.items():
            leak_count = len(sweeps['leak'])
            test_count = len(sweeps['test'])
            
            if leak_count >= 1 and test_count >= 1:
                valid_groups[group_num] = sweeps
                logger.debug(
                    f"Group {group_num}: {leak_count} LEAK, {test_count} TEST"
                )
            else:
                logger.debug(
                    f"Skipping incomplete group {group_num}: "
                    f"{leak_count} LEAK, {test_count} TEST"
                )
        
        if not valid_groups:
            raise ValueError(
                "No valid LEAK/TEST pairs found. "
                "Each group must have at least 1 LEAK and 1 TEST sweep."
            )
        
        logger.info(f"Found {len(valid_groups)} valid groups")
        return valid_groups
    
    def calculate_cursor_average(
        self,
        data_array: np.ndarray,
        cursor_idx: int,
        n_avg: int = None
    ) -> float:
        """
        Calculate average around cursor position.
        
        Averages n_avg samples starting at cursor_idx.
        
        Args:
            data_array: 1D data array
            cursor_idx: Sample index of cursor
            n_avg: Number of samples to average (default: NAVG)
            
        Returns:
            Average value around cursor
        """
        if n_avg is None:
            n_avg = self.NAVG
        
        num_samples = len(data_array)
        
        # Calculate index range
        i0 = cursor_idx
        i1 = min(cursor_idx + n_avg - 1, num_samples - 1)
        
        # Calculate average using numpy
        avg_value = np.mean(data_array[i0:i1+1])
        
        logger.debug(
            f"Cursor average: idx={cursor_idx}, range=[{i0}, {i1}], "
            f"n={(i1-i0+1)}, avg={avg_value:.6f}"
        )
        
        return float(avg_value)
    
    def calculate_voltage_scaling(
        self,
        v_leak_bc: np.ndarray,
        v_test_bc: np.ndarray,
        vhold_idx: int,
        vtest_idx: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate voltage scaling factor.
        
        Scaling factor = (V_test_step) / (V_leak_step)
        where steps are measured between VHold and VTest cursors.
        
        Args:
            v_leak_bc: Baseline-corrected LEAK voltage (1D array)
            v_test_bc: Baseline-corrected TEST voltage (1D array)
            vhold_idx: VHold cursor sample index
            vtest_idx: VTest cursor sample index
            
        Returns:
            Tuple of (leak_scale, details_dict)
            
        Raises:
            ValueError: If voltage step is too small
        """
        # Calculate VHold averages
        v_hold_leak = self.calculate_cursor_average(v_leak_bc, vhold_idx)
        v_hold_test = self.calculate_cursor_average(v_test_bc, vhold_idx)
        
        # Calculate VTest averages
        v_pulse_leak = self.calculate_cursor_average(v_leak_bc, vtest_idx)
        v_pulse_test = self.calculate_cursor_average(v_test_bc, vtest_idx)
        
        # Calculate voltage steps
        v_pulse_step = v_pulse_test - v_hold_test
        v_leak_step = v_pulse_leak - v_hold_leak
        
        logger.info(
            f"Voltage measurements:\n"
            f"  LEAK: VHold={v_hold_leak:.4f}, VPulse={v_pulse_leak:.4f}, "
            f"Step={v_leak_step:.4f} mV\n"
            f"  TEST: VHold={v_hold_test:.4f}, VPulse={v_pulse_test:.4f}, "
            f"Step={v_pulse_step:.4f} mV"
        )
        
        # Validate voltage step
        if abs(v_leak_step) <= self.VLIMIT:
            raise ValueError(
                f"LEAK voltage step too small: {abs(v_leak_step):.6f} mV "
                f"(minimum: {self.VLIMIT} mV)"
            )
        
        # Calculate scaling factor
        leak_scale = v_pulse_step / v_leak_step
        
        # Validate result
        if not np.isfinite(leak_scale):
            raise ValueError(f"Invalid leak_scale: {leak_scale}")
        
        logger.info(f"Voltage scaling factor: {leak_scale:.6f}")
        
        # Return details for inspection
        details = {
            'v_hold_leak': float(v_hold_leak),
            'v_hold_test': float(v_hold_test),
            'v_pulse_leak': float(v_pulse_leak),
            'v_pulse_test': float(v_pulse_test),
            'v_leak_step': float(v_leak_step),
            'v_pulse_step': float(v_pulse_step),
            'leak_scale': float(leak_scale)
        }
        
        return leak_scale, details
    
    def average_sweeps(
        self,
        dataset,
        sweep_indices: List[str],
        vhold_idx: int,
        current_ch: int,
        voltage_ch: int,
        baseline_mode: str = "cursor"
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Average multiple sweeps after baseline correction.
        
        Uses VHold cursor position as baseline reference (single point).
        
        Args:
            dataset: ElectrophysiologyDataset
            sweep_indices: List of sweep IDs to average
            vhold_idx: VHold cursor sample index
            current_ch: Current channel index
            voltage_ch: Voltage channel index
            baseline_mode: "cursor" or "fixed" (for future expansion)
            
        Returns:
            Tuple of (i_avg, v_avg, baseline_dict)
        """
        i_sum = None
        v_sum = None
        n_sweeps = len(sweep_indices)
        
        # Store individual baselines for metadata
        i_baselines = []
        v_baselines = []
        
        for sweep_idx in sweep_indices:
            # Get calibrated data from dataset
            time_ms, data = dataset.get_sweep(sweep_idx)
            
            # Extract channels
            i_data = data[:, current_ch]
            v_data = data[:, voltage_ch]
            
            # Get baseline value at VHold cursor (single point)
            i_zero = float(i_data[vhold_idx])
            v_zero = float(v_data[vhold_idx])
            
            i_baselines.append(i_zero)
            v_baselines.append(v_zero)
            
            # Apply baseline correction
            i_bc = i_data - i_zero
            v_bc = v_data - v_zero
            
            # Accumulate
            if i_sum is None:
                i_sum = i_bc.copy()
                v_sum = v_bc.copy()
            else:
                i_sum += i_bc
                v_sum += v_bc
        
        # Calculate averages
        i_avg = i_sum / n_sweeps
        v_avg = v_sum / n_sweeps
        
        # Return baseline info for metadata
        baseline_dict = {
            'i_baselines': i_baselines,
            'v_baselines': v_baselines,
            'n_sweeps': n_sweeps
        }
        
        logger.debug(f"Averaged {n_sweeps} sweeps with baseline correction")
        
        return i_avg, v_avg, baseline_dict
    
    def perform_leak_subtraction(
        self,
        dataset,
        vhold_ms: float,
        vtest_ms: float,
        start_group: int,
        end_group: int,
        current_channel: Optional[int] = None,
        voltage_channel: Optional[int] = None,
        scaling_mode: str = "voltage",
        fixed_scale: float = 1.0,
        baseline_mode: str = "cursor",
        rejected_sweeps: Optional[Set[int]] = None
    ):
        """
        Perform leak subtraction
        
        Algorithm:
        1. Load calibrated data from dataset
        2. Apply baseline correction at VHold cursor
        3. Average LEAK and TEST sweeps per group
        4. Calculate voltage-based or fixed scaling
        5. Subtract: I_sub = I_test - scale * I_leak
        6. Return new dataset
        
        Args:
            dataset: ElectrophysiologyDataset
            vhold_ms: VHold cursor position (milliseconds)
            vtest_ms: VTest cursor position (milliseconds)
            start_group: First group to process (inclusive)
            end_group: Last group to process (inclusive)
            current_channel: Current channel index (auto-detect if None)
            voltage_channel: Voltage channel index (auto-detect if None)
            scaling_mode: "voltage" or "fixed"
            fixed_scale: Fixed scaling factor (if scaling_mode="fixed")
            baseline_mode: "cursor" or "fixed"
            rejected_sweeps: Set of sweep indices to exclude (1-based)
            
        Returns:
            New ElectrophysiologyDataset with subtracted TEST sweeps
        """
        # Validate dataset
        self.validate_dataset(dataset)
        
        # Auto-detect channels from metadata if not provided
        if current_channel is None or voltage_channel is None:
            channel_config = dataset.metadata.get('channel_config')
            if not channel_config:
                raise ValueError(
                    "Cannot auto-detect channels: dataset missing 'channel_config' metadata. "
                    "Please provide current_channel and voltage_channel explicitly."
                )
            
            if current_channel is None:
                current_channel = channel_config.get('current_channel')
                if current_channel is None:
                    raise ValueError(
                        "Cannot auto-detect current channel. "
                        "Please provide current_channel explicitly."
                    )
                logger.info(f"Auto-detected current_channel: {current_channel}")
            
            if voltage_channel is None:
                voltage_channel = channel_config.get('voltage_channel')
                if voltage_channel is None:
                    raise ValueError(
                        "Cannot auto-detect voltage channel. "
                        "Please provide voltage_channel explicitly."
                    )
                logger.info(f"Auto-detected voltage_channel: {voltage_channel}")
        
        # Group sweeps
        all_groups = self.group_sweeps(dataset, rejected_sweeps)
        
        # Filter by group range
        groups_in_range = {
            g: sweeps for g, sweeps in all_groups.items()
            if start_group <= g <= end_group
        }
        
        if not groups_in_range:
            raise ValueError(
                f"No valid groups in range [{start_group}, {end_group}]. "
                f"Available groups: {sorted(all_groups.keys())}"
            )
        
        logger.info(
            f"Processing {len(groups_in_range)} groups: "
            f"{sorted(groups_in_range.keys())}"
        )
        
        # Get sample rate to convert time to indices
        first_sweep_idx = list(dataset.metadata['sweep_info'].keys())[0]
        time_ms, _ = dataset.get_sweep(first_sweep_idx)
        
        # Convert cursor positions to sample indices
        vhold_idx = int(np.argmin(np.abs(time_ms - vhold_ms)))
        vtest_idx = int(np.argmin(np.abs(time_ms - vtest_ms)))
        
        logger.info(
            f"Cursor positions: VHold={vhold_ms:.2f} ms (idx={vhold_idx}), "
            f"VTest={vtest_ms:.2f} ms (idx={vtest_idx})"
        )
        
        # Create new dataset for results
        from data_analysis_gui.core.dataset import ElectrophysiologyDataset
        new_dataset = ElectrophysiologyDataset()
        new_dataset.metadata = deepcopy(dataset.metadata)
        new_dataset.metadata['leak_subtraction_applied'] = True
        new_dataset.metadata['leak_subtraction_params'] = {
            'vhold_ms': vhold_ms,
            'vtest_ms': vtest_ms,
            'vhold_idx': vhold_idx,
            'vtest_idx': vtest_idx,
            'current_channel': current_channel,
            'voltage_channel': voltage_channel,
            'scaling_mode': scaling_mode,
            'baseline_mode': baseline_mode,
            'fixed_scale': fixed_scale if scaling_mode == "fixed" else None,
            'groups_processed': sorted(groups_in_range.keys())
        }
        new_dataset.metadata['sweep_info'] = {}
        
        # Process each group
        success_count = 0
        fail_count = 0
        
        for group_num in sorted(groups_in_range.keys()):
            leak_indices = groups_in_range[group_num]['leak']
            test_indices = groups_in_range[group_num]['test']
            
            logger.info(
                f"Processing group {group_num}: "
                f"{len(leak_indices)} LEAK, {len(test_indices)} TEST"
            )
            
            try:
                # Average LEAK sweeps with baseline correction
                i_leak_bc, v_leak_bc, leak_baselines = self.average_sweeps(
                    dataset, leak_indices, vhold_idx,
                    current_channel, voltage_channel, baseline_mode
                )
                
                # Average TEST sweeps with baseline correction
                i_test_bc, v_test_bc, test_baselines = self.average_sweeps(
                    dataset, test_indices, vhold_idx,
                    current_channel, voltage_channel, baseline_mode
                )
                
                # Calculate or use fixed scaling factor
                if scaling_mode == "voltage":
                    try:
                        leak_scale, scaling_details = self.calculate_voltage_scaling(
                            v_leak_bc, v_test_bc, vhold_idx, vtest_idx
                        )
                    except ValueError as e:
                        logger.warning(f"Group {group_num}: {e}, skipping")
                        fail_count += 1
                        continue
                else:
                    leak_scale = fixed_scale
                    scaling_details = {'leak_scale': leak_scale, 'mode': 'fixed'}
                
                # Perform subtraction (core mathematical operation)
                i_subtracted_bc = i_test_bc - leak_scale * i_leak_bc
                
                logger.info(
                    f"Group {group_num}: Subtraction complete with scale={leak_scale:.6f}"
                )
                
                # Get time axis and template data from last TEST sweep
                last_test_idx = test_indices[-1]
                time_ms, template_data = dataset.get_sweep(last_test_idx)
                
                # Create new sweep data with subtracted current
                # Keep original voltage, replace current with subtracted values
                new_sweep_data = template_data.copy()
                new_sweep_data[:, current_channel] = i_subtracted_bc
                
                # Add to new dataset
                new_dataset.add_sweep(last_test_idx, time_ms, new_sweep_data)
                
                # Store comprehensive metadata
                original_info = dataset.metadata['sweep_info'][last_test_idx]
                new_dataset.metadata['sweep_info'][last_test_idx] = {
                    'time': original_info['time'],
                    'rec_type': 'TEST',
                    'group': group_num,
                    'status': 'ACCEPTED',
                    'leak_subtracted': True,
                    'source_leak_sweeps': leak_indices,
                    'source_test_sweeps': test_indices,
                    'leak_scale': float(leak_scale),
                    'scaling_details': scaling_details,
                    'leak_baselines': leak_baselines,
                    'test_baselines': test_baselines
                }
                
                success_count += 1
                
            except Exception as e:
                logger.error(
                    f"Failed to process group {group_num}: {e}",
                    exc_info=True
                )
                fail_count += 1
        
        # Verify success
        if success_count == 0:
            raise ValueError(
                f"Failed to process any groups. {fail_count} groups failed."
            )
        
        logger.info(
            f"Leak subtraction complete: "
            f"{success_count} successful, {fail_count} failed"
        )
        
        return new_dataset
    
    def get_group_range(
        self,
        dataset,
        rejected_sweeps: Optional[Set[int]] = None
    ) -> Tuple[int, int]:
        """
        Get range of valid group numbers.
        
        Args:
            dataset: ElectrophysiologyDataset
            rejected_sweeps: Set of sweep indices to exclude
            
        Returns:
            Tuple of (min_group, max_group)
        """
        groups = self.group_sweeps(dataset, rejected_sweeps)
        group_nums = sorted(groups.keys())
        return group_nums[0], group_nums[-1]