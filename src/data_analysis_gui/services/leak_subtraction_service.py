"""
Leak Subtraction Service - ROUND-TRIP SIMULATION

This version simulates WinWCP's complete save/load cycle:
1. Perform subtraction in physical units
2. Convert to ADC (with truncation): ADC_save = Trunc(I_subtracted/IScale) + IZero
3. Simulate loader's calibration (dynamic or fixed baseline)

This matches WinWCP's actual behavior where subtracted data is saved as ADC
and then re-loaded with the file's calibration settings.

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
    WinWCP-compliant leak subtraction service with round-trip simulation.
    
    Implements the exact algorithm from LEAKSUB.PAS including the
    save-to-ADC and reload-with-calibration cycle.
    """
    
    # Constants from LEAKSUB.PAS
    VLIMIT = 0.001  # Line 538: Minimum voltage step for valid scaling
    NAVG = 20       # Line 529: Number of samples for cursor averaging
    
    def __init__(self):
        """Initialize the service."""
        pass
    
    def validate_dataset(self, dataset) -> None:
        """
        Validate dataset for leak subtraction.
        
        Args:
            dataset: ElectrophysiologyDataset
            
        Raises:
            ValidationError: If dataset is invalid
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
                "Please classify sweeps in WinWCP first."
            )
        
        logger.info(f"Validation passed: {classified} classified sweeps")
    
    def group_sweeps(
        self,
        dataset,
        rejected_sweeps: Optional[Set[int]] = None
    ) -> Dict[int, Dict[str, List[str]]]:
        """
        Group sweeps by RH.Number (group number).
        
        This matches WinWCP's grouping logic from LEAKSUB.PAS lines 397-520.
        Each group can have multiple LEAK and/or TEST sweeps that will be averaged.
        
        Args:
            dataset: ElectrophysiologyDataset
            rejected_sweeps: Set of sweep indices to exclude (1-based)
            
        Returns:
            Dict mapping group_number -> {'leak': [sweep_ids], 'test': [sweep_ids]}
            Only groups with at least 1 LEAK and 1 TEST are included.
        """
        if rejected_sweeps is None:
            rejected_sweeps = set()
        
        sweep_info = dataset.metadata['sweep_info']
        groups = {}  # group_num -> {'leak': [], 'test': []}
        
        # Group sweeps by group number
        for sweep_idx, info in sweep_info.items():
            sweep_num = int(sweep_idx)
            
            # Skip rejected sweeps
            if sweep_num in rejected_sweeps:
                logger.debug(f"Skipping rejected sweep {sweep_idx}")
                continue
            
            # Skip non-accepted sweeps (from file)
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
        Calculate average around cursor using WinWCP method.
        
        Matches LEAKSUB.PAS lines 531-537, 539-545:
        ```pascal
        i0 := VHoldCursor;
        i1 := Min(VHoldCursor + nAvg - 1, NumSamples-1);
        VHoldLeak := 0.;
        for i := i0 to i1 do begin
            VHoldLeak := VHoldLeak + VLeak^[i];
        end;
        VHoldLeak := VHoldLeak / (i1 - i0 + 1);
        ```
        
        Args:
            data_array: 1D baseline-corrected data array
            cursor_idx: Sample index of cursor
            n_avg: Number of samples to average (default: NAVG)
            
        Returns:
            Average value around cursor
        """
        if n_avg is None:
            n_avg = self.NAVG
        
        num_samples = len(data_array)
        
        # Calculate indices (Pascal-style loop bounds)
        i0 = cursor_idx
        i1 = min(cursor_idx + n_avg - 1, num_samples - 1)
        
        # Calculate sum manually (matching Pascal loop)
        value_sum = 0.0
        for i in range(i0, i1 + 1):
            value_sum += data_array[i]
        
        # Calculate average with exact denominator
        avg_value = value_sum / (i1 - i0 + 1)
        
        logger.debug(
            f"Cursor average: idx={cursor_idx}, range=[{i0}, {i1}], "
            f"n={(i1-i0+1)}, avg={avg_value:.6f}"
        )
        
        return avg_value
    
    def calculate_voltage_scaling(
        self,
        v_leak_bc: np.ndarray,
        v_test_bc: np.ndarray,
        vhold_idx: int,
        vtest_idx: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate voltage scaling factor using WinWCP algorithm.
        
        Exact implementation of LEAKSUB.PAS lines 529-566.
        
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
        # Calculate VHold averages (lines 531-537)
        v_hold_leak = self.calculate_cursor_average(v_leak_bc, vhold_idx)
        v_hold_test = self.calculate_cursor_average(v_test_bc, vhold_idx)
        
        # Calculate VTest averages (lines 539-545)
        v_pulse_leak = self.calculate_cursor_average(v_leak_bc, vtest_idx)
        v_pulse_test = self.calculate_cursor_average(v_test_bc, vtest_idx)
        
        # Calculate voltage steps (lines 549-550)
        v_pulse_step = v_pulse_test - v_hold_test
        v_leak_step = v_pulse_leak - v_hold_leak
        
        logger.info(
            f"Voltage measurements:\n"
            f"  LEAK: VHold={v_hold_leak:.4f}, VPulse={v_pulse_leak:.4f}, "
            f"Step={v_leak_step:.4f} mV\n"
            f"  TEST: VHold={v_hold_test:.4f}, VPulse={v_pulse_test:.4f}, "
            f"Step={v_pulse_step:.4f} mV"
        )
        
        # Validate voltage step (line 553)
        if abs(v_leak_step) <= self.VLIMIT:
            raise ValueError(
                f"LEAK voltage step too small: {abs(v_leak_step):.6f} V "
                f"(minimum: {self.VLIMIT} V)"
            )
        
        # Calculate scaling factor (line 554)
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
        
        Uses VHold cursor position as baseline (LEAKSUB.PAS lines 412-413, 446-447).
        
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
        # Import here to avoid circular dependency
        from data_analysis_gui.core.loaders.wcp_loader import WCPParser
        
        i_sum = None
        v_sum = None
        n_sweeps = len(sweep_indices)
        
        # Store individual baselines for debugging
        i_baselines_adc = []
        v_baselines_adc = []
        
        # Need to re-read RAW data from file
        filepath = dataset.metadata['source_file']
        
        with WCPParser(filepath) as wcp:
            for sweep_idx in sweep_indices:
                # Read RAW uncalibrated data
                record_num = int(sweep_idx)
                header, raw_data = wcp.read_record(record_num, calibrated=False)
                
                # Extract RAW ADC values for each channel
                i_raw_adc = raw_data[:, current_ch].astype(np.float64)
                v_raw_adc = raw_data[:, voltage_ch].astype(np.float64)
                
                # Get WinWCP-style baseline (single point at VHold cursor)
                # LEAKSUB.PAS line 412-413, 446-447
                i_zero_adc = float(i_raw_adc[vhold_idx])
                v_zero_adc = float(v_raw_adc[vhold_idx])
                
                # Store raw ADC baselines
                i_baselines_adc.append(i_zero_adc)
                v_baselines_adc.append(v_zero_adc)
                
                # Apply baseline correction in ADC space
                i_bc_adc = i_raw_adc - i_zero_adc
                v_bc_adc = v_raw_adc - v_zero_adc
                
                # Get calibration factors for THIS sweep
                # LEAKSUB.PAS lines 405-406, 443-444
                ch_current = wcp.file_header.channels[current_ch]
                ch_voltage = wcp.file_header.channels[voltage_ch]
                
                i_scale = (abs(header.adc_voltage_range[current_ch]) / 
                          (ch_current.calibration_factor * (wcp.file_header.max_adc_value + 1)))
                v_scale = (abs(header.adc_voltage_range[voltage_ch]) / 
                          (ch_voltage.calibration_factor * (wcp.file_header.max_adc_value + 1)))
                
                # Convert to physical units (AFTER baseline correction)
                # LEAKSUB.PAS line 418-419, 455-456
                i_bc = i_bc_adc * i_scale
                v_bc = v_bc_adc * v_scale
                
                # Accumulate
                if i_sum is None:
                    i_sum = i_bc.copy()
                    v_sum = v_bc.copy()
                else:
                    i_sum += i_bc
                    v_sum += v_bc
        
        # Average (LEAKSUB.PAS lines 470-475, 513-518)
        i_avg = i_sum / n_sweeps
        v_avg = v_sum / n_sweeps
        
        # Return baseline info
        baseline_dict = {
            'i_baselines_adc': i_baselines_adc,
            'v_baselines_adc': v_baselines_adc,
            'n_sweeps': n_sweeps
        }
        
        logger.debug(f"Averaged {n_sweeps} sweeps")
        
        return i_avg, v_avg, baseline_dict
    
    def simulate_wcp_roundtrip(
        self,
        i_subtracted_bc: np.ndarray,
        i_zero_adc: float,
        i_scale: float,
        channel_adc_zero: int,
        channel_adc_zero_at: int,
        num_zero_avg: int
    ) -> np.ndarray:
        """
        Simulate WinWCP's save-to-ADC and reload-with-calibration cycle.
        
        CRITICAL: This is what WinWCP actually does:
        1. Save: ADC_save = Trunc(I_subtracted / IScale) + IZero
        2. Reload with calibration based on file header:
           - If adc_zero_at >= 0: Calculate zero from baseline region of saved data
           - If adc_zero_at < 0: Use fixed channel.adc_zero
           - Physical = (ADC_save - calculated_zero) * IScale
        
        Args:
            i_subtracted_bc: Baseline-corrected subtracted current (physical units)
            i_zero_adc: Raw ADC baseline from last TEST sweep's VHold cursor
            i_scale: Calibration factor (physical units per ADC)
            channel_adc_zero: Fixed zero level from file header
            channel_adc_zero_at: Starting index for dynamic zero calculation (-1 = fixed)
            num_zero_avg: Number of samples to average for dynamic zero
            
        Returns:
            Final physical current values as WinWCP would display them
        """
        # Step 1: Convert to ADC (LEAKSUB.PAS line 577)
        # ADC^[j] := Trunc( ITest^[i]/IScale ) + IZero
        adc_saved = np.trunc(i_subtracted_bc / i_scale) + i_zero_adc
        
        logger.debug(
            f"Round-trip: IZero_adc={i_zero_adc:.2f}, "
            f"ADC range=[{adc_saved.min():.1f}, {adc_saved.max():.1f}]"
        )
        
        # Step 2: Simulate loader's calibration (from wcp_loader.py read_record)
        if channel_adc_zero_at >= 0:
            # Dynamic zero: Calculate from baseline region of SAVED ADC data
            # This matches wcp_loader.py _calculate_dynamic_zero()
            num_samples = len(adc_saved)
            i0 = max(0, min(channel_adc_zero_at, num_samples - 1))
            i1 = i0 + num_zero_avg - 1
            i1 = max(0, min(i1, num_samples - 1))
            
            # Calculate mean of baseline region (from saved ADC values)
            zero_level = np.mean(adc_saved[i0:i1+1])
            
            logger.info(
                f"Round-trip: Dynamic zero calculated from ADC_saved[{i0}:{i1+1}] = {zero_level:.2f}"
            )
        else:
            # Fixed zero: Use channel.adc_zero from file header
            zero_level = channel_adc_zero
            
            logger.info(
                f"Round-trip: Using fixed zero = {zero_level:.2f}"
            )
        
        # Step 3: Apply calibration (wcp_loader.py line in read_record)
        # Physical = (Raw - Zero) * Scale
        i_final = (adc_saved - zero_level) * i_scale
        
        logger.debug(
            f"Round-trip: Final range=[{i_final.min():.2f}, {i_final.max():.2f}] pA"
        )
        
        return i_final
    
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
        Perform leak subtraction matching WinWCP exactly with round-trip simulation.
        
        Complete implementation of LEAKSUB.PAS lines 340-605 including the
        save-to-ADC and reload-with-calibration cycle.
        
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
        
        # Import WCPParser for reading raw data
        from data_analysis_gui.core.loaders.wcp_loader import WCPParser
        
        # Get filepath for raw data access
        filepath = dataset.metadata.get('source_file')
        if not filepath:
            raise ValueError("Dataset missing 'source_file' in metadata")
        
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
        
        # Get file header info for round-trip simulation
        with WCPParser(filepath) as wcp:
            channel_adc_zero = wcp.file_header.channels[current_channel].adc_zero
            channel_adc_zero_at = wcp.file_header.channels[current_channel].adc_zero_at
            num_zero_avg = wcp.file_header.num_zero_avg
            
            logger.info(
                f"File header: channel.adc_zero={channel_adc_zero}, "
                f"adc_zero_at={channel_adc_zero_at}, num_zero_avg={num_zero_avg}"
            )
        
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
        # Load first sweep to get time array
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
                # Average LEAK sweeps (LEAKSUB.PAS lines 397-475)
                i_leak_bc, v_leak_bc, leak_baselines = self.average_sweeps(
                    dataset, leak_indices, vhold_idx,
                    current_channel, voltage_channel, baseline_mode
                )
                
                # Average TEST sweeps (LEAKSUB.PAS lines 483-518)
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
                
                # Perform subtraction (LEAKSUB.PAS line 559)
                i_subtracted_bc = i_test_bc - leak_scale * i_leak_bc
                
                # Get parameters for round-trip simulation
                last_test_idx = test_indices[-1]
                i_zero_adc_last_test = test_baselines['i_baselines_adc'][-1]
                
                # Get calibration factor from last TEST sweep
                with WCPParser(filepath) as wcp:
                    header, _ = wcp.read_record(int(last_test_idx), calibrated=False)
                    ch_current = wcp.file_header.channels[current_channel]
                    i_scale = (abs(header.adc_voltage_range[current_channel]) / 
                              (ch_current.calibration_factor * (wcp.file_header.max_adc_value + 1)))
                
                # CRITICAL: Simulate WinWCP's round-trip (save as ADC, reload with calibration)
                i_subtracted_final = self.simulate_wcp_roundtrip(
                    i_subtracted_bc,
                    i_zero_adc_last_test,
                    i_scale,
                    channel_adc_zero,
                    channel_adc_zero_at,
                    num_zero_avg
                )
                
                logger.info(
                    f"Group {group_num}: leak_scale={leak_scale:.6f}, "
                    f"IZero_adc={i_zero_adc_last_test:.2f}"
                )
                
                # Create new sweep with subtracted current
                # Use the voltage from the last TEST sweep (re-read as calibrated)
                with WCPParser(filepath) as wcp:
                    _, template_data = wcp.read_record(int(last_test_idx), calibrated=True)
                
                # Create new data array with subtracted current and original voltage
                new_sweep_data = template_data.copy()
                new_sweep_data[:, current_channel] = i_subtracted_final
                
                # Get time axis
                time_ms, _ = dataset.get_sweep(last_test_idx)
                
                # Add to new dataset
                new_dataset.add_sweep(last_test_idx, time_ms, new_sweep_data)
                
                # Store metadata
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