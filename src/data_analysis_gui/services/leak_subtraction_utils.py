"""
Utility functions for leak subtraction feature.

Provides helper functions for validation, sweep classification,
and metadata inspection used by both the service and dialog.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import logging
from typing import Dict, List, Tuple, Set, Optional
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


def is_leak_subtraction_available(dataset: ElectrophysiologyDataset) -> Tuple[bool, str]:
    """
    Check if leak subtraction is available for this dataset.
    
    Args:
        dataset: The dataset to check
        
    Returns:
        Tuple of (is_available, message)
        where is_available is True if leak subtraction can be performed,
        and message explains why or confirms availability
    """
    # Check if dataset exists
    if dataset is None or dataset.is_empty():
        return False, "No data loaded"
    
    # Check file format
    file_format = dataset.metadata.get('format', 'unknown')
    if file_format != 'wcp':
        return False, (
            "Leak subtraction only works with WCP files. "
            "ABF files do not contain RecType metadata."
        )
    
    # Check for sweep_info metadata
    if 'sweep_info' not in dataset.metadata:
        return False, (
            "This WCP file does not contain sweep classification metadata. "
            "File may be from an older WCP version."
        )
    
    # Check that at least some sweeps are classified
    sweep_info = dataset.metadata['sweep_info']
    classified_count = sum(
        1 for info in sweep_info.values()
        if info.get('rec_type') in ['LEAK', 'TEST']
    )
    
    if classified_count == 0:
        return False, (
            "No sweeps are classified as LEAK or TEST. "
            "Please classify sweeps in WinWCP before leak subtraction."
        )
    
    # All checks passed
    return True, f"Leak subtraction available ({classified_count} classified sweeps)"


def get_sweep_classification_summary(
    dataset: ElectrophysiologyDataset
) -> Dict[str, int]:
    """
    Get a summary of sweep classifications in the dataset.
    
    Args:
        dataset: The dataset to analyze
        
    Returns:
        Dictionary mapping rec_type -> count
        
    Example:
        {'LEAK': 10, 'TEST': 10, '': 5}
    """
    sweep_info = dataset.metadata.get('sweep_info', {})
    
    summary = {}
    for info in sweep_info.values():
        rec_type = info.get('rec_type', '')
        summary[rec_type] = summary.get(rec_type, 0) + 1
    
    return summary


def get_group_summary(
    dataset: ElectrophysiologyDataset,
    rejected_sweeps: Optional[Set[int]] = None
) -> Dict[int, Dict[str, List[str]]]:
    """
    Get a summary of how sweeps are grouped.
    
    Args:
        dataset: The dataset to analyze
        rejected_sweeps: Set of sweep indices to exclude
        
    Returns:
        Dictionary mapping group_number -> {'leak': [sweep_indices], 'test': [sweep_indices]}
        
    Example:
        {
            1: {'leak': ['1'], 'test': ['2']},
            2: {'leak': ['3'], 'test': ['4', '5']},  # Invalid - multiple TEST
            3: {'leak': [], 'test': ['6']}  # Invalid - no LEAK
        }
    """
    if rejected_sweeps is None:
        rejected_sweeps = set()
    
    sweep_info = dataset.metadata.get('sweep_info', {})
    
    groups = {}
    
    for sweep_idx, info in sweep_info.items():
        # Skip rejected sweeps
        if int(sweep_idx) in rejected_sweeps:
            continue
        
        # Skip non-accepted sweeps
        if info.get('status') != 'ACCEPTED':
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
    
    return groups


def validate_group_pairing(
    groups: Dict[int, Dict[str, List[str]]]
) -> Tuple[List[int], List[int]]:
    """
    Validate that groups have proper LEAK/TEST pairing.
    
    Args:
        groups: Group summary from get_group_summary()
        
    Returns:
        Tuple of (valid_group_numbers, invalid_group_numbers)
    """
    valid_groups = []
    invalid_groups = []
    
    for group_num, sweeps in groups.items():
        leak_count = len(sweeps['leak'])
        test_count = len(sweeps['test'])
        
        if leak_count == 1 and test_count == 1:
            valid_groups.append(group_num)
        else:
            invalid_groups.append(group_num)
    
    return sorted(valid_groups), sorted(invalid_groups)


def get_sweep_by_group_and_type(
    dataset: ElectrophysiologyDataset,
    group_num: int,
    rec_type: str,
    rejected_sweeps: Optional[Set[int]] = None
) -> Optional[str]:
    """
    Find sweep index for a specific group and record type.
    
    Args:
        dataset: The dataset to search
        group_num: Group number to find
        rec_type: 'LEAK' or 'TEST'
        rejected_sweeps: Set of sweep indices to exclude
        
    Returns:
        Sweep index (as string) or None if not found
    """
    if rejected_sweeps is None:
        rejected_sweeps = set()
    
    sweep_info = dataset.metadata.get('sweep_info', {})
    
    for sweep_idx, info in sweep_info.items():
        # Skip rejected sweeps
        if int(sweep_idx) in rejected_sweeps:
            continue
        
        # Skip non-accepted sweeps
        if info.get('status') != 'ACCEPTED':
            continue
        
        # Check for match
        if info.get('group') == group_num and info.get('rec_type') == rec_type:
            return sweep_idx
    
    return None


def format_scale_factor_display(
    leak_scale: float,
    details: Optional[Dict[str, float]] = None
) -> str:
    """
    Format scaling factor for display in UI.
    
    Args:
        leak_scale: The calculated scaling factor
        details: Optional dictionary with intermediate values
        
    Returns:
        Formatted string for display
    """
    if details is None:
        return f"Scale Factor: {leak_scale:.4f}"
    
    return (
        f"Scale Factor: {leak_scale:.4f}\n"
        f"VHold (LEAK): {details['v_hold_leak']:.2f} mV\n"
        f"VHold (TEST): {details['v_hold_test']:.2f} mV\n"
        f"VTest (LEAK): {details['v_pulse_leak']:.2f} mV\n"
        f"VTest (TEST): {details['v_pulse_test']:.2f} mV\n"
        f"LEAK Step: {details['v_leak_step']:.2f} mV\n"
        f"TEST Step: {details['v_pulse_step']:.2f} mV"
    )


def check_cursor_positions(
    vhold_ms: float,
    vtest_ms: float,
    max_time_ms: float
) -> Tuple[bool, str]:
    """
    Validate cursor positions are within bounds and properly ordered.
    
    Args:
        vhold_ms: VHold cursor position
        vtest_ms: VTest cursor position
        max_time_ms: Maximum time in sweep
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check bounds
    if vhold_ms < 0 or vhold_ms > max_time_ms:
        return False, f"VHold position ({vhold_ms:.2f} ms) is out of bounds [0, {max_time_ms:.2f}]"
    
    if vtest_ms < 0 or vtest_ms > max_time_ms:
        return False, f"VTest position ({vtest_ms:.2f} ms) is out of bounds [0, {max_time_ms:.2f}]"
    
    # Check ordering (VHold should typically be before VTest)
    if vhold_ms >= vtest_ms:
        return False, "VHold must be positioned before VTest"
    
    return True, "Cursor positions valid"


def get_available_groups(
    dataset: ElectrophysiologyDataset,
    rejected_sweeps: Optional[Set[int]] = None
) -> List[int]:
    """
    Get list of valid group numbers that can be processed.
    
    Args:
        dataset: The dataset to analyze
        rejected_sweeps: Set of sweep indices to exclude
        
    Returns:
        Sorted list of valid group numbers
    """
    groups = get_group_summary(dataset, rejected_sweeps)
    valid_groups, _ = validate_group_pairing(groups)
    return valid_groups