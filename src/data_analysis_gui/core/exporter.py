# src/data_analysis_gui/core/exporter.py

import os
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .batch_processor import BatchResult

@dataclass(frozen=True)
class ExportOutcome:
    """Represents the result of a single file export operation."""
    path: str
    success: bool
    error_message: Optional[str] = None

def _sanitize_filename(name: str) -> str:
    """Removes invalid characters from a potential filename."""
    name = re.sub(r'[\[\]]', '', name)  # Remove brackets
    name = re.sub(r'[<>:"/\\|?*]', '_', name)  # Replace other invalid chars
    return name

def _get_next_available_path(path: str) -> str:
    """
    Finds an available filename by appending _1, _2, etc. if the path exists.
    Example: C:/data/file.csv -> C:/data/file_1.csv
    """
    if not os.path.exists(path):
        return path
    
    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    
    counter = 1
    while True:
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(directory, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def _write_csv(path: str, table: Dict[str, Any]) -> None:
    """Writes a single table dictionary to a CSV file using numpy."""
    headers = table.get('headers', [])
    data = table.get('data', np.array([[]]))
    fmt = table.get('format_spec', '%.6f')
    
    header_str = ','.join(headers)
    np.savetxt(path, data, delimiter=',', header=header_str, fmt=fmt, comments='')

def write_single_table(
    table: Dict[str, Any],
    base_name: str,
    destination_folder: str
) -> ExportOutcome:
    """
    Writes a single analysis table to a CSV file in the given folder.
    Handles filename sanitization and collision.
    """
    sanitized_name = _sanitize_filename(base_name)
    target_path = os.path.join(destination_folder, f"{sanitized_name}.csv")
    
    try:
        if not os.path.isdir(destination_folder):
            raise FileNotFoundError(f"Destination folder does not exist: {destination_folder}")

        final_path = _get_next_available_path(target_path)
        _write_csv(final_path, table)
        
        return ExportOutcome(path=final_path, success=True)
    except Exception as e:
        return ExportOutcome(path=target_path, success=False, error_message=str(e))

def write_tables(result: BatchResult, destination_folder: str) -> List[ExportOutcome]:
    """
    Writes all successful file results from a BatchResult to a destination folder.
    
    Args:
        result: The BatchResult containing processed file data.
        destination_folder: The folder path to save CSV files to.

    Returns:
        A list of ExportOutcome objects detailing the result of each write operation.
    """
    outcomes: List[ExportOutcome] = []
    
    try:
        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)
    except OSError as e:
        error_msg = f"Could not create destination folder '{destination_folder}': {e}"
        return [ExportOutcome(path=res.file_path, success=False, error_message=error_msg) for res in result.successful_results]

    for file_result in result.successful_results:
        outcome = write_single_table(
            table=file_result.export_table,
            base_name=file_result.base_name,
            destination_folder=destination_folder
        )
        outcomes.append(outcome)
            
    return outcomes

def write_single_table(
    table: Dict[str, Any],
    base_name: str,
    destination_folder: str,
    peak_type: Optional[str] = None  # NEW parameter
) -> ExportOutcome:
    """
    Writes a single analysis table to a CSV file in the given folder.
    Handles filename sanitization and collision.
    
    Args:
        table: Table data dictionary
        base_name: Base filename
        destination_folder: Output folder
        peak_type: Optional peak type to append to filename
    """
    sanitized_name = _sanitize_filename(base_name)
    
    # Add peak type suffix if provided
    if peak_type and peak_type != "Absolute":  # Only add suffix for non-default peak types
        peak_suffix_map = {
            "Positive": "_positive",
            "Negative": "_negative", 
            "Peak-Peak": "_peak-peak"
        }
        suffix = peak_suffix_map.get(peak_type, "")
        sanitized_name = f"{sanitized_name}{suffix}"
    
    target_path = os.path.join(destination_folder, f"{sanitized_name}.csv")
    
    try:
        if not os.path.isdir(destination_folder):
            raise FileNotFoundError(f"Destination folder does not exist: {destination_folder}")

        final_path = _get_next_available_path(target_path)
        _write_csv(final_path, table)
        
        return ExportOutcome(path=final_path, success=True)
    except Exception as e:
        return ExportOutcome(path=target_path, success=False, error_message=str(e))

def write_tables(result: BatchResult, destination_folder: str) -> List[ExportOutcome]:
    """
    Writes all successful file results from a BatchResult to a destination folder.
    """
    outcomes: List[ExportOutcome] = []
    
    try:
        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)
    except OSError as e:
        error_msg = f"Could not create destination folder '{destination_folder}': {e}"
        return [ExportOutcome(path=res.file_path, success=False, error_message=error_msg) 
                for res in result.successful_results]

    for file_result in result.successful_results:
        # Extract peak type from the result if available
        peak_type = getattr(file_result, 'peak_type', None)
        
        outcome = write_single_table(
            table=file_result.export_table,
            base_name=file_result.base_name,
            destination_folder=destination_folder,
            peak_type=peak_type  # Pass peak type
        )
        outcomes.append(outcome)
            
    return outcomes