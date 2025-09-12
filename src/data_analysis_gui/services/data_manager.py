"""
Unified data management for loading, validating, and exporting data.

This module combines all data-related operations into a single, easy-to-understand
class that scientist-programmers can extend and modify.

Author: Data Analysis GUI Contributors
License: MIT
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from data_analysis_gui.core.dataset import DatasetLoader, ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.models import ExportResult
from data_analysis_gui.core.exceptions import (
    ValidationError, FileError, DataError
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class DataManager:
    """
    Manages all data operations: loading, validation, and export.
    
    This class provides simple, direct methods for working with electrophysiology data.
    Scientists can easily add new file formats or export methods by extending the
    appropriate methods.
    """
    
    def __init__(self):
        """Initialize the data manager."""
        logger.info("DataManager initialized")
    
    # =========================================================================
    # Dataset Loading
    # =========================================================================
    
    def load_dataset(self, 
                    filepath: str,
                    channel_config: Optional[ChannelDefinitions] = None) -> ElectrophysiologyDataset:
        """
        Load a dataset from a file.
        
        To add support for a new file format:
        1. Add format detection in core/dataset.py DatasetLoader.detect_format()
        2. Add loading method in core/dataset.py DatasetLoader.load_[format]()
        
        Args:
            filepath: Path to the data file
            channel_config: Optional channel configuration
            
        Returns:
            Loaded and validated dataset
            
        Raises:
            FileError: If file cannot be loaded
            DataError: If data is invalid
        """
        # Check file exists
        if not os.path.exists(filepath):
            raise FileError(f"File not found: {filepath}")
        
        if not os.access(filepath, os.R_OK):
            raise FileError(f"File not readable: {filepath}")
        
        # Check file not empty
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            raise DataError(f"File is empty: {filepath}")
        
        logger.info(f"Loading dataset from {Path(filepath).name}")
        
        # Load using DatasetLoader
        try:
            dataset = DatasetLoader.load(filepath, channel_config)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise FileError(f"Failed to load {filepath}: {str(e)}")
        
        # Validate
        if dataset is None or dataset.is_empty():
            raise DataError(f"No valid data found in {filepath}")
        
        logger.info(f"Successfully loaded {dataset.sweep_count()} sweeps")
        return dataset
    
    # =========================================================================
    # Data Export
    # =========================================================================
    
    def export_to_csv(self,
                     data: Dict[str, Any],
                     filepath: str) -> ExportResult:
        """
        Export data to a CSV file.
        
        Args:
            data: Dictionary with 'headers' and 'data' keys
            filepath: Output file path
            
        Returns:
            ExportResult with status
        """
        try:
            # Validate data
            if not data or 'headers' not in data or 'data' not in data:
                raise ValidationError("Invalid data structure")
            
            headers = data['headers']
            data_array = np.array(data['data'])
            
            if data_array.size == 0:
                raise DataError("No data to export")
            
            # Ensure directory exists
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Write CSV
            format_spec = data.get('format_spec', '%.6f')
            header_str = ','.join(headers)
            np.savetxt(filepath, data_array, delimiter=',', 
                      fmt=format_spec, header=header_str, comments='')
            
            # Verify file was created
            if not os.path.exists(filepath):
                raise FileError("File was not created")
            
            records = len(data_array)
            logger.info(f"Exported {records} records to {Path(filepath).name}")
            
            return ExportResult(
                success=True,
                file_path=filepath,
                records_exported=records
            )
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                error_message=str(e)
            )
    
    def export_multiple_files(self,
                            data_list: List[Dict[str, Any]],
                            output_dir: str,
                            base_name: str = "export") -> List[ExportResult]:
        """
        Export multiple data tables to separate CSV files.
        
        Args:
            data_list: List of data dictionaries to export
            output_dir: Output directory
            base_name: Base filename for exports
            
        Returns:
            List of ExportResult objects
        """
        results = []
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        for i, data in enumerate(data_list):
            # Generate unique filename
            suffix = data.get('suffix', f"_{i+1}")
            filename = f"{base_name}{suffix}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Make unique if file exists
            filepath = self.make_unique_path(filepath)
            
            # Export
            result = self.export_to_csv(data, filepath)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Exported {successful}/{len(data_list)} files")
        
        return results
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def suggest_filename(self,
                        source_path: str,
                        suffix: str = "_",
                        params: Optional[Any] = None) -> str:
        """
        Generate a suggested filename for export.
        
        Args:
            source_path: Original file path
            suffix: Suffix to add
            params: Optional analysis parameters for context
            
        Returns:
            Suggested filename
        """
        if not source_path:
            return f"analysis{suffix}.csv"
        
        # Get base name and clean it
        base_name = Path(source_path).stem
        base_name = re.sub(r'\[.*?\]', '', base_name)  # Remove brackets
        base_name = base_name.strip(' _')
        
        # Add parameter-specific suffix if available
        if params and hasattr(params, 'y_axis'):
            if params.y_axis.measure == "Peak" and params.y_axis.peak_type:
                peak_suffixes = {
                    "Absolute": "_absolute",
                    "Positive": "_positive", 
                    "Negative": "_negative",
                    "Peak-Peak": "_peak-peak"
                }
                suffix = peak_suffixes.get(params.y_axis.peak_type, suffix)
        
        return f"{base_name}{suffix}.csv"
    
    def make_unique_path(self, filepath: str) -> str:
        """
        Make a filepath unique by appending numbers if it exists.
        
        Args:
            filepath: Desired filepath
            
        Returns:
            Unique filepath
        """
        if not os.path.exists(filepath):
            return filepath
        
        path = Path(filepath)
        directory = path.parent
        stem = path.stem
        suffix = path.suffix
        
        counter = 1
        while counter <= 10000:
            new_path = directory / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return str(new_path)
            counter += 1
        
        raise FileError(f"Could not create unique filename for {filepath}")
    
    def validate_export_path(self, filepath: str) -> bool:
        """
        Check if a filepath is valid for export.
        
        Args:
            filepath: Path to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If path is invalid
        """
        if not filepath or not filepath.strip():
            raise ValidationError("Export path cannot be empty")
        
        path = Path(filepath)
        
        # Must have an extension
        if not path.suffix:
            raise ValidationError("Export file must have an extension")
        
        # Check for invalid characters
        invalid_chars = '<>:"|?*' if os.name == 'nt' else '\0'
        invalid_found = [c for c in path.name if c in invalid_chars]
        if invalid_found:
            raise ValidationError(f"Filename contains invalid characters: {invalid_found}")
        
        # Check if directory is writable
        directory = path.parent
        if directory.exists() and not os.access(directory, os.W_OK):
            raise ValidationError(f"No write permission for directory: {directory}")
        
        return True