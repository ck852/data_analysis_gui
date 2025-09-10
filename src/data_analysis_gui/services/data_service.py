"""
Unified data service for dataset, export, and file operations.

This module consolidates all data operations into a single, easy-to-understand service
that scientist-programmers can extend and modify. It replaces the complex multi-service
architecture with direct, clear implementations.

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
    ValidationError, FileError, DataError, validate_not_none
)
from data_analysis_gui.config.logging import get_logger, log_performance

logger = get_logger(__name__)


class DataService:
    """
    Unified service for all data operations.
    
    This service consolidates dataset loading, validation, export, and file operations
    into a single, easy-to-understand class. Scientists can easily add new file formats
    or export methods by extending the appropriate methods.
    """
    
    def __init__(self):
        """Initialize the data service."""
        logger.info("DataService initialized")
    
    # =========================================================================
    # Dataset Operations (from dataset_service.py)
    # =========================================================================
    
    def load_dataset(self, 
                    filepath: str,
                    channel_config: Optional[ChannelDefinitions] = None) -> ElectrophysiologyDataset:
        """
        Load and validate a dataset from a file.
        
        To add support for a new file format:
        1. Add format detection in core/dataset.py DatasetLoader.detect_format()
        2. Add loading method in core/dataset.py DatasetLoader.load_[format]()
        
        Args:
            filepath: Path to the data file
            channel_config: Optional channel configuration
            
        Returns:
            Validated dataset ready for analysis
            
        Raises:
            ValidationError: If inputs are invalid
            FileError: If file is not accessible
            DataError: If dataset is invalid or empty
        """
        # Validate inputs
        validate_not_none(filepath, "filepath")
        if not filepath.strip():
            raise ValidationError("Filepath cannot be empty")
        
        # Check file exists and is readable
        if not self.file_exists(filepath):
            raise FileError(
                f"File not found: {filepath}",
                details={'filepath': filepath}
            )
        
        if not os.access(filepath, os.R_OK):
            raise FileError(
                f"File is not readable: {filepath}",
                details={'filepath': filepath, 'permission': 'read'}
            )
        
        # Get file info
        file_size = os.path.getsize(filepath)
        file_name = Path(filepath).name
        
        # Check file not empty
        if file_size == 0:
            raise DataError(
                f"File is empty: {file_name}",
                details={'filepath': filepath, 'size': 0}
            )
        
        logger.info(f"Loading dataset from {file_name} (size: {file_size:,} bytes)")
        
        # Load the dataset using DatasetLoader
        with log_performance(logger, f"load dataset from {file_name}"):
            dataset = DatasetLoader.load(filepath, channel_config)
        
        # Validate the loaded dataset
        self.validate_dataset(dataset, file_name)
        
        # Log success
        sweep_count = dataset.sweep_count() if hasattr(dataset, 'sweep_count') else 'unknown'
        channel_count = dataset.channel_count() if hasattr(dataset, 'channel_count') else 'unknown'
        
        logger.info(
            f"Successfully loaded {file_name}: "
            f"{sweep_count} sweeps, {channel_count} channels"
        )
        
        return dataset
    
    def validate_dataset(self, dataset: ElectrophysiologyDataset, 
                        file_name: str = "unknown") -> bool:
        """
        Validate a loaded dataset.
        
        Args:
            dataset: Dataset to validate
            file_name: Name of source file for error messages
            
        Returns:
            True if valid
            
        Raises:
            DataError: If dataset violates validation rules
        """
        # Check dataset not None
        if dataset is None:
            raise DataError(
                f"Failed to load dataset from {file_name}",
                details={'file_name': file_name}
            )
        
        # Check dataset has data
        if dataset.is_empty():
            sweep_count = dataset.sweep_count() if hasattr(dataset, 'sweep_count') else 0
            raise DataError(
                f"Dataset is empty - no valid sweeps found in {file_name}",
                details={'file_name': file_name, 'sweep_count': sweep_count}
            )
        
        # Check dataset has channels
        if hasattr(dataset, 'channel_count'):
            channel_count = dataset.channel_count()
            if channel_count < 1:
                raise DataError(
                    f"Dataset has no channels in {file_name}",
                    details={'file_name': file_name, 'channel_count': channel_count}
                )
        
        logger.debug(f"Dataset validation passed for {file_name}")
        return True
    
    # =========================================================================
    # Export Operations (from export_service.py)
    # =========================================================================
    
    def export_analysis_data(self,
                            analysis_data: Dict[str, Any],
                            file_path: str,
                            format_spec: str = '%.6f') -> ExportResult:
        """
        Export analysis data to a CSV file.
        
        To add support for new export formats:
        1. Add format detection based on file extension
        2. Add new export method (e.g., export_to_xlsx, export_to_hdf5)
        3. Call appropriate method based on extension
        
        Args:
            analysis_data: Data to export with 'headers' and 'data' keys
            file_path: Target file path
            format_spec: Number formatting specification
            
        Returns:
            ExportResult with operation status
        """
        logger.info(f"Starting export to {Path(file_path).name}")
        
        try:
            # Validate data structure
            if not analysis_data or 'headers' not in analysis_data or 'data' not in analysis_data:
                raise ValidationError("Invalid data structure for export")
            
            headers = analysis_data['headers']
            data = analysis_data['data']
            
            # Convert to numpy array if needed
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Check data not empty
            if data.size == 0:
                raise DataError("Cannot export empty data array")
            
            # Validate dimensions
            expected_cols = data.shape[1] if data.ndim > 1 else 1
            if len(headers) != expected_cols:
                raise DataError(
                    f"Header count ({len(headers)}) doesn't match data columns ({expected_cols})"
                )
            
            # Calculate metrics
            records = data.shape[0] if data.ndim > 1 else len(data)
            
            # Ensure directory exists
            directory = str(Path(file_path).parent)
            if directory and directory != '.':
                self.ensure_directory(directory)
            
            # Write CSV file
            fmt = analysis_data.get('format_spec', format_spec)
            header_str = ','.join(headers) if headers else ''
            np.savetxt(file_path, data, delimiter=',', fmt=fmt,
                      header=header_str, comments='')
            
            # Verify file was created
            if not self.file_exists(file_path):
                raise FileError(
                    "Export appeared to succeed but file was not created",
                    details={'file_path': file_path}
                )
            
            file_size = os.path.getsize(file_path)
            logger.info(
                f"Successfully exported {records} records to {Path(file_path).name} "
                f"({file_size:,} bytes)"
            )
            
            return ExportResult(
                success=True,
                file_path=file_path,
                records_exported=records
            )
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                error_message=str(e)
            )
    
    def get_suggested_filename(self,
                              source_file_path: str,
                              analysis_params: Optional[Any] = None,
                              suffix: str = "_analyzed") -> str:
        """
        Generate a suggested filename for export.
        
        Args:
            source_file_path: Original data file path
            analysis_params: Optional analysis parameters for context
            suffix: Suffix to append
            
        Returns:
            Suggested filename
        """
        if not source_file_path:
            logger.warning("No source file path provided, using default filename")
            return f"analysis{suffix}.csv"
        
        # Extract base name and remove bracketed indices
        base_name = Path(source_file_path).stem
        base_name = re.sub(r'\[.*?\]', '', base_name)
        base_name = base_name.strip(' _')
        
        # Add context-specific suffix if available
        if analysis_params and hasattr(analysis_params, 'y_axis'):
            y_axis = analysis_params.y_axis
            if hasattr(y_axis, 'peak_type') and y_axis.measure == "Peak":
                peak_suffix_map = {
                    "Absolute": "_absolute",
                    "Positive": "_positive",
                    "Negative": "_negative",
                    "Peak-Peak": "_peak-peak"
                }
                suffix = peak_suffix_map.get(y_axis.peak_type, suffix)
        
        suggested = f"{base_name}{suffix}.csv"
        logger.debug(f"Suggested filename: {suggested}")
        
        return suggested
    
    def prepare_export_path(self, file_path: str, ensure_unique: bool = True) -> str:
        """
        Prepare and validate an export path.
        
        Args:
            file_path: Desired export path
            ensure_unique: Whether to ensure path is unique
            
        Returns:
            Prepared path (possibly made unique)
        """
        # Ensure unique if requested
        if ensure_unique and self.file_exists(file_path):
            base = Path(file_path)
            directory = base.parent
            stem = base.stem
            suffix = base.suffix
            
            counter = 1
            while counter <= 10000:
                new_name = f"{stem}_{counter}{suffix}"
                new_path = directory / new_name
                if not self.file_exists(str(new_path)):
                    logger.info(f"Using unique path: {new_path.name}")
                    return str(new_path)
                counter += 1
            
            raise FileError(f"Could not find unique filename after 10000 attempts")
        
        return file_path
    
    def export_multiple_tables(self,
                              tables: List[Dict[str, Any]],
                              output_directory: str,
                              base_name: str = "export") -> List[ExportResult]:
        """
        Export multiple data tables to separate files.
        
        Args:
            tables: List of export data dictionaries
            output_directory: Target directory
            base_name: Base filename
            
        Returns:
            List of ExportResult objects
        """
        if not tables:
            logger.warning("No tables provided for batch export")
            return []
        
        logger.info(f"Starting batch export of {len(tables)} tables")
        
        # Ensure output directory exists
        self.ensure_directory(output_directory)
        
        results = []
        for i, table in enumerate(tables):
            # Generate filename
            suffix = table.get('suffix', f"_{i+1}")
            filename = f"{base_name}{suffix}.csv"
            file_path = str(Path(output_directory) / filename)
            
            # Ensure unique path
            file_path = self.prepare_export_path(file_path, ensure_unique=True)
            
            # Export table
            result = self.export_analysis_data(table, file_path)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch export complete: {successful}/{len(tables)} successful")
        
        return results
    
    # =========================================================================
    # File Operations (from file_io.py)
    # =========================================================================
    
    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            path: File path to check
            
        Returns:
            True if file exists
        """
        return os.path.exists(path)
    
    def ensure_directory(self, path: str) -> None:
        """
        Ensure a directory exists, creating if necessary.
        
        Args:
            path: Directory path
            
        Raises:
            FileError: If directory cannot be created
        """
        if path and not os.path.exists(path):
            logger.debug(f"Creating directory: {path}")
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                raise FileError(
                    f"Could not create directory",
                    details={'directory': path},
                    cause=e
                )
    
    def validate_export_path(self, file_path: str) -> None:
        """
        Validate that a file path is suitable for export.
        
        Args:
            file_path: Path to validate
            
        Raises:
            ValidationError: If path is invalid
        """
        validate_not_none(file_path, "file_path")
        
        if not file_path.strip():
            raise ValidationError("File path cannot be empty")
        
        path = Path(file_path)
        
        # Must have an extension
        if not path.suffix:
            raise ValidationError("Export file must have an extension")
        
        # Check for invalid characters
        invalid_chars = '<>:"|?*' if os.name == 'nt' else '\0'
        filename = path.name
        invalid_found = [c for c in filename if c in invalid_chars]
        
        if invalid_found:
            raise ValidationError(
                f"Filename contains invalid characters: {invalid_found}"
            )
        
        # Check if existing file is writable
        if self.file_exists(file_path):
            if not os.access(file_path, os.W_OK):
                raise ValidationError(
                    f"Cannot overwrite file (no write permission)"
                )