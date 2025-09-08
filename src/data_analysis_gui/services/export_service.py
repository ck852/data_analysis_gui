"""
Refactored export service with proper separation of concerns.

This service handles the business logic for data export, delegating all
infrastructure operations to injected dependencies.

Phase 5 Refactor: Complete rewrite following SOLID principles.

Author: Data Analysis GUI Contributors
License: MIT
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

from data_analysis_gui.core.interfaces import IFileWriter, IFileSystem, IPathUtilities
from data_analysis_gui.core.exceptions import (
    ExportError, ValidationError, DataError,
    validate_not_none
)
from data_analysis_gui.config.logging import get_logger, log_performance

logger = get_logger(__name__)


@dataclass
class ExportResult:
    """Result of an export operation."""
    success: bool
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    records_exported: int = 0


class ExportService:
    """
    Business service for data export operations.
    
    This service contains only business logic and validation,
    delegating all I/O operations to injected dependencies.
    This follows the Single Responsibility and Dependency Inversion principles.
    """
    
    def __init__(self,
                 file_writer: IFileWriter,
                 file_system: IFileSystem,
                 path_utilities: IPathUtilities):
        """
        Initialize with injected dependencies.
        
        Args:
            file_writer: Implementation for writing files
            file_system: Implementation for file system operations
            path_utilities: Implementation for path manipulation
        """
        validate_not_none(file_writer, "file_writer")
        validate_not_none(file_system, "file_system")
        validate_not_none(path_utilities, "path_utilities")
        
        self.file_writer = file_writer
        self.file_system = file_system
        self.path_utilities = path_utilities
        
        logger.info("ExportService initialized")
    
    def export_analysis_data(self,
                            analysis_data: Dict[str, Any],
                            file_path: str,
                            format_spec: str = '%.6f') -> ExportResult:
        """
        Export analysis data to a file.
        
        This method contains the business logic for export:
        1. Validate the data structure
        2. Validate the export path
        3. Prepare the data for export
        4. Delegate writing to infrastructure
        
        Args:
            analysis_data: Data to export with 'headers' and 'data' keys
            file_path: Target file path
            format_spec: Number formatting specification
            
        Returns:
            ExportResult with operation status
        """
        logger.info(f"Starting export to {Path(file_path).name}")
        
        try:
            # Business validation
            validated_data = self._validate_export_data(analysis_data)
            self._validate_export_path(file_path)
            
            # Extract components
            headers = validated_data['headers']
            data_array = validated_data['data']
            fmt = validated_data.get('format_spec', format_spec)
            
            # Calculate metrics
            records = data_array.shape[0] if data_array.ndim > 1 else len(data_array)
            columns = data_array.shape[1] if data_array.ndim > 1 else 1
            
            logger.info(
                f"Exporting {records} records × {columns} columns to {Path(file_path).name}"
            )
            
            # Ensure directory exists (delegate to infrastructure)
            directory = str(Path(file_path).parent)
            if directory and directory != '.':
                self.file_writer.ensure_directory(directory)
            
            # Delegate actual writing to infrastructure
            with log_performance(logger, f"write {records} records"):
                self.file_writer.write_csv(file_path, data_array, headers, fmt)
            
            # Verify the file was created (business rule)
            if not self.file_system.exists(file_path):
                raise ExportError(
                    "Export appeared to succeed but file was not created",
                    details={'file_path': file_path}
                )
            
            file_size = self.file_system.get_size(file_path)
            logger.info(
                f"Successfully exported {records} records to {Path(file_path).name} "
                f"({file_size:,} bytes)"
            )
            
            return ExportResult(
                success=True,
                file_path=file_path,
                records_exported=records
            )
            
        except (ValidationError, DataError, ExportError) as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                error_message=str(e)
            )
            
        except Exception as e:
            logger.error(f"Unexpected export error: {e}", exc_info=True)
            return ExportResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _validate_export_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and prepare data for export (business logic).
        
        Args:
            analysis_data: Data to validate
            
        Returns:
            Validated data with numpy array conversion
            
        Raises:
            ValidationError: If data structure is invalid
            DataError: If data is empty or corrupted
        """
        validate_not_none(analysis_data, "analysis_data")
        
        if not analysis_data:
            raise ValidationError("No data provided for export")
        
        # Business rule: must have headers and data
        if 'headers' not in analysis_data:
            raise ValidationError(
                "Export data missing 'headers' key",
                details={'available_keys': list(analysis_data.keys())}
            )
        
        if 'data' not in analysis_data:
            raise ValidationError(
                "Export data missing 'data' key",
                details={'available_keys': list(analysis_data.keys())}
            )
        
        headers = analysis_data['headers']
        data = analysis_data['data']
        
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            logger.debug(f"Converting data from {type(data).__name__} to numpy array")
            try:
                data = np.array(data)
            except Exception as e:
                raise DataError(
                    "Could not convert data to numpy array",
                    details={'data_type': type(data).__name__},
                    cause=e
                )
        
        # Business rule: data must not be empty
        if data.size == 0:
            raise DataError("Cannot export empty data array")
        
        # Business rule: headers must match data dimensions
        expected_cols = data.shape[1] if data.ndim > 1 else 1
        if len(headers) != expected_cols:
            raise DataError(
                f"Header count ({len(headers)}) doesn't match data columns ({expected_cols})",
                details={
                    'header_count': len(headers),
                    'data_columns': expected_cols
                }
            )
        
        return {
            'headers': headers,
            'data': data,
            'format_spec': analysis_data.get('format_spec')
        }
    
    def _validate_export_path(self, file_path: str) -> None:
        """
        Validate the export file path (business rules).
        
        Args:
            file_path: Path to validate
            
        Raises:
            ValidationError: If path violates business rules
        """
        validate_not_none(file_path, "file_path")
        
        if not file_path.strip():
            raise ValidationError("File path cannot be empty")
        
        path = Path(file_path)
        
        # Business rule: must have an extension
        if not path.suffix:
            raise ValidationError(
                "Export file must have an extension",
                details={'file_path': file_path}
            )
        
        # Business rule: filename must be valid
        invalid_chars = '<>:"|?*' if os.name == 'nt' else '\0'
        filename = path.name
        invalid_found = [c for c in filename if c in invalid_chars]
        
        if invalid_found:
            raise ValidationError(
                f"Filename contains invalid characters: {invalid_found}",
                details={'filename': filename, 'invalid_chars': invalid_found}
            )
        
        # Check if existing file is writable
        if self.file_system.exists(file_path):
            if not self.file_system.is_writable(file_path):
                raise ValidationError(
                    f"Cannot overwrite file (no write permission)",
                    details={'file_path': file_path}
                )
            logger.debug(f"Will overwrite existing file: {path.name}")
    
    def get_suggested_filename(self,
                              source_file_path: str,
                              analysis_params: Optional[Any] = None,
                              suffix: str = "_analyzed") -> str:
        """
        Generate a suggested filename for export (business logic).
        
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
        
        # Extract base name
        base_name = Path(source_file_path).stem
        
        # Business rule: remove bracketed indices
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
            
        Raises:
            ValidationError: If path is invalid
        """
        # Validate the path
        self._validate_export_path(file_path)
        
        # Optionally ensure uniqueness
        if ensure_unique and self.file_system.exists(file_path):
            unique_path = self.path_utilities.ensure_unique_path(file_path)
            logger.info(f"Using unique path: {Path(unique_path).name}")
            return unique_path
        
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
        validate_not_none(tables, "tables")
        validate_not_none(output_directory, "output_directory")
        
        if not tables:
            logger.warning("No tables provided for batch export")
            return []
        
        logger.info(f"Starting batch export of {len(tables)} tables")
        
        # Ensure output directory exists
        self.file_writer.ensure_directory(output_directory)
        
        results = []
        successful = 0
        total_records = 0
        
        with log_performance(logger, f"batch export {len(tables)} tables"):
            for i, table in enumerate(tables):
                # Generate filename
                suffix = table.get('suffix', f"_{i+1}")
                filename = f"{base_name}{suffix}.csv"
                file_path = str(Path(output_directory) / filename)
                
                # Ensure unique path
                file_path = self.path_utilities.ensure_unique_path(file_path)
                
                # Export table
                result = self.export_analysis_data(table, file_path)
                results.append(result)
                
                if result.success:
                    successful += 1
                    total_records += result.records_exported
        
        logger.info(
            f"Batch export complete: {successful}/{len(tables)} successful, "
            f"{total_records} total records"
        )
        
        return results