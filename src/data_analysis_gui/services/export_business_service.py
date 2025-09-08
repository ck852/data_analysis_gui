"""
Pure business logic export service with no GUI dependencies.
Handles all data export operations independently of the presentation layer.

Phase 2 Refactor: Converted from static methods to instance methods for proper
dependency injection. This allows for better testing, mocking, and future
configuration options.

Phase 5 Refactor: Added comprehensive logging, converted validation to exceptions,
and implemented fail-fast principles. All methods now raise specific exceptions
instead of returning error tuples or None.

Author: Data Analysis GUI Contributors
License: MIT
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from data_analysis_gui.services.loader_service import LoaderService
from data_analysis_gui.core.exceptions import (
    ExportError, ValidationError, FileError, DataError,
    validate_not_none, validate_positive
)
from data_analysis_gui.config.logging import (
    get_logger, log_performance, log_error_with_context
)

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
    Business logic for data export operations.
    
    Phase 2 Refactor: Converted to instance methods from static methods
    to enable proper dependency injection and testing.
    
    Phase 5 Refactor: All methods now follow fail-fast principles with
    comprehensive logging and specific exceptions.
    """
    
    def __init__(self):
        """Initialize the export service."""
        logger.info("ExportService initialized")
    
    def export_analysis_data(
        self,
        analysis_data: Dict[str, Any], 
        file_path: str,
        format_spec: str = '%.6f'
    ) -> ExportResult:
        """
        Export analysis data to file.
        
        Phase 5: Added comprehensive logging and performance metrics.
        Still returns ExportResult for backward compatibility with UI,
        but internally uses exceptions for validation.
        
        Args:
            analysis_data: Table data from AnalysisEngine.get_export_table()
                          Expected keys: 'headers', 'data', 'format_spec' (optional)
            file_path: Complete file path for export
            format_spec: Number formatting specification
            
        Returns:
            ExportResult indicating success/failure
        """
        logger.info(f"Starting export to {Path(file_path).name}")
        
        try:
            # Validate inputs using exceptions internally
            validate_not_none(analysis_data, "analysis_data")
            validate_not_none(file_path, "file_path")
            
            if not analysis_data:
                raise ValidationError("No data provided for export")
            
            if 'headers' not in analysis_data or 'data' not in analysis_data:
                raise ValidationError(
                    "Invalid data structure: missing 'headers' or 'data'",
                    details={'keys': list(analysis_data.keys())}
                )
            
            headers = analysis_data.get('headers', [])
            data = analysis_data.get('data', np.array([[]]))
            
            # Use provided format spec from data or fallback to parameter
            fmt = analysis_data.get('format_spec', format_spec)
            
            # Validate and convert data
            if not isinstance(data, np.ndarray):
                logger.debug(f"Converting data from {type(data).__name__} to numpy array")
                try:
                    data = np.array(data)
                except Exception as e:
                    raise DataError(
                        f"Could not convert data to numpy array",
                        details={'data_type': type(data).__name__},
                        cause=e
                    )
            
            if data.size == 0:
                raise DataError("No data to export (empty array)")
            
            # Calculate metrics for logging
            records = data.shape[0] if data.ndim > 1 else len(data)
            columns = data.shape[1] if data.ndim > 1 else 1
            
            logger.info(
                f"Exporting {records} records × {columns} columns to {Path(file_path).name}"
            )
            
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                logger.debug(f"Creating export directory: {directory}")
                try:
                    os.makedirs(directory, exist_ok=True)
                except OSError as e:
                    raise FileError(
                        f"Could not create directory '{directory}'",
                        details={'directory': directory},
                        cause=e
                    )
            
            # Export with performance logging
            with log_performance(logger, f"write {records} records to CSV"):
                header_str = ','.join(headers) if headers else ''
                np.savetxt(
                    file_path, 
                    data, 
                    delimiter=',', 
                    header=header_str, 
                    fmt=fmt, 
                    comments=''
                )
            
            # Verify export
            if not os.path.exists(file_path):
                raise ExportError(
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
            
        except (ValidationError, DataError, FileError, ExportError) as e:
            # Log the specific error
            log_error_with_context(
                logger, e, "export_analysis_data",
                file_path=file_path,
                data_shape=data.shape if isinstance(data, np.ndarray) else None
            )
            
            return ExportResult(
                success=False,
                error_message=str(e)
            )
            
        except Exception as e:
            # Unexpected error
            log_error_with_context(
                logger, e, "export_analysis_data",
                file_path=file_path,
                error_type=type(e).__name__
            )
            
            return ExportResult(
                success=False,
                error_message=f"Unexpected export error: {str(e)}"
            )
    
    def get_suggested_filename(
        self,
        source_file_path: str, 
        analysis_params: Optional[Any] = None,
        suffix: str = "_analyzed"
    ) -> str:
        """
        Generate suggested export filename.
        
        Phase 5: Added validation and logging.
        
        Args:
            source_file_path: Path to original data file
            analysis_params: Analysis configuration (optional, for advanced naming)
            suffix: Filename suffix
            
        Returns:
            Suggested filename (not full path)
            
        Raises:
            ValidationError: If inputs are invalid
        """
        if not source_file_path:
            logger.warning("No source file path provided, using default filename")
            return f"analysis{suffix}.csv"
        
        # Extract base name without extension
        base_name = Path(source_file_path).stem
        
        # Remove bracketed content (e.g., "[1-234]")
        base_name = re.sub(r'\[.*?\]', '', base_name)
        
        # Clean up any trailing/leading whitespace or underscores
        base_name = base_name.strip(' _')
        
        # Add peak type suffix if specified in parameters
        if analysis_params:
            # Check for peak type in y_axis configuration
            if hasattr(analysis_params, 'y_axis') and hasattr(analysis_params.y_axis, 'peak_type'):
                peak_type = analysis_params.y_axis.peak_type
                if peak_type and analysis_params.y_axis.measure == "Peak":
                    peak_suffix_map = {
                        "Absolute": "_absolute",
                        "Positive": "_positive", 
                        "Negative": "_negative",
                        "Peak-Peak": "_peak-peak"
                    }
                    suffix = peak_suffix_map.get(peak_type, suffix)
                    logger.debug(f"Using peak-specific suffix: {suffix}")
        
        suggested_name = f"{base_name}{suffix}.csv"
        logger.debug(f"Suggested filename: {suggested_name}")
        
        return suggested_name
    
    def validate_export_path(self, file_path: str) -> None:
        """
        Validate export file path.
        
        Phase 5: Converted to raise exceptions instead of returning tuple.
        For backward compatibility, a wrapper method returns the tuple.
        
        Args:
            file_path: Path to validate
            
        Raises:
            ValidationError: If path is invalid
            FileError: If path exists but isn't writable
        """
        validate_not_none(file_path, "file_path")
        
        if not file_path.strip():
            raise ValidationError("File path cannot be empty")
        
        try:
            path = Path(file_path)
            
            # Check for invalid characters in filename
            invalid_chars = '<>:"|?*' if os.name == 'nt' else '\0'
            filename = path.name
            
            if any(char in filename for char in invalid_chars):
                raise ValidationError(
                    f"Filename contains invalid characters",
                    details={
                        'filename': filename,
                        'invalid_chars': [c for c in filename if c in invalid_chars]
                    }
                )
            
            # Check if parent directory exists or can be created
            parent = path.parent
            if parent and not parent.exists():
                # Check if we can create it (dry run)
                try:
                    # Check if parent of parent exists and is writable
                    if parent.parent.exists():
                        # Check write permission
                        test_file = parent.parent / '.test_write_permission'
                        try:
                            test_file.touch()
                            test_file.unlink()
                            logger.debug(f"Directory {parent} can be created")
                        except:
                            raise FileError(
                                f"No write permission in directory",
                                details={'directory': str(parent.parent)}
                            )
                except Exception as e:
                    logger.warning(f"Cannot verify directory creation: {e}")
            
            # Check if file already exists and is writable
            if path.exists():
                if not os.access(str(path), os.W_OK):
                    raise FileError(
                        f"File exists and is not writable",
                        details={'file_path': str(path)}
                    )
                logger.debug(f"File {path.name} exists and will be overwritten")
            
            # Validate extension
            if not path.suffix:
                raise ValidationError(
                    "File must have an extension",
                    details={'file_path': str(path)}
                )
            
            logger.debug(f"Export path validation successful: {file_path}")
            
        except (ValidationError, FileError):
            # Re-raise our exceptions
            raise
            
        except Exception as e:
            # Wrap unexpected errors
            logger.error(f"Unexpected error validating path: {e}")
            raise ValidationError(
                f"Invalid file path",
                details={'file_path': file_path},
                cause=e
            )
    
    def validate_export_path_compat(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Backward compatibility wrapper for validate_export_path.
        
        This method maintains the old tuple return interface while using
        the new exception-based validation internally.
        
        Args:
            file_path: Path to validate
            
        Returns:
            (is_valid, error_message) tuple for backward compatibility
        """
        try:
            self.validate_export_path(file_path)
            return True, None
        except (ValidationError, FileError) as e:
            logger.debug(f"Path validation failed: {e}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Remove invalid characters from a filename.
        
        Phase 5: Added validation and logging.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for all platforms
            
        Raises:
            ValidationError: If filename is None
        """
        validate_not_none(filename, "filename")
        
        original = filename
        
        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')
        
        # Remove other invalid characters
        if os.name == 'nt':  # Windows
            invalid_chars = '<>:"|?*'
        else:  # Unix-like
            invalid_chars = '\0'
        
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure filename is not empty
        if not filename:
            logger.warning("Sanitization resulted in empty filename, using default")
            filename = 'exported_data'
        
        if filename != original:
            logger.debug(f"Sanitized filename: '{original}' -> '{filename}'")
        
        return filename
    
    def ensure_unique_path(self, file_path: str) -> str:
        """
        Ensure file path is unique by appending numbers if necessary.
        
        Phase 5: Added validation, logging, and fail-safe limit.
        
        Args:
            file_path: Desired file path
            
        Returns:
            Unique file path (may have _1, _2, etc. appended)
            
        Raises:
            ValidationError: If file_path is invalid
            FileError: If cannot find unique name after max attempts
        """
        validate_not_none(file_path, "file_path")
        
        if not os.path.exists(file_path):
            logger.debug(f"Path is already unique: {file_path}")
            return file_path
        
        path = Path(file_path)
        directory = path.parent
        stem = path.stem
        suffix = path.suffix
        
        counter = 1
        max_attempts = 10000
        
        logger.debug(f"Finding unique path for existing file: {path.name}")
        
        while counter <= max_attempts:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = directory / new_name
            if not new_path.exists():
                logger.info(f"Found unique path: {new_name} (attempt {counter})")
                return str(new_path)
            counter += 1
        
        # Safety check to prevent infinite loop
        raise FileError(
            f"Could not find unique filename after {max_attempts} attempts",
            details={
                'original_path': file_path,
                'attempts': max_attempts
            }
        )
    
    def export_multiple_tables(
        self,
        tables: List[Dict[str, Any]],
        output_directory: str,
        base_name: str = "export"
    ) -> List[ExportResult]:
        """
        Export multiple data tables to separate files.
        
        Phase 5: Added validation, logging, and performance metrics.
        
        Args:
            tables: List of table dictionaries, each with 'headers', 'data', and optional 'suffix'
            output_directory: Directory to save files
            base_name: Base name for all files
            
        Returns:
            List of ExportResult objects for each export
            
        Raises:
            ValidationError: If inputs are invalid
        """
        validate_not_none(tables, "tables")
        validate_not_none(output_directory, "output_directory")
        validate_not_none(base_name, "base_name")
        
        if not tables:
            logger.warning("No tables provided for export")
            return []
        
        logger.info(f"Starting batch export of {len(tables)} tables to {output_directory}")
        
        results = []
        
        # Ensure output directory exists
        try:
            os.makedirs(output_directory, exist_ok=True)
            logger.debug(f"Output directory ready: {output_directory}")
        except OSError as e:
            # Return error for all tables
            logger.error(f"Could not create output directory: {e}")
            error_result = ExportResult(
                success=False,
                error_message=f"Could not create output directory: {str(e)}"
            )
            return [error_result] * len(tables)
        
        # Track performance
        successful_exports = 0
        total_records = 0
        
        with log_performance(logger, f"export {len(tables)} tables"):
            for i, table in enumerate(tables):
                # Generate filename
                suffix = table.get('suffix', f"_{i+1}")
                filename = f"{base_name}{suffix}.csv"
                file_path = os.path.join(output_directory, filename)
                
                # Ensure unique path
                try:
                    file_path = self.ensure_unique_path(file_path)
                except FileError as e:
                    logger.error(f"Could not find unique path for table {i+1}: {e}")
                    results.append(ExportResult(
                        success=False,
                        error_message=str(e)
                    ))
                    continue
                
                # Export table
                result = self.export_analysis_data(table, file_path)
                results.append(result)
                
                if result.success:
                    successful_exports += 1
                    total_records += result.records_exported
        
        # Log summary
        logger.info(
            f"Batch export complete: {successful_exports}/{len(tables)} successful, "
            f"{total_records} total records exported"
        )
        
        if successful_exports < len(tables):
            failed_count = len(tables) - successful_exports
            logger.warning(f"{failed_count} exports failed")
        
        return results