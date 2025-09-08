"""
Centralized file loading and I/O service.

This service consolidates all file operations, providing a clean interface
for loading various data formats and exporting results. It replaces the
scattered file I/O utilities that were previously in utils/file_io.py.

Phase 2 Refactor: Created to break the utils god module and establish
proper domain boundaries for file operations.

Phase 5 Refactor: Added comprehensive error handling, logging, and fail-fast
validation. All methods now raise specific exceptions instead of returning
None or error tuples.

Author: Data Analysis GUI Contributors
License: MIT
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

from data_analysis_gui.core.dataset import DatasetLoader, ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.exceptions import (
    FileError, ValidationError, DataError, 
    validate_file_exists, validate_not_none
)
from data_analysis_gui.config.logging import get_logger, log_performance

logger = get_logger(__name__)


class LoaderService:
    """
    Service for all file loading and I/O operations.
    
    Phase 5 Refactor: All methods now follow fail-fast principles with
    explicit exception handling and comprehensive logging.
    
    This service provides static methods for file operations, maintaining
    a clean separation between file I/O and business logic. All methods
    are stateless and thread-safe.
    """
    
    @staticmethod
    def load_dataset(filepath: str, 
                    channel_config: Optional[ChannelDefinitions] = None) -> ElectrophysiologyDataset:
        """
        Load a data file into an ElectrophysiologyDataset.
        
        Phase 5: Added comprehensive error handling and logging.
        Fails fast with specific exceptions for different failure modes.
        
        Args:
            filepath: Path to the data file
            channel_config: Optional channel configuration
            
        Returns:
            Loaded dataset (never None)
            
        Raises:
            ValidationError: If filepath is None or empty
            FileError: If file doesn't exist, isn't readable, or has invalid format
            DataError: If file structure is invalid or data is corrupted
        """
        # Validate inputs - fail fast
        validate_not_none(filepath, "filepath")
        if not filepath.strip():
            raise ValidationError("Filepath cannot be empty")
        
        # Validate file exists and is readable
        validate_file_exists(filepath)
        
        # Get file info for logging
        file_size = os.path.getsize(filepath)
        file_name = Path(filepath).name
        
        logger.info(f"Loading dataset from {file_name} (size: {file_size:,} bytes)")
        
        try:
            with log_performance(logger, f"load dataset from {file_name}"):
                dataset = DatasetLoader.load(filepath, channel_config)
            
            # Validate loaded dataset
            if dataset is None:
                raise DataError(
                    f"DatasetLoader returned None for {file_name}",
                    details={'filepath': filepath, 'file_size': file_size}
                )
            
            if dataset.is_empty():
                raise DataError(
                    f"Dataset is empty - no valid sweeps found in {file_name}",
                    details={
                        'filepath': filepath,
                        'file_size': file_size,
                        'sweep_count': dataset.sweep_count() if hasattr(dataset, 'sweep_count') else 0
                    }
                )
            
            # Log successful load
            sweep_count = dataset.sweep_count() if hasattr(dataset, 'sweep_count') else 'unknown'
            channel_count = dataset.channel_count() if hasattr(dataset, 'channel_count') else 'unknown'
            
            logger.info(
                f"Successfully loaded {file_name}: "
                f"{sweep_count} sweeps, {channel_count} channels"
            )
            
            return dataset
            
        except (FileError, DataError, ValidationError):
            # Re-raise our specific exceptions
            raise
            
        except IOError as e:
            # Convert I/O errors to FileError
            logger.error(f"I/O error loading {file_name}: {e}")
            raise FileError(
                f"Failed to read file {file_name}: {str(e)}",
                details={'filepath': filepath, 'error_type': 'IOError'},
                cause=e
            )
            
        except ValueError as e:
            # Convert value errors to DataError
            logger.error(f"Invalid data structure in {file_name}: {e}")
            raise DataError(
                f"Invalid file structure in {file_name}: {str(e)}",
                details={'filepath': filepath, 'error_type': 'ValueError'},
                cause=e
            )
            
        except Exception as e:
            # Catch any unexpected errors and wrap them
            logger.error(f"Unexpected error loading {file_name}: {e}", exc_info=True)
            raise FileError(
                f"Unexpected error loading {file_name}",
                details={
                    'filepath': filepath,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                cause=e
            )
    
    @staticmethod
    def export_to_csv(filepath: str, data: np.ndarray, headers: List[str], 
                     format_spec: str = '%.6f') -> None:
        """
        Export numpy array to CSV with headers.
        
        Phase 5: Added validation, logging, and specific exceptions.
        
        Args:
            filepath: Output file path
            data: Numpy array to export
            headers: List of column headers
            format_spec: Format specification for numbers
            
        Raises:
            ValidationError: If inputs are invalid
            FileError: If file cannot be written
            DataError: If data structure is invalid
        """
        # Validate inputs
        validate_not_none(filepath, "filepath")
        validate_not_none(data, "data")
        validate_not_none(headers, "headers")
        
        if not filepath.strip():
            raise ValidationError("Filepath cannot be empty")
        
        if not isinstance(data, np.ndarray):
            raise ValidationError(
                f"Data must be numpy array, got {type(data).__name__}",
                details={'data_type': type(data).__name__}
            )
        
        if data.size == 0:
            raise DataError(
                "Cannot export empty data array",
                details={'shape': data.shape}
            )
        
        # Validate headers match data dimensions
        expected_cols = data.shape[1] if data.ndim > 1 else 1
        if len(headers) != expected_cols:
            raise DataError(
                f"Header count ({len(headers)}) doesn't match data columns ({expected_cols})",
                details={
                    'header_count': len(headers),
                    'data_shape': data.shape,
                    'expected_columns': expected_cols
                }
            )
        
        file_name = Path(filepath).name
        records = data.shape[0] if data.ndim > 1 else len(data)
        
        logger.info(f"Exporting {records} records to {file_name}")
        
        try:
            # Ensure directory exists
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                logger.debug(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
            
            # Write CSV
            with log_performance(logger, f"export {records} records to CSV"):
                header_str = ','.join(headers)
                np.savetxt(filepath, data, delimiter=',', fmt=format_spec,
                          header=header_str, comments='')
            
            # Verify file was written
            if not os.path.exists(filepath):
                raise FileError(
                    f"File was not created: {filepath}",
                    details={'filepath': filepath}
                )
            
            file_size = os.path.getsize(filepath)
            logger.info(f"Successfully exported to {file_name} ({file_size:,} bytes)")
            
        except FileError:
            raise
            
        except (IOError, OSError) as e:
            logger.error(f"Failed to write {file_name}: {e}")
            raise FileError(
                f"Failed to write file {file_name}: {str(e)}",
                details={
                    'filepath': filepath,
                    'error_type': type(e).__name__
                },
                cause=e
            )
            
        except Exception as e:
            logger.error(f"Unexpected error exporting to {file_name}: {e}", exc_info=True)
            raise FileError(
                f"Unexpected error during export",
                details={
                    'filepath': filepath,
                    'error_type': type(e).__name__
                },
                cause=e
            )
    
    @staticmethod
    def get_next_available_filename(path: str) -> str:
        """
        Find available filename by appending _1, _2, etc.
        
        Phase 5: Added validation and logging.
        
        Args:
            path: Initial file path to check
        
        Returns:
            Available file path (original or with suffix)
            
        Raises:
            ValidationError: If path is invalid
        """
        validate_not_none(path, "path")
        if not path.strip():
            raise ValidationError("Path cannot be empty")
        
        if not os.path.exists(path):
            logger.debug(f"Path available: {path}")
            return path
        
        base, ext = os.path.splitext(path)
        i = 1
        max_attempts = 10000
        
        while i <= max_attempts:
            new_path = f"{base}_{i}{ext}"
            if not os.path.exists(new_path):
                logger.debug(f"Found available path: {new_path} (attempt {i})")
                return new_path
            i += 1
        
        # If we've tried 10000 times, something is wrong
        raise FileError(
            f"Could not find available filename after {max_attempts} attempts",
            details={'base_path': path, 'attempts': max_attempts}
        )
    
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        Sanitize string for use as filename.
        
        Phase 5: Added validation and fail-fast behavior.
        
        Args:
            name: String to sanitize
        
        Returns:
            Safe filename string with problematic characters replaced
            
        Raises:
            ValidationError: If name is None
        """
        validate_not_none(name, "name")
        
        # Handle empty string case
        if not name.strip():
            logger.warning("Empty filename provided, using default")
            return "unnamed_file"
        
        original_name = name
        
        # Handle parentheses with special content
        def replacer(match):
            content = match.group(1)
            if '+' in content or '-' in content:
                return '_' + content
            return ''
        
        name_after_parens = re.sub(r'\s*\((.*?)\)', replacer, name).strip()
        safe_name = re.sub(r'[^\w+-]', '_', name_after_parens).replace('__', '_')
        
        # Ensure result is not empty
        if not safe_name:
            safe_name = "sanitized_file"
        
        if safe_name != original_name:
            logger.debug(f"Sanitized filename: '{original_name}' -> '{safe_name}'")
        
        return safe_name
    
    @staticmethod
    def extract_file_number(filepath: str) -> int:
        """
        Extract number from filename for sorting.
        
        Phase 5: Added validation and logging.
        
        Args:
            filepath: File path to extract number from
        
        Returns:
            Integer extracted from filename, or 0 if not found
            
        Raises:
            ValidationError: If filepath is None
        """
        validate_not_none(filepath, "filepath")
        
        filename = os.path.basename(filepath)
        try:
            # Look for numbers in the filename
            number_part = filename.split('_')[-1].split('.')[0]
            number = int(number_part)
            logger.debug(f"Extracted number {number} from {filename}")
            return number
        except (IndexError, ValueError):
            logger.debug(f"No number found in {filename}, returning 0")
            return 0
    
    @staticmethod
    def validate_file_exists(filepath: str) -> bool:
        """
        Check if a file exists and is readable.
        
        Phase 5: Now raises exception instead of returning bool.
        
        Args:
            filepath: Path to check
            
        Returns:
            True if file exists and is readable
            
        Raises:
            ValidationError: If filepath is None
            FileError: If file doesn't exist or isn't readable
        """
        validate_not_none(filepath, "filepath")
        
        try:
            # This will raise FileError if file doesn't exist or isn't readable
            validate_file_exists(filepath)
            return True
        except FileError:
            raise
    
    @staticmethod
    def get_file_info(filepath: str) -> Dict[str, Any]:
        """
        Get metadata about a file.
        
        Phase 5: Added validation and specific exceptions.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dictionary with file metadata (size, modified time, etc.)
            
        Raises:
            ValidationError: If filepath is None
            FileError: If file doesn't exist
        """
        validate_not_none(filepath, "filepath")
        validate_file_exists(filepath)
        
        try:
            stat = os.stat(filepath)
            info = {
                'path': filepath,
                'name': os.path.basename(filepath),
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'extension': Path(filepath).suffix.lower()
            }
            
            logger.debug(
                f"File info for {info['name']}: "
                f"size={info['size']:,} bytes, "
                f"extension={info['extension']}"
            )
            
            return info
            
        except OSError as e:
            logger.error(f"Failed to get file info for {filepath}: {e}")
            raise FileError(
                f"Failed to get file information",
                details={'filepath': filepath},
                cause=e
            )
    
    @staticmethod
    def create_export_path(base_path: str, suffix: str = "_analyzed", 
                          extension: str = ".csv") -> str:
        """
        Create an export path based on the input file path.
        
        Phase 5: Added validation and logging.
        
        Args:
            base_path: Original file path
            suffix: Suffix to add before extension
            extension: New file extension
            
        Returns:
            Export file path
            
        Raises:
            ValidationError: If base_path is None or empty
        """
        validate_not_none(base_path, "base_path")
        if not base_path.strip():
            raise ValidationError("Base path cannot be empty")
        
        base_name = Path(base_path).stem
        directory = Path(base_path).parent
        export_name = f"{base_name}{suffix}{extension}"
        export_path = str(directory / export_name)
        
        logger.debug(f"Created export path: {export_path}")
        return export_path