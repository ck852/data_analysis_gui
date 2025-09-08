"""
Infrastructure layer for file I/O operations.

This module contains concrete implementations of file system operations,
separated from business logic to maintain clean architecture.

Phase 5 Refactor: Created to separate infrastructure concerns from business logic.

Author: Data Analysis GUI Contributors
License: MIT
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from data_analysis_gui.core.dataset import DatasetLoader, ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.exceptions import FileError, DataError
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class FileDatasetLoader:
    """Concrete implementation of dataset loading from files."""
    
    def load(self, filepath: str, channel_config: Optional[ChannelDefinitions]) -> ElectrophysiologyDataset:
        """
        Load a dataset from a file using the existing DatasetLoader.
        
        This is a thin wrapper around the existing DatasetLoader,
        keeping infrastructure separate from business logic.
        """
        logger.debug(f"Loading dataset from {filepath}")
        
        try:
            dataset = DatasetLoader.load(filepath, channel_config)
            
            if dataset is None:
                raise DataError(f"DatasetLoader returned None for {filepath}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            if isinstance(e, (FileError, DataError)):
                raise
            raise FileError(
                f"Failed to load dataset from {filepath}",
                details={'filepath': filepath},
                cause=e
            )


class CsvFileWriter:
    """Concrete implementation for writing CSV files."""
    
    def write_csv(self, filepath: str, data: np.ndarray, headers: List[str], 
                  format_spec: str = '%.6f') -> None:
        """Write data to CSV file."""
        logger.debug(f"Writing CSV to {filepath}")
        
        try:
            header_str = ','.join(headers) if headers else ''
            np.savetxt(filepath, data, delimiter=',', fmt=format_spec,
                      header=header_str, comments='')
                      
        except (IOError, OSError) as e:
            raise FileError(
                f"Failed to write CSV file",
                details={'filepath': filepath},
                cause=e
            )
    
    def ensure_directory(self, directory: str) -> None:
        """Ensure directory exists, creating if necessary."""
        if directory and not os.path.exists(directory):
            logger.debug(f"Creating directory: {directory}")
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                raise FileError(
                    f"Could not create directory",
                    details={'directory': directory},
                    cause=e
                )


class FileSystemOperations:
    """Concrete implementation of file system operations."""
    
    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        return os.path.exists(path)
    
    def is_readable(self, path: str) -> bool:
        """Check if a file is readable."""
        return os.path.isfile(path) and os.access(path, os.R_OK)
    
    def is_writable(self, path: str) -> bool:
        """Check if a file/directory is writable."""
        if os.path.exists(path):
            return os.access(path, os.W_OK)
        # Check parent directory for new files
        parent = os.path.dirname(path)
        return not parent or os.access(parent, os.W_OK)
    
    def get_size(self, path: str) -> int:
        """Get file size in bytes."""
        try:
            return os.path.getsize(path)
        except OSError as e:
            raise FileError(
                f"Could not get file size",
                details={'path': path},
                cause=e
            )
    
    def get_info(self, path: str) -> Dict[str, Any]:
        """Get file metadata."""
        try:
            stat = os.stat(path)
            return {
                'path': path,
                'name': os.path.basename(path),
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'extension': Path(path).suffix.lower()
            }
        except OSError as e:
            raise FileError(
                f"Could not get file info",
                details={'path': path},
                cause=e
            )


class PathUtilities:
    """Concrete implementation of path manipulation utilities."""
    
    def __init__(self, file_system: Optional[FileSystemOperations] = None):
        """
        Initialize with optional file system dependency.
        
        Args:
            file_system: File system operations (defaults to FileSystemOperations)
        """
        self.file_system = file_system or FileSystemOperations()
    
    def sanitize_filename(self, filename: str) -> str:
        """Remove invalid characters from filename."""
        if not filename.strip():
            return "unnamed_file"
        
        # Handle parentheses with special content
        def replacer(match):
            content = match.group(1)
            if '+' in content or '-' in content:
                return '_' + content
            return ''
        
        name_after_parens = re.sub(r'\s*\((.*?)\)', replacer, filename).strip()
        safe_name = re.sub(r'[^\w+-]', '_', name_after_parens).replace('__', '_')
        
        return safe_name if safe_name else "sanitized_file"
    
    def ensure_unique_path(self, path: str) -> str:
        """Ensure path is unique by appending numbers if needed."""
        if not self.file_system.exists(path):
            return path
        
        base = Path(path)
        directory = base.parent
        stem = base.stem
        suffix = base.suffix
        
        counter = 1
        max_attempts = 10000
        
        while counter <= max_attempts:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = directory / new_name
            if not self.file_system.exists(str(new_path)):
                return str(new_path)
            counter += 1
        
        raise FileError(
            f"Could not find unique filename after {max_attempts} attempts",
            details={'original_path': path}
        )
    
    def extract_file_number(self, filepath: str) -> int:
        """Extract number from filename for sorting."""
        filename = os.path.basename(filepath)
        try:
            number_part = filename.split('_')[-1].split('.')[0]
            return int(number_part)
        except (IndexError, ValueError):
            return 0
    
    def create_export_path(self, base_path: str, suffix: str = "_analyzed", 
                          extension: str = ".csv") -> str:
        """Create an export path based on input file."""
        base_name = Path(base_path).stem
        directory = Path(base_path).parent
        export_name = f"{base_name}{suffix}{extension}"
        return str(directory / export_name)