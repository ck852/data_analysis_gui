"""
Protocol definitions for dependency inversion.

This module defines abstract interfaces that allow business logic to depend
on abstractions rather than concrete implementations, following the Dependency
Inversion Principle.

Phase 5 Refactor: Created to properly separate business logic from infrastructure
and enable testability through dependency injection.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Protocol, Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions


class IDatasetLoader(Protocol):
    """Interface for loading electrophysiology datasets."""
    
    def load(self, filepath: str, channel_config: Optional[ChannelDefinitions]) -> ElectrophysiologyDataset:
        """
        Load a dataset from a file.
        
        Args:
            filepath: Path to the data file
            channel_config: Optional channel configuration
            
        Returns:
            Loaded dataset
            
        Raises:
            FileError: If file cannot be loaded
            DataError: If data is corrupted
        """
        ...


class IFileWriter(Protocol):
    """Interface for writing data to files."""
    
    def write_csv(self, filepath: str, data: np.ndarray, headers: List[str], 
                  format_spec: str = '%.6f') -> None:
        """
        Write data to CSV file.
        
        Args:
            filepath: Output file path
            data: Data array to write
            headers: Column headers
            format_spec: Number format specification
            
        Raises:
            FileError: If file cannot be written
        """
        ...
    
    def ensure_directory(self, directory: str) -> None:
        """
        Ensure a directory exists, creating if necessary.
        
        Args:
            directory: Directory path
            
        Raises:
            FileError: If directory cannot be created
        """
        ...


class IFileSystem(Protocol):
    """Interface for file system operations."""
    
    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        ...
    
    def is_readable(self, path: str) -> bool:
        """Check if a file is readable."""
        ...
    
    def is_writable(self, path: str) -> bool:
        """Check if a file/directory is writable."""
        ...
    
    def get_size(self, path: str) -> int:
        """Get file size in bytes."""
        ...
    
    def get_info(self, path: str) -> Dict[str, Any]:
        """Get file metadata."""
        ...


class IPathUtilities(Protocol):
    """Interface for path manipulation utilities."""
    
    def sanitize_filename(self, filename: str) -> str:
        """Remove invalid characters from filename."""
        ...
    
    def ensure_unique_path(self, path: str) -> str:
        """Ensure path is unique by appending numbers if needed."""
        ...
    
    def extract_file_number(self, filepath: str) -> int:
        """Extract number from filename for sorting."""
        ...
    
    def create_export_path(self, base_path: str, suffix: str, extension: str) -> str:
        """Create an export path based on input file."""
        ...


class IDataValidator(Protocol):
    """Interface for data validation."""
    
    def validate_export_data(self, data: Dict[str, Any]) -> None:
        """
        Validate data structure for export.
        
        Raises:
            ValidationError: If data is invalid
        """
        ...
    
    def validate_dataset(self, dataset: ElectrophysiologyDataset) -> None:
        """
        Validate a loaded dataset.
        
        Raises:
            DataError: If dataset is invalid or empty
        """
        ...