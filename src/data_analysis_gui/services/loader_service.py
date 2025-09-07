"""
Centralized file loading and I/O service.

This service consolidates all file operations, providing a clean interface
for loading various data formats and exporting results. It replaces the
scattered file I/O utilities that were previously in utils/file_io.py.

Phase 2 Refactor: Created to break the utils god module and establish
proper domain boundaries for file operations.
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

from data_analysis_gui.core.dataset import DatasetLoader, ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions


class LoaderService:
    """
    Service for all file loading and I/O operations.
    
    This service provides static methods for file operations, maintaining
    a clean separation between file I/O and business logic. All methods
    are stateless and thread-safe.
    """
    
    @staticmethod
    def load_dataset(filepath: str, 
                    channel_config: Optional[ChannelDefinitions] = None) -> ElectrophysiologyDataset:
        """
        Load a data file into an ElectrophysiologyDataset.
        
        Args:
            filepath: Path to the data file
            channel_config: Optional channel configuration
            
        Returns:
            Loaded dataset
            
        Raises:
            IOError: If file cannot be loaded
            ValueError: If file structure is invalid
        """
        return DatasetLoader.load(filepath, channel_config)
    
    @staticmethod
    def export_to_csv(filepath: str, data: np.ndarray, headers: List[str], 
                     format_spec: str = '%.6f') -> None:
        """
        Export numpy array to CSV with headers.
        
        Args:
            filepath: Output file path
            data: Numpy array to export
            headers: List of column headers
            format_spec: Format specification for numbers
            
        Raises:
            IOError: If file cannot be written
        """
        header_str = ','.join(headers)
        np.savetxt(filepath, data, delimiter=',', fmt=format_spec,
                  header=header_str, comments='')
    
    @staticmethod
    def get_next_available_filename(path: str) -> str:
        """
        Find available filename by appending _1, _2, etc.
        
        Args:
            path: Initial file path to check
        
        Returns:
            Available file path (original or with suffix)
        """
        if not os.path.exists(path):
            return path
        
        base, ext = os.path.splitext(path)
        i = 1
        while True:
            new_path = f"{base}_{i}{ext}"
            if not os.path.exists(new_path):
                return new_path
            i += 1
    
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        Sanitize string for use as filename.
        
        Args:
            name: String to sanitize
        
        Returns:
            Safe filename string with problematic characters replaced
        """
        # Handle parentheses with special content
        def replacer(match):
            content = match.group(1)
            if '+' in content or '-' in content:
                return '_' + content
            return ''
        
        name_after_parens = re.sub(r'\s*\((.*?)\)', replacer, name).strip()
        safe_name = re.sub(r'[^\w+-]', '_', name_after_parens).replace('__', '_')
        return safe_name
    
    @staticmethod
    def extract_file_number(filepath: str) -> int:
        """
        Extract number from filename for sorting.
        
        Args:
            filepath: File path to extract number from
        
        Returns:
            Integer extracted from filename, or 0 if not found
        """
        filename = os.path.basename(filepath)
        try:
            # Look for numbers in the filename
            number_part = filename.split('_')[-1].split('.')[0]
            return int(number_part)
        except (IndexError, ValueError):
            return 0
    
    @staticmethod
    def validate_file_exists(filepath: str) -> bool:
        """
        Check if a file exists and is readable.
        
        Args:
            filepath: Path to check
            
        Returns:
            True if file exists and is readable
        """
        return os.path.isfile(filepath) and os.access(filepath, os.R_OK)
    
    @staticmethod
    def get_file_info(filepath: str) -> Dict[str, Any]:
        """
        Get metadata about a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dictionary with file metadata (size, modified time, etc.)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        stat = os.stat(filepath)
        return {
            'path': filepath,
            'name': os.path.basename(filepath),
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'extension': Path(filepath).suffix.lower()
        }
    
    @staticmethod
    def create_export_path(base_path: str, suffix: str = "_analyzed", 
                          extension: str = ".csv") -> str:
        """
        Create an export path based on the input file path.
        
        Args:
            base_path: Original file path
            suffix: Suffix to add before extension
            extension: New file extension
            
        Returns:
            Export file path
        """
        base_name = Path(base_path).stem
        directory = Path(base_path).parent
        export_name = f"{base_name}{suffix}{extension}"
        return str(directory / export_name)