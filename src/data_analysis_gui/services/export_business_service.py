# src/data_analysis_gui/services/export_business_service.py
"""
Pure business logic export service with no GUI dependencies.
Handles all data export operations independently of the presentation layer.

Phase 2 Refactor: Converted from static methods to instance methods for proper
dependency injection. This allows for better testing, mocking, and future
configuration options.
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from data_analysis_gui.services.loader_service import LoaderService

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
    """
    
    def __init__(self):
        """Initialize the export service."""
        # Future: Could accept configuration options here
        pass
    
    def export_analysis_data(
        self,
        analysis_data: Dict[str, Any], 
        file_path: str,
        format_spec: str = '%.6f'
    ) -> ExportResult:
        """
        Export analysis data to file.
        
        Args:
            analysis_data: Table data from AnalysisEngine.get_export_table()
                          Expected keys: 'headers', 'data', 'format_spec' (optional)
            file_path: Complete file path for export
            format_spec: Number formatting specification
            
        Returns:
            ExportResult indicating success/failure
        """
        try:
            # Validate input data structure
            if not analysis_data:
                return ExportResult(
                    success=False,
                    error_message="No data provided for export"
                )
            
            if 'headers' not in analysis_data or 'data' not in analysis_data:
                return ExportResult(
                    success=False,
                    error_message="Invalid data structure: missing 'headers' or 'data'"
                )
            
            headers = analysis_data.get('headers', [])
            data = analysis_data.get('data', np.array([[]]))
            
            # Use provided format spec from data or fallback to parameter
            fmt = analysis_data.get('format_spec', format_spec)
            
            # Validate data
            if not isinstance(data, np.ndarray):
                try:
                    data = np.array(data)
                except Exception as e:
                    return ExportResult(
                        success=False,
                        error_message=f"Could not convert data to array: {str(e)}"
                    )
            
            if data.size == 0:
                return ExportResult(
                    success=False,
                    error_message="No data to export (empty array)"
                )
            
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except OSError as e:
                    return ExportResult(
                        success=False,
                        error_message=f"Could not create directory '{directory}': {str(e)}"
                    )
            
            # Write CSV file
            header_str = ','.join(headers) if headers else ''
            np.savetxt(
                file_path, 
                data, 
                delimiter=',', 
                header=header_str, 
                fmt=fmt, 
                comments=''
            )
            
            # Calculate records exported
            records = data.shape[0] if data.ndim > 1 else len(data)
            
            return ExportResult(
                success=True,
                file_path=file_path,
                records_exported=records
            )
            
        except Exception as e:
            return ExportResult(
                success=False,
                error_message=f"Export failed: {str(e)}"
            )
    
    def get_suggested_filename(
        self,
        source_file_path: str, 
        analysis_params: Optional[Any] = None,
        suffix: str = "_analyzed"
    ) -> str:
        """
        Generate suggested export filename.
        
        Args:
            source_file_path: Path to original data file
            analysis_params: Analysis configuration (optional, for advanced naming)
            suffix: Filename suffix
            
        Returns:
            Suggested filename (not full path)
        """
        if not source_file_path:
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
        
        return f"{base_name}{suffix}.csv"
    
    def validate_export_path(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate export file path.
        
        Args:
            file_path: Path to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not file_path:
            return False, "File path cannot be empty"
        
        try:
            path = Path(file_path)
            
            # Check for invalid characters in filename
            invalid_chars = '<>:"|?*' if os.name == 'nt' else '\0'
            filename = path.name
            if any(char in filename for char in invalid_chars):
                return False, f"Filename contains invalid characters: {invalid_chars}"
            
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
                        except:
                            return False, f"No write permission in directory: {parent.parent}"
                except:
                    pass  # Will be caught when actually trying to create
            
            # Check if file already exists and is writable
            if path.exists():
                if not os.access(str(path), os.W_OK):
                    return False, f"File exists and is not writable: {file_path}"
            
            # Validate extension
            if not path.suffix:
                return False, "File must have an extension (e.g., .csv)"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid file path: {str(e)}"
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Remove invalid characters from a filename.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for all platforms
        """
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
            filename = 'exported_data'
        
        return filename
    
    def ensure_unique_path(self, file_path: str) -> str:
        """
        Ensure file path is unique by appending numbers if necessary.
        
        Args:
            file_path: Desired file path
            
        Returns:
            Unique file path (may have _1, _2, etc. appended)
        """
        if not os.path.exists(file_path):
            return file_path
        
        path = Path(file_path)
        directory = path.parent
        stem = path.stem
        suffix = path.suffix
        
        counter = 1
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = directory / new_name
            if not new_path.exists():
                return str(new_path)
            counter += 1
            
            # Safety check to prevent infinite loop
            if counter > 10000:
                raise ValueError(f"Could not find unique filename after 10000 attempts for: {file_path}")
    
    def export_multiple_tables(
        self,
        tables: List[Dict[str, Any]],
        output_directory: str,
        base_name: str = "export"
    ) -> List[ExportResult]:
        """
        Export multiple data tables to separate files.
        
        Args:
            tables: List of table dictionaries, each with 'headers', 'data', and optional 'suffix'
            output_directory: Directory to save files
            base_name: Base name for all files
            
        Returns:
            List of ExportResult objects for each export
        """
        results = []
        
        # Ensure output directory exists
        try:
            os.makedirs(output_directory, exist_ok=True)
        except OSError as e:
            # Return error for all tables
            error_result = ExportResult(
                success=False,
                error_message=f"Could not create output directory: {str(e)}"
            )
            return [error_result] * len(tables)
        
        for i, table in enumerate(tables):
            # Generate filename
            suffix = table.get('suffix', f"_{i+1}")
            filename = f"{base_name}{suffix}.csv"
            file_path = os.path.join(output_directory, filename)
            
            # Ensure unique path
            file_path = self.ensure_unique_path(file_path)
            
            # Export table
            result = self.export_analysis_data(table, file_path)
            results.append(result)
        
        return results