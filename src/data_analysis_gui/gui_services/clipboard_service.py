"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Clipboard service for copying analysis data to system clipboard.

Provides utilities to format data tables and copy them to clipboard
for easy pasting into other applications (Excel, Prism, etc.).
"""

from typing import Dict, Any, List
import numpy as np
from PySide6.QtWidgets import QApplication

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ClipboardService:
    """
    Service for copying data tables to system clipboard.
    
    Formats data as tab-separated values (TSV) which Excel, Prism,
    and most spreadsheet applications handle well when pasting.
    """
    
    @staticmethod
    def format_data_as_text(data: Dict[str, Any], separator: str = "\t") -> str:
        """
        Format a data dictionary as delimited text.
        
        Args:
            data: Dictionary containing 'headers' (list of strings) and 
                  'data' (2D array or list of lists)
            separator: Delimiter to use between columns (default: tab)
            
        Returns:
            Formatted string ready for clipboard, or empty string on error
            
        Example:
            >>> data = {
            ...     "headers": ["Time (ms)", "Current (pA)"],
            ...     "data": [[100.0, -45.2], [200.0, -52.1]]
            ... }
            >>> text = ClipboardService.format_data_as_text(data)
        """
        try:
            # Validate data structure
            if not data or "headers" not in data or "data" not in data:
                logger.error("Invalid data structure for clipboard format")
                return ""
            
            headers = data["headers"]
            data_array = np.array(data["data"])
            
            if data_array.size == 0:
                logger.warning("No data to format for clipboard")
                return ""
            
            # Build text output
            lines = []
            
            # Add header row
            header_line = separator.join(str(h) for h in headers)
            lines.append(header_line)
            
            # Add data rows
            for row in data_array:
                # Convert each value to string, handling numpy types
                row_values = [str(val) for val in row]
                row_line = separator.join(row_values)
                lines.append(row_line)
            
            result = "\n".join(lines)
            logger.debug(f"Formatted {len(data_array)} rows with {len(headers)} columns")
            
            return result
            
        except Exception as e:
            logger.error(f"Error formatting data for clipboard: {e}")
            return ""
    
    @staticmethod
    def copy_to_clipboard(text: str) -> bool:
        """
        Copy text to system clipboard.
        
        Args:
            text: Text to copy to clipboard
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not text:
                logger.warning("Attempted to copy empty text to clipboard")
                return False
            
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            
            logger.info("Data copied to clipboard successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error copying to clipboard: {e}")
            return False
    
    @staticmethod
    def copy_data_to_clipboard(data: Dict[str, Any], separator: str = "\t") -> bool:
        """
        Format and copy data to clipboard in one step.
        
        Args:
            data: Dictionary containing 'headers' and 'data'
            separator: Delimiter to use (default: tab for TSV)
            
        Returns:
            True if successful, False otherwise
        """
        text = ClipboardService.format_data_as_text(data, separator)
        
        if not text:
            return False
        
        return ClipboardService.copy_to_clipboard(text)