"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Clipboard service for copying analysis data to system clipboard.
"""

from typing import Dict, Any, List
import numpy as np
from PySide6.QtWidgets import QApplication

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ClipboardService:
    """
    Formats analysis data as tab-separated text and copies to system clipboard.
    
    Uses TSV instead of CSV for easier paste operations into spreadsheets.
    """
    
    @staticmethod
    def format_data_as_text(data: Dict[str, Any], separator: str = "\t") -> str:
        """
        Convert data dictionary to delimited text with headers.
        
        Args:
            data: Dictionary with 'headers' (list) and 'data' (array-like) keys
            separator: Column delimiter (default tab for TSV)
            
        Returns:
            Formatted text string, or empty string on error
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
        """Copy text to system clipboard using Qt."""
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
        """Format data and copy to clipboard. Returns True if successful."""
        text = ClipboardService.format_data_as_text(data, separator)
        
        if not text:
            return False
        
        return ClipboardService.copy_to_clipboard(text)