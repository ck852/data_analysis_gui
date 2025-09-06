"""
GUI Service for file dialog operations.

This service encapsulates all file dialog interactions, keeping them separate
from business logic while providing a clean interface for the GUI layer.
Part of the presentation layer - handles user interaction for file selection.
"""

import os
from typing import Optional, List, Tuple
from PyQt5.QtWidgets import QFileDialog, QWidget


class FileDialogService:
    """
    Centralized service for all file dialog operations.
    
    This service maintains separation of concerns by handling all file dialog
    UI interactions in one place, making the main window cleaner and the
    file operations more testable and maintainable.
    """
    
    @staticmethod
    def get_export_path(
        parent: QWidget,
        suggested_name: str,
        default_directory: Optional[str] = None,
        file_types: str = "CSV files (*.csv);;All files (*.*)"
    ) -> Optional[str]:
        """
        Show a save file dialog and return the selected path.
        
        Args:
            parent: Parent widget for the dialog
            suggested_name: Suggested filename (without path)
            default_directory: Directory to open dialog in (defaults to last used)
            file_types: File type filter string
            
        Returns:
            Selected file path or None if cancelled
        """
        # Construct the suggested full path
        if default_directory and os.path.isdir(default_directory):
            suggested_path = os.path.join(default_directory, suggested_name)
        else:
            suggested_path = suggested_name
        
        # Show the dialog and return result
        file_path, _ = QFileDialog.getSaveFileName(
            parent,
            "Export Analysis Data",
            suggested_path,
            file_types
        )
        
        return file_path if file_path else None
    
    @staticmethod
    def get_import_path(
        parent: QWidget,
        title: str = "Open File",
        default_directory: Optional[str] = None,
        file_types: str = "All files (*.*)"
    ) -> Optional[str]:
        """
        Show an open file dialog and return the selected path.
        
        Args:
            parent: Parent widget for the dialog
            title: Dialog window title
            default_directory: Directory to open dialog in
            file_types: File type filter string
            
        Returns:
            Selected file path or None if cancelled
        """
        file_path, _ = QFileDialog.getOpenFileName(
            parent,
            title,
            default_directory or "",
            file_types
        )
        
        return file_path if file_path else None
    
    @staticmethod
    def get_import_paths(
        parent: QWidget,
        title: str = "Select Files",
        default_directory: Optional[str] = None,
        file_types: str = "All files (*.*)"
    ) -> List[str]:
        """
        Show a multi-file selection dialog and return selected paths.
        
        Args:
            parent: Parent widget for the dialog
            title: Dialog window title
            default_directory: Directory to open dialog in
            file_types: File type filter string
            
        Returns:
            List of selected file paths (empty if cancelled)
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            parent,
            title,
            default_directory or "",
            file_types
        )
        
        return file_paths if file_paths else []
    
    @staticmethod
    def get_directory(
        parent: QWidget,
        title: str = "Select Directory",
        default_directory: Optional[str] = None
    ) -> Optional[str]:
        """
        Show a directory selection dialog and return the selected path.
        
        Args:
            parent: Parent widget for the dialog
            title: Dialog window title
            default_directory: Directory to open dialog in
            
        Returns:
            Selected directory path or None if cancelled
        """
        directory = QFileDialog.getExistingDirectory(
            parent,
            title,
            default_directory or "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        return directory if directory else None