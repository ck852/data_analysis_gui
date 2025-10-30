"""
PatchBatch Electrophysiology Data Analysis Tool

GUI Service for file dialog operations with directory memory.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This service encapsulates all file dialog interactions for the GUI.

Features:
- Remembers the last used directory for each dialog type independently.
- Provides methods for importing, exporting, batch selection, and directory selection.
- Ensures a consistent user experience across sessions.
"""

import os
from typing import Optional, List, Dict
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QWidget

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class FileDialogService:
    """
    Centralized service for all file dialog operations with directory memory.

    Each dialog type remembers its last used directory independently:
        - 'import_data': Opening data files
        - 'export_analysis': Exporting analysis results
        - 'export_batch': Batch exports
        - 'import_batch': Batch file selection
        - 'select_directory': Directory selection
    """

    def __init__(self):
        """
        Initialize the service with empty directory memory.
        """
        # Dictionary to track last used directories by dialog type
        self._last_directories: Dict[str, str] = {}
        logger.debug("FileDialogService initialized")

    def set_last_directories(self, directories: Dict[str, str]) -> None:
        """
        Set the last used directories, typically loaded from session settings.

        Args:
            directories (Dict[str, str]): Mapping of dialog types to directory paths.
        """
        # Only set directories that actually exist
        self._last_directories = {}
        valid_count = 0
        invalid_count = 0

        for dialog_type, directory in directories.items():
            if directory and os.path.isdir(directory):
                self._last_directories[dialog_type] = directory
                valid_count += 1
            else:
                invalid_count += 1
                logger.debug(f"Skipped invalid directory for {dialog_type}: {directory}")

        logger.info(f"Loaded directory memory: {valid_count} valid, {invalid_count} invalid")

    def get_last_directories(self) -> Dict[str, str]:
        """
        Get the current last used directories for saving to session settings.

        Returns:
            Dict[str, str]: Mapping of dialog types to directory paths.
        """
        logger.debug(f"Retrieved {len(self._last_directories)} stored directories")
        return self._last_directories.copy()

    def _get_default_directory(
        self, dialog_type: str, fallback: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the default directory for a dialog type.

        Args:
            dialog_type (str): Type of dialog (e.g., 'import_data', 'export_analysis').
            fallback (Optional[str]): Fallback directory if no stored directory exists.

        Returns:
            Optional[str]: Directory path to use as default, or None.
        """
        # First try the stored directory for this dialog type
        if dialog_type in self._last_directories:
            stored_dir = self._last_directories[dialog_type]
            if os.path.isdir(stored_dir):
                logger.debug(f"Using stored directory for {dialog_type}: {stored_dir}")
                return stored_dir
            else:
                logger.warning(f"Stored directory no longer exists for {dialog_type}: {stored_dir}")

        # Then try the fallback
        if fallback and os.path.isdir(fallback):
            logger.debug(f"Using fallback directory for {dialog_type}: {fallback}")
            return fallback

        # No valid directory found
        logger.debug(f"No valid directory found for {dialog_type}")
        return None

    def _remember_directory(self, dialog_type: str, file_path: str) -> None:
        """
        Remember the directory from a selected file path.

        Args:
            dialog_type (str): Type of dialog.
            file_path (str): Full path to the selected file.
        """
        if file_path:
            directory = str(Path(file_path).parent)
            if os.path.isdir(directory):
                self._last_directories[dialog_type] = directory
                logger.debug(f"Remembered directory for {dialog_type}: {directory}")
            else:
                logger.warning(f"Cannot remember invalid directory for {dialog_type}: {directory}")

    def get_export_path(
        self,
        parent: QWidget,
        suggested_name: str,
        default_directory: Optional[str] = None,
        file_types: str = "CSV files (*.csv);;All files (*.*)",
        dialog_type: str = "export_analysis",
    ) -> Optional[str]:
        """
        Show a save file dialog and return the selected path.

        Args:
            parent (QWidget): Parent widget for the dialog.
            suggested_name (str): Suggested filename (without path).
            default_directory (Optional[str]): Directory to open dialog in (overrides remembered directory).
            file_types (str): File type filter string.
            dialog_type (str): Type of dialog for directory memory.

        Returns:
            Optional[str]: Selected file path or None if cancelled.
        """
        logger.debug(f"Opening export dialog: type={dialog_type}, suggested={suggested_name}")

        # Determine the default directory
        if default_directory and os.path.isdir(default_directory):
            start_dir = default_directory
            logger.debug(f"Using override directory: {start_dir}")
        else:
            start_dir = self._get_default_directory(dialog_type)

        # Construct the suggested full path
        if start_dir:
            suggested_path = os.path.join(start_dir, suggested_name)
        else:
            suggested_path = suggested_name
            logger.debug("No start directory available, using suggested name only")

        # Show the dialog
        file_path, _ = QFileDialog.getSaveFileName(
            parent, "Export Analysis Data", suggested_path, file_types
        )

        # Remember the directory if a file was selected
        if file_path:
            self._remember_directory(dialog_type, file_path)
            logger.info(f"Export path selected: {Path(file_path).name}")
            return file_path

        logger.debug(f"Export dialog cancelled for {dialog_type}")
        return None

    def get_import_path(
        self,
        parent: QWidget,
        title: str = "Open File",
        default_directory: Optional[str] = None,
        file_types: str = "All files (*.*)",
        dialog_type: str = "import_data",
    ) -> Optional[str]:
        """
        Show an open file dialog and return the selected path.

        Args:
            parent (QWidget): Parent widget for the dialog.
            title (str): Dialog window title.
            default_directory (Optional[str]): Directory to open dialog in (overrides remembered directory).
            file_types (str): File type filter string.
            dialog_type (str): Type of dialog for directory memory.

        Returns:
            Optional[str]: Selected file path or None if cancelled.
        """
        logger.debug(f"Opening import dialog: type={dialog_type}, title='{title}'")

        # Determine the default directory
        start_dir = self._get_default_directory(dialog_type, default_directory)

        file_path, _ = QFileDialog.getOpenFileName(
            parent, title, start_dir or "", file_types
        )

        # Remember the directory if a file was selected
        if file_path:
            self._remember_directory(dialog_type, file_path)
            logger.info(f"Import file selected: {Path(file_path).name}")
            return file_path

        logger.debug(f"Import dialog cancelled for {dialog_type}")
        return None

    def get_import_paths(
        self,
        parent: QWidget,
        title: str = "Select Files",
        default_directory: Optional[str] = None,
        file_types: str = "All files (*.*)",
        dialog_type: str = "import_batch",
    ) -> List[str]:
        """
        Show a multi-file selection dialog and return selected paths.

        Args:
            parent (QWidget): Parent widget for the dialog.
            title (str): Dialog window title.
            default_directory (Optional[str]): Directory to open dialog in (overrides remembered directory).
            file_types (str): File type filter string.
            dialog_type (str): Type of dialog for directory memory.

        Returns:
            List[str]: List of selected file paths (empty if cancelled).
        """
        logger.debug(f"Opening multi-file import dialog: type={dialog_type}, title='{title}'")

        # Determine the default directory
        start_dir = self._get_default_directory(dialog_type, default_directory)

        file_paths, _ = QFileDialog.getOpenFileNames(
            parent, title, start_dir or "", file_types
        )

        # Remember the directory if files were selected
        if file_paths:
            self._remember_directory(dialog_type, file_paths[0])
            logger.info(f"Selected {len(file_paths)} files for import")
            return file_paths

        logger.debug(f"Multi-file import dialog cancelled for {dialog_type}")
        return []

    def get_directory(
        self,
        parent: QWidget,
        title: str = "Select Directory",
        default_directory: Optional[str] = None,
        dialog_type: str = "select_directory",
    ) -> Optional[str]:
        """
        Show a directory selection dialog and return the selected path.

        Args:
            parent (QWidget): Parent widget for the dialog.
            title (str): Dialog window title.
            default_directory (Optional[str]): Directory to open dialog in (overrides remembered directory).
            dialog_type (str): Type of dialog for directory memory.

        Returns:
            Optional[str]: Selected directory path or None if cancelled.
        """
        logger.debug(f"Opening directory selection dialog: type={dialog_type}, title='{title}'")

        # Determine the default directory
        start_dir = self._get_default_directory(dialog_type, default_directory)

        directory = QFileDialog.getExistingDirectory(
            parent,
            title,
            start_dir or "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )

        # Remember the directory if one was selected
        if directory:
            self._last_directories[dialog_type] = directory
            logger.info(f"Directory selected: {Path(directory).name}")
            return directory

        logger.debug(f"Directory selection cancelled for {dialog_type}")
        return None
