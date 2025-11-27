"""
PatchBatch Electrophysiology Data Analysis Tool

For handling all file dialog interactions.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import os
from typing import Optional, List, Dict
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QWidget

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class FileDialogService:
    """
    Manages file dialogs with persistent directory memory.

    Tracks last-used directories separately for 'import_data' (MainWindow file opening),
    'batch_import' (batch file selection), and 'export' (all exports). The batch_import
    type automatically falls back to import_data location for convenience.
    """

    def __init__(self):
        self._last_directories: Dict[str, str] = {}
        logger.debug("FileDialogService initialized")

    def set_last_directories(self, directories: Dict[str, str]) -> None:
        """Load directory memory from session settings, validating paths exist."""
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
        logger.debug(f"Retrieved {len(self._last_directories)} stored directories")
        return self._last_directories.copy()

    def _get_fallback_for_dialog_type(self, dialog_type: str) -> Optional[str]:
        """Return fallback directory based on dialog type relationships."""
        if dialog_type == "batch_import":
            if "import_data" in self._last_directories:
                import_dir = self._last_directories["import_data"]
                if os.path.isdir(import_dir):
                    logger.debug(f"batch_import falling back to import_data: {import_dir}")
                    return import_dir
        return None

    def _get_default_directory(
        self, dialog_type: str, fallback: Optional[str] = None
    ) -> Optional[str]:
        """Resolve default directory using stored memory, fallback, or Qt defaults."""
        # Stored directory for this type
        if dialog_type in self._last_directories:
            stored_dir = self._last_directories[dialog_type]
            if os.path.isdir(stored_dir):
                logger.debug(f"Using stored directory for {dialog_type}: {stored_dir}")
                return stored_dir
            else:
                logger.warning(f"Stored directory no longer exists for {dialog_type}: {stored_dir}")

        # Explicit fallback
        if fallback and os.path.isdir(fallback):
            logger.debug(f"Using explicit fallback for {dialog_type}: {fallback}")
            return fallback

        # Type-specific fallback
        type_fallback = self._get_fallback_for_dialog_type(dialog_type)
        if type_fallback:
            return type_fallback

        logger.debug(f"No valid directory found for {dialog_type}, using Qt default")
        return None

    def _remember_directory(self, dialog_type: str, file_path: str) -> None:
        """Store parent directory of file_path for this dialog type."""
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
        dialog_type: str = "export",
    ) -> Optional[str]:
        """Show save dialog with suggested filename. Updates directory memory."""
        logger.debug(f"Opening export dialog: type={dialog_type}, suggested={suggested_name}")

        start_dir = self._get_default_directory(dialog_type, default_directory)

        if start_dir:
            suggested_path = os.path.join(start_dir, suggested_name)
        else:
            suggested_path = suggested_name
            logger.debug("No start directory available, using suggested name only")

        file_path, _ = QFileDialog.getSaveFileName(
            parent, "Export File", suggested_path, file_types
        )

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
        """Show single-file selection dialog."""
        logger.debug(f"Opening import dialog: type={dialog_type}, title='{title}'")

        start_dir = self._get_default_directory(dialog_type, default_directory)

        file_path, _ = QFileDialog.getOpenFileName(
            parent, title, start_dir or "", file_types
        )

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
        dialog_type: str = "batch_import",
    ) -> List[str]:
        """Show multi-file selection dialog. Returns empty list if cancelled."""
        logger.debug(f"Opening multi-file import dialog: type={dialog_type}, title='{title}'")

        start_dir = self._get_default_directory(dialog_type, default_directory)

        file_paths, _ = QFileDialog.getOpenFileNames(
            parent, title, start_dir or "", file_types
        )

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
        dialog_type: str = "export",
    ) -> Optional[str]:
        """Show directory selection dialog."""
        logger.debug(f"Opening directory selection dialog: type={dialog_type}, title='{title}'")

        start_dir = self._get_default_directory(dialog_type, default_directory)

        directory = QFileDialog.getExistingDirectory(
            parent,
            title,
            start_dir or "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )

        if directory:
            self._last_directories[dialog_type] = directory
            logger.info(f"Directory selected: {Path(directory).name}")
            return directory

        logger.debug(f"Directory selection cancelled for {dialog_type}")
        return None