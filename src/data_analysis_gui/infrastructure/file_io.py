"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Infrastructure layer for file I/O operations.

Provides concrete implementations for dataset loading, CSV writing, file system
operations, and path utilities, supporting the data analysis workflow.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from data_analysis_gui.core.dataset import DatasetLoader, ElectrophysiologyDataset
from data_analysis_gui.core.exceptions import FileError, DataError
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class FileDatasetLoader:
    """
    Concrete implementation for loading electrophysiology datasets from files.

    Wraps DatasetLoader to separate infrastructure from business logic.
    """

    def load(self, filepath: str) -> ElectrophysiologyDataset:
        """
        Load a dataset from a file using DatasetLoader.
        
        Channel configuration is automatically detected from file metadata.

        Args:
            filepath (str): Path to the data file.

        Returns:
            ElectrophysiologyDataset: Loaded dataset with auto-detected channel config.

        Raises:
            DataError: If DatasetLoader returns None.
            FileError: If loading fails.
        """
        logger.debug(f"Loading dataset from {filepath}")

        try:
            dataset = DatasetLoader.load(filepath)

            if dataset is None:
                raise DataError(f"DatasetLoader returned None for {filepath}")

            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            if isinstance(e, (FileError, DataError)):
                raise
            raise FileError(
                f"Failed to load dataset from {filepath}",
                details={"filepath": filepath},
                cause=e,
            )


class CsvFileWriter:
    """
    Concrete implementation for writing CSV files.

    Provides methods to write data arrays to CSV and ensure output directories exist.
    """

    def write_csv(
        self,
        filepath: str,
        data: np.ndarray,
        headers: List[str],
        format_spec: str = "%.6f",
    ) -> None:
        """
        Write a NumPy array to a CSV file.

        Args:
            filepath (str): Output file path.
            data (np.ndarray): Data to write.
            headers (List[str]): List of column headers.
            format_spec (str, optional): Format specification for data values.

        Raises:
            FileError: If writing fails.
        """
        logger.debug(f"Writing CSV to {filepath}")

        try:
            header_str = ",".join(headers) if headers else ""
            np.savetxt(
                filepath,
                data,
                delimiter=",",
                fmt=format_spec,
                header=header_str,
                comments="",
            )

        except (IOError, OSError) as e:
            raise FileError(
                "Failed to write CSV file", details={"filepath": filepath}, cause=e
            )

    def ensure_directory(self, directory: str) -> None:
        """
        Ensure that a directory exists, creating it if necessary.

        Args:
            directory (str): Directory path.

        Raises:
            FileError: If directory creation fails.
        """
        if directory and not os.path.exists(directory):
            logger.debug(f"Creating directory: {directory}")
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                raise FileError(
                    "Could not create directory",
                    details={"directory": directory},
                    cause=e,
                )


class FileSystemOperations:
    """
    Concrete implementation of file system operations.

    Provides methods for checking existence, readability, writability, and
    retrieving file metadata.
    """

    def exists(self, path: str) -> bool:
        """
        Check if a path exists.

        Args:
            path (str): Path to check.

        Returns:
            bool: True if path exists, False otherwise.
        """
        return os.path.exists(path)

    def is_readable(self, path: str) -> bool:
        """
        Check if a file is readable.

        Args:
            path (str): File path.

        Returns:
            bool: True if file is readable, False otherwise.
        """
        return os.path.isfile(path) and os.access(path, os.R_OK)

    def is_writable(self, path: str) -> bool:
        """
        Check if a file or directory is writable.

        Args:
            path (str): Path to check.

        Returns:
            bool: True if writable, False otherwise.
        """
        if os.path.exists(path):
            return os.access(path, os.W_OK)
        # Check parent directory for new files
        parent = os.path.dirname(path)
        return not parent or os.access(parent, os.W_OK)

    def get_size(self, path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            path (str): File path.

        Returns:
            int: File size in bytes.

        Raises:
            FileError: If file size cannot be determined.
        """
        try:
            return os.path.getsize(path)
        except OSError as e:
            raise FileError("Could not get file size", details={"path": path}, cause=e)

    def get_info(self, path: str) -> Dict[str, Any]:
        """
        Get metadata for a file.

        Args:
            path (str): File path.

        Returns:
            Dict[str, Any]: Dictionary with file metadata.

        Raises:
            FileError: If file info cannot be retrieved.
        """
        try:
            stat = os.stat(path)
            return {
                "path": path,
                "name": os.path.basename(path),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "extension": Path(path).suffix.lower(),
            }
        except OSError as e:
            raise FileError("Could not get file info", details={"path": path}, cause=e)


class PathUtilities:
    """
    Concrete implementation of path manipulation utilities.

    Provides methods for sanitizing filenames, ensuring unique paths,
    extracting file numbers, and creating export paths.
    """

    def __init__(self, file_system: Optional[FileSystemOperations] = None):
        """
        Initialize PathUtilities with an optional file system dependency.

        Args:
            file_system (Optional[FileSystemOperations]): File system operations instance.
        """
        self.file_system = file_system or FileSystemOperations()

    def sanitize_filename(self, filename: str) -> str:
        """
        Remove invalid characters from a filename.

        Args:
            filename (str): Filename to sanitize.

        Returns:
            str: Sanitized filename.
        """
        if not filename.strip():
            return "unnamed_file"

        # Handle parentheses with special content
        def replacer(match):
            content = match.group(1)
            if "+" in content or "-" in content:
                return "_" + content
            return ""

        name_after_parens = re.sub(r"\s*\((.*?)\)", replacer, filename).strip()
        safe_name = re.sub(r"[^\w+-]", "_", name_after_parens).replace("__", "_")

        return safe_name if safe_name else "sanitized_file"

    def ensure_unique_path(self, path: str) -> str:
        """
        Ensure a file path is unique by appending a number if needed.

        Args:
            path (str): Desired file path.

        Returns:
            str: Unique file path.

        Raises:
            FileError: If a unique filename cannot be found.
        """
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
            details={"original_path": path},
        )

    def extract_file_number(self, filepath: str) -> int:
        """
        Extract a numeric identifier from a filename for sorting.

        Args:
            filepath (str): File path.

        Returns:
            int: Extracted number, or 0 if not found.
        """
        filename = os.path.basename(filepath)
        try:
            number_part = filename.split("_")[-1].split(".")[0]
            return int(number_part)
        except (IndexError, ValueError):
            return 0

    def create_export_path(
        self, base_path: str, suffix: str = "", extension: str = ".csv"
    ) -> str:
        """
        Create an export file path based on an input file.

        Args:
            base_path (str): Base file path.
            suffix (str, optional): Suffix to append to filename.
            extension (str, optional): File extension.

        Returns:
            str: Constructed export file path.
        """
        base_name = Path(base_path).stem
        directory = Path(base_path).parent
        export_name = f"{base_name}{suffix}{extension}"
        return str(directory / export_name)
