"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Use this for CSV exports.

Coordinates the loading of data files via DatasetLoader, validation of datasets,
and exporting of analysis results to CSV files.
Handles file paths, file validation, and export formatting. Used heavily by batch analysis for 
managing file names, multiple file exports.

Make adjustments here to add support for new export types (ie Excel, TSV).
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from data_analysis_gui.core.dataset import DatasetLoader, ElectrophysiologyDataset
from data_analysis_gui.core.models import ExportResult
from data_analysis_gui.config.exceptions import ValidationError, FileError, DataError
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class DataManager:
    """
    Extend this class to add support for new export types; add new file
    formats in core/dataset.py's DatasetLoader.
    """

    def __init__(self):

        logger.info("DataManager initialized")

    # =========================================================================
    # Dataset Loading
    # =========================================================================

    def load_dataset(self, filepath: str) -> ElectrophysiologyDataset:
        """
        Load and validate an electrophysiology dataset from a file.
        
        Channel configuration is automatically detected from file metadata.

        To add support for a new file format:
            1. Update DatasetLoader.FORMAT_EXTENSIONS/detect_format() in core/dataset.py.
            2. Add a corresponding loader in DatasetLoader (e.g., load_abf, load_wcp).
        """
        # Check file exists
        if not os.path.exists(filepath):
            raise FileError(f"File not found: {filepath}")

        if not os.access(filepath, os.R_OK):
            raise FileError(f"File not readable: {filepath}")

        # Check file not empty
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            raise DataError(f"File is empty: {filepath}")

        logger.info(f"Loading dataset from {Path(filepath).name}")

        # Load using DatasetLoader - channel config auto-detected
        try:
            dataset = DatasetLoader.load(filepath)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise FileError(f"Failed to load {filepath}: {str(e)}")

        # Validate
        if dataset is None or dataset.is_empty():
            raise DataError(f"No valid data found in {filepath}")

        logger.info(f"Successfully loaded {dataset.sweep_count()} sweeps")
        return dataset

    # =========================================================================
    # Data Export
    # =========================================================================

    def export_to_csv(self, data: Dict[str, Any], filepath: str) -> ExportResult:

        try:
            # Validate data
            if not data or "headers" not in data or "data" not in data:
                raise ValidationError("Invalid data structure")

            headers = data["headers"]
            data_array = np.array(data["data"])

            if data_array.size == 0:
                raise DataError("No data to export")

            # Ensure directory exists
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            # Write CSV
            format_spec = data.get("format_spec", "%.6f")
            header_str = ",".join(headers)
            np.savetxt(
                filepath,
                data_array,
                delimiter=",",
                fmt=format_spec,
                header=header_str,
                comments="",
                encoding="utf-8",
            )

            # Verify file was created
            if not os.path.exists(filepath):
                raise FileError("File was not created")

            records = len(data_array)
            logger.info(f"Exported {records} records to {Path(filepath).name}")

            return ExportResult(
                success=True, file_path=filepath, records_exported=records
            )

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(success=False, error_message=str(e))

    def export_multiple_files(
        self,
        data_list: List[Dict[str, Any]],
        output_dir: str,
        base_name: str = "export",
    ) -> List[ExportResult]:

        results = []

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for i, data in enumerate(data_list):
            # Generate unique filename
            suffix = data.get("suffix", f"_{i+1}")
            filename = f"{base_name}{suffix}.csv"
            filepath = os.path.join(output_dir, filename)

            # Make unique if file exists
            filepath = self.make_unique_path(filepath)

            # Export
            result = self.export_to_csv(data, filepath)
            results.append(result)

        successful = sum(1 for r in results if r.success)
        logger.info(f"Exported {successful}/{len(data_list)} files")

        return results

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def suggest_filename(
        self, source_path: str, suffix: str = "_", params: Optional[Any] = None
    ) -> str:
        """
        Generate a suggested filename for exporting analysis results.
        """
        if not source_path:
            return f"analysis{suffix}.csv"

        # Get base name and clean it
        base_name = Path(source_path).stem
        base_name = re.sub(r"\[.*?\]", "", base_name)  # Remove brackets
        base_name = base_name.strip(" _")

        # Add parameter-specific suffix if available
        if params and hasattr(params, "y_axis"):
            if params.y_axis.measure == "Peak" and params.y_axis.peak_type:
                peak_suffixes = {
                    "Absolute": "_absolute",
                    "Positive": "_positive",
                    "Negative": "_negative",
                    "Peak-Peak": "_peak-peak",
                }
                suffix = peak_suffixes.get(params.y_axis.peak_type, suffix)

        return f"{base_name}{suffix}.csv"

    def make_unique_path(self, filepath: str) -> str:
        """
        Ensure a filepath is unique by appending a numeric suffix if needed.
        """
        if not os.path.exists(filepath):
            return filepath

        path = Path(filepath)
        directory = path.parent
        stem = path.stem
        suffix = path.suffix

        counter = 1
        while counter <= 10000:
            new_path = directory / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return str(new_path)
            counter += 1

        raise FileError(f"Could not create unique filename for {filepath}")

    def validate_export_path(self, filepath: str) -> bool:

        if not filepath or not filepath.strip():
            raise ValidationError("Export path cannot be empty")

        path = Path(filepath)

        # Must have an extension
        if not path.suffix:
            raise ValidationError("Export file must have an extension")

        # Check for invalid characters
        invalid_chars = '<>:"|?*' if os.name == "nt" else "\0"
        invalid_found = [c for c in path.name if c in invalid_chars]
        if invalid_found:
            raise ValidationError(
                f"Filename contains invalid characters: {invalid_found}"
            )

        # Check if directory is writable
        directory = path.parent
        if directory.exists() and not os.access(directory, os.W_OK):
            raise ValidationError(f"No write permission for directory: {directory}")

        return True
