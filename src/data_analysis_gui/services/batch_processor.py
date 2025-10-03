"""
PatchBatch Electrophysiology Data Analysis Tool

This module provides the batch processing functionality for PatchBatch,
an electrophysiology data analysis tool. It enables sequential analysis
of multiple data files using consistent parameters, manages progress
reporting, and supports exporting results to CSV files. Designed for
clarity, maintainability, and accessibility in electrophysiology workflows.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import time
import re
from pathlib import Path
from typing import List, Callable, Optional

from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.models import (
    FileAnalysisResult,
    BatchAnalysisResult,
    BatchExportResult,
)
from data_analysis_gui.config.logging import get_logger
from data_analysis_gui.core.exceptions import ValidationError

# Direct imports of managers
from data_analysis_gui.services.data_manager import DataManager
from data_analysis_gui.services.analysis_manager import AnalysisManager

logger = get_logger(__name__)


class BatchProcessor:
    """
    Processes multiple files with the same analysis parameters.

    Implements simple, sequential batch processing for electrophysiology data.
    Designed for clarity and accessibility.
    """

    def __init__(self):
        """
        Initialize the BatchProcessor.
        """
        self.data_manager = DataManager()  # Direct instantiation

        # Progress callbacks (optional)
        self.on_progress: Optional[Callable] = None
        self.on_file_complete: Optional[Callable] = None

        logger.info("BatchProcessor initialized")

    def process_files(
        self, file_paths: List[str], params: AnalysisParameters
    ) -> BatchAnalysisResult:
        """
        Process multiple files sequentially using the provided analysis parameters.

        Args:
            file_paths (List[str]): List of file paths to process.
            params (AnalysisParameters): Analysis parameters to apply.

        Returns:
            BatchAnalysisResult: Object containing results for all processed files.

        Raises:
            ValueError: If no files are provided.
        """
        if not file_paths:
            raise ValueError("No files provided")

        # Validate file formats before processing
        self._validate_file_formats(file_paths)

        logger.info(f"Processing {len(file_paths)} files")
        start_time = time.time()

        successful_results = []
        failed_results = []

        # Simple sequential processing
        for i, path in enumerate(file_paths):
            # Update progress
            if self.on_progress:
                self.on_progress(i + 1, len(file_paths), Path(path).name)

            # Process the file
            result = self._process_single_file(path, params)

            # Store result
            if result.success:
                successful_results.append(result)
            else:
                failed_results.append(result)

            # Notify completion
            if self.on_file_complete:
                self.on_file_complete(result)

        end_time = time.time()

        logger.info(
            f"Batch complete: {len(successful_results)} succeeded, "
            f"{len(failed_results)} failed in {end_time - start_time:.2f}s"
        )

        return BatchAnalysisResult(
            successful_results=successful_results,
            failed_results=failed_results,
            parameters=params,
            start_time=start_time,
            end_time=end_time,
        )

    def _process_single_file(
        self, file_path: str, params: AnalysisParameters
    ) -> FileAnalysisResult:
        """
        Process a single file and perform analysis.
        
        Channel configuration is automatically detected from each file's metadata.

        Args:
            file_path (str): Path to the file to process.
            params (AnalysisParameters): Analysis parameters to apply.

        Returns:
            FileAnalysisResult: Result object containing analysis outcome and data.
        """
        base_name = self._clean_filename(file_path)
        start_time = time.time()

        try:
            # Load dataset - channel config auto-detected from file
            dataset = self.data_manager.load_dataset(file_path)

            # Create analysis manager (no channel defs needed)
            analysis_manager = AnalysisManager()

            # Perform analysis
            analysis_result = analysis_manager.analyze(dataset, params)

            # Get export table
            export_table = analysis_manager.get_export_table(dataset, params)

            processing_time = time.time() - start_time

            return FileAnalysisResult(
                file_path=file_path,
                base_name=base_name,
                success=True,
                x_data=analysis_result.x_data,
                y_data=analysis_result.y_data,
                x_data2=analysis_result.x_data2 if params.use_dual_range else None,
                y_data2=analysis_result.y_data2 if params.use_dual_range else None,
                export_table=export_table,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Failed to process {base_name}: {e}")
            return FileAnalysisResult(
                file_path=file_path,
                base_name=base_name,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )

    def export_results(
        self, batch_result: BatchAnalysisResult, output_dir: str
    ) -> BatchExportResult:
        """
        Export all successful analysis results to individual CSV files.

        Args:
            batch_result (BatchAnalysisResult): Batch analysis results to export.
            output_dir (str): Directory to save exported files.

        Returns:
            BatchExportResult: Object containing export status and summary.
        """
        export_results = []
        total_records = 0

        for file_result in batch_result.successful_results:
            if file_result.export_table:
                output_path = Path(output_dir) / f"{file_result.base_name}.csv"

                # Export using DataManager
                export_result = self.data_manager.export_to_csv(
                    file_result.export_table, str(output_path)
                )

                export_results.append(export_result)
                if export_result.success:
                    total_records += export_result.records_exported

        logger.info(
            f"Exported {len(export_results)} files, {total_records} total records"
        )

        return BatchExportResult(
            export_results=export_results,
            output_directory=output_dir,
            total_records=total_records,
        )

    def _validate_file_formats(self, file_paths: List[str]) -> None:
        """
        Validate that all files in batch have the same format.
        
        Args:
            file_paths: List of file paths to validate
            
        Raises:
            ValidationError: If mixed file formats are detected
        """
        if not file_paths:
            return
        
        # Get extensions
        extensions = set(Path(fp).suffix.lower() for fp in file_paths)
        
        if len(extensions) > 1:
            raise ValidationError(
                f"Mixed file formats detected in batch: {extensions}. "
                "All files in a batch must have the same format."
            )
        
        logger.debug(f"Batch format validated: {extensions.pop()}")

    @staticmethod
    def _clean_filename(file_path: str) -> str:
        """
        Clean a filename for display or export by removing extension and bracketed content.

        Args:
            file_path (str): Full file path.

        Returns:
            str: Cleaned filename without extension or brackets.
        """
        stem = Path(file_path).stem
        # Remove bracketed content
        cleaned = re.sub(r"\[.*?\]", "", stem).strip()
        return cleaned
