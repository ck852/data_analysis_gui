"""
PatchBatch Batch Processor

Coordinates the batch analysis process. This is to be used for straight batch analysis with 
simple parameters (ie avg current vs avg voltage), but is expandable for more complex scenarios 
such as background subtraction. Processes one file at a time sequentially to assure accuracy.
Sweep rejection is disallowed for simplicity (all sweeps in all files will be analyzed regardless 
of any sweeps rejected from MainWindow).

Each file is loaded, analyzed, and results collected in complete isolation before the next file is loaded. This
follows the same backend processes as single file analysis (DataManager and DataExtractor load raw data,
params are passed through AnalysisManager to AnalysisEngine to MetricsCalculator and PlotFormatter, then AnalysisManager 
formats results for export). 

Optionally, background subtraction can be applied to each file before analysis via
BackgroundSubtractionService (must be initiated from "Batch Analysis with BG Subtraction" dialog in MainWindow).

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

# Direct imports of managers
from data_analysis_gui.services.data_manager import DataManager
from data_analysis_gui.services.analysis_manager import AnalysisManager

logger = get_logger(__name__)


class BatchProcessor:
    """
    Processes multiple files with the same analysis parameters as defined by user in ControlPanel.
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
        self, file_paths: List[str], params: AnalysisParameters, bg_subtraction_range=None
    ) -> BatchAnalysisResult:

        if not file_paths:
            raise ValueError("No files provided")

        bg_info = f" with BG subtraction [{bg_subtraction_range[0]:.1f}-{bg_subtraction_range[1]:.1f} ms]" if bg_subtraction_range else ""
        logger.info(f"Processing {len(file_paths)} files{bg_info}")
        start_time = time.time()

        successful_results = []
        failed_results = []

        # Simple sequential processing
        for i, path in enumerate(file_paths):
            # Update progress
            if self.on_progress:
                self.on_progress(i + 1, len(file_paths), Path(path).name)

            # Process the file with optional BG subtraction
            result = self._process_single_file(path, params, bg_subtraction_range)

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
        self, file_path: str, params: AnalysisParameters, bg_subtraction_range=None
    ) -> FileAnalysisResult:

        base_name = self._clean_filename(file_path)
        start_time = time.time()

        try:
            # Load dataset - channel config auto-detected from file
            dataset = self.data_manager.load_dataset(file_path)

            # Apply background subtraction if requested
            if bg_subtraction_range:
                from data_analysis_gui.services.bg_subtraction_service import BackgroundSubtractionService
                
                start_ms, end_ms = bg_subtraction_range
                
                bg_result = BackgroundSubtractionService.apply_background_subtraction(
                    dataset, start_ms, end_ms
                )
                
                if not bg_result.success:
                    logger.warning(
                        f"Background subtraction failed for {base_name}: {bg_result.error_message}"
                    )
                    # Return failed result
                    return FileAnalysisResult(
                        file_path=file_path,
                        base_name=base_name,
                        success=False,
                        error_message=f"BG subtraction failed: {bg_result.error_message}",
                        processing_time=time.time() - start_time,
                    )
                
                logger.debug(
                    f"Applied BG subtraction to {base_name}: "
                    f"{bg_result.processed_sweeps}/{bg_result.total_sweeps} sweeps"
                )

            # Create analysis manager (no channel defs needed)
            analysis_manager = AnalysisManager()

            # Extract current units from this file's metadata (like single-file analysis does)
            channel_config = dataset.metadata.get("channel_config", {})
            current_units = channel_config.get("current_units", "pA")

            # Update params with file-specific current units
            file_params = params.with_updates(
                channel_config={
                    **params.channel_config,
                    "current_units": current_units,
                }
            )

            # Perform analysis with file-specific params
            analysis_result = analysis_manager.analyze(dataset, file_params)

            # Get export table (also use file-specific params)
            export_table = analysis_manager.get_export_table(dataset, file_params)

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

    @staticmethod
    def _clean_filename(file_path: str) -> str:
        """
        Clean a filename for display or export by removing extension and bracketed content.
        Initially removed bracketed content for abf exports that had bracketed metadata in names. 
        """
        stem = Path(file_path).stem
        # Remove bracketed content
        cleaned = re.sub(r"\[.*?\]", "", stem).strip()
        return cleaned
