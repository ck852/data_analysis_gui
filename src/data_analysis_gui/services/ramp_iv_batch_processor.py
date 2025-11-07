"""
Ramp IV Batch Processor for PatchBatch Electrophysiology Data Analysis Tool

This module provides batch processing functionality for ramp IV analysis,
enabling sequential analysis of multiple data files using consistent ramp
parameters. Results are transformed to match the standard batch analysis
structure for compatibility with existing export and current density workflows.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import time
import re
from pathlib import Path
from typing import List, Callable, Optional
import numpy as np

from data_analysis_gui.core.models import (
    FileAnalysisResult,
    BatchAnalysisResult,
    BatchExportResult,
)
from data_analysis_gui.config.logging import get_logger
from data_analysis_gui.services.data_manager import DataManager
from data_analysis_gui.services.ramp_iv_service import RampIVService, RampIVResult

logger = get_logger(__name__)


class RampIVBatchProcessor:
    """
    Processes multiple files with ramp IV analysis parameters.

    Transforms ramp IV results into standard batch analysis format for
    compatibility with existing export and current density infrastructure.
    """

    def __init__(self):
        """Initialize the RampIVBatchProcessor."""
        self.data_manager = DataManager()

        # Progress callbacks (optional)
        self.on_progress: Optional[Callable] = None
        self.on_file_complete: Optional[Callable] = None

        logger.info("RampIVBatchProcessor initialized")

    def process_files(
        self,
        file_paths: List[str],
        voltage_targets: List[float],
        start_ms: float,
        end_ms: float,
        current_units: str = "pA",
        sweep_selection_mode: str = "all",
        selected_sweeps: Optional[List[str]] = None,
    ) -> BatchAnalysisResult:
        """
        Process multiple files sequentially using ramp IV analysis.

        Args:
            file_paths: List of file paths to process.
            voltage_targets: Target voltages to extract current at.
            start_ms: Start of analysis range in milliseconds.
            end_ms: End of analysis range in milliseconds.
            current_units: Units for current measurements (pA, nA, etc.).
            sweep_selection_mode: "all" or "same" - how to select sweeps per file.
            selected_sweeps: List of sweep indices to use if mode is "same".

        Returns:
            BatchAnalysisResult with is_ramp_iv=True for CD compatibility.

        Raises:
            ValueError: If no files are provided or invalid sweep mode.
        """
        if not file_paths:
            raise ValueError("No files provided")

        if sweep_selection_mode not in ["all", "same"]:
            raise ValueError(f"Invalid sweep_selection_mode: {sweep_selection_mode}")

        if sweep_selection_mode == "same" and not selected_sweeps:
            raise ValueError("selected_sweeps required when mode is 'same'")

        logger.info(
            f"Processing {len(file_paths)} files with ramp IV analysis "
            f"(sweep mode: {sweep_selection_mode})"
        )
        start_time = time.time()

        successful_results = []
        failed_results = []

        # Sequential processing
        for i, path in enumerate(file_paths):
            # Update progress
            if self.on_progress:
                self.on_progress(i + 1, len(file_paths), Path(path).name)

            # Process the file
            result = self._process_single_file(
                path,
                voltage_targets,
                start_ms,
                end_ms,
                current_units,
                sweep_selection_mode,
                selected_sweeps,
            )

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
            f"Ramp IV batch complete: {len(successful_results)} succeeded, "
            f"{len(failed_results)} failed in {end_time - start_time:.2f}s"
        )

        return BatchAnalysisResult(
            successful_results=successful_results,
            failed_results=failed_results,
            parameters=None,  # Ramp IV doesn't use standard AnalysisParameters
            start_time=start_time,
            end_time=end_time,
            is_ramp_iv=True,  # Flag for CD compatibility
        )

    def _process_single_file(
        self,
        file_path: str,
        voltage_targets: List[float],
        start_ms: float,
        end_ms: float,
        current_units: str,
        sweep_selection_mode: str,
        selected_sweeps: Optional[List[str]],
    ) -> FileAnalysisResult:
        """
        Process a single file with ramp IV analysis.

        Args:
            file_path: Path to the file to process.
            voltage_targets: Target voltages to extract current at.
            start_ms: Start of analysis range.
            end_ms: End of analysis range.
            current_units: Current measurement units.
            sweep_selection_mode: "all" or "same".
            selected_sweeps: Sweep indices if mode is "same".

        Returns:
            FileAnalysisResult with averaged ramp IV data.
        """
        base_name = self._clean_filename(file_path)
        start_time = time.time()

        try:
            # Load dataset
            dataset = self.data_manager.load_dataset(file_path)

            # Determine which sweeps to use
            if sweep_selection_mode == "all":
                # Use all sweeps in the file
                sweeps_to_use = sorted(
                    dataset.sweeps(), key=lambda x: int(x) if x.isdigit() else 0
                )
            else:
                # Use same sweeps as preview (if they exist in this file)
                available_sweeps = set(dataset.sweeps())
                sweeps_to_use = [
                    s for s in selected_sweeps if s in available_sweeps
                ]

                if not sweeps_to_use:
                    logger.warning(
                        f"None of the selected sweeps found in {base_name}, "
                        f"falling back to all sweeps"
                    )
                    sweeps_to_use = sorted(
                        dataset.sweeps(), key=lambda x: int(x) if x.isdigit() else 0
                    )

            # Perform ramp IV analysis
            ramp_result = RampIVService.extract_ramp_iv_data(
                dataset=dataset,
                selected_sweeps=sweeps_to_use,
                target_voltages=voltage_targets,
                start_ms=start_ms,
                end_ms=end_ms,
                current_units=current_units,
            )

            if not ramp_result.success:
                return FileAnalysisResult(
                    file_path=file_path,
                    base_name=base_name,
                    success=False,
                    error_message=ramp_result.error_message,
                    processing_time=time.time() - start_time,
                )

            # Transform to standard FileAnalysisResult
            file_result = self._transform_ramp_to_file_result(
                ramp_result, file_path, base_name, time.time() - start_time
            )

            logger.debug(
                f"Processed {base_name}: {len(ramp_result.processed_sweeps)} sweeps"
            )
            return file_result

        except Exception as e:
            logger.error(f"Failed to process {base_name}: {e}")
            return FileAnalysisResult(
                file_path=file_path,
                base_name=base_name,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )

    def _transform_ramp_to_file_result(
        self,
        ramp_result: RampIVResult,
        file_path: str,
        base_name: str,
        processing_time: float,
    ) -> FileAnalysisResult:
        """
        Transform RampIVResult to FileAnalysisResult for batch compatibility.

        Averages current values across all processed sweeps for each voltage
        to create x_data (voltages) and y_data (averaged currents).

        Args:
            ramp_result: Result from RampIVService.
            file_path: Full path to the file.
            base_name: Cleaned filename.
            processing_time: Time taken to process.

        Returns:
            FileAnalysisResult compatible with batch infrastructure.
        """
        # Get sorted voltages
        voltages = sorted(ramp_result.voltage_targets)

        # Average currents across sweeps for each voltage
        averaged_currents = []

        for voltage in voltages:
            # Collect all valid current values for this voltage
            currents_at_voltage = []

            for sweep_idx in ramp_result.processed_sweeps:
                if sweep_idx in ramp_result.extracted_data:
                    sweep_data = ramp_result.extracted_data[sweep_idx]
                    if voltage in sweep_data:
                        current_value = sweep_data[voltage]
                        if not np.isnan(current_value):
                            currents_at_voltage.append(current_value)

            # Calculate average or NaN if no valid values
            if currents_at_voltage:
                avg_current = np.mean(currents_at_voltage)
            else:
                avg_current = np.nan

            averaged_currents.append(avg_current)

        # Create export table in standard format
        export_table = {
            "headers": ["Voltage (mV)", f"Current ({ramp_result.current_units})"],
            "data": np.column_stack([voltages, averaged_currents]),
            "format_spec": "%.6f",
        }

        # Log summary
        valid_count = sum(1 for c in averaged_currents if not np.isnan(c))
        logger.debug(
            f"{base_name}: {valid_count}/{len(voltages)} voltage points have valid data"
        )

        return FileAnalysisResult(
            file_path=file_path,
            base_name=base_name,
            success=True,
            x_data=np.array(voltages),
            y_data=np.array(averaged_currents),
            x_data2=None,  # Ramp IV doesn't use dual range
            y_data2=None,
            export_table=export_table,
            processing_time=processing_time,
        )

    def export_results(
        self, batch_result: BatchAnalysisResult, output_dir: str
    ) -> BatchExportResult:
        """
        Export all successful ramp IV results to individual CSV files.

        Args:
            batch_result: Batch analysis results to export.
            output_dir: Directory to save exported files.

        Returns:
            BatchExportResult with export status and summary.
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
            f"Exported {len(export_results)} ramp IV files, {total_records} total records"
        )

        return BatchExportResult(
            export_results=export_results,
            output_directory=output_dir,
            total_records=total_records,
        )

    @staticmethod
    def _clean_filename(file_path: str) -> str:
        """
        Clean a filename for display by removing extension and bracketed content.

        Args:
            file_path: Full file path.

        Returns:
            Cleaned filename without extension or brackets.
        """
        stem = Path(file_path).stem
        # Remove bracketed content
        cleaned = re.sub(r"\[.*?\]", "", stem).strip()
        return cleaned