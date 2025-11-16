"""
Takes the AnalysisManager output (analysis parameters formatted for NumPy) and raw data arrays from input file via DataExtractor
and orchestrates actual analysis calculations via MetricsCalculator. PlotFormatter dictates 
what data is desired for plotting (Average Voltage, Peak Current, etc.) and formats the 
output accordingly.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import Dict, List, Optional, Any, Set

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.metrics_calculator import MetricsCalculator, SweepMetrics
from data_analysis_gui.core.data_extractor import DataExtractor
from data_analysis_gui.core.plot_formatter import PlotFormatter
from data_analysis_gui.core.exceptions import (
    ValidationError,
    DataError,
    ProcessingError,
)
from data_analysis_gui.config.logging import (
    get_logger,
    log_performance,
    log_analysis_request,
)

logger = get_logger(__name__)


class AnalysisEngine:

    def __init__(
        self,
        data_extractor: DataExtractor,
        metrics_calculator: MetricsCalculator,
        plot_formatter: PlotFormatter,
    ):
        """Wire up the analysis pipeline with its required components via dependency injection."""

        logger.info("Initializing AnalysisEngine")

        # Validate required dependencies
        if data_extractor is None:
            raise ValidationError("data_extractor cannot be None")
        if metrics_calculator is None:
            raise ValidationError("metrics_calculator cannot be None")
        if plot_formatter is None:
            raise ValidationError("plot_formatter cannot be None")

        # Store injected dependencies
        self.data_extractor = data_extractor
        self.metrics_calculator = metrics_calculator
        self.plot_formatter = plot_formatter

        logger.debug("AnalysisEngine initialized successfully")

    def analyze_dataset(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters,
        rejected_sweeps: Optional[Set[int]] = None,
    ) -> List[SweepMetrics]:
        """
        Orchestrates extraction of sweep data and computation of metrics for all valid sweeps,
        excluding any sweeps in the rejected_sweeps set.
        """
        # Validate inputs
        if dataset is None:
            raise ValidationError("Dataset cannot be None")
        if params is None:
            raise ValidationError("Parameters cannot be None")

        if dataset.is_empty():
            raise DataError("Dataset is empty, no sweeps to analyze")

        # Default to empty set if None
        if rejected_sweeps is None:
            rejected_sweeps = set()

        # Log the analysis request
        dataset_info = {
            "sweep_count": dataset.sweep_count(),
            "identifier": f"{dataset.source_file if hasattr(dataset, 'source_file') else 'unknown'}",
            "rejected_count": len(rejected_sweeps),
        }
        log_analysis_request(logger, params.to_export_dict(), dataset_info)

        # Log rejected sweeps if any
        if rejected_sweeps:
            logger.info(f"Excluding {len(rejected_sweeps)} rejected sweeps: {sorted(rejected_sweeps)}")

        # Perform analysis 
        with log_performance(logger, f"analyze {dataset.sweep_count()} sweeps"):
            metrics = self._compute_all_metrics(dataset, params, rejected_sweeps)

        return metrics

    def get_plot_data(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters,
        rejected_sweeps: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compute metrics and format them for plotting.
        
        Returns empty plot structure on failure rather than raising exceptions,
        allowing the GUI to display a blank plot instead of crashing.
        """

        try:
            # Default to empty set if None
            if rejected_sweeps is None:
                rejected_sweeps = set()

            # Get metrics through main analysis method
            metrics = self.analyze_dataset(dataset, params, rejected_sweeps)

            # Format for plotting
            return self.plot_formatter.format_for_plot(metrics, params)

        except (DataError, ProcessingError) as e:
            logger.error(f"Failed to generate plot data: {e}")
            # Return empty structure rather than propagating exception
            return self.plot_formatter.empty_plot_data()

    def get_export_table(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters,
        rejected_sweeps: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        """Format analysis results as a table structure suitable for CSV/Excel export."""
        # Default to empty set if None
        if rejected_sweeps is None:
            rejected_sweeps = set()

        # Get plot data first (with rejected sweeps filtering)
        plot_data = self.get_plot_data(dataset, params, rejected_sweeps)

        # Format for export
        return self.plot_formatter.format_for_export(plot_data, params)

    def get_sweep_plot_data(
        self, dataset: ElectrophysiologyDataset, sweep_index: str, channel_type: str
    ) -> Dict[str, Any]:
        """Extract raw time-series data for a single sweep's channel, formatted for plotting."""
        # Extract channel data
        time_ms, data_matrix, channel_id = self.data_extractor.extract_channel_for_plot(
            dataset, sweep_index, channel_type
        )

        # Return formatted for plot manager
        return {
            "time_ms": time_ms,
            "data_matrix": data_matrix,
            "channel_id": channel_id,
            "sweep_index": sweep_index,
            "channel_type": channel_type,
        }

    def get_peak_analysis_data(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters,
        peak_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate peak analysis data for multiple peak detection methods.
        
        Default peak_types cover the standard analysis modes: absolute, positive-only,
        negative-only, and peak-to-peak measurements.
        """

        if peak_types is None:
            peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]

        with log_performance(logger, f"peak analysis for {len(peak_types)} types"):
            # Get base metrics
            metrics = self.analyze_dataset(dataset, params)

            if not metrics:
                logger.warning("No metrics available for peak analysis")
                return {}

            # Format peak analysis data
            return self.plot_formatter.format_peak_analysis(metrics, params, peak_types)

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _compute_all_metrics(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters,
        rejected_sweeps: Optional[Set[int]] = None,
    ) -> List[SweepMetrics]:
        """
        Core analysis loop that processes each sweep sequentially.
        
        Sweep times must be present in dataset metadata (required for both ABF and WCP files).
        Individual sweep failures are logged but don't halt processing of remaining sweeps.
        """

        metrics = []
        failed_sweeps = []
        skipped_sweeps = []

        # Default to empty set if None
        if rejected_sweeps is None:
            rejected_sweeps = set()

        # Get sweep times from metadata (required for all files)
        sweep_times = dataset.metadata.get('sweep_times', {})
        file_format = dataset.metadata.get('format', 'unknown')
        
        if not sweep_times:
            raise DataError(
                f"No sweep time metadata found in {file_format.upper()} file. "
                "File may be corrupted or incompletely loaded."
            )
        
        logger.info(f"Using sweep times from {file_format.upper()} file metadata")

        # Process sweeps in sorted order
        sweep_list = sorted(
            dataset.sweeps(), key=lambda x: int(x) if x.isdigit() else 0
        )

        for sweep_number, sweep_index in enumerate(sweep_list):
            # Convert sweep_index to int for rejection check
            try:
                sweep_idx_int = int(sweep_index)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert sweep index to int: {sweep_index}")
                sweep_idx_int = None
            
            # Skip rejected sweeps
            if sweep_idx_int is not None and sweep_idx_int in rejected_sweeps:
                logger.debug(f"Skipping rejected sweep {sweep_index}")
                skipped_sweeps.append(sweep_index)
                continue

            try:
                # Extract sweep data
                sweep_data = self.data_extractor.extract_sweep_data(
                    dataset, sweep_index
                )

                # Get actual sweep time from metadata (required)
                actual_time = sweep_times.get(sweep_index)
                
                if actual_time is None:
                    raise DataError(
                        f"Sweep {sweep_index}: Missing sweep time in metadata. "
                        f"File may be corrupted or incompletely loaded."
                    )

                # Compute metrics
                metric = self.metrics_calculator.compute_sweep_metrics(
                    time_ms=sweep_data["time_ms"],
                    voltage=sweep_data["voltage"],
                    current=sweep_data["current"],
                    sweep_index=sweep_index,
                    sweep_number=sweep_number,
                    range1_start=params.range1_start,
                    range1_end=params.range1_end,
                    actual_sweep_time=actual_time,
                    range2_start=params.range2_start if params.use_dual_range else None,
                    range2_end=params.range2_end if params.use_dual_range else None,
                )

                metrics.append(metric)

            except (DataError, ProcessingError) as e:
                logger.warning(f"Failed to process sweep {sweep_index}: {e}")
                failed_sweeps.append(sweep_index)

        # Log summary
        if skipped_sweeps:
            logger.info(f"Skipped {len(skipped_sweeps)} rejected sweeps: {skipped_sweeps}")
        
        if failed_sweeps:
            logger.warning(
                f"Failed to process {len(failed_sweeps)} of {len(sweep_list)} sweeps. "
                f"Failed sweeps: {failed_sweeps[:10]}"  # Show first 10
            )

        # Ensure we have at least some valid metrics
        if not metrics:
            raise ProcessingError(
                "No valid metrics computed for any sweep",
                details={
                    "total_sweeps": len(sweep_list),
                    "failed_sweeps": len(failed_sweeps),
                    "rejected_sweeps": len(skipped_sweeps),
                },
            )

        logger.info(f"Successfully computed metrics for {len(metrics)} sweeps")
        return metrics


# ===========================================================================
# Factory function for convenient creation with default components
# ===========================================================================


def create_analysis_engine() -> AnalysisEngine:
    """Create an AnalysisEngine with standard components wired up."""

    from data_analysis_gui.core.data_extractor import DataExtractor
    from data_analysis_gui.core.metrics_calculator import MetricsCalculator
    from data_analysis_gui.core.plot_formatter import PlotFormatter

    # Create components
    data_extractor = DataExtractor()
    metrics_calculator = MetricsCalculator()
    plot_formatter = PlotFormatter()

    # Create and return engine
    return AnalysisEngine(
        data_extractor=data_extractor,
        metrics_calculator=metrics_calculator,
        plot_formatter=plot_formatter,
    )
