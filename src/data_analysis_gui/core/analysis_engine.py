"""
PatchBatch Electrophysiology Data Analysis Tool - Analysis Engine Module

This module provides the core orchestration engine for electrophysiology data analysis workflows.
It coordinates injected components for data extraction, metrics calculation, and plot/export formatting,
enabling flexible, testable, and thread-safe analysis operations.

Features:
- Stateless orchestration of analysis, metrics computation, and result formatting.
- Dependency injection for all major components (DataExtractor, MetricsCalculator, PlotFormatter).
- Robust error handling and logging for all analysis steps.
- No internal caching; each analysis is independent and suitable for concurrent execution.
- Factory function for convenient engine instantiation with default components.

Usage:
Instantiate AnalysisEngine with required dependencies, or use create_analysis_engine for default configuration.
Call analysis methods to process datasets and retrieve metrics, plot data, or export tables.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import Dict, List, Optional, Any

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
    """
    Orchestrator for electrophysiology data analysis workflow.

    Coordinates between injected components to perform analysis, compute metrics, and format results for plotting and export.
    All dependencies are injected, making the engine highly testable and flexible. No caching is performed; each analysis is independent and thread-safe.

    Responsibilities:
        - Orchestrate analysis workflow
        - Coordinate between components

    Limitations:
        - Does NOT create its own dependencies
        - Does NOT cache results
        - Does NOT format data, compute metrics, or extract data directly

    Args:
        data_extractor (DataExtractor): Component for extracting data from datasets.
        metrics_calculator (MetricsCalculator): Component for computing metrics.
        plot_formatter (PlotFormatter): Component for formatting data for plots/exports.

    Example:
        >>> engine = AnalysisEngine(
        ...     data_extractor=DataExtractor(channel_defs),
        ...     metrics_calculator=MetricsCalculator(),
        ...     plot_formatter=PlotFormatter()
        ... )
    """

    def __init__(
        self,
        data_extractor: DataExtractor,
        metrics_calculator: MetricsCalculator,
        plot_formatter: PlotFormatter,
    ):
        """
        Initialize the AnalysisEngine with injected dependencies.

        Args:
            data_extractor (DataExtractor): Component for extracting data from datasets.
            metrics_calculator (MetricsCalculator): Component for computing metrics.
            plot_formatter (PlotFormatter): Component for formatting data for plots/exports.

        Raises:
            ValidationError: If any required dependency is None.
        """
        logger.info("Initializing AnalysisEngine with injected dependencies")

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
        self, dataset: ElectrophysiologyDataset, params: AnalysisParameters
    ) -> List[SweepMetrics]:
        """
        Perform complete analysis of an electrophysiology dataset.

        Orchestrates extraction of sweep data and computation of metrics for all valid sweeps.

        Args:
            dataset (ElectrophysiologyDataset): The dataset to analyze.
            params (AnalysisParameters): Analysis parameters defining ranges, measures, etc.

        Returns:
            List[SweepMetrics]: List of computed metrics for all valid sweeps.

        Raises:
            ValidationError: If inputs are invalid.
            DataError: If dataset is empty or corrupted.
            ProcessingError: If no valid metrics could be computed.
        """
        # Validate inputs
        if dataset is None:
            raise ValidationError("Dataset cannot be None")
        if params is None:
            raise ValidationError("Parameters cannot be None")

        if dataset.is_empty():
            raise DataError("Dataset is empty, no sweeps to analyze")

        # Log the analysis request
        dataset_info = {
            "sweep_count": dataset.sweep_count(),
            "identifier": f"{dataset.source_file if hasattr(dataset, 'source_file') else 'unknown'}",
        }
        log_analysis_request(logger, params.to_export_dict(), dataset_info)

        # Perform analysis directly (no caching)
        with log_performance(logger, f"analyze {dataset.sweep_count()} sweeps"):
            metrics = self._compute_all_metrics(dataset, params)

        return metrics

    def get_plot_data(
        self, dataset: ElectrophysiologyDataset, params: AnalysisParameters
    ) -> Dict[str, Any]:
        """
        Get analysis results formatted for plotting.

        Args:
            dataset (ElectrophysiologyDataset): The dataset to analyze.
            params (AnalysisParameters): Analysis parameters.

        Returns:
            Dict[str, Any]: Dictionary containing plot-ready data.
        """
        try:
            # Get metrics through main analysis method
            metrics = self.analyze_dataset(dataset, params)

            # Format for plotting
            return self.plot_formatter.format_for_plot(metrics, params)

        except (DataError, ProcessingError) as e:
            logger.error(f"Failed to generate plot data: {e}")
            # Return empty structure rather than propagating exception
            return self.plot_formatter.empty_plot_data()

    def get_export_table(
        self, dataset: ElectrophysiologyDataset, params: AnalysisParameters
    ) -> Dict[str, Any]:
        """
        Get analysis results formatted for export.

        Args:
            dataset (ElectrophysiologyDataset): The dataset to analyze.
            params (AnalysisParameters): Analysis parameters.

        Returns:
            Dict[str, Any]: Dictionary with 'headers', 'data', and 'format_spec' for export.
        """
        # Get plot data first
        plot_data = self.get_plot_data(dataset, params)

        # Format for export
        return self.plot_formatter.format_for_export(plot_data, params)

    def get_sweep_plot_data(
        self, dataset: ElectrophysiologyDataset, sweep_index: str, channel_type: str
    ) -> Dict[str, Any]:
        """
        Get single sweep data formatted for plotting.

        Args:
            dataset (ElectrophysiologyDataset): The dataset containing the sweep.
            sweep_index (str): Identifier of the sweep to plot.
            channel_type (str): Channel type to plot ("Voltage" or "Current").

        Returns:
            Dict[str, Any]: Dictionary with sweep plot data.

        Raises:
            ValidationError: If inputs are invalid.
            DataError: If sweep not found or data extraction fails.
        """
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
        Get comprehensive peak analysis across multiple peak types.

        Args:
            dataset (ElectrophysiologyDataset): The dataset to analyze.
            params (AnalysisParameters): Analysis parameters.
            peak_types (Optional[List[str]]): List of peak types to analyze. Defaults to all types.

        Returns:
            Dict[str, Any]: Dictionary with peak analysis data for each type.
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

    def clear_caches(self) -> None:
        """
        No-op method maintained for compatibility.

        Since caching has been removed, this method does nothing.
        Kept to avoid breaking existing code that calls it.
        """
        logger.debug("clear_caches called (no-op - caching removed)")

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _compute_all_metrics(
        self, dataset: ElectrophysiologyDataset, params: AnalysisParameters
    ) -> List[SweepMetrics]:
        """
        Compute metrics for all sweeps in the dataset.

        Orchestrates extraction and computation for each sweep, delegating work to injected components.

        Args:
            dataset (ElectrophysiologyDataset): Dataset to analyze.
            params (AnalysisParameters): Analysis parameters.

        Returns:
            List[SweepMetrics]: List of computed metrics for all valid sweeps.

        Raises:
            ProcessingError: If no valid metrics could be computed.
            DataError: If sweep time metadata is missing from file.
        """
        metrics = []
        failed_sweeps = []

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
                },
            )

        logger.info(f"Successfully computed metrics for {len(metrics)} sweeps")
        return metrics


# ===========================================================================
# Factory function for convenient creation with default components
# ===========================================================================


def create_analysis_engine() -> AnalysisEngine:
    """
    Factory function to create an AnalysisEngine with default components.

    Provides a convenient way to create a fully configured engine for production use, while still allowing for dependency injection in tests.

    Returns:
        AnalysisEngine: Configured AnalysisEngine instance.

    Example:
        >>> channel_defs = ChannelDefinitions()
        >>> engine = create_analysis_engine(channel_defs)
    """
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
