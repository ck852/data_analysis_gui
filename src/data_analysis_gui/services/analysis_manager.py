"""
Coordinates dataset analysis operations between the application controller and analysis engine.

Receives datasets and analysis parameters from ApplicationController, prepares them 
for processing by AnalysisEngine, and returns AnalysisEngine results back to the controller.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import Dict, Any, List, Optional, Set
import numpy as np

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.analysis_engine import create_analysis_engine
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.models import (
    AnalysisResult,
    PlotData,
    PeakAnalysisResult,
    ExportResult,
)
from data_analysis_gui.core.exceptions import ValidationError, DataError
from data_analysis_gui.config.logging import get_logger

# Direct import of DataManager
from data_analysis_gui.services.data_manager import DataManager

logger = get_logger(__name__)


class AnalysisManager:
    """Manages analysis workflows and coordinates between controllers and the analysis engine."""

    def __init__(self):
        self.engine = create_analysis_engine()
        self.data_manager = DataManager()  # Direct instantiation

        logger.info("AnalysisManager initialized")

    def analyze(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters,
        rejected_sweeps: Optional[Set[int]] = None,
    ) -> AnalysisResult:
        """
        Perform analysis on the dataset and return formatted results.

        Args:
            dataset: Electrophysiology data to analyze
            params: Analysis configuration parameters
            rejected_sweeps: Set of sweep indices to exclude from analysis

        Returns:
            AnalysisResult containing plot data and metadata

        Raises:
            DataError: If dataset is empty or analysis produces no results
        """

        if not dataset or dataset.is_empty():
            raise DataError("Cannot analyze empty dataset")

        if rejected_sweeps is None:
            rejected_sweeps = set()

        logger.debug(f"Analyzing {dataset.sweep_count()} sweeps (excluding {len(rejected_sweeps)} rejected)")

        # Get plot data from engine, passing rejected_sweeps for filtering
        plot_data = self.engine.get_plot_data(dataset, params, rejected_sweeps=rejected_sweeps)

        if not plot_data or "x_data" not in plot_data:
            raise DataError("Analysis produced no results")

        # Prepare all data before creating the frozen AnalysisResult
        x_data = np.array(plot_data["x_data"])
        y_data = np.array(plot_data["y_data"])
        x_label = plot_data.get("x_label", "")
        y_label = plot_data.get("y_label", "")
        sweep_indices = plot_data.get("sweep_indices", [])

        # Prepare dual range data if needed
        x_data2 = None
        y_data2 = None
        y_label_r1 = None
        y_label_r2 = None

        if params.use_dual_range:
            x_data2 = np.array(plot_data.get("x_data2", []))
            y_data2 = np.array(plot_data.get("y_data2", []))
            y_label_r1 = plot_data.get("y_label_r1")
            y_label_r2 = plot_data.get("y_label_r2")

        # Create result with all data at once
        result = AnalysisResult(
            x_data=x_data,
            y_data=y_data,
            x_label=x_label,
            y_label=y_label,
            x_data2=x_data2,
            y_data2=y_data2,
            y_label_r1=y_label_r1,
            y_label_r2=y_label_r2,
            sweep_indices=sweep_indices,
            use_dual_range=params.use_dual_range,
        )

        logger.info(f"Analysis complete: {len(result.x_data)} data points")
        return result

    def get_sweep_plot_data(
        self, dataset: ElectrophysiologyDataset, sweep_index: str, channel_type: str
    ) -> PlotData:
        """
        Retrieve plot data for a specific sweep and channel.

        Args:
            dataset: Source dataset
            sweep_index: Identifier for the sweep to retrieve
            channel_type: Either "Voltage" or "Current"

        Returns:
            PlotData with time series and channel information

        Raises:
            ValidationError: If channel_type is invalid
            DataError: If sweep data cannot be retrieved
        """

        if channel_type not in ["Voltage", "Current"]:
            raise ValidationError(f"Invalid channel type: {channel_type}")

        # Get data from engine
        data = self.engine.get_sweep_plot_data(dataset, sweep_index, channel_type)

        if not data:
            raise DataError(f"No data for sweep {sweep_index}")

        return PlotData(
            time_ms=np.array(data["time_ms"]),
            data_matrix=np.array(data["data_matrix"]),
            channel_id=data["channel_id"],
            sweep_index=data["sweep_index"],
            channel_type=data["channel_type"],
        )

    def export_analysis(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters,
        filepath: str,
        rejected_sweeps: Optional[Set[int]] = None,
    ) -> ExportResult:
        """
        Export analysis results to a CSV file.

        Args:
            dataset: Dataset to export
            params: Analysis parameters for processing
            filepath: Destination file path for CSV export
            rejected_sweeps: Set of sweep indices to exclude

        Returns:
            ExportResult indicating success or failure with error details
        """

        if dataset.is_empty():
            return ExportResult(success=False, error_message="Dataset is empty")

        try:
            if rejected_sweeps is None:
                rejected_sweeps = set()

            # Get export table from engine, passing rejected_sweeps for filtering
            table_data = self.engine.get_export_table(dataset, params, rejected_sweeps=rejected_sweeps)

            if not table_data or not table_data.get("data", []).size:
                return ExportResult(success=False, error_message="No data to export")

            # Export using DataManager
            return self.data_manager.export_to_csv(table_data, filepath)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(success=False, error_message=str(e))

    def get_peak_analysis(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters,
        peak_types: List[str] = None,
    ) -> PeakAnalysisResult:
        """
        Analyze peak characteristics across sweeps.

        Args:
            dataset: Dataset to analyze
            params: Analysis parameters
            peak_types: Types of peaks to analyze (defaults to all types)

        Returns:
            PeakAnalysisResult with peak data and x-axis information

        Raises:
            DataError: If dataset is empty or peak analysis fails
        """

        if dataset.is_empty():
            raise DataError("Cannot analyze empty dataset")

        if peak_types is None:
            peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]

        # Get peak data from engine
        peak_data = self.engine.get_peak_analysis_data(dataset, params, peak_types)

        if not peak_data:
            raise DataError("Peak analysis failed")

        return PeakAnalysisResult(
            peak_data=peak_data.get("peak_data", {}),
            x_data=np.array(peak_data["x_data"]),
            x_label=peak_data.get("x_label", ""),
            sweep_indices=peak_data.get("sweep_indices", []),
        )

    def get_export_table(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters,
        rejected_sweeps: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        """
        Generate formatted table data for export.

        Args:
            dataset: Dataset to process
            params: Analysis parameters
            rejected_sweeps: Set of sweep indices to exclude

        Returns:
            Dictionary with headers, data array, and format specifications
        """

        if dataset.is_empty():
            return {"headers": [], "data": np.array([[]]), "format_spec": "%.6f"}

        if rejected_sweeps is None:
            rejected_sweeps = set()

        # Pass rejected_sweeps to engine for filtering
        return self.engine.get_export_table(dataset, params, rejected_sweeps=rejected_sweeps)
