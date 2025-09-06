"""
Core batch processing service for electrophysiology data.

This module provides a GUI-independent service that orchestrates the batch
analysis of multiple electrophysiology datasets. It uses the AnalysisEngine
to compute metrics for each file based on a single set of parameters.

Author: Data Analysis GUI Contributors
License: MIT
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

import numpy as np

# Internal imports
from data_analysis_gui.core.dataset import DatasetLoader
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.utils import extract_file_number


@dataclass
class FileResult:
    """Holds the results and metadata for a single processed file."""
    file_path: str
    base_name: str
    success: bool
    x_data: np.ndarray = field(default_factory=lambda: np.array([]))
    x_data2: np.ndarray = field(default_factory=lambda: np.array([]))
    y_data: np.ndarray = field(default_factory=lambda: np.array([]))
    y_data2: np.ndarray = field(default_factory=lambda: np.array([]))
    export_table: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    peak_type: Optional[str] = None  # Add peak type information
    x_label: Optional[str] = None  # Add axis labels for clarity
    y_label: Optional[str] = None

@dataclass
class BatchResult:
    """Holds the aggregated results from a complete batch analysis run."""
    params: AnalysisParameters
    results: List[FileResult] = field(default_factory=list)

    @property
    def successful_results(self) -> List[FileResult]:
        """Returns a list of only the successfully processed file results."""
        return [res for res in self.results if res.success]

    @property
    def failed_results(self) -> List[FileResult]:
        """Returns a list of only the failed file results."""
        return [res for res in self.results if not res.success]


class BatchProcessor:
    """
    A stateless service to orchestrate batch analysis of datasets.
    """

    def __init__(self, channel_definitions: ChannelDefinitions):
        """
        Initializes the processor with the current channel configuration.

        Args:
            channel_definitions: The channel mapping configuration to be used
                                 for loading all datasets in the batch.
        """
        self.channel_definitions = channel_definitions

    def run(self,
            file_paths: List[str],
            params: AnalysisParameters,
            on_progress: Optional[Callable[[int, int], None]] = None,
            on_file_done: Optional[Callable[[FileResult], None]] = None
            ) -> BatchResult:
        """
        Synchronously processes a list of files with the given parameters.
        """
        all_results: List[FileResult] = []
        total_files = len(file_paths)
        sorted_paths = sorted(file_paths, key=extract_file_number)
        
        # Extract peak type from params if available
        x_peak = getattr(params.x_axis, 'peak_type', None) if params.x_axis.measure == "Peak" else None
        y_peak = getattr(params.y_axis, 'peak_type', None) if params.y_axis.measure == "Peak" else None
        peak_type = y_peak or x_peak  # Prioritize Y-axis peak type

        for i, file_path in enumerate(sorted_paths):
            if on_progress:
                on_progress(i, total_files)

            base_name = os.path.basename(file_path).split('.mat')[0]
            if '[' in base_name:
                base_name = base_name.split('[')[0]

            file_result = None
            try:
                dataset = DatasetLoader.load(file_path, self.channel_definitions)
                if dataset.is_empty():
                    raise ValueError("Dataset is empty or could not be loaded correctly.")
                
                engine = AnalysisEngine(dataset, self.channel_definitions)
                plot_data = engine.get_plot_data(params)
                export_table = engine.get_export_table(params)

                file_result = FileResult(
                    file_path=file_path,
                    base_name=base_name,
                    success=True,
                    x_data=plot_data['x_data'],
                    y_data=plot_data['y_data'],
                    x_data2=plot_data.get('x_data2', np.array([])),  # Add x_data2
                    y_data2=plot_data.get('y_data2', np.array([])),
                    export_table=export_table,
                    peak_type=peak_type,
                    x_label=plot_data.get('x_label'),
                    y_label=plot_data.get('y_label')
                )

            except Exception as e:
                file_result = FileResult(
                    file_path=file_path,
                    base_name=base_name,
                    success=False,
                    error_message=str(e),
                    peak_type=peak_type
                )
            
            finally:
                if file_result:
                    all_results.append(file_result)
                    if on_file_done:
                        on_file_done(file_result)

        if on_progress:
            on_progress(total_files, total_files)

        return BatchResult(params=params, results=all_results)
    
    def run_peak_analysis(self,
                        file_paths: List[str],
                        params: AnalysisParameters,
                        peak_types: List[str] = None,
                        on_progress: Optional[Callable[[int, int], None]] = None,
                        on_file_done: Optional[Callable[[FileResult], None]] = None
                        ) -> Dict[str, BatchResult]:
        """
        Process files with multiple peak analysis types.
        
        Args:
            file_paths: List of file paths to process
            params: Base analysis parameters
            peak_types: List of peak types to analyze (default: all types)
            on_progress: Progress callback
            on_file_done: File completion callback
        
        Returns:
            Dictionary mapping peak type to BatchResult
        """
        if peak_types is None:
            peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]
        
        results_by_peak = {}
        
        for peak_idx, peak_type in enumerate(peak_types):
            # Create modified parameters for this peak type
            modified_params = AnalysisParameters(
                range1_start=params.range1_start,
                range1_end=params.range1_end,
                use_dual_range=params.use_dual_range,
                range2_start=params.range2_start,
                range2_end=params.range2_end,
                stimulus_period=params.stimulus_period,
                x_axis=AxisConfig(
                    measure=params.x_axis.measure,
                    channel=params.x_axis.channel,
                    peak_type=peak_type if params.x_axis.measure == "Peak" else params.x_axis.peak_type
                ),
                y_axis=AxisConfig(
                    measure=params.y_axis.measure,
                    channel=params.y_axis.channel,
                    peak_type=peak_type if params.y_axis.measure == "Peak" else params.y_axis.peak_type
                ),
                channel_config=params.channel_config
            )
            
            all_results: List[FileResult] = []
            total_files = len(file_paths)
            sorted_paths = sorted(file_paths, key=extract_file_number)
            
            for i, file_path in enumerate(sorted_paths):
                # Adjust progress for multiple peak types
                overall_progress = peak_idx * total_files + i
                total_operations = len(peak_types) * total_files
                if on_progress:
                    on_progress(overall_progress, total_operations)
                
                base_name = os.path.basename(file_path).split('.mat')[0]
                if '[' in base_name:
                    base_name = base_name.split('[')[0]
                
                file_result = None
                try:
                    # Load and analyze
                    dataset = DatasetLoader.load(file_path, self.channel_definitions)
                    if dataset.is_empty():
                        raise ValueError("Dataset is empty or could not be loaded correctly.")
                    
                    engine = AnalysisEngine(dataset, self.channel_definitions)
                    plot_data = engine.get_plot_data(modified_params)
                    export_table = engine.get_export_table(modified_params)
                    
                    # Package result with peak type info
                    file_result = FileResult(
                        file_path=file_path,
                        base_name=base_name,
                        success=True,
                        x_data=plot_data['x_data'],
                        y_data=plot_data['y_data'],
                        y_data2=plot_data.get('y_data2', np.array([])),
                        export_table=export_table,
                        peak_type=peak_type,
                        x_label=plot_data.get('x_label'),
                        y_label=plot_data.get('y_label')
                    )
                    
                except Exception as e:
                    file_result = FileResult(
                        file_path=file_path,
                        base_name=base_name,
                        success=False,
                        error_message=str(e),
                        peak_type=peak_type
                    )
                
                finally:
                    if file_result:
                        all_results.append(file_result)
                        if on_file_done:
                            on_file_done(file_result)
            
            results_by_peak[peak_type] = BatchResult(params=modified_params, results=all_results)
        
        if on_progress:
            on_progress(len(peak_types) * total_files, len(peak_types) * total_files)
        
        return results_by_peak