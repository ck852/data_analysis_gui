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
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.utils import extract_file_number


@dataclass
class FileResult:
    """Holds the results and metadata for a single processed file."""
    file_path: str
    base_name: str
    success: bool
    x_data: np.ndarray = field(default_factory=lambda: np.array([]))
    y_data: np.ndarray = field(default_factory=lambda: np.array([]))
    y_data2: np.ndarray = field(default_factory=lambda: np.array([]))
    export_table: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


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

        Args:
            file_paths: A list of absolute paths to the .mat files.
            params: The AnalysisParameters to apply to all files.
            on_progress: An optional callback for progress updates,
                         called with (current_index, total_files).
            on_file_done: An optional callback executed after each file is
                          processed, receiving the FileResult object.

        Returns:
            A BatchResult object containing the outcomes for all files.
        """
        all_results: List[FileResult] = []
        total_files = len(file_paths)
        sorted_paths = sorted(file_paths, key=extract_file_number)

        for i, file_path in enumerate(sorted_paths):
            if on_progress:
                on_progress(i, total_files)

            base_name = os.path.basename(file_path).split('.mat')[0]
            if '[' in base_name:
                base_name = base_name.split('[')[0]

            file_result = None
            try:
                # 1. Load data for the current file
                dataset = DatasetLoader.load(file_path, self.channel_definitions)
                if dataset.is_empty():
                    raise ValueError("Dataset is empty or could not be loaded correctly.")
                
                # 2. Run analysis
                engine = AnalysisEngine(dataset, self.channel_definitions)
                plot_data = engine.get_plot_data(params)
                export_table = engine.get_export_table(params)

                # 3. Package successful result
                file_result = FileResult(
                    file_path=file_path,
                    base_name=base_name,
                    success=True,
                    x_data=plot_data['x_data'],
                    y_data=plot_data['y_data'],
                    y_data2=plot_data.get('y_data2', np.array([])),
                    export_table=export_table
                )

            except Exception as e:
                # 4. Package failed result
                file_result = FileResult(
                    file_path=file_path,
                    base_name=base_name,
                    success=False,
                    error_message=str(e)
                )
            
            finally:
                if file_result:
                    all_results.append(file_result)
                    if on_file_done:
                        on_file_done(file_result)

        if on_progress:
            on_progress(total_files, total_files)  # Signal completion

        return BatchResult(params=params, results=all_results)