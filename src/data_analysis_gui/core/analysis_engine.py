"""
Core analysis engine - pure orchestration of analysis workflow.

PHASE 5 REFACTOR: Simplified to pure orchestration with dependency injection.
- All dependencies are injected, not created internally
- Cache key generation is externalized
- Engine is now easily testable with mock components
- Follows Single Responsibility and Dependency Inversion principles

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, List, Optional, Any
import numpy as np

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.metrics_calculator import MetricsCalculator, SweepMetrics
from data_analysis_gui.core.data_extractor import DataExtractor
from data_analysis_gui.core.cache_manager import CacheManager
from data_analysis_gui.core.plot_formatter import PlotFormatter
from data_analysis_gui.core.dataset_identifier import DatasetIdentifier
from data_analysis_gui.core.exceptions import ValidationError, DataError, ProcessingError
from data_analysis_gui.config.logging import get_logger, log_performance, log_analysis_request

logger = get_logger(__name__)


class AnalysisEngine:
    """
    Pure orchestrator for analysis workflow.
    
    This class coordinates between specialized components to perform analysis.
    All dependencies are injected, making it highly testable and flexible.
    
    Responsibilities:
    - Orchestrate analysis workflow
    - Coordinate between components
    - Manage caching strategy
    
    Does NOT:
    - Create its own dependencies
    - Generate cache keys
    - Format data
    - Compute metrics
    - Extract data
    
    Example:
        >>> # Production usage with real components
        >>> engine = AnalysisEngine(
        ...     data_extractor=DataExtractor(channel_defs),
        ...     metrics_calculator=MetricsCalculator(),
        ...     plot_formatter=PlotFormatter(),
        ...     dataset_identifier=DatasetIdentifier(),
        ...     metrics_cache=CacheManager(100, "metrics"),
        ...     series_cache=CacheManager(200, "series")
        ... )
        
        >>> # Test usage with mocks
        >>> engine = AnalysisEngine(
        ...     data_extractor=mock_extractor,
        ...     metrics_calculator=mock_calculator,
        ...     plot_formatter=mock_formatter,
        ...     dataset_identifier=mock_identifier,
        ...     metrics_cache=mock_cache,
        ...     series_cache=mock_cache
        ... )
    """
    
    def __init__(
        self,
        data_extractor: DataExtractor,
        metrics_calculator: MetricsCalculator,
        plot_formatter: PlotFormatter,
        dataset_identifier: DatasetIdentifier,
        metrics_cache: Optional[CacheManager] = None,
        series_cache: Optional[CacheManager] = None
    ):
        """
        Initialize engine with injected dependencies.
        
        Args:
            data_extractor: Component for extracting data from datasets
            metrics_calculator: Component for computing metrics
            plot_formatter: Component for formatting data for plots/exports
            dataset_identifier: Component for generating cache keys
            metrics_cache: Optional cache for computed metrics
            series_cache: Optional cache for time series data
        
        Raises:
            ValidationError: If required dependencies are None
        """
        logger.info("Initializing AnalysisEngine with injected dependencies")
        
        # Validate required dependencies
        if data_extractor is None:
            raise ValidationError("data_extractor cannot be None")
        if metrics_calculator is None:
            raise ValidationError("metrics_calculator cannot be None")
        if plot_formatter is None:
            raise ValidationError("plot_formatter cannot be None")
        if dataset_identifier is None:
            raise ValidationError("dataset_identifier cannot be None")
        
        # Store injected dependencies
        self.data_extractor = data_extractor
        self.metrics_calculator = metrics_calculator
        self.plot_formatter = plot_formatter
        self.dataset_identifier = dataset_identifier
        
        # Caches are optional (can be None for testing or when caching is disabled)
        self.metrics_cache = metrics_cache
        self.series_cache = series_cache
        
        logger.debug("AnalysisEngine initialized successfully")
    
    def analyze_dataset(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters
    ) -> List[SweepMetrics]:
        """
        Perform complete analysis of a dataset.
        
        This is the main entry point for analysis. It orchestrates the entire
        workflow of extracting data, computing metrics, and managing caches.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters defining ranges, measures, etc.
        
        Returns:
            List of computed metrics for all valid sweeps
        
        Raises:
            ValidationError: If inputs are invalid
            DataError: If dataset is empty or corrupted
            ProcessingError: If no valid metrics could be computed
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
            'sweep_count': dataset.sweep_count(),
            'identifier': self.dataset_identifier.get_identifier(dataset)
        }
        log_analysis_request(logger, params.to_export_dict(), dataset_info)
        
        # Check cache if available
        if self.metrics_cache is not None:
            cache_key = self._generate_cache_key(dataset, params)
            cached_metrics = self.metrics_cache.get(cache_key)
            if cached_metrics is not None:
                logger.debug(f"Returning cached metrics for key {cache_key[:8]}...")
                return cached_metrics
        
        # Perform analysis
        with log_performance(logger, f"analyze {dataset.sweep_count()} sweeps"):
            metrics = self._compute_all_metrics(dataset, params)
        
        # Cache results if cache is available
        if self.metrics_cache is not None and metrics:
            self.metrics_cache.set(cache_key, metrics)
        
        return metrics
    
    def get_plot_data(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters
    ) -> Dict[str, Any]:
        """
        Get analysis results formatted for plotting.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
        
        Returns:
            Dictionary with plot-ready data
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
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters
    ) -> Dict[str, Any]:
        """
        Get analysis results formatted for export.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
        
        Returns:
            Dictionary with 'headers', 'data', and 'format_spec'
        """
        # Get plot data first
        plot_data = self.get_plot_data(dataset, params)
        
        # Format for export
        return self.plot_formatter.format_for_export(plot_data, params)
    
    def get_sweep_plot_data(
        self,
        dataset: ElectrophysiologyDataset,
        sweep_index: str,
        channel_type: str
    ) -> Dict[str, Any]:
        """
        Get single sweep data formatted for plotting.
        
        Args:
            dataset: The dataset containing the sweep
            sweep_index: Identifier of the sweep to plot
            channel_type: "Voltage" or "Current"
        
        Returns:
            Dictionary with sweep plot data
        
        Raises:
            ValidationError: If inputs are invalid
            DataError: If sweep not found or data extraction fails
        """
        # Extract channel data
        time_ms, data_matrix, channel_id = self.data_extractor.extract_channel_for_plot(
            dataset, sweep_index, channel_type
        )
        
        # Return formatted for plot manager
        return {
            'time_ms': time_ms,
            'data_matrix': data_matrix,
            'channel_id': channel_id,
            'sweep_index': sweep_index,
            'channel_type': channel_type
        }
    
    def get_peak_analysis_data(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters,
        peak_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive peak analysis across multiple peak types.
        
        Args:
            dataset: The dataset to analyze
            params: Analysis parameters
            peak_types: List of peak types to analyze (default: all types)
        
        Returns:
            Dictionary with peak analysis data for each type
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
        Clear all caches.
        
        This should be called when loading a new dataset or when
        analysis parameters change fundamentally (e.g., channel swap).
        """
        if self.metrics_cache is not None:
            self.metrics_cache.clear()
        
        if self.series_cache is not None:
            self.series_cache.clear()
        
        logger.info("All caches cleared")
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _compute_all_metrics(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters
    ) -> List[SweepMetrics]:
        """
        Compute metrics for all sweeps in the dataset.
        
        This method orchestrates the computation but delegates all actual
        work to the injected components.
        
        Args:
            dataset: Dataset to analyze
            params: Analysis parameters
        
        Returns:
            List of computed metrics
        
        Raises:
            ProcessingError: If no valid metrics could be computed
        """
        metrics = []
        failed_sweeps = []
        
        # Process sweeps in sorted order
        sweep_list = sorted(
            dataset.sweeps(),
            key=lambda x: int(x) if x.isdigit() else 0
        )
        
        for sweep_number, sweep_index in enumerate(sweep_list):
            try:
                # Extract sweep data
                sweep_data = self.data_extractor.extract_sweep_data(dataset, sweep_index)
                
                # Compute metrics
                metric = self.metrics_calculator.compute_sweep_metrics(
                    time_ms=sweep_data['time_ms'],
                    voltage=sweep_data['voltage'],
                    current=sweep_data['current'],
                    sweep_index=sweep_index,
                    sweep_number=sweep_number,
                    range1_start=params.range1_start,
                    range1_end=params.range1_end,
                    stimulus_period=params.stimulus_period,
                    range2_start=params.range2_start if params.use_dual_range else None,
                    range2_end=params.range2_end if params.use_dual_range else None
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
                    'total_sweeps': len(sweep_list),
                    'failed_sweeps': len(failed_sweeps)
                }
            )
        
        logger.info(f"Successfully computed metrics for {len(metrics)} sweeps")
        return metrics
    
    def _generate_cache_key(
        self,
        dataset: ElectrophysiologyDataset,
        params: AnalysisParameters
    ) -> str:
        """
        Generate cache key using the injected identifier.
        
        Args:
            dataset: Dataset to identify
            params: Parameters to include in key
        
        Returns:
            Cache key string
        """
        dataset_id = self.dataset_identifier.get_identifier(dataset)
        params_key = str(params.cache_key())
        combined = f"{dataset_id}:{params_key}"
        
        # Use the identifier's hashing method
        return self.dataset_identifier.create_hash(combined)


# ===========================================================================
# Factory function for convenient creation with default components
# ===========================================================================

def create_analysis_engine(
    channel_definitions,
    enable_caching: bool = True,
    cache_sizes: Optional[Dict[str, int]] = None
) -> AnalysisEngine:
    """
    Factory function to create an AnalysisEngine with default components.
    
    This provides a convenient way to create a fully configured engine
    while still allowing for dependency injection in tests.
    
    Args:
        channel_definitions: Channel configuration
        enable_caching: Whether to enable caching
        cache_sizes: Optional dict with 'metrics' and 'series' cache sizes
    
    Returns:
        Configured AnalysisEngine instance
    
    Example:
        >>> from data_analysis_gui.core.channel_definitions import ChannelDefinitions
        >>> channel_defs = ChannelDefinitions()
        >>> engine = create_analysis_engine(channel_defs)
    """
    from data_analysis_gui.core.data_extractor import DataExtractor
    from data_analysis_gui.core.metrics_calculator import MetricsCalculator
    from data_analysis_gui.core.plot_formatter import PlotFormatter
    from data_analysis_gui.core.dataset_identifier import DatasetIdentifier
    from data_analysis_gui.core.cache_manager import CacheManager
    
    # Create components
    data_extractor = DataExtractor(channel_definitions)
    metrics_calculator = MetricsCalculator()
    plot_formatter = PlotFormatter()
    dataset_identifier = DatasetIdentifier()
    
    # Create caches if enabled
    if enable_caching:
        sizes = cache_sizes or {'metrics': 100, 'series': 200}
        metrics_cache = CacheManager(sizes.get('metrics', 100), "metrics")
        series_cache = CacheManager(sizes.get('series', 200), "series")
    else:
        metrics_cache = None
        series_cache = None
    
    # Create and return engine
    return AnalysisEngine(
        data_extractor=data_extractor,
        metrics_calculator=metrics_calculator,
        plot_formatter=plot_formatter,
        dataset_identifier=dataset_identifier,
        metrics_cache=metrics_cache,
        series_cache=series_cache
    )