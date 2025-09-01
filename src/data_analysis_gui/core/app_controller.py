"""
Application Controller - Mediates between GUI and business logic.
This is the ONLY class the GUI should interact with for business operations.
"""

import os
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass

# Business logic imports (no GUI dependencies)
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.dataset import ElectrophysiologyDataset, DatasetLoader
from data_analysis_gui.core.analysis_engine import AnalysisEngine
from data_analysis_gui.core.params import AnalysisParameters, AxisConfig
from data_analysis_gui.core.batch_processor import BatchProcessor, FileResult
from data_analysis_gui.core import exporter
from data_analysis_gui.core.iv_analysis import IVAnalysisService


@dataclass
class FileInfo:
    """Information about loaded file"""
    path: str
    name: str
    sweep_count: int
    sweep_names: List[str]


@dataclass
class PlotData:
    """Data for plotting a single sweep"""
    time_ms: Any  # numpy array
    data_matrix: Any  # numpy array
    channel_id: int
    sweep_index: int
    channel_type: str


@dataclass
class AnalysisPlotData:
    """Data for analysis plots"""
    x_data: List[float]
    y_data: List[float]
    y_data2: Optional[List[float]]
    x_label: str
    y_label: str
    sweep_indices: List[int]
    use_dual_range: bool


@dataclass
class BatchAnalysisResult:
    """Result from batch analysis operation"""
    success: bool
    batch_result: Any  # BatchResult object
    batch_data: Dict[str, Dict[str, Any]]
    iv_data: Any
    iv_file_mapping: Any
    successful_count: int
    failed_count: int
    export_outcomes: List[Any]
    x_label: str
    y_label: str


class ApplicationController:
    """
    Central controller that manages all business logic.
    GUI interacts only with this controller, never directly with business logic.
    """
    
    def __init__(self, get_save_path_callback=None):
        # Core business objects
        self.channel_definitions = ChannelDefinitions()
        self.analysis_engine = AnalysisEngine(channel_definitions=self.channel_definitions)
        self.current_dataset: Optional[ElectrophysiologyDataset] = None
        self.loaded_file_path: Optional[str] = None
        
        # Callbacks for GUI updates (dependency injection)
        self.get_save_path_callback = get_save_path_callback
        self.on_file_loaded = None
        self.on_error = None
        self.on_status_update = None
    
    def trigger_export_dialog(self, params):
        """
        Triggers the UI to show a save dialog via the callback,
        then proceeds with the export if a path is returned.
        """
        if not self.get_save_path_callback:
            print("Error: No callback available to show a save dialog.")
            return False

        # CORRECTED: Call this method without arguments.
        suggested_path = self.get_suggested_export_filename()

        # The rest of the function remains the same.
        file_path = self.get_save_path_callback(suggested_path)

        if file_path:
            return self.export_analysis_data_to_file(params, file_path)

        return False

    def get_channel_configuration(self) -> Dict[str, Any]:
        """Get channel configuration without exposing the internal object"""
        return self.channel_definitions.get_configuration()

    def create_parameters_from_dict(self, gui_state: Dict[str, Any]) -> AnalysisParameters:
        """Create parameters from a simple dictionary of GUI values"""
        return self.build_parameters(
            range1_start=gui_state['range1_start'],
            range1_end=gui_state['range1_end'],
            use_dual_range=gui_state['use_dual_range'],
            range2_start=gui_state.get('range2_start'),
            range2_end=gui_state.get('range2_end'),
            stimulus_period=gui_state['stimulus_period'],
            x_measure=gui_state['x_measure'],
            x_channel=gui_state.get('x_channel'),
            x_peak_type=gui_state.get('x_peak_type', 'Absolute'),  # Added
            y_measure=gui_state['y_measure'],
            y_channel=gui_state.get('y_channel'),
            y_peak_type=gui_state.get('y_peak_type', 'Absolute'),  # Added
            channel_config=self.get_channel_configuration()
        )
    
    def perform_batch_analysis(
        self,
        file_paths: List[str],
        params: AnalysisParameters,
        destination_folder: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchAnalysisResult:
        """
        Perform batch analysis and return data only (no plotting).
        
        Args:
            file_paths: List of file paths to analyze
            params: Analysis parameters
            destination_folder: Where to save results
            progress_callback: Optional progress callback
            
        Returns:
            BatchAnalysisResult containing all data needed for visualization
        """
        # Process files
        processor = BatchProcessor(self.channel_definitions)
        batch_result = processor.run(
            file_paths,
            params,
            on_progress=progress_callback
        )
        
        # Export results
        export_outcomes = []
        if batch_result.successful_results:
            export_outcomes = exporter.write_tables(batch_result, destination_folder)
        
        # Prepare batch data for dialog
        batch_data = {
            res.base_name: {
                'x_values': res.x_data.tolist() if hasattr(res.x_data, 'tolist') else res.x_data,
                'y_values': res.y_data.tolist() if hasattr(res.y_data, 'tolist') else res.y_data,
                'y_values2': res.y_data2.tolist() if res.y_data2 is not None and hasattr(res.y_data2, 'tolist') else []
            } for res in batch_result.successful_results
        }
        
        # Prepare IV data
        iv_data, iv_file_mapping = IVAnalysisService.prepare_iv_data(batch_data, params)
        
        # Create axis labels
        x_label, y_label = self.create_axis_labels(params)
        
        return BatchAnalysisResult(
            success=len(batch_result.successful_results) > 0,
            batch_result=batch_result,
            batch_data=batch_data,
            iv_data=iv_data,
            iv_file_mapping=iv_file_mapping,
            successful_count=len(batch_result.successful_results),
            failed_count=len(batch_result.failed_results),
            export_outcomes=export_outcomes,
            x_label=x_label,
            y_label=y_label
        )
    
    # ============ File Operations ============
    
    def load_file(self, file_path: str) -> Optional[FileInfo]:
        """
        Load a MAT file and return file information.
        
        Args:
            file_path: Path to the MAT file
            
        Returns:
            FileInfo object if successful, None otherwise
        """
        try:
            self.current_dataset = DatasetLoader.load(file_path, self.channel_definitions)
            self.loaded_file_path = file_path
            self.analysis_engine.set_dataset(self.current_dataset)
            
            # Prepare file info
            sweep_names = [f"Sweep {idx}" for idx in sorted(
                self.current_dataset.sweeps(), key=lambda x: int(x)
            )]
            
            file_info = FileInfo(
                path=file_path,
                name=os.path.basename(file_path),
                sweep_count=self.current_dataset.sweep_count(),
                sweep_names=sweep_names
            )
            
            # Notify GUI if callback is set
            if self.on_file_loaded:
                self.on_file_loaded(file_info)
            
            return file_info
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Error loading file: {str(e)}")
            return None
        
    def export_analysis_data_to_file(self, params: AnalysisParameters, file_path: str) -> bool:
        """
        Export analysis data to a specific CSV file.
        """
        if not self.has_data() or not self.loaded_file_path:
            return False
        
        try:
            self.analysis_engine.set_dataset(self.current_dataset)
            table_data = self.analysis_engine.get_export_table(params)
            
            if not table_data or len(table_data.get('data', [])) == 0:
                if self.on_error:
                    self.on_error("No data to export")
                return False
            
            # Extract peak type from parameters
            peak_type = None
            if params.y_axis.measure == "Peak":
                peak_type = getattr(params.y_axis, 'peak_type', None)
            elif params.x_axis.measure == "Peak":
                peak_type = getattr(params.x_axis, 'peak_type', None)
            
            # Extract the directory and filename from the full path
            destination_folder = os.path.dirname(file_path)
            filename_with_ext = os.path.basename(file_path)
            
            # Remove .csv extension if present (write_single_table adds it)
            base_name = filename_with_ext.replace('.csv', '')
            
            outcome = exporter.write_single_table(
                table=table_data,
                base_name=base_name,
                destination_folder=destination_folder,
                peak_type=peak_type  # Pass peak type
            )
            
            if outcome.success and self.on_status_update:
                self.on_status_update(f"Data exported to: {outcome.path}")
            elif not outcome.success and self.on_error:
                self.on_error(f"Export failed: {outcome.error_message}")
            
            return outcome.success
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Export error: {str(e)}")
            return False
        
    def get_suggested_export_filename(self, suffix="_analyzed") -> str:
        """Generate suggested filename for exports"""
        if self.loaded_file_path:
            base_name = os.path.basename(self.loaded_file_path).split('.mat')[0]
            if '[' in base_name:
                base_name = base_name.split('[')[0]
            
            # Get current parameters to check peak type
            # This would need the current params passed in, or stored
            # For now, just return the basic name
            return f"{base_name}{suffix}.csv"
        return f"analyzed.csv"
    
    def has_data(self) -> bool:
        """Check if data is loaded"""
        return self.current_dataset is not None and not self.current_dataset.is_empty()
    
    # ============ Channel Operations ============
    
    def swap_channels(self) -> Dict[str, Any]:
        """
        Swap voltage and current channel assignments.
        
        Returns:
            Dictionary with swap status and configuration
        """
        if not self.has_data():
            return {'success': False, 'reason': 'No data loaded'}
        
        if self.current_dataset.channel_count() < 2:
            return {'success': False, 'reason': 'Data has fewer than 2 channels'}
        
        self.channel_definitions.swap_channels()
        
        return {
            'success': True,
            'is_swapped': self.channel_definitions.is_swapped(),
            'configuration': self.channel_definitions.get_configuration()
        }
    
    # ============ Sweep Data Operations ============
    
    def get_sweep_plot_data(self, sweep_name: str, channel_type: str) -> Optional[PlotData]:
        """
        Get data for plotting a single sweep.
        
        Args:
            sweep_name: Name of the sweep (e.g., "Sweep 1")
            channel_type: Type of channel to plot ("Voltage" or "Current")
            
        Returns:
            PlotData object if successful, None otherwise
        """
        if not self.has_data():
            return None
        
        try:
            sweep_idx = sweep_name.split()[-1]
            plot_data_dict = self.analysis_engine.get_sweep_plot_data(sweep_idx, channel_type)
            
            if plot_data_dict is None:
                return None
            
            return PlotData(
                time_ms=plot_data_dict['time_ms'],
                data_matrix=plot_data_dict['data_matrix'],
                channel_id=plot_data_dict['channel_id'],
                sweep_index=plot_data_dict['sweep_index'],
                channel_type=plot_data_dict['channel_type']
            )
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Error getting sweep data: {str(e)}")
            return None
    
    # ============ Analysis Operations ============
    
    def perform_analysis(self, params: AnalysisParameters) -> Optional[AnalysisPlotData]:
        """
        Perform analysis with given parameters.
        
        Args:
            params: Analysis parameters
            
        Returns:
            AnalysisPlotData if successful, None otherwise
        """
        if not self.has_data():
            return None
        
        try:
            self.analysis_engine.set_dataset(self.current_dataset)
            result = self.analysis_engine.get_plot_data(params)
            
            return AnalysisPlotData(
                x_data=result['x_data'],
                y_data=result['y_data'],
                y_data2=result.get('y_data2'),
                x_label=result['x_label'],
                y_label=result['y_label'],
                sweep_indices=result['sweep_indices'],
                use_dual_range=params.use_dual_range
            )
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Error performing analysis: {str(e)}")
            return None
    
    # ============ Export Operations ============
    
    def get_suggested_export_filename(self, suffix="_analyzed") -> str:
        """Generate suggested filename for exports"""
        if self.loaded_file_path:
            base_name = os.path.basename(self.loaded_file_path).split('.mat')[0]
            if '[' in base_name:
                base_name = base_name.split('[')[0]
            return f"{base_name}{suffix}.csv"
        return f"analyzed.csv"
    
    # ============ Parameter Building ============
    
    @staticmethod
    def build_parameters(
        range1_start: float,
        range1_end: float,
        use_dual_range: bool,
        range2_start: Optional[float],
        range2_end: Optional[float],
        stimulus_period: float,
        x_measure: str,
        x_channel: Optional[str],
        x_peak_type: Optional[str],  # Added parameter
        y_measure: str,
        y_channel: Optional[str],
        y_peak_type: Optional[str],  # Added parameter
        channel_config: Dict[str, int]
    ) -> AnalysisParameters:
        """
        Build AnalysisParameters from individual values.
        This is a static method so it can be used anywhere.
        """
        x_axis_config = AxisConfig(
            measure=x_measure,
            channel=x_channel if x_measure != "Time" else None,
            peak_type=x_peak_type if x_measure == "Peak" else None  # Added
        )
        
        y_axis_config = AxisConfig(
            measure=y_measure,
            channel=y_channel if y_measure != "Time" else None,
            peak_type=y_peak_type if y_measure == "Peak" else None  # Added
        )
        
        return AnalysisParameters(
            range1_start=range1_start,
            range1_end=range1_end,
            use_dual_range=use_dual_range,
            range2_start=range2_start if use_dual_range else None,
            range2_end=range2_end if use_dual_range else None,
            stimulus_period=stimulus_period,
            x_axis=x_axis_config,
            y_axis=y_axis_config,
            channel_config=channel_config
        )
    
    @staticmethod
    def create_axis_labels(params: AnalysisParameters) -> Tuple[str, str]:
        """Create axis labels from parameters"""
        # X-axis label
        if params.x_axis.measure == "Time":
            x_label = "Time (s)"
        elif params.x_axis.measure == "Peak":
            unit = "(pA)" if params.x_axis.channel == "Current" else "(mV)"
            peak_type = params.x_axis.peak_type or "Absolute"
            peak_label_map = {
                "Absolute": "Peak",
                "Positive": "Peak (+)",
                "Negative": "Peak (-)",
                "Peak-Peak": "Peak-Peak"
            }
            x_label = f"{peak_label_map.get(peak_type, 'Peak')} {params.x_axis.channel} {unit}"
        else:
            unit = "(pA)" if params.x_axis.channel == "Current" else "(mV)"
            x_label = f"{params.x_axis.measure} {params.x_axis.channel} {unit}"
        
        # Y-axis label
        if params.y_axis.measure == "Time":
            y_label = "Time (s)"
        elif params.y_axis.measure == "Peak":
            unit = "(pA)" if params.y_axis.channel == "Current" else "(mV)"
            peak_type = params.y_axis.peak_type or "Absolute"
            peak_label_map = {
                "Absolute": "Peak",
                "Positive": "Peak (+)",
                "Negative": "Peak (-)",
                "Peak-Peak": "Peak-Peak"
            }
            y_label = f"{peak_label_map.get(peak_type, 'Peak')} {params.y_axis.channel} {unit}"
        else:
            unit = "(pA)" if params.y_axis.channel == "Current" else "(mV)"
            y_label = f"{params.y_axis.measure} {params.y_axis.channel} {unit}"
        
        return x_label, y_label

    def perform_peak_analysis(self, base_params: AnalysisParameters, 
                            peak_types: List[str] = None) -> Dict[str, AnalysisPlotData]:
        """
        Perform analysis for multiple peak types.
        
        Args:
            base_params: Base analysis parameters
            peak_types: List of peak types to analyze
            
        Returns:
            Dictionary mapping peak type to AnalysisPlotData
        """
        if not self.has_data():
            return {}
        
        if peak_types is None:
            peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]
        
        results = {}
        
        for peak_type in peak_types:
            # Create modified parameters with specific peak type
            modified_params = AnalysisParameters(
                range1_start=base_params.range1_start,
                range1_end=base_params.range1_end,
                use_dual_range=base_params.use_dual_range,
                range2_start=base_params.range2_start,
                range2_end=base_params.range2_end,
                stimulus_period=base_params.stimulus_period,
                x_axis=AxisConfig(
                    measure=base_params.x_axis.measure,
                    channel=base_params.x_axis.channel,
                    peak_type=peak_type if base_params.x_axis.measure == "Peak" else None
                ),
                y_axis=AxisConfig(
                    measure=base_params.y_axis.measure,
                    channel=base_params.y_axis.channel,
                    peak_type=peak_type if base_params.y_axis.measure == "Peak" else None
                ),
                channel_config=base_params.channel_config
            )
            
            # Perform analysis with modified parameters
            plot_data = self.perform_analysis(modified_params)
            if plot_data:
                results[peak_type] = plot_data
        
        return results

    def export_peak_analysis(self, base_params: AnalysisParameters, 
                            destination_folder: str,
                            peak_types: List[str] = None) -> Dict[str, bool]:
        """
        Export analysis data for multiple peak types.
        
        Args:
            base_params: Base analysis parameters
            destination_folder: Where to save the files
            peak_types: List of peak types to export
            
        Returns:
            Dictionary mapping peak type to export success status
        """
        if not self.has_data() or not self.loaded_file_path:
            return {}
        
        if peak_types is None:
            peak_types = ["Absolute", "Positive", "Negative", "Peak-Peak"]
        
        results = {}
        base_name = os.path.basename(self.loaded_file_path).split('.mat')[0]
        if '[' in base_name:
            base_name = base_name.split('[')[0]
        
        for peak_type in peak_types:
            # Create filename with peak type suffix
            peak_suffix_map = {
                "Absolute": "_absolute",
                "Positive": "_positive",
                "Negative": "_negative",
                "Peak-Peak": "_peak-peak"
            }
            suffix = peak_suffix_map.get(peak_type, "")
            filename = f"{base_name}{suffix}.csv"
            file_path = os.path.join(destination_folder, filename)
            
            # Create modified parameters
            modified_params = AnalysisParameters(
                range1_start=base_params.range1_start,
                range1_end=base_params.range1_end,
                use_dual_range=base_params.use_dual_range,
                range2_start=base_params.range2_start,
                range2_end=base_params.range2_end,
                stimulus_period=base_params.stimulus_period,
                x_axis=AxisConfig(
                    measure=base_params.x_axis.measure,
                    channel=base_params.x_axis.channel,
                    peak_type=peak_type if base_params.x_axis.measure == "Peak" else None
                ),
                y_axis=AxisConfig(
                    measure=base_params.y_axis.measure,
                    channel=base_params.y_axis.channel,
                    peak_type=peak_type if base_params.y_axis.measure == "Peak" else None
                ),
                channel_config=base_params.channel_config
            )
            
            # Export with modified parameters
            success = self.export_analysis_data_to_file(modified_params, file_path)
            results[peak_type] = success
        
        return results