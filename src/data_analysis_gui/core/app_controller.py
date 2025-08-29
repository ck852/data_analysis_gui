"""
Application Controller - Mediates between GUI and business logic.
This is the ONLY class the GUI should interact with for business operations.
"""

import os
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from io import BytesIO
import base64
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

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


class ApplicationController:
    """
    Central controller that manages all business logic.
    GUI interacts only with this controller, never directly with business logic.
    """
    
    def __init__(self):
        # Core business objects
        self.channel_definitions = ChannelDefinitions()
        self.analysis_engine = AnalysisEngine(self.channel_definitions)
        self.current_dataset: Optional[ElectrophysiologyDataset] = None
        self.loaded_file_path: Optional[str] = None
        
        # Callbacks for GUI updates (dependency injection)
        self.on_file_loaded: Optional[Callable[[FileInfo], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_status_update: Optional[Callable[[str], None]] = None
    
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
            y_measure=gui_state['y_measure'],
            y_channel=gui_state.get('y_channel'),
            channel_config=self.get_channel_configuration()
        )
    
    def perform_batch_analysis_with_plot(
        self,
        file_paths: List[str],
        params: AnalysisParameters,
        destination_folder: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Perform batch analysis and generate the plot internally.
        Returns the plot as serialized figure data.
        """
        # Create figure internally
        fig = Figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        x_label, y_label = self.create_axis_labels(params)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        
        # Process files and plot
        successful_files = []
        failed_files = []
        
        def handle_file_result(result: FileResult):
            if result.success:
                successful_files.append(result)
                # Plot the data internally
                ax.plot(result.x_data, result.y_data, 
                       'o-', label=f"{result.base_name} (Range 1)",
                       markersize=4, alpha=0.7)
                
                if params.use_dual_range and len(result.y_data2) > 0:
                    ax.plot(result.x_data, result.y_data2,
                           's--', label=f"{result.base_name} (Range 2)",
                           markersize=4, alpha=0.7)
            else:
                failed_files.append(result)
        
        # Run the batch processor
        processor = BatchProcessor(self.channel_definitions)
        batch_result = processor.run(
            file_paths,
            params,
            on_progress=progress_callback,
            on_file_done=handle_file_result
        )
        
        # Export results
        export_outcomes = []
        if batch_result.successful_results:
            export_outcomes = exporter.write_tables(batch_result, destination_folder)
        
        # Finalize plot
        if successful_files:
            ax.legend(loc='best', fontsize=8)
            fig.tight_layout()
        
        # Serialize figure to bytes
        buf = BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(buf, format='png')
        buf.seek(0)
        figure_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        # Prepare batch data for dialog
        batch_data = {
            res.base_name: {
                'x_values': res.x_data.tolist(),
                'y_values': res.y_data.tolist(),
                'y_values2': res.y_data2.tolist() if res.y_data2 is not None else []
            } for res in batch_result.successful_results
        }
        
        # Prepare IV data
        iv_data, iv_file_mapping = IVAnalysisService.prepare_iv_data(batch_data, params)
        
        return {
            'success': len(successful_files) > 0,
            'figure_data': figure_data,  # Base64 encoded PNG
            'figure_size': (12, 8),
            'batch_data': batch_data,
            'iv_data': iv_data,
            'iv_file_mapping': iv_file_mapping,
            'successful_count': len(successful_files),
            'failed_count': len(failed_files),
            'export_outcomes': export_outcomes,
            'x_label': x_label,
            'y_label': y_label
        }
        
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
    
    def export_analysis_data(self, params: AnalysisParameters, destination_folder: str) -> bool:
        """
        Export analysis data to CSV.
        
        Args:
            params: Analysis parameters
            destination_folder: Where to save the file
            
        Returns:
            True if successful, False otherwise
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
            
            base_name = os.path.basename(self.loaded_file_path).split('.mat')[0]
            outcome = exporter.write_single_table(
                table=table_data,
                base_name=f"{base_name}_analyzed",
                destination_folder=destination_folder
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
    
    # ============ Batch Operations ============
    
    def perform_batch_analysis(
        self,
        file_paths: List[str],
        params: AnalysisParameters,
        destination_folder: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_file_complete: Optional[Callable[[FileResult], None]] = None
    ) -> Dict[str, Any]:
        """
        Perform batch analysis on multiple files.
        
        Args:
            file_paths: List of file paths to analyze
            params: Analysis parameters
            destination_folder: Where to save results
            on_progress: Progress callback (current, total)
            on_file_complete: Callback for each file completion
            
        Returns:
            Dictionary with results and statistics
        """
        try:
            processor = BatchProcessor(self.channel_definitions)
            batch_result = processor.run(
                file_paths,
                params,
                on_progress=on_progress,
                on_file_done=on_file_complete
            )
            
            # Export results
            export_outcomes = []
            if batch_result.successful_results:
                export_outcomes = exporter.write_tables(batch_result, destination_folder)
                success_count = sum(1 for o in export_outcomes if o.success)
                
                if self.on_status_update:
                    self.on_status_update(
                        f"Batch complete. Exported {success_count} files to {os.path.basename(destination_folder)}"
                    )
            
            # Prepare IV data if applicable
            batch_data = {
                res.base_name: {
                    'x_values': res.x_data,
                    'y_values': res.y_data,
                    'y_values2': res.y_data2
                } for res in batch_result.successful_results
            }
            
            iv_data, iv_file_mapping = IVAnalysisService.prepare_iv_data(batch_data, params)
            
            return {
                'success': True,
                'batch_data': batch_data,
                'iv_data': iv_data,
                'iv_file_mapping': iv_file_mapping,
                'export_outcomes': export_outcomes,
                'results': batch_result.results
            }
            
        except Exception as e:
            if self.on_error:
                self.on_error(f"Batch analysis error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
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
        y_measure: str,
        y_channel: Optional[str],
        channel_config: Dict[str, int]
    ) -> AnalysisParameters:
        """
        Build AnalysisParameters from individual values.
        This is a static method so it can be used anywhere.
        """
        x_axis_config = AxisConfig(
            measure=x_measure,
            channel=x_channel if x_measure != "Time" else None
        )
        
        y_axis_config = AxisConfig(
            measure=y_measure,
            channel=y_channel if y_measure != "Time" else None
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
        else:
            unit = "(pA)" if params.x_axis.channel == "Current" else "(mV)"
            x_label = f"{params.x_axis.measure} {params.x_axis.channel} {unit}"
        
        # Y-axis label
        if params.y_axis.measure == "Time":
            y_label = "Time (s)"
        else:
            unit = "(pA)" if params.y_axis.channel == "Current" else "(mV)"
            y_label = f"{params.y_axis.measure} {params.y_axis.channel} {unit}"
        
        return x_label, y_label