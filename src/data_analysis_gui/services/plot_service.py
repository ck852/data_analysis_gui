"""
Plot Service - Handles all plotting operations for the application.
Separates visualization logic from business logic.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from io import BytesIO
import base64

from data_analysis_gui.core.plotting_interface import PlotBackend, PlotBackendFactory
from data_analysis_gui.core.batch_processor import BatchResult
from data_analysis_gui.core.params import AnalysisParameters
from matplotlib.figure import Figure

class PlotService:
    """
    Service class that handles all plotting operations.
    Uses the abstract PlotBackend to remain testable in CI environments.
    """
    
    def __init__(self, backend: Optional[PlotBackend] = None):
        """
        Initialize the plot service.
        
        Args:
            backend: Optional PlotBackend instance. If None, creates one using factory.
        """
        self.backend = backend or PlotBackendFactory.create_backend()
    
    def create_batch_plot(
        self,
        batch_result: BatchResult,
        params: AnalysisParameters,
        x_label: str,
        y_label: str,
        figsize: Tuple[float, float] = (12, 8)
    ) -> Dict[str, Any]:
        """
        Create a batch analysis plot and return it as serialized data.
        
        Args:
            batch_result: The BatchResult containing all file results
            params: Analysis parameters used
            x_label: Label for x-axis
            y_label: Label for y-axis
            figsize: Figure size tuple
            
        Returns:
            Dictionary containing:
                - figure_data: Base64 encoded PNG image
                - figure_size: Original figure size
                - plot_count: Number of files plotted
        """
        # Create figure using backend
        fig = self.backend.create_figure(figsize)
        ax = self._add_axes_to_figure(fig)
        
        # Set up axes
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        
        # Plot each successful file result
        plot_count = 0
        for result in batch_result.successful_results:
            if len(result.x_data) > 0 and len(result.y_data) > 0:
                # Plot Range 1 data
                ax.plot(result.x_data, result.y_data,
                       'o-', label=f"{result.base_name} (Range 1)",
                       markersize=4, alpha=0.7)
                plot_count += 1
                
                # Plot Range 2 data if dual range is enabled
                if params.use_dual_range and len(result.y_data2) > 0:
                    ax.plot(result.x_data, result.y_data2,
                           's--', label=f"{result.base_name} (Range 2)",
                           markersize=4, alpha=0.7)
        
        # Add legend if there's data
        if plot_count > 0:
            ax.legend(loc='best', fontsize=8)
        
        # Finalize layout
        self._tight_layout(fig)
        
        # Serialize figure to base64
        figure_bytes = self.backend.save_figure(fig, format='png')
        figure_data = base64.b64encode(figure_bytes).decode('utf-8')
        
        return {
            'figure_data': figure_data,
            'figure_size': figsize,
            'plot_count': plot_count
        }
    
    def create_analysis_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_label: str,
        y_label: str,
        y_data2: Optional[np.ndarray] = None,
        y_label2: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 6)
    ) -> Dict[str, Any]:
        """
        Create a single analysis plot.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            x_label: X-axis label
            y_label: Y-axis label
            y_data2: Optional second Y-axis dataset
            y_label2: Optional label for second dataset
            title: Optional plot title
            figsize: Figure size
            
        Returns:
            Dictionary containing figure_data and metadata
        """
        # Create figure
        fig = self.backend.create_figure(figsize)
        ax = self._add_axes_to_figure(fig)
        
        # Plot primary data
        ax.plot(x_data, y_data, 'o-', label=y_label, markersize=6)
        
        # Plot secondary data if provided
        if y_data2 is not None:
            label2 = y_label2 or f"{y_label} (Range 2)"
            ax.plot(x_data, y_data2, 's--', label=label2, markersize=6)
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        if y_data2 is not None:
            ax.legend()
        
        # Finalize layout
        self._tight_layout(fig)
        
        # Serialize
        figure_bytes = self.backend.save_figure(fig, format='png')
        figure_data = base64.b64encode(figure_bytes).decode('utf-8')
        
        return {
            'figure_data': figure_data,
            'figure_size': figsize
        }
    
    def create_sweep_plot(
        self,
        time_ms: np.ndarray,
        data: np.ndarray,
        channel_type: str,
        sweep_index: int,
        figsize: Tuple[float, float] = (8, 6)
    ) -> Dict[str, Any]:
        """
        Create a plot for a single sweep.
        
        Args:
            time_ms: Time array in milliseconds
            data: Data array
            channel_type: Type of channel ("Voltage" or "Current")
            sweep_index: Index of the sweep
            figsize: Figure size
            
        Returns:
            Dictionary containing figure_data and metadata
        """
        fig = self.backend.create_figure(figsize)
        ax = self._add_axes_to_figure(fig)
        
        # Plot the data
        ax.plot(time_ms, data, linewidth=2)
        
        # Set labels and title
        ax.set_title(f"Sweep {sweep_index} - {channel_type}")
        ax.set_xlabel("Time (ms)")
        
        # Use proper units based on channel type
        unit = "mV" if channel_type == "Voltage" else "pA"
        ax.set_ylabel(f"{channel_type} ({unit})")
        
        ax.grid(True, alpha=0.3)
        
        # Auto-scale with padding
        self._autoscale_with_padding(ax, time_ms, data)
        
        # Finalize
        self._tight_layout(fig)
        
        # Serialize
        figure_bytes = self.backend.save_figure(fig, format='png')
        figure_data = base64.b64encode(figure_bytes).decode('utf-8')
        
        return {
            'figure_data': figure_data,
            'figure_size': figsize,
            'sweep_index': sweep_index,
            'channel_type': channel_type
        }
    
    def build_batch_figure(
        self,
        batch_result: BatchResult,
        params: AnalysisParameters,
        x_label: str,
        y_label: str,
        figsize: Tuple[float, float] = (12, 8)
    ) -> Tuple[Figure, int]:
        """
        Build and return the actual matplotlib Figure for batch results.
        
        Returns:
            Tuple of (Figure object, plot_count)
        """
        # Create figure using backend
        fig = self.backend.create_figure(figsize)
        ax = self._add_axes_to_figure(fig)
        
        # Set up axes
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        
        # Plot each successful file result
        plot_count = 0
        for result in batch_result.successful_results:
            if len(result.x_data) > 0 and len(result.y_data) > 0:
                ax.plot(result.x_data, result.y_data,
                    'o-', label=f"{result.base_name} (Range 1)",
                    markersize=4, alpha=0.7)
                plot_count += 1
                
                if params.use_dual_range and len(result.y_data2) > 0:
                    ax.plot(result.x_data, result.y_data2,
                        's--', label=f"{result.base_name} (Range 2)",
                        markersize=4, alpha=0.7)
        
        if plot_count > 0:
            ax.legend(loc='best', fontsize=8)
        
        self._tight_layout(fig)
        
        return fig, plot_count
    
    # ============ Helper Methods ============
    
    def _add_axes_to_figure(self, fig):
        """Add axes to figure (backend-agnostic)"""
        # This is a simplified version - the actual implementation
        # would depend on the backend
        return fig.add_subplot(111)
    
    def _tight_layout(self, fig):
        """Apply tight layout if supported by backend"""
        if hasattr(fig, 'tight_layout'):
            fig.tight_layout()
    
    def _autoscale_with_padding(self, ax, x_data, y_data, padding=0.05):
        """Auto-scale axes with padding"""
        if len(x_data) > 0 and len(y_data) > 0:
            x_range = np.max(x_data) - np.min(x_data)
            y_range = np.max(y_data) - np.min(y_data)
            
            x_pad = x_range * padding
            y_pad = y_range * padding
            
            ax.set_xlim(np.min(x_data) - x_pad, np.max(x_data) + x_pad)
            ax.set_ylim(np.min(y_data) - y_pad, np.max(y_data) + y_pad)