"""
Simplified plotting service for creating visualizations.

This module provides straightforward plotting functionality without
complex backend abstractions.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend by default
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from data_analysis_gui.core.models import BatchAnalysisResult, FileAnalysisResult
from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class PlotService:
    """
    Creates plots for analysis results.
    
    Simple, direct plotting methods that scientists can easily understand
    and modify for their specific visualization needs.
    """
    
    def __init__(self):
        """Initialize the plot service."""
        logger.info("PlotService initialized")
    
    def create_analysis_plot(self,
                            x_data: np.ndarray,
                            y_data: np.ndarray,
                            x_label: str,
                            y_label: str,
                            title: str = None,
                            y_data2: np.ndarray = None,
                            figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Create a simple analysis plot.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            x_label: X-axis label
            y_label: Y-axis label
            title: Optional plot title
            y_data2: Optional second dataset for dual range
            figsize: Figure size in inches
            
        Returns:
            Matplotlib Figure object
        """
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Plot primary data
        ax.plot(x_data, y_data, 'o-', label="Range 1", markersize=6)
        
        # Plot secondary data if provided
        if y_data2 is not None:
            ax.plot(x_data, y_data2, 's--', label="Range 2", markersize=6)
            ax.legend()
        
        # Labels and formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def create_batch_plot(self,
                         results: List[FileAnalysisResult],
                         params: AnalysisParameters,
                         x_label: str,
                         y_label: str,
                         figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Create a plot showing multiple file results.
        
        Args:
            results: List of file analysis results
            params: Analysis parameters
            x_label: X-axis label
            y_label: Y-axis label
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Get color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Plot each result
        for i, result in enumerate(results):
            if not result.success or len(result.x_data) == 0:
                continue
            
            color = colors[i % len(colors)]
            
            # Plot Range 1
            ax.plot(result.x_data, result.y_data,
                   'o-', label=f"{result.base_name}",
                   markersize=4, alpha=0.7, color=color)
            
            # Plot Range 2 if available
            if params.use_dual_range and result.y_data2 is not None:
                ax.plot(result.x_data, result.y_data2,
                       's--', label=f"{result.base_name} (R2)",
                       markersize=4, alpha=0.7, color=color)
        
        # Formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        
        # Add legend if there's data
        if ax.has_data():
            ax.legend(loc='best', fontsize=8)
        
        fig.tight_layout()
        return fig
    
    def create_sweep_plot(self,
                         time_ms: np.ndarray,
                         data: np.ndarray,
                         channel_type: str,
                         sweep_index: int,
                         figsize: Tuple[int, int] = (8, 6)) -> Figure:
        """
        Create a plot for a single sweep.
        
        Args:
            time_ms: Time array in milliseconds
            data: Data array
            channel_type: "Voltage" or "Current"
            sweep_index: Sweep number
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Plot the sweep
        ax.plot(time_ms, data, linewidth=2)
        
        # Labels
        ax.set_title(f"Sweep {sweep_index} - {channel_type}")
        ax.set_xlabel("Time (ms)")
        
        unit = "mV" if channel_type == "Voltage" else "pA"
        ax.set_ylabel(f"{channel_type} ({unit})")
        
        ax.grid(True, alpha=0.3)
        
        # Add some padding to axes
        if len(time_ms) > 0:
            x_margin = (time_ms[-1] - time_ms[0]) * 0.02
            ax.set_xlim(time_ms[0] - x_margin, time_ms[-1] + x_margin)
        
        if len(data) > 0:
            y_margin = (np.max(data) - np.min(data)) * 0.05
            ax.set_ylim(np.min(data) - y_margin, np.max(data) + y_margin)
        
        fig.tight_layout()
        return fig
    
    def save_figure(self, figure: Figure, filepath: str, dpi: int = 300):
        """
        Save a figure to file.
        
        Args:
            figure: Matplotlib Figure to save
            filepath: Output file path
            dpi: Resolution in dots per inch
        """
        try:
            figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save figure: {e}")
            raise
    
    @staticmethod
    def get_axis_labels(params: AnalysisParameters) -> Tuple[str, str]:
        """
        Get axis labels from analysis parameters.
        
        Args:
            params: Analysis parameters
            
        Returns:
            Tuple of (x_label, y_label)
        """
        # X-axis label
        if params.x_axis.measure == "Time":
            x_label = "Time (s)"
        elif params.x_axis.measure == "Average":
            unit = "mV" if params.x_axis.channel == "Voltage" else "pA"
            x_label = f"Average {params.x_axis.channel} ({unit})"
        else:  # Peak
            unit = "mV" if params.x_axis.channel == "Voltage" else "pA"
            x_label = f"Peak {params.x_axis.channel} ({unit})"
        
        # Y-axis label
        if params.y_axis.measure == "Time":
            y_label = "Time (s)"
        elif params.y_axis.measure == "Average":
            unit = "mV" if params.y_axis.channel == "Voltage" else "pA"
            y_label = f"Average {params.y_axis.channel} ({unit})"
        else:  # Peak
            unit = "mV" if params.y_axis.channel == "Voltage" else "pA"
            y_label = f"Peak {params.y_axis.channel} ({unit})"
        
        return x_label, y_label