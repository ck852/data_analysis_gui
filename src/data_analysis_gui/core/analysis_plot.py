# src/data_analysis_gui/core/analysis_plot.py
"""
Core analysis plot functionality that can be used independently of GUI.
This module provides all the data processing and plotting logic without
any GUI dependencies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


@dataclass
class AnalysisPlotData:
    """Data structure for analysis plots"""
    x_data: np.ndarray
    y_data: np.ndarray
    sweep_indices: List[int]
    use_dual_range: bool = False
    y_data2: Optional[np.ndarray] = None
    y_label_r1: Optional[str] = None
    y_label_r2: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisPlotData':
        """Create from dictionary for backward compatibility"""
        return cls(
            x_data=np.array(data.get('x_data', [])),
            y_data=np.array(data.get('y_data', [])),
            sweep_indices=data.get('sweep_indices', []),
            use_dual_range=data.get('use_dual_range', False),
            y_data2=np.array(data.get('y_data2', [])) if 'y_data2' in data else None,
            y_label_r1=data.get('y_label_r1'),
            y_label_r2=data.get('y_label_r2')
        )


class AnalysisPlotter:
    """Handles creation and configuration of analysis plots"""
    
    def __init__(self, plot_data: AnalysisPlotData, x_label: str, y_label: str, title: str):
        self.plot_data = plot_data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        
    def create_figure(self, figsize: Tuple[int, int] = (8, 6)) -> Tuple[Figure, Axes]:
        """
        Create and configure a matplotlib figure with the analysis plot.
        
        Returns:
            Tuple of (Figure, Axes) objects
        """
        figure = Figure(figsize=figsize)
        ax = figure.add_subplot(111)
        self._configure_plot(ax)
        return figure, ax
    
    def _configure_plot(self, ax: Axes) -> None:
        """Configure the plot on the given axes"""
        x_data = self.plot_data.x_data
        y_data = self.plot_data.y_data
        sweep_indices = self.plot_data.sweep_indices
        
        if len(x_data) > 0 and len(y_data) > 0:
            # Create scatter plot with connecting lines for Range 1
            ax.plot(x_data, y_data, 'o-', linewidth=2, markersize=6, label="Range 1")
            
            # Add sweep labels to the points
            for i, sweep_idx in enumerate(sweep_indices):
                if i < len(x_data) and i < len(y_data):
                    ax.annotate(f"{sweep_idx}",
                                (x_data[i], y_data[i]),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha='center')
        
        # Plot Range 2 if available
        if self.plot_data.use_dual_range and self.plot_data.y_data2 is not None:
            y_data2 = self.plot_data.y_data2
            if len(x_data) > 0 and len(y_data2) > 0:
                ax.plot(x_data, y_data2, 's--', linewidth=2, markersize=6, label="Range 2")
        
        # Format plot
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(self.title)
        ax.grid(True, alpha=0.3)
        
        # Add legend if needed
        if self.plot_data.use_dual_range:
            ax.legend()
        
        # Apply padding
        self._apply_axis_padding(ax)
    
    def _apply_axis_padding(self, ax: Axes, padding_factor: float = 0.05) -> None:
        """Apply padding to both axes for better visualization"""
        ax.relim()
        ax.autoscale_view()
        
        x_data = self.plot_data.x_data
        y_data = self.plot_data.y_data
        
        if len(x_data) > 0 and len(y_data) > 0:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            x_padding = x_range * padding_factor if x_range > 0 else 0.1
            y_padding = y_range * padding_factor if y_range > 0 else 0.1
            
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    def save_figure(self, figure: Figure, filepath: str, dpi: int = 300) -> None:
        """Save figure to file"""
        figure.tight_layout()
        figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
    
    def get_export_data(self) -> Tuple[np.ndarray, str]:
        """
        Prepare data for CSV export.
        
        Returns:
            Tuple of (data array, header string)
        """
        x_data = self.plot_data.x_data
        y_data = self.plot_data.y_data
        
        if self.plot_data.use_dual_range and self.plot_data.y_data2 is not None:
            y_data2 = self.plot_data.y_data2
            export_data = np.column_stack((x_data, y_data, y_data2))
            
            # Use descriptive labels if available
            y_label_r1 = self.plot_data.y_label_r1 or f"{self.y_label} (Range 1)"
            y_label_r2 = self.plot_data.y_label_r2 or f"{self.y_label} (Range 2)"
            header = f"{self.x_label},{y_label_r1},{y_label_r2}"
        else:
            export_data = np.column_stack((x_data, y_data))
            header = f"{self.x_label},{self.y_label}"
        
        return export_data, header


# CLI-friendly functions
def create_analysis_plot(plot_data_dict: Dict[str, Any], 
                         x_label: str, 
                         y_label: str, 
                         title: str,
                         output_path: Optional[str] = None,
                         show: bool = False) -> Optional[Figure]:
    """
    Create an analysis plot from data dictionary.
    
    Args:
        plot_data_dict: Dictionary containing plot data
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        output_path: Optional path to save the plot
        show: Whether to display the plot (requires GUI backend)
    
    Returns:
        Figure object if created, None otherwise
    """
    plot_data = AnalysisPlotData.from_dict(plot_data_dict)
    plotter = AnalysisPlotter(plot_data, x_label, y_label, title)
    
    # Use regular matplotlib for CLI
    fig, ax = plt.subplots(figsize=(8, 6))
    plotter._configure_plot(ax)
    
    if output_path:
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def export_analysis_data(plot_data_dict: Dict[str, Any],
                         x_label: str,
                         y_label: str,
                         output_path: str,
                         format_spec: str = '%.6f') -> None:
    """
    Export analysis data to CSV file.
    
    Args:
        plot_data_dict: Dictionary containing plot data
        x_label: Label for x column
        y_label: Label for y column(s)
        output_path: Path to save CSV file
        format_spec: Format specification for numpy.savetxt
    """
    plot_data = AnalysisPlotData.from_dict(plot_data_dict)
    plotter = AnalysisPlotter(plot_data, x_label, y_label, "")
    
    export_data, header = plotter.get_export_data()
    
    # Direct export without GUI dependencies
    np.savetxt(output_path, export_data, delimiter=',', 
               header=header, fmt=format_spec, comments='')