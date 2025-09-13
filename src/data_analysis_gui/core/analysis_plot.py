# src/data_analysis_gui/core/analysis_plot.py
"""
Core analysis plot functionality that can be used independently of GUI.
This module provides all the data processing and plotting logic without
any GUI dependencies.

PHASE 3 REFACTOR: Converted to stateless pure functions for thread safety
and memory efficiency. All methods are now static and receive data as parameters.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib
# Set thread-safe backend as default for non-GUI operations
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from data_analysis_gui.config.plot_style import (
    apply_plot_style, format_analysis_plot, get_line_styles, COLORS
)


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
    """
    PHASE 3 REFACTOR: Stateless plotter for analysis plots.
    
    All methods are static and pure functions that transform inputs to outputs
    without side effects. This design ensures:
    - Thread safety: No shared mutable state between calls
    - Memory efficiency: No data stored in instances
    - Testability: Pure functions are easy to test in isolation
    - Parallelization: Safe for concurrent execution
    
    Thread Safety Guarantees:
    - All methods are thread-safe when using 'Agg' backend
    - Matplotlib Figure creation is thread-local
    - No global state modifications
    - Safe for parallel batch processing
    
    Note: When using GUI backends, external synchronization may be required.
    """
    
    @staticmethod
    def create_figure(plot_data: AnalysisPlotData, 
                     x_label: str, 
                     y_label: str, 
                     title: str,
                     figsize: Tuple[int, int] = (8, 6)) -> Tuple[Figure, Axes]:
        """
        Create and configure a matplotlib figure with modern styling.
        """
        # Apply global style
        apply_plot_style()
        
        # Create figure with styled background
        figure = Figure(figsize=figsize, facecolor='#FAFAFA')
        ax = figure.add_subplot(111)
        
        # Configure plot with modern styling
        AnalysisPlotter._configure_plot(ax, plot_data, x_label, y_label, title)
        
        # Apply analysis-specific formatting
        format_analysis_plot(ax, x_label, y_label, title)
        
        # Ensure proper layout
        figure.tight_layout(pad=1.5)
        
        return figure, ax
    
    @staticmethod
    def _configure_plot(ax: Axes, 
                       plot_data: AnalysisPlotData,
                       x_label: str,
                       y_label: str, 
                       title: str) -> None:
        """
        Configure the plot with modern styling.
        """
        x_data = plot_data.x_data
        y_data = plot_data.y_data
        sweep_indices = plot_data.sweep_indices
        
        # Get line styles
        line_styles = get_line_styles()
        
        if len(x_data) > 0 and len(y_data) > 0:
            # Create plot with modern styling for Range 1
            primary_style = line_styles['primary']
            line1 = ax.plot(
                x_data, y_data,
                marker=primary_style['marker'],
                markersize=primary_style['markersize'],
                markeredgewidth=primary_style['markeredgewidth'],
                linewidth=primary_style['linewidth'],
                color=primary_style['color'],
                alpha=primary_style['alpha'],
                label="Range 1"
            )[0]
            
            # Add subtle sweep labels with better positioning
            for i, sweep_idx in enumerate(sweep_indices):
                if i < len(x_data) and i < len(y_data):
                    ax.annotate(
                        f"{sweep_idx}",
                        (x_data[i], y_data[i]),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha='center',
                        fontsize=8,
                        color='#606060',
                        alpha=0.8
                    )
        
        # Plot Range 2 if available with contrasting style
        if plot_data.use_dual_range and plot_data.y_data2 is not None:
            y_data2 = plot_data.y_data2
            if len(x_data) > 0 and len(y_data2) > 0:
                secondary_style = line_styles['secondary']
                line2 = ax.plot(
                    x_data, y_data2,
                    marker=secondary_style['marker'],
                    markersize=secondary_style['markersize'],
                    markeredgewidth=secondary_style['markeredgewidth'],
                    linewidth=secondary_style['linewidth'],
                    linestyle=secondary_style.get('linestyle', '-'),
                    color=secondary_style['color'],
                    alpha=secondary_style['alpha'],
                    label="Range 2"
                )[0]
        
        # Modern legend styling if dual range
        if plot_data.use_dual_range:
            ax.legend(
                loc='best',
                frameon=True,
                fancybox=False,
                shadow=False,
                framealpha=0.95,
                edgecolor='#D0D0D0',
                facecolor='white',
                fontsize=9
            )
        
        # Apply axis padding with subtle animation-ready margins
        AnalysisPlotter._apply_axis_padding(ax, x_data, y_data)
    
    @staticmethod
    def _apply_axis_padding(ax: Axes, 
                           x_data: np.ndarray,
                           y_data: np.ndarray,
                           padding_factor: float = 0.05) -> None:
        """
        Apply padding to both axes for better visualization.
        """
        ax.relim()
        ax.autoscale_view()
        
        if len(x_data) > 0 and len(y_data) > 0:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Slightly asymmetric padding for visual balance
            x_padding = x_range * padding_factor if x_range > 0 else 0.1
            y_padding_bottom = y_range * padding_factor if y_range > 0 else 0.1
            y_padding_top = y_range * (padding_factor * 1.2) if y_range > 0 else 0.1
            
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding_bottom, y_max + y_padding_top)
    
    @staticmethod
    def save_figure(figure: Figure, 
                   filepath: str, 
                   dpi: int = 300) -> None:
        """
        Save figure to file.
        
        Pure function that performs I/O operation.
        
        Args:
            figure: Matplotlib figure to save
            filepath: Output file path
            dpi: Resolution in dots per inch
            
        Thread Safety: File I/O may require external synchronization
                      if multiple threads write to same directory
        """
        figure.tight_layout()
        figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
    
    @staticmethod
    def create_and_save_plot(plot_data: AnalysisPlotData,
                           x_label: str,
                           y_label: str,
                           title: str,
                           filepath: str,
                           figsize: Tuple[int, int] = (8, 6),
                           dpi: int = 300) -> Figure:
        """
        Convenience method to create and save a plot in one operation.
        
        Combines figure creation and saving for common use case.
        Pure function that returns the created figure.
        
        Args:
            plot_data: Data to plot
            x_label: X-axis label
            y_label: Y-axis label
            title: Plot title
            filepath: Output file path
            figsize: Figure size
            dpi: Resolution
            
        Returns:
            The created Figure object
            
        Thread Safety: Safe with 'Agg' backend, file I/O may need synchronization
        """
        figure, _ = AnalysisPlotter.create_figure(
            plot_data, x_label, y_label, title, figsize
        )
        AnalysisPlotter.save_figure(figure, filepath, dpi)
        return figure


# CLI-friendly functions updated for stateless operation
def create_analysis_plot(plot_data_dict: Dict[str, Any], 
                         x_label: str, 
                         y_label: str, 
                         title: str,
                         output_path: Optional[str] = None,
                         show: bool = False) -> Optional[Figure]:
    """
    Create an analysis plot from data dictionary.
    
    PHASE 3 UPDATE: Now uses stateless AnalysisPlotter methods.
    This function serves as a CLI interface to the plotting functionality.
    
    Args:
        plot_data_dict: Dictionary containing plot data
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        output_path: Optional path to save the plot
        show: Whether to display the plot (requires GUI backend)
    
    Returns:
        Figure object if created, None otherwise
        
    Thread Safety: Safe with 'Agg' backend when show=False
                  GUI display (show=True) requires main thread
    """
    plot_data = AnalysisPlotData.from_dict(plot_data_dict)
    
    # Use stateless methods
    if output_path:
        # Use the combined method for efficiency
        fig = AnalysisPlotter.create_and_save_plot(
            plot_data, x_label, y_label, title, output_path
        )
    else:
        # Just create without saving
        fig, ax = AnalysisPlotter.create_figure(
            plot_data, x_label, y_label, title
        )
    
    if show:
        # Note: This requires GUI backend and is NOT thread-safe
        # Should only be called from main thread
        import warnings
        warnings.warn(
            "Displaying plots with show=True is not thread-safe. "
            "Use only from main thread.",
            RuntimeWarning
        )
        plt.show()
    
    return fig