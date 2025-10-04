"""
PatchBatch Electrophysiology Data Analysis Tool - Analysis Plot Module

This module provides core functionality for generating, configuring, and saving analysis plots
using matplotlib for electrophysiology data. It defines data structures and stateless plotting utilities
to support both GUI and CLI workflows.

Features:
- AnalysisPlotData dataclass for structured plot data representation.
- Stateless AnalysisPlotter class for thread-safe figure creation, configuration, and export.
- Modern plot styling and dual-range support for comparative analysis.
- CLI-friendly functions for quick plot generation and saving.
- Designed for integration with both GUI and batch processing pipelines.

Usage:
Import and use AnalysisPlotter or create_analysis_plot to generate publication-ready figures
from analysis results, with support for custom styling and export.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib

# Set thread-safe backend as default for non-GUI operations
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from data_analysis_gui.config.plot_style import (
    apply_plot_style,
    format_analysis_plot,
    get_line_styles,
)


@dataclass
class AnalysisPlotData:
    """
    Data structure for analysis plots.

    Attributes:
        x_data (np.ndarray): X-axis data points.
        y_data (np.ndarray): Y-axis data points for primary range.
        sweep_indices (List[int]): Indices of sweeps included in the plot.
        use_dual_range (bool): Whether dual range plotting is enabled.
        y_data2 (Optional[np.ndarray]): Y-axis data for secondary range, if applicable.
        y_label_r1 (Optional[str]): Label for primary range Y-axis.
        y_label_r2 (Optional[str]): Label for secondary range Y-axis.
    """

    x_data: np.ndarray
    y_data: np.ndarray
    sweep_indices: List[int]
    use_dual_range: bool = False
    y_data2: Optional[np.ndarray] = None
    y_label_r1: Optional[str] = None
    y_label_r2: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisPlotData":
        """
        Create an AnalysisPlotData instance from a dictionary for backward compatibility.

        Args:
            data (Dict[str, Any]): Dictionary containing plot data fields.

        Returns:
            AnalysisPlotData: Instance populated from the dictionary.
        """
        return cls(
            x_data=np.array(data.get("x_data", [])),
            y_data=np.array(data.get("y_data", [])),
            sweep_indices=data.get("sweep_indices", []),
            use_dual_range=data.get("use_dual_range", False),
            y_data2=np.array(data.get("y_data2", [])) if "y_data2" in data else None,
            y_label_r1=data.get("y_label_r1"),
            y_label_r2=data.get("y_label_r2"),
        )


class AnalysisPlotter:
    """
    Provides static methods for creating, configuring, and saving analysis plots using matplotlib.
    All methods are stateless and thread-safe for non-GUI operations.
    """

    @staticmethod
    def create_figure(
        plot_data: AnalysisPlotData,
        x_label: str,
        y_label: str,
        title: str,
        figsize: Tuple[int, int] = (8, 6),
    ) -> Tuple[Figure, Axes]:
        """
        Create and configure a matplotlib figure for analysis plots with modern styling.

        Args:
            plot_data (AnalysisPlotData): Data to plot.
            x_label (str): Label for the X-axis.
            y_label (str): Label for the Y-axis.
            title (str): Title of the plot.
            figsize (Tuple[int, int], optional): Size of the figure in inches. Defaults to (8, 6).

        Returns:
            Tuple[Figure, Axes]: The created matplotlib Figure and Axes objects.
        """
        # Apply global style
        apply_plot_style()

        # Create figure with styled background
        figure = Figure(figsize=figsize, facecolor="#FAFAFA")
        ax = figure.add_subplot(111)

        # Extract voltage-annotated labels from plot_data for legend
        y_label_r1 = plot_data.y_label_r1  # ADDED
        y_label_r2 = plot_data.y_label_r2  # ADDED

        # Configure plot with modern styling
        AnalysisPlotter._configure_plot(
            ax, 
            plot_data, 
            x_label, 
            y_label, 
            title,
            y_label_r1=y_label_r1,  # ADDED
            y_label_r2=y_label_r2   # ADDED
        )

        # Apply analysis-specific formatting
        format_analysis_plot(ax, x_label, y_label, title)

        # Ensure proper layout
        figure.tight_layout(pad=1.5)

        return figure, ax

    @staticmethod
    def _configure_plot(
        ax: Axes, 
        plot_data: AnalysisPlotData, 
        x_label: str, 
        y_label: str, 
        title: str,
        y_label_r1: Optional[str] = None,  # NEW
        y_label_r2: Optional[str] = None   # NEW
    ) -> None:
        """
        Configure the matplotlib Axes object with analysis plot data and modern styling.

        Args:
            ax (Axes): Matplotlib Axes to configure.
            plot_data (AnalysisPlotData): Data to plot.
            x_label (str): Label for the X-axis.
            y_label (str): Label for the Y-axis.
            title (str): Title of the plot.
            y_label_r1 (Optional[str]): Label for Range 1 legend (with voltage annotation).
            y_label_r2 (Optional[str]): Label for Range 2 legend (with voltage annotation).
        """
        x_data = plot_data.x_data
        y_data = plot_data.y_data
        sweep_indices = plot_data.sweep_indices

        # Get line styles
        line_styles = get_line_styles()

        if len(x_data) > 0 and len(y_data) > 0:
            # Use voltage-annotated label if provided, otherwise fallback
            range1_label = y_label_r1 or "Range 1"  # CHANGED
            
            # Create plot with modern styling for Range 1
            primary_style = line_styles["primary"]
            line1 = ax.plot(
                x_data,
                y_data,
                marker=primary_style["marker"],
                markersize=primary_style["markersize"],
                markeredgewidth=primary_style["markeredgewidth"],
                linewidth=primary_style["linewidth"],
                color=primary_style["color"],
                alpha=primary_style["alpha"],
                label=range1_label,  # CHANGED
            )[0]

        # Plot Range 2 if available with contrasting style
        if plot_data.use_dual_range and plot_data.y_data2 is not None:
            y_data2 = plot_data.y_data2
            if len(x_data) > 0 and len(y_data2) > 0:
                # Use voltage-annotated label if provided, otherwise fallback
                range2_label = y_label_r2 or "Range 2"  # CHANGED
                
                secondary_style = line_styles["secondary"]
                line2 = ax.plot(
                    x_data,
                    y_data2,
                    marker=secondary_style["marker"],
                    markersize=secondary_style["markersize"],
                    markeredgewidth=secondary_style["markeredgewidth"],
                    linewidth=secondary_style["linewidth"],
                    linestyle=secondary_style.get("linestyle", "-"),
                    color=secondary_style["color"],
                    alpha=secondary_style["alpha"],
                    label=range2_label,  # CHANGED
                )[0]

        # Modern legend styling if dual range
        if plot_data.use_dual_range:
            ax.legend(
                loc="best",
                frameon=True,
                fancybox=False,
                shadow=False,
                framealpha=0.95,
                edgecolor="#D0D0D0",
                facecolor="white",
                fontsize=9,
            )

        # Apply axis padding with subtle animation-ready margins
        AnalysisPlotter._apply_axis_padding(ax, x_data, y_data)

    @staticmethod
    def _apply_axis_padding(
        ax: Axes, x_data: np.ndarray, y_data: np.ndarray, padding_factor: float = 0.05
    ) -> None:
        """
        Apply padding to both axes for improved visualization and layout.

        Args:
            ax (Axes): Matplotlib Axes to adjust.
            x_data (np.ndarray): X-axis data points.
            y_data (np.ndarray): Y-axis data points.
            padding_factor (float, optional): Fractional padding to apply. Defaults to 0.05.
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
    def save_figure(figure: Figure, filepath: str, dpi: int = 300) -> None:
        """
        Save a matplotlib figure to a file.

        Args:
            figure (Figure): Matplotlib figure to save.
            filepath (str): Output file path.
            dpi (int, optional): Resolution in dots per inch. Defaults to 300.

        Note:
            File I/O may require external synchronization if multiple threads write to the same directory.
        """
        figure.tight_layout()
        figure.savefig(filepath, dpi=dpi, bbox_inches="tight")

    @staticmethod
    def create_and_save_plot(
        plot_data: AnalysisPlotData,
        x_label: str,
        y_label: str,
        title: str,
        filepath: str,
        figsize: Tuple[int, int] = (8, 6),
        dpi: int = 300,
    ) -> Figure:
        """
        Create and save an analysis plot in a single operation.

        Args:
            plot_data (AnalysisPlotData): Data to plot.
            x_label (str): X-axis label.
            y_label (str): Y-axis label.
            title (str): Plot title.
            filepath (str): Output file path for saving the plot.
            figsize (Tuple[int, int], optional): Figure size in inches. Defaults to (8, 6).
            dpi (int, optional): Resolution in dots per inch. Defaults to 300.

        Returns:
            Figure: The created matplotlib Figure object.
        """
        figure, _ = AnalysisPlotter.create_figure(
            plot_data, x_label, y_label, title, figsize
        )
        AnalysisPlotter.save_figure(figure, filepath, dpi)
        return figure


# CLI-friendly functions updated for stateless operation
def create_analysis_plot(
    plot_data_dict: Dict[str, Any],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[Figure]:
    """
    Create an analysis plot from a data dictionary and optionally save or display it.

    Args:
        plot_data_dict (Dict[str, Any]): Dictionary containing plot data fields.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        title (str): Title of the plot.
        output_path (Optional[str], optional): Path to save the plot. If None, plot is not saved.
        show (bool, optional): Whether to display the plot (requires GUI backend). Defaults to False.

    Returns:
        Optional[Figure]: The created matplotlib Figure object if successful, None otherwise.

    Note:
        Displaying plots with show=True is not thread-safe and should only be called from the main thread.
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
        fig, ax = AnalysisPlotter.create_figure(plot_data, x_label, y_label, title)

    if show:
        # Note: This requires GUI backend and is NOT thread-safe
        # Should only be called from main thread
        import warnings

        warnings.warn(
            "Displaying plots with show=True is not thread-safe. "
            "Use only from main thread.",
            RuntimeWarning,
        )
        plt.show()

    return fig
