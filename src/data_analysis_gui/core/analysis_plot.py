"""
PatchBatch Electrophysiology Data Analysis Tool

Matplotlib-based plotting module. Main use is to display the plot called
by "Generate Analysis Plot" in MainWindow. Also used to export plots to image files.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from data_analysis_gui.config.plot_style import (
    apply_plot_style,
    format_analysis_plot,
    get_line_styles,
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisPlotData:
    """
    Container for analysis plot data with optional dual-range support.
    
    y_label_r1/r2 store voltage-annotated legend labels (e.g., "Range 1 (-70 mV)")
    when X-axis is Time and Y-axis is Current.
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
        """Create from dictionary. Used for backward compatibility."""
        logger.debug("Creating AnalysisPlotData from dictionary")
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
    Stateless plotting utilities using matplotlib's Agg backend.
    
    All methods are static. Use returned Figure objects for GUI canvas or export.
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
        Create a styled matplotlib figure ready for display or export.
        
        Applies styling from plot_style config and handles dual-range plots
        with distinct visual styles for each range.

        Returns:
            Tuple[Figure, Axes]: Use figure.savefig() to export or embed in GUI canvas.
        """
        logger.debug(f"Creating analysis figure: {title}, size={figsize}")
        
        apply_plot_style()

        figure = Figure(figsize=figsize, facecolor="#FAFAFA")
        ax = figure.add_subplot(111)

        # Extract voltage-annotated labels from plot_data for legend (very nice to have for time-course plots)
        y_label_r1 = plot_data.y_label_r1
        y_label_r2 = plot_data.y_label_r2

        logger.debug(
            f"Configuring plot with {len(plot_data.x_data)} data points, "
            f"dual_range={plot_data.use_dual_range}"
        )

        # Configure plot with data and v labels
        AnalysisPlotter._configure_plot(
            ax, 
            plot_data, 
            y_label_r1=y_label_r1,
            y_label_r2=y_label_r2
        )

        # Apply analysis-specific formatting
        format_analysis_plot(ax, x_label, y_label, title)

        # Ensure proper layout
        figure.tight_layout(pad=1.5)

        logger.info(
            f"Created analysis figure: '{title}' with {len(plot_data.sweep_indices)} sweeps"
        )

        return figure, ax

    @staticmethod
    def _configure_plot(
        ax: Axes, 
        plot_data: AnalysisPlotData,
        y_label_r1: Optional[str] = None,
        y_label_r2: Optional[str] = None
    ) -> None:
        """
        Plot data with styling from plot_style config.
        
        Voltage-annotated labels in legend for time-course plots.
        """
        x_data = plot_data.x_data
        y_data = plot_data.y_data

        line_styles = get_line_styles()

        if len(x_data) > 0 and len(y_data) > 0:
            # Use voltage-annotated label if provided, otherwise fallback
            range1_label = y_label_r1 or "Range 1"
            
            logger.debug(f"Plotting Range 1 data: {len(x_data)} points, label='{range1_label}'")
            
            # Plot Range 1
            primary_style = line_styles["primary"]
            ax.plot(
                x_data,
                y_data,
                marker=primary_style["marker"],
                markersize=primary_style["markersize"],
                markeredgewidth=primary_style["markeredgewidth"],
                linewidth=primary_style["linewidth"],
                color=primary_style["color"],
                alpha=primary_style["alpha"],
                label=range1_label,
            )

        # Plot Range 2 if applicable with contrasting style
        if plot_data.use_dual_range and plot_data.y_data2 is not None:
            y_data2 = plot_data.y_data2
            if len(x_data) > 0 and len(y_data2) > 0:
                # Use voltage-annotated label if provided, otherwise fallback
                range2_label = y_label_r2 or "Range 2"
                
                logger.debug(f"Plotting Range 2 data: {len(y_data2)} points, label='{range2_label}'")
                
                secondary_style = line_styles["secondary"]
                ax.plot(
                    x_data,
                    y_data2,
                    marker=secondary_style["marker"],
                    markersize=secondary_style["markersize"],
                    markeredgewidth=secondary_style["markeredgewidth"],
                    linewidth=secondary_style["linewidth"],
                    linestyle=secondary_style.get("linestyle", "-"),
                    color=secondary_style["color"],
                    alpha=secondary_style["alpha"],
                    label=range2_label,
                )

        # Dual range legend
        if plot_data.use_dual_range:
            logger.debug("Adding legend for dual-range plot")
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

        AnalysisPlotter._apply_axis_padding(ax, x_data, y_data)

    @staticmethod
    def _apply_axis_padding(
        ax: Axes, x_data: np.ndarray, y_data: np.ndarray, padding_factor: float = 0.05
    ) -> None:

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

            logger.debug(
                f"Applied axis padding: X=[{x_min - x_padding:.2f}, {x_max + x_padding:.2f}], "
                f"Y=[{y_min - y_padding_bottom:.2f}, {y_max + y_padding_top:.2f}]"
            )

            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding_bottom, y_max + y_padding_top)

    @staticmethod
    def save_figure(figure: Figure, filepath: str, dpi: int = 300) -> None:
        """Save figure to file with tight layout."""
        logger.debug(f"Saving figure to {filepath} at {dpi} DPI")
        
        try:
            figure.tight_layout()
            figure.savefig(filepath, dpi=dpi, bbox_inches="tight")
            logger.info(f"Successfully saved figure to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save figure to {filepath}: {e}", exc_info=True)
            raise

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
        """Create and save a plot in one operation."""
        logger.info(f"Creating and saving plot: '{title}' to {filepath}")
        
        figure, _ = AnalysisPlotter.create_figure(
            plot_data, x_label, y_label, title, figsize
        )
        AnalysisPlotter.save_figure(figure, filepath, dpi)
        
        return figure
