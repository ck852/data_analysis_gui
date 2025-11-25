"""
PatchBatch Electrophysiology Data Analysis Tool

Reusable visual style definitions for matplotlib plots. 

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any

from data_analysis_gui.config.themes import TYPOGRAPHY

# Define a modern scientific color palette
COLORS = {
    "primary": "#2E86AB",  # Deep blue
    "secondary": "#A23B72",  # Muted purple
    "accent": "#F18F01",  # Orange
    "success": "#73AB84",  # Sage green
    "warning": "#C73E1D",  # Rust red
    "info": "#6C91BF",  # Light blue
    "dark": "#2D3436",  # Near black
    "light": "#F7F9FB",  # Off white
    "grid": "#E1E5E8",  # Light gray
}

# Color cycle for multiple data series
COLOR_CYCLE = [
    "#2E86AB",  # Deep blue
    "#A23B72",  # Muted purple
    "#73AB84",  # Sage green
    "#F18F01",  # Orange
    "#C73E1D",  # Rust red
    "#6C91BF",  # Light blue
    "#8B6F90",  # Dusty purple
    "#4A7C59",  # Forest green
]

# TOOLBAR CONFIGURATION - Centralized here
TOOLBAR_CONFIG = {
    "button_font_size": 12,
    "button_padding": "4px 8px",
    "icon_size_multiplier": 1.0,
    "mode_label_font_size": 10,
    "button_min_height": 22,
    "icon_size": 20,
}


def get_plot_style() -> Dict[str, Any]:

    # Extract font family from theme (handle the CSS font-family string)
    font_family_str = TYPOGRAPHY["font_family"]
    font_list = [f.strip() for f in font_family_str.split(",")]
    # Filter out generic families and -apple-system
    font_list = [
        f
        for f in font_list
        if f not in ["-apple-system", "BlinkMacSystemFont", "sans-serif"]
    ]

    # INCREASED FONT SIZES FOR BETTER READABILITY
    plot_font_sizes = {
        "tick_size": 9,
        "label_size": 10,
        "title_size": 12,
        "legend_size": 9,
    }

    return {
        # Figure
        "figure.facecolor": "#FAFAFA",
        "figure.edgecolor": "none",
        "figure.frameon": True,
        "figure.autolayout": False,
        "figure.dpi": 100,
        "figure.titlesize": plot_font_sizes["title_size"],
        "figure.titleweight": "normal",
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#B0B0B0",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.titlesize": plot_font_sizes["title_size"],
        "axes.titleweight": "normal",
        "axes.titlepad": 10,
        "axes.labelsize": plot_font_sizes["label_size"],
        "axes.labelweight": "normal",
        "axes.labelcolor": "#2D3436",
        "axes.labelpad": 6,  # Added explicit padding
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.use_mathtext": True,
        "axes.formatter.useoffset": False,
        # Grid
        "grid.color": "#E1E5E8",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        # Lines
        "lines.linewidth": 1.5,
        "lines.antialiased": True,
        "lines.markersize": 6,
        "lines.markeredgewidth": 0,
        "lines.markeredgecolor": "auto",
        "lines.solid_capstyle": "round",
        # Patches
        "patch.linewidth": 0,
        "patch.facecolor": COLORS["primary"],
        "patch.edgecolor": "none",
        "patch.antialiased": True,
        # Ticks
        "xtick.major.size": 5,
        "xtick.minor.size": 3,
        "xtick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "xtick.major.pad": 6,
        "xtick.minor.pad": 6,
        "xtick.color": "#606060",
        "xtick.labelsize": plot_font_sizes["tick_size"],
        "xtick.direction": "out",
        "xtick.top": False,
        "ytick.major.size": 5,
        "ytick.minor.size": 3,
        "ytick.major.width": 0.8,
        "ytick.minor.width": 0.6,
        "ytick.major.pad": 6,
        "ytick.minor.pad": 6,
        "ytick.color": "#606060",
        "ytick.labelsize": plot_font_sizes["tick_size"],
        "ytick.direction": "out",
        "ytick.right": False,
        # Font - synchronized with theme
        "font.family": ["sans-serif"],
        "font.sans-serif": font_list + ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": plot_font_sizes["label_size"],  # Base size from theme
        "font.weight": "normal",
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.edgecolor": "#D0D0D0",
        "legend.fancybox": False,
        "legend.shadow": False,
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,
        "legend.markerscale": 1.0,
        "legend.fontsize": plot_font_sizes["legend_size"],
        "legend.title_fontsize": plot_font_sizes["label_size"],
        "legend.borderpad": 0.5,  # Increased slightly
        "legend.columnspacing": 1.2,
        "legend.loc": "best",
        # Savefig
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        # Error bars
        "errorbar.capsize": 3,
        # Scatter plots
        "scatter.marker": "o",
        "scatter.edgecolors": "face",
        # Histogram
        "hist.bins": "auto",
    }


def apply_plot_style():
    """
    Updates matplotlib's rcParams with the style returned by get_plot_style().
    """
    plt.rcParams.update(get_plot_style())

def add_zero_axis_lines(ax, color=None, linewidth=0.8, alpha=0.4, linestyle=':'):
    """Add prominent gridlines at x=0 and y=0."""
    if color is None:
        color = COLORS["dark"]
    ax.axhline(y=0, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=1)
    ax.axvline(x=0, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=1)

def style_axis(
    ax,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    remove_top_right: bool = True,
):
    """
    Applies consistent styling to a single matplotlib axis using theme settings.
    """
    # Use the increased font sizes
    plot_font_sizes = {
        "tick_size": 9,
        "label_size": 10,
        "title_size": 12,
        "legend_size": 9,
    }

    if title:
        ax.set_title(
            title, fontsize=plot_font_sizes["title_size"], fontweight="normal", pad=10
        )
    if xlabel:
        ax.set_xlabel(
            xlabel, fontsize=plot_font_sizes["label_size"], fontweight="normal"
        )
    if ylabel:
        ax.set_ylabel(
            ylabel, fontsize=plot_font_sizes["label_size"], fontweight="normal"
        )

    if remove_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Subtle spine styling
    for spine in ax.spines.values():
        if spine.get_visible():
            spine.set_linewidth(0.8)
            spine.set_color("#B0B0B0")

    # Grid styling
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="#E1E5E8")
    ax.set_axisbelow(True)

    # Tick styling with increased font sizes
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=plot_font_sizes["tick_size"],
        colors="#606060",
        length=5,
        width=0.8,
        direction="out",
    )


def get_line_styles():
    """
    Returns a dictionary of consistent line styles for different plot types.
    """
    return {
        "primary": {
            "color": COLORS["primary"],
            "linewidth": 0.9,
            "marker": "o",
            "markersize": 5,
            "markeredgewidth": 0,
            "alpha": 1.0,
        },
        "secondary": {
            "color": COLORS["secondary"],
            "linewidth": 1.5,
            "marker": "s",
            "markersize": 5,
            "markeredgewidth": 0,
            "alpha": 1.0,
            "linestyle": "--",
        },
        "range_line": {"linewidth": 2, "alpha": 0.7, "linestyle": "-"},
        "range1": {
            "color": "#32CD32",  # Sage green
            "linewidth": 1.2,
            "alpha": 1.0,
            "linestyle": "-",
        },
        "range2": {
            "color": "#C73E1D",  # Rust red
            "linewidth": 1.2,
            "alpha": 1.0,
            "linestyle": "-",
        },
    }


def format_sweep_plot(ax, sweep_index: int, channel_type: str):

    unit = "mV" if channel_type == "Voltage" else "pA"

    style_axis(
        ax,
        title=f"Sweep {sweep_index} - {channel_type}",
        xlabel="Time (ms)",
        ylabel=f"{channel_type} ({unit})",
    )

    # Make sweep plots slightly different
    ax.set_facecolor("#FAFBFC")


def format_analysis_plot(ax, x_label: str, y_label: str, title: str = None):

    style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)

    # Analysis plots get a subtle background
    ax.set_facecolor("#FFFFFF")


def format_batch_plot(ax, x_label: str, y_label: str):
    """
    Applies specific formatting for batch plots with multiple series, using theme fonts and legend styling.
    """
    style_axis(ax, xlabel=x_label, ylabel=y_label)

    # Batch plots need clear differentiation
    ax.set_facecolor("#FFFFFF")

    # Get the increased font sizes
    plot_font_sizes = {
        "legend_size": 12,
    }

    # Ensure legend is well-positioned with increased font size
    if ax.get_lines():  # Check if there are any lines plotted
        ax.legend(
            loc="best",
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=0.95,
            edgecolor="#D0D0D0",
            fontsize=plot_font_sizes["legend_size"],
        )


def get_toolbar_style() -> str:
    """
    Returns the CSS stylesheet string for the plot toolbar, with proper font sizes and spacing.

    Centralized here to maintain a single source of truth for plot-related styling.

    Returns:
        str: CSS stylesheet string for use with Qt toolbars.
    """
    config = TOOLBAR_CONFIG

    return f"""
        QToolBar {{
            background-color: #F5F5F5;
            border: none;
            border-bottom: 1px solid #D0D0D0;
            padding: 4px;
            spacing: 4px;
        }}
        
        QToolBar::separator {{
            background-color: #D0D0D0;
            width: 1px;
            margin: 6px 8px;
        }}
        
        QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 3px;
            padding: {config['button_padding']};
            margin: 2px;
            font-size: {config['button_font_size']}px;
            font-weight: 500;
            min-height: {config['button_min_height']}px;
        }}
        
        QToolButton:hover {{
            background-color: #E0E0E0;
            border: 1px solid #C0C0C0;
        }}
        
        QToolButton:pressed {{
            background-color: #D0D0D0;
            border: 1px solid #B0B0B0;
        }}
        
        QToolButton:checked {{
            background-color: #D8E4F0;
            border: 1px solid #2E86AB;
        }}
        
        QLabel {{
            color: #606060;
            font-size: {config['mode_label_font_size']}px;
            margin: 0px 10px;
        }}
    """
