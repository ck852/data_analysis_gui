# config/plot_style.py
"""
Modern matplotlib style configuration for scientific plots.
Provides a clean, professional appearance that complements the GUI.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any

# Define a modern scientific color palette
COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',     # Muted purple
    'accent': '#F18F01',        # Orange
    'success': '#73AB84',       # Sage green
    'warning': '#C73E1D',       # Rust red
    'info': '#6C91BF',          # Light blue
    'dark': '#2D3436',          # Near black
    'light': '#F7F9FB',         # Off white
    'grid': '#E1E5E8',          # Light gray
}

# Color cycle for multiple data series
COLOR_CYCLE = [
    '#2E86AB',  # Deep blue
    '#A23B72',  # Muted purple
    '#73AB84',  # Sage green
    '#F18F01',  # Orange
    '#C73E1D',  # Rust red
    '#6C91BF',  # Light blue
    '#8B6F90',  # Dusty purple
    '#4A7C59',  # Forest green
]

def get_plot_style() -> Dict[str, Any]:
    """
    Get matplotlib rcParams for modern scientific plots.
    
    Returns:
        Dictionary of matplotlib rcParams
    """
    return {
        # Figure
        'figure.facecolor': '#FAFAFA',
        'figure.edgecolor': 'none',
        'figure.frameon': True,
        'figure.autolayout': False,
        'figure.dpi': 100,
        'figure.titlesize': 12,
        'figure.titleweight': 'normal',
        
        # Axes
        'axes.facecolor': 'white',
        'axes.edgecolor': '#B0B0B0',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.titlesize': 11,
        'axes.titleweight': 'normal',
        'axes.titlepad': 8,
        'axes.labelsize': 10,
        'axes.labelweight': 'normal',
        'axes.labelcolor': '#2D3436',
        'axes.axisbelow': True,
        'axes.prop_cycle': mpl.cycler(color=COLOR_CYCLE),
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.formatter.use_mathtext': True,
        'axes.formatter.useoffset': False,
        
        # Grid
        'grid.color': '#E1E5E8',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.antialiased': True,
        'lines.markersize': 5,
        'lines.markeredgewidth': 0,
        'lines.markeredgecolor': 'auto',
        'lines.solid_capstyle': 'round',
        
        # Patches
        'patch.linewidth': 0,
        'patch.facecolor': COLORS['primary'],
        'patch.edgecolor': 'none',
        'patch.antialiased': True,
        
        # Ticks
        'xtick.major.size': 4,
        'xtick.minor.size': 2,
        'xtick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'xtick.major.pad': 5,
        'xtick.minor.pad': 5,
        'xtick.color': '#606060',
        'xtick.labelsize': 9,
        'xtick.direction': 'out',
        'xtick.top': False,
        
        'ytick.major.size': 4,
        'ytick.minor.size': 2,
        'ytick.major.width': 0.8,
        'ytick.minor.width': 0.6,
        'ytick.major.pad': 5,
        'ytick.minor.pad': 5,
        'ytick.color': '#606060',
        'ytick.labelsize': 9,
        'ytick.direction': 'out',
        'ytick.right': False,
        
        # Font
        'font.family': ['sans-serif'],
        'font.sans-serif': ['Segoe UI', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'font.weight': 'normal',
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#D0D0D0',
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
        'legend.markerscale': 1.0,
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        'legend.borderpad': 0.4,
        'legend.columnspacing': 1.0,
        'legend.loc': 'best',
        
        # Savefig
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Error bars
        'errorbar.capsize': 3,
        
        # Scatter plots
        'scatter.marker': 'o',
        'scatter.edgecolors': 'face',
        
        # Histogram
        'hist.bins': 'auto',
    }

def apply_plot_style():
    """Apply the modern scientific plot style to matplotlib."""
    plt.rcParams.update(get_plot_style())

def style_axis(ax, title: str = None, xlabel: str = None, ylabel: str = None,
               remove_top_right: bool = True):
    """
    Apply consistent styling to a single axis.
    
    Args:
        ax: Matplotlib axis object
        title: Optional title for the plot
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
        remove_top_right: Whether to remove top and right spines
    """
    if title:
        ax.set_title(title, fontsize=11, fontweight='normal', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontweight='normal')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight='normal')
    
    if remove_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Subtle spine styling
    for spine in ax.spines.values():
        if spine.get_visible():
            spine.set_linewidth(0.8)
            spine.set_color('#B0B0B0')
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#E1E5E8')
    ax.set_axisbelow(True)
    
    # Tick styling
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=9,
        colors='#606060',
        length=4,
        width=0.8,
        direction='out'
    )

def get_line_styles():
    """
    Get consistent line styles for different plot types.
    
    Returns:
        Dictionary of line style configurations
    """
    return {
        'primary': {
            'color': COLORS['primary'],
            'linewidth': 1.5,
            'marker': 'o',
            'markersize': 4,
            'markeredgewidth': 0,
            'alpha': 0.9
        },
        'secondary': {
            'color': COLORS['secondary'],
            'linewidth': 1.5,
            'marker': 's',
            'markersize': 4,
            'markeredgewidth': 0,
            'alpha': 0.9,
            'linestyle': '--'
        },
        'range_line': {
            'linewidth': 2,
            'alpha': 0.7,
            'linestyle': '-'
        },
        'range1': {
            'color': '#73AB84',  # Sage green
            'linewidth': 2,
            'alpha': 0.7,
            'linestyle': '-'
        },
        'range2': {
            'color': '#C73E1D',  # Rust red
            'linewidth': 2,
            'alpha': 0.7,
            'linestyle': '-'
        }
    }

def format_sweep_plot(ax, sweep_index: int, channel_type: str):
    """
    Apply specific formatting for sweep plots.
    
    Args:
        ax: Matplotlib axis
        sweep_index: The sweep number
        channel_type: "Voltage" or "Current"
    """
    unit = "mV" if channel_type == "Voltage" else "pA"
    
    style_axis(
        ax,
        title=f"Sweep {sweep_index} - {channel_type}",
        xlabel="Time (ms)",
        ylabel=f"{channel_type} ({unit})"
    )
    
    # Make sweep plots slightly different
    ax.set_facecolor('#FAFBFC')

def format_analysis_plot(ax, x_label: str, y_label: str, title: str = None):
    """
    Apply specific formatting for analysis plots.
    
    Args:
        ax: Matplotlib axis
        x_label: X-axis label
        y_label: Y-axis label
        title: Optional plot title
    """
    style_axis(ax, title=title, xlabel=x_label, ylabel=y_label)
    
    # Analysis plots get a subtle background
    ax.set_facecolor('#FFFFFF')

def format_batch_plot(ax, x_label: str, y_label: str):
    """
    Apply specific formatting for batch plots with multiple series.
    
    Args:
        ax: Matplotlib axis
        x_label: X-axis label
        y_label: Y-axis label
    """
    style_axis(ax, xlabel=x_label, ylabel=y_label)
    
    # Batch plots need clear differentiation
    ax.set_facecolor('#FFFFFF')
    
    # Ensure legend is well-positioned
    ax.legend(
        loc='best',
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        edgecolor='#D0D0D0',
        fontsize=8
    )