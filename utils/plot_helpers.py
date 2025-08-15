import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def setup_plot_style(ax, title="", xlabel="", ylabel="", grid=True):
    """Configure plot appearance.
    Common plot setup used throughout the code"""
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)


def add_range_indicators(ax, ranges, colors=None):
    """Add shaded regions and boundary lines for ranges.
    Extracted from ConcentrationResponseDialog.draw_range_indicators()"""
    if colors is None:
        colors = {
            'analysis': {'line': '#2E7D32', 'fill': (0.18, 0.49, 0.20, 0.2)},
            'background': {'line': '#1565C0', 'fill': (0.08, 0.40, 0.75, 0.2)}
        }
    
    patches = []
    lines = []
    
    for range_info in ranges:
        color_set = colors[range_info['type']]
        start, end = range_info['start'], range_info['end']
        
        # Add shaded region
        ylim = ax.get_ylim()
        patch = mpatches.Rectangle(
            (start, ylim[0]), 
            end - start,
            ylim[1] - ylim[0],
            facecolor=color_set['fill'],
            edgecolor='none', 
            zorder=1
        )
        ax.add_patch(patch)
        patches.append(patch)
        
        # Add boundary lines
        start_line = ax.axvline(start, color=color_set['line'], 
                                ls='--', lw=1.5, picker=5, alpha=0.7)
        end_line = ax.axvline(end, color=color_set['line'],
                             ls='--', lw=1.5, picker=5, alpha=0.7)
        lines.extend([start_line, end_line])
    
    return patches, lines


def update_range_lines(lines, positions):
    """Update positions of range indicator lines.
    From update_lines_from_entries()"""
    for line, pos in zip(lines, positions):
        line.set_xdata([pos, pos])


def add_padding_to_axes(ax, x_padding_pct=0.05, y_padding_pct=0.05):
    """Add padding to plot axes.
    From multiple plot methods for consistent padding"""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_padding = x_range * x_padding_pct if x_range > 0 else 0.1
    y_padding = y_range * y_padding_pct if y_range > 0 else 0.1
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)


def create_batch_figure(title, xlabel, ylabel):
    """Create a figure for batch analysis.
    From batch_analyze()"""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax