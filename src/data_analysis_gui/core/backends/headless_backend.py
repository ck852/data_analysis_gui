# core/backends/headless_backend.py
"""
Headless plotting backend for CI/testing environments.
"""

from typing import Tuple
from io import BytesIO

from data_analysis_gui.core.plotting_interface import PlotBackend


class HeadlessPlotBackend(PlotBackend):
    """Headless backend for plotting operations in CI/testing"""
    
    def __init__(self):
        """Initialize headless backend with Agg matplotlib backend"""
        # Set matplotlib to use non-interactive backend before importing
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # Store classes for later use
        self._Figure = Figure
        self._FigureCanvas = FigureCanvasAgg
    
    def create_figure(self, figsize: Tuple[float, float]):
        """
        Create a matplotlib figure with Agg backend.
        
        Args:
            figsize: Tuple of (width, height) in inches
            
        Returns:
            matplotlib.figure.Figure instance
        """
        return self._Figure(figsize=figsize)
    
    def create_canvas(self, figure):
        """
        Create an Agg canvas for the figure.
        
        Args:
            figure: matplotlib.figure.Figure instance
            
        Returns:
            FigureCanvasAgg instance
        """
        return self._FigureCanvas(figure)
    
    def save_figure(self, figure, format: str = 'png') -> bytes:
        """
        Save figure to bytes.
        
        Args:
            figure: matplotlib.figure.Figure instance
            format: Output format (png, pdf, svg, etc.)
            
        Returns:
            Bytes containing the figure data
        """
        buf = BytesIO()
        figure.savefig(buf, format=format, bbox_inches='tight')
        buf.seek(0)
        data = buf.read()
        buf.close()
        return data
    
    def get_backend_name(self) -> str:
        """Return the backend name"""
        return "Agg"