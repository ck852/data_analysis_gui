# core/backends/qt_backend.py
"""
Qt-based plotting backend for GUI environments.
"""

from typing import Tuple
from io import BytesIO

from data_analysis_gui.core.plotting_interface import PlotBackend


class QtPlotBackend(PlotBackend):
    """Qt backend for plotting operations in GUI mode"""
    
    def __init__(self):
        """Initialize Qt backend"""
        # Import here to avoid import errors in headless environments
        import matplotlib
        matplotlib.use('Qt5Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        
        # Store classes for later use
        self._Figure = Figure
        self._FigureCanvas = FigureCanvasQTAgg
    
    def create_figure(self, figsize: Tuple[float, float]):
        """
        Create a matplotlib figure with Qt backend.
        
        Args:
            figsize: Tuple of (width, height) in inches
            
        Returns:
            matplotlib.figure.Figure instance
        """
        return self._Figure(figsize=figsize)
    
    def create_canvas(self, figure):
        """
        Create a Qt canvas for the figure.
        
        Args:
            figure: matplotlib.figure.Figure instance
            
        Returns:
            FigureCanvasQTAgg instance
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
        return "Qt5Agg"