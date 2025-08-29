# core/plotting_interface.py
"""
Abstract plotting interface for backend-agnostic plotting operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class PlotBackend(ABC):
    """Abstract base class for plotting backends"""
    
    @abstractmethod
    def create_figure(self, figsize: Tuple[float, float]) -> Any:
        """Create a new figure with given size"""
        pass
    
    @abstractmethod
    def create_canvas(self, figure: Any) -> Any:
        """Create a canvas for the figure"""
        pass
    
    @abstractmethod
    def save_figure(self, figure: Any, format: str = 'png') -> bytes:
        """Save figure to bytes in specified format"""
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of this backend"""
        pass


class PlotBackendFactory:
    """Factory for creating appropriate plot backends based on environment"""
    
    @staticmethod
    def create_backend() -> PlotBackend:
        """
        Create the appropriate backend based on the current environment.
        
        Returns:
            PlotBackend instance suitable for the environment
        """
        from data_analysis_gui.config.environment import Environment
        
        if Environment.is_headless():
            from data_analysis_gui.core.backends.headless_backend import HeadlessPlotBackend
            return HeadlessPlotBackend()
        else:
            # Try to import Qt backend, fall back to headless if it fails
            try:
                from data_analysis_gui.core.backends.qt_backend import QtPlotBackend
                return QtPlotBackend()
            except ImportError:
                # Qt not available, use headless backend
                from data_analysis_gui.core.backends.headless_backend import HeadlessPlotBackend
                return HeadlessPlotBackend()