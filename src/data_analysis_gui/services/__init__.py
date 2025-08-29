# services/__init__.py
"""
Service layer for the application.
Services handle operations that coordinate between business logic and presentation.
"""

from data_analysis_gui.services.plot_service import PlotService

__all__ = ['PlotService']