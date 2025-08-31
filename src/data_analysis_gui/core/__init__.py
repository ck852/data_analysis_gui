# src/data_analysis_gui/core/__init__.py
"""
Core business logic module for the data analysis GUI.
"""

from .channel_definitions import ChannelDefinitions
from .current_density_exporter import CurrentDensityExporter

__all__ = ['ChannelDefinitions', 'CurrentDensityExporter']