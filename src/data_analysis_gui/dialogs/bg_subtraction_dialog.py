"""
Background Subtraction Dialog for PatchBatch Electrophysiology Data Analysis Tool
Refactored to use BackgroundSubtractionService for business logic separation.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import numpy as np


from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QMessageBox, QFormLayout
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from data_analysis_gui.config.themes import apply_modern_theme, create_styled_button
from data_analysis_gui.config.plot_style import (apply_plot_style, style_axis, get_line_styles, 
                                                 add_zero_axis_lines, COLORS
)
from data_analysis_gui.widgets.custom_inputs import SelectAllSpinBox
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.data_extractor import DataExtractor
from data_analysis_gui.services.bg_subtraction_service import BackgroundSubtractionService
from data_analysis_gui.widgets.cursor_spinbox import CursorSpinbox

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class BackgroundSubtractionDialog(QDialog):
    """
    Dialog for defining background range and applying background subtraction.
    Uses BackgroundSubtractionService for business logic separation.
    """
    
    def __init__(self, dataset: ElectrophysiologyDataset, sweep_index: str, parent=None):
        """
        Initialize the background subtraction dialog with centralized styling.
        
        Args:
            dataset: The electrophysiology dataset
            sweep_index: Current sweep index to display
            default_start: Default start time for background range (ms)
            default_end: Default end time for background range (ms)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.dataset = dataset
        self.sweep_index = sweep_index
        self.data_extractor = DataExtractor()
        
        self.setWindowTitle("Background Subtraction")
        self.setModal(True)
        self.setFixedSize(600, 500)
        
        # Apply global plot style for consistency
        apply_plot_style()
        
        # Get styling constants from plot_style.py
        self.colors = COLORS
        self.line_styles = get_line_styles()
        
        # Get current channel data for the plot
        self.time_ms, self.current_data = self._get_current_channel_data()
        
        if self.time_ms is None or self.current_data is None:
            QMessageBox.warning(self, "No Data", "No current channel data available.")
            self.reject()
            return
            
        self.max_time = self.time_ms[-1] if len(self.time_ms) > 0 else 10000
        
        self._init_ui()
        self._connect_signals()
        self._update_plot()
        
        # Apply modern theme
        apply_modern_theme(self)
        
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Create matplotlib plot
        self._create_plot()
        layout.addWidget(self.canvas)
        
        # Range controls
        range_widget = QWidget()
        range_layout = QFormLayout(range_widget)
        range_layout.setSpacing(8)

        self.default_start = 5  # Default start time for background range (ms)
        self.default_end = 45  # Default end time for background range (ms)
        
        self.start_spinbox = SelectAllSpinBox()
        self.start_spinbox.setDecimals(1)
        self.start_spinbox.setSuffix(" ms")
        self.start_spinbox.setMinimum(0)
        self.start_spinbox.setMaximum(self.max_time)
        self.start_spinbox.setValue(self.default_start)
        self.start_spinbox.setMinimumWidth(120)
        
        self.end_spinbox = SelectAllSpinBox()
        self.end_spinbox.setDecimals(1)
        self.end_spinbox.setSuffix(" ms")
        self.end_spinbox.setMinimum(0)
        self.end_spinbox.setMaximum(self.max_time)
        self.end_spinbox.setValue(self.default_end)
        self.end_spinbox.setMinimumWidth(120)
        
        range_layout.addRow("Background Start:", self.start_spinbox)
        range_layout.addRow("Background End:", self.end_spinbox)
        
        # Add cursor manager for draggable cursors
        self.cursor_manager = CursorSpinbox(self.ax, self.canvas)
        self.cursor_manager.add_cursor("start", self.start_spinbox, self.default_start, color="#73AB84")
        self.cursor_manager.add_cursor("end", self.end_spinbox, self.default_end, color="#73AB84")
        self.cursor_manager.enable_shading(alpha=0.1)


        layout.addWidget(range_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = create_styled_button("Cancel", "secondary")
        self.apply_button = create_styled_button("Apply Background Subtraction", "primary")
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.apply_button)
        
        layout.addLayout(button_layout)
        
    def _create_plot(self):
        """Create the matplotlib plot widget with centralized styling."""
        self.figure = Figure(figsize=(7, 3.5), facecolor=self.colors["light"])
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Basic axis setup will be handled by style_axis() in _update_plot()
        
    def _connect_signals(self):
        """Connect UI signals to their handlers."""
        self.start_spinbox.valueChanged.connect(self._update_plot)
        self.end_spinbox.valueChanged.connect(self._update_plot)
        self.apply_button.clicked.connect(self._apply_background_subtraction)
        self.cancel_button.clicked.connect(self.reject)
        
    def _get_current_channel_data(self):
        """
        Get current channel data for the specified sweep.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time and current data arrays, or (None, None) if failed
        """
        try:
            sweep_data = self.data_extractor.extract_sweep_data(self.dataset, self.sweep_index)
            return sweep_data["time_ms"], sweep_data["current"]
        except Exception as e:
            logger.error(f"Failed to extract current channel data: {e}")
            return None, None
            
    def _update_plot(self):
        """Update the plot using centralized styling functions."""
        if self.time_ms is None or self.current_data is None:
            return
            
        self.ax.clear()
        
        # Plot current data
        primary_style = self.line_styles["primary"]
        self.ax.plot(
            self.time_ms, 
            self.current_data, 
            color=primary_style["color"],
            linewidth=primary_style["linewidth"], 
            alpha=primary_style["alpha"]
        )
        
        # Re-add cursor lines after clear
        for cursor_data in self.cursor_manager.cursors.values():
            self.ax.add_line(cursor_data['line'])
        
        # Re-add shading after clear - let CursorSpinbox handle it
        self.cursor_manager.recreate_shading_after_clear()
        
        # Style axis
        style_axis(
            self.ax,
            title=f"Current Trace - Sweep {self.sweep_index}",
            xlabel="Time (ms)",
            ylabel="Current (pA)",
            remove_top_right=True
        )
        
        self.ax.relim()
        self.ax.autoscale_view(tight=True)
        self.ax.margins(x=0.02, y=0.05)
        add_zero_axis_lines(self.ax)
        
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw_idle()
        
    def _validate_range(self):
        """
        Validate the background range using the service validation.
        
        Returns:
            bool: True if range is valid, False otherwise
        """
        start_time = self.start_spinbox.value()
        end_time = self.end_spinbox.value()
        
        # Use the service's validation method
        is_valid, error_message = BackgroundSubtractionService.validate_background_range(
            self.dataset, start_time, end_time
        )
        
        if not is_valid:
            QMessageBox.warning(self, "Invalid Range", error_message)
            return False
            
        # Additional check specific to the dialog - ensure range contains data points
        if len(self.time_ms) > 0:
            mask = (self.time_ms >= start_time) & (self.time_ms <= end_time)
            if not np.any(mask):
                QMessageBox.warning(self, "Invalid Range", 
                                  "Background range contains no data points.")
                return False
        
        return True
        
    def _apply_background_subtraction(self):
        """Apply background subtraction using the BackgroundSubtractionService."""
        if not self._validate_range():
            return
            
        try:
            start_time = self.start_spinbox.value()
            end_time = self.end_spinbox.value()
            
            # Show confirmation dialog
            num_sweeps = self.dataset.sweep_count()
            reply = QMessageBox.question(
                self, "Confirm Background Subtraction",
                f"Apply background subtraction to all {num_sweeps} sweeps?\n"
                f"Range: {start_time:.1f} - {end_time:.1f} ms\n\n"
                f"This will modify the dataset permanently.\n"
                f"To undo, you will need to reload the original file.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Apply background subtraction using the service
            result = BackgroundSubtractionService.apply_background_subtraction(
                self.dataset, start_time, end_time
            )
            
            if result.success:
                logger.info(f"Background subtraction applied: range [{start_time}, {end_time}] ms "
                           f"to {result.processed_sweeps}/{result.total_sweeps} sweeps")
                
                
                self.accept()
            else:
                error_message = f"Background subtraction failed: {result.error_message}"
                if result.failed_sweeps:
                    error_message += f"\n\nFailed sweeps: {result.failed_sweeps}"
                    
                logger.error(error_message)
                QMessageBox.critical(self, "Error", error_message)
                
        except Exception as e:
            logger.error(f"Unexpected error during background subtraction: {e}")
            QMessageBox.critical(
                self, "Unexpected Error", 
                f"An unexpected error occurred: {str(e)}\n\n"
                f"Please check the logs for more details."
            )