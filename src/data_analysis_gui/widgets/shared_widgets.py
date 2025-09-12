"""
Shared widget components for batch analysis windows.

This module provides reusable components for displaying batch analysis results
with consistent behavior across different windows. It includes a dynamic plot
widget that updates smoothly without recreating figures, and a file list widget
that maintains selection state across windows.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Dict, List, Set, Optional, Tuple, Callable
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QCheckBox, QHeaderView, QLabel,
                             QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QPixmap, QPainter, QBrush

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from data_analysis_gui.core.models import FileAnalysisResult
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class FileSelectionState:
    """
    Manages file selection state that can be shared across windows.
    
    This class maintains a set of selected files and notifies observers
    when the selection changes, enabling synchronized selection across
    multiple UI components.
    """
    
    def __init__(self, initial_files: Optional[Set[str]] = None):
        """
        Initialize with optional set of initially selected files.
        
        Args:
            initial_files: Set of filenames to select initially
        """
        self._selected_files: Set[str] = initial_files.copy() if initial_files else set()
        self._observers: List[Callable[[Set[str]], None]] = []
        
    def toggle_file(self, filename: str, selected: bool) -> None:
        """Toggle selection state for a file."""
        if selected:
            self._selected_files.add(filename)
        else:
            self._selected_files.discard(filename)
        self._notify_observers()
    
    def set_files(self, filenames: Set[str]) -> None:
        """Set the complete selection state."""
        self._selected_files = filenames.copy()
        self._notify_observers()
    
    def is_selected(self, filename: str) -> bool:
        """Check if a file is selected."""
        return filename in self._selected_files
    
    def get_selected_files(self) -> Set[str]:
        """Get copy of currently selected files."""
        return self._selected_files.copy()
    
    def add_observer(self, callback: Callable[[Set[str]], None]) -> None:
        """Add observer to be notified of selection changes."""
        self._observers.append(callback)
    
    def remove_observer(self, callback: Callable[[Set[str]], None]) -> None:
        """Remove an observer."""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self) -> None:
        """Notify all observers of selection change."""
        selected = self.get_selected_files()
        for observer in self._observers:
            try:
                observer(selected)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")


class DynamicBatchPlotWidget(QWidget):
    """
    Reusable plot widget for batch results with smooth dynamic updates.
    
    This widget maintains a persistent matplotlib figure and updates only
    the data or visibility of plot lines, avoiding the flicker of complete
    redraws. It's designed to work with both batch results and current
    density displays.
    """
    
    # Signals
    plot_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the plot widget."""
        super().__init__(parent)
        
        # Plot components (created lazily)
        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvas] = None
        self.ax = None
        self.toolbar: Optional[NavigationToolbar] = None
        
        # Data management
        self.line_objects: Dict[str, Dict[str, Line2D]] = {}  # {filename: {range: Line2D}}
        self.file_colors: Dict[str, Tuple[float, ...]] = {}
        self.plot_initialized = False
        
        # Configuration
        self.use_dual_range = False
        self.x_label = "X"
        self.y_label = "Y"
        self.title = ""
        self.legend_fontsize = 8
        
        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Initialize with empty message
        self.empty_label = QLabel("No data to display")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.empty_label)
    
    def initialize_plot(self, x_label: str, y_label: str, title: str = "") -> None:
        """
        Initialize the plot with labels and title.
        
        Args:
            x_label: Label for x-axis
            y_label: Label for y-axis  
            title: Plot title (optional)
        """
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        
        if not self.plot_initialized:
            self._create_plot_components()
    
    def _create_plot_components(self) -> None:
        """Create matplotlib figure, canvas, and toolbar once."""
        # Remove empty label
        if self.empty_label:
            self.empty_label.setParent(None)
            self.empty_label = None
        
        # Create figure with constrained layout for better sizing
        self.figure = Figure(figsize=(12, 8), constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvas(self.figure)
        
        # Create toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add to layout
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        
        # Configure axes
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        if self.title:
            self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.3)
        
        self.plot_initialized = True
        logger.debug("Plot components created")
    
    def set_data(self, results: List[FileAnalysisResult], 
                 use_dual_range: bool = False,
                 color_mapping: Optional[Dict[str, Tuple[float, ...]]] = None) -> None:
        """
        Set the data to be plotted.
        
        Args:
            results: List of analysis results
            use_dual_range: Whether dual range data should be shown
            color_mapping: Optional pre-defined color mapping
        """
        if not self.plot_initialized:
            logger.warning("Plot not initialized. Call initialize_plot first.")
            return
        
        self.use_dual_range = use_dual_range
        
        # Generate color mapping if not provided
        if color_mapping is None:
            color_mapping = self._generate_color_mapping(results)
        self.file_colors = color_mapping
        
        # Clear existing lines
        for lines_dict in self.line_objects.values():
            for line in lines_dict.values():
                line.remove()
        self.line_objects.clear()
        
        # Create line objects for each result
        for result in results:
            self._create_lines_for_result(result)
        
        # Update plot appearance
        self._update_plot_appearance()
        
        # Draw
        self.canvas.draw_idle()
        self.plot_updated.emit()
    
    def _generate_color_mapping(self, results: List[FileAnalysisResult]) -> Dict[str, Tuple[float, ...]]:
        """Generate consistent color mapping for files."""
        # Use matplotlib color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        color_mapping = {}
        for idx, result in enumerate(results):
            color_str = colors[idx % len(colors)]
            # Convert to RGB tuple
            if color_str.startswith('#'):
                color = tuple(int(color_str[i:i+2], 16)/255 for i in (1, 3, 5))
            else:
                import matplotlib.colors as mcolors
                color = mcolors.to_rgb(color_str)
            color_mapping[result.base_name] = color
        
        return color_mapping
    
    def _create_lines_for_result(self, result: FileAnalysisResult) -> None:
        """Create line objects for a single result."""
        color = self.file_colors.get(result.base_name, (0, 0, 0))
        
        # Range 1 line
        if len(result.x_data) > 0 and len(result.y_data) > 0:
            line_r1, = self.ax.plot(
                result.x_data, result.y_data,
                'o-', label=f"{result.base_name}",
                markersize=4, alpha=0.8, color=color,
                visible=True  # Start visible
            )
            
            if result.base_name not in self.line_objects:
                self.line_objects[result.base_name] = {}
            self.line_objects[result.base_name]['range1'] = line_r1
        
        # Range 2 line if applicable
        if self.use_dual_range and result.y_data2 is not None:
            if len(result.x_data) > 0 and len(result.y_data2) > 0:
                line_r2, = self.ax.plot(
                    result.x_data if result.x_data2 is None else result.x_data2, 
                    result.y_data2,
                    's--', label=f"{result.base_name} (Range 2)",
                    markersize=4, alpha=0.8, color=color,
                    visible=True
                )
                self.line_objects[result.base_name]['range2'] = line_r2
    
    def update_visibility(self, selected_files: Set[str]) -> None:
        """
        Update line visibility based on selected files.
        
        Args:
            selected_files: Set of filenames that should be visible
        """
        if not self.plot_initialized:
            return
        
        # Update line visibility
        for filename, lines_dict in self.line_objects.items():
            visible = filename in selected_files
            for line in lines_dict.values():
                line.set_visible(visible)
        
        # Update legend to show only visible lines
        self._update_plot_appearance()
        
        # Redraw
        self.canvas.draw_idle()
        self.plot_updated.emit()
    
    def update_line_data(self, filename: str, y_data: np.ndarray, 
                        y_data2: Optional[np.ndarray] = None) -> None:
        """
        Update Y data for a specific file's lines.
        
        Args:
            filename: Name of file to update
            y_data: New Y data for range 1
            y_data2: New Y data for range 2 (if applicable)
        """
        if filename not in self.line_objects:
            logger.warning(f"No line objects for file: {filename}")
            return
        
        lines = self.line_objects[filename]
        
        # Update range 1
        if 'range1' in lines:
            lines['range1'].set_ydata(y_data)
        
        # Update range 2
        if self.use_dual_range and y_data2 is not None and 'range2' in lines:
            lines['range2'].set_ydata(y_data2)
        
        # Update axis limits if needed
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw
        self.canvas.draw_idle()
        self.plot_updated.emit()
    
    def _update_plot_appearance(self) -> None:
        """Update legend and other plot appearance elements."""
        # Get visible lines for legend
        visible_lines = []
        visible_labels = []
        
        for filename, lines_dict in self.line_objects.items():
            for range_key, line in lines_dict.items():
                if line.get_visible():
                    visible_lines.append(line)
                    visible_labels.append(line.get_label())
        
        # Update legend with only visible lines
        if visible_lines:
            self.ax.legend(visible_lines, visible_labels, 
                          loc='best', fontsize=self.legend_fontsize)
        else:
            # Remove legend if no lines visible
            legend = self.ax.get_legend()
            if legend:
                legend.remove()
    
    def clear_plot(self) -> None:
        """Clear all plot data."""
        if self.plot_initialized:
            for lines_dict in self.line_objects.values():
                for line in lines_dict.values():
                    line.remove()
            self.line_objects.clear()
            self.file_colors.clear()
            self.ax.clear()
            self.canvas.draw_idle()
    
    def export_figure(self, filepath: str, dpi: int = 300) -> None:
        """Export the current figure to a file."""
        if self.figure:
            self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight')


class BatchFileListWidget(QTableWidget):
    """
    Enhanced file list widget that maintains selection state across windows.
    
    This widget displays files with checkboxes and optional additional columns
    (like Cslow values). It uses a FileSelectionState object to maintain
    consistency across different windows.
    """
    
    # Signals
    selection_changed = pyqtSignal()
    cslow_value_changed = pyqtSignal(str, float)  # filename, new_value
    
    def __init__(self, selection_state: Optional[FileSelectionState] = None,
                 show_cslow: bool = False, parent=None):
        """
        Initialize the file list widget.
        
        Args:
            selection_state: Shared selection state object
            show_cslow: Whether to show Cslow column
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.selection_state = selection_state or FileSelectionState()
        self.show_cslow = show_cslow
        self.file_colors: Dict[str, Tuple[float, ...]] = {}
        
        # Prevent signal cascades
        self._updating_checkboxes = False
        
        # Configure table
        self._setup_table()
        
        # Connect to selection state
        self.selection_state.add_observer(self._on_external_selection_change)
    
    def _setup_table(self) -> None:
        """Set up table structure and appearance."""
        # Column setup
        if self.show_cslow:
            self.setColumnCount(4)
            self.setHorizontalHeaderLabels(["", "Color", "File", "Cslow (pF)"])
        else:
            self.setColumnCount(3)
            self.setHorizontalHeaderLabels(["", "Color", "File"])
        
        # Column sizing
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        
        self.setColumnWidth(0, 30)  # Checkbox
        self.setColumnWidth(1, 40)  # Color
        
        if self.show_cslow:
            self.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
            self.setColumnWidth(3, 100)
        
        # Appearance
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.verticalHeader().setVisible(False)
        
        # Make file column non-editable
        self.setEditTriggers(QTableWidget.NoEditTriggers)
    
    def add_file(self, file_name: str, color: Tuple[float, ...], 
                 cslow_val: Optional[float] = None) -> None:
        """
        Add a file to the list.
        
        Args:
            file_name: Name of the file
            color: RGB color tuple
            cslow_val: Cslow value (if show_cslow is True)
        """
        row = self.rowCount()
        self.insertRow(row)
        
        # Checkbox
        checkbox = QCheckBox()
        checkbox.setChecked(self.selection_state.is_selected(file_name))
        checkbox.stateChanged.connect(lambda: self._on_checkbox_changed(file_name, checkbox.isChecked()))
        
        checkbox_widget = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.setAlignment(Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.setCellWidget(row, 0, checkbox_widget)
        
        # Color indicator
        self.setCellWidget(row, 1, self._create_color_indicator(color))
        
        # File name
        file_item = QTableWidgetItem(file_name)
        file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
        self.setItem(row, 2, file_item)
        
        # Cslow value (if applicable)
        if self.show_cslow and cslow_val is not None:
            from data_analysis_gui.widgets.custom_inputs import SelectAllLineEdit
            
            cslow_edit = SelectAllLineEdit()
            cslow_edit.setText(f"{cslow_val:.2f}")
            cslow_edit.editingFinished.connect(
                lambda: self._on_cslow_changed(file_name, cslow_edit)
            )
            self.setCellWidget(row, 3, cslow_edit)
        
        # Store color
        self.file_colors[file_name] = color
    
    def _create_color_indicator(self, color: Tuple[float, ...]) -> QWidget:
        """Create a colored square widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Convert to QColor
        qcolor = QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        
        painter.setBrush(QBrush(qcolor))
        painter.setPen(Qt.black)
        painter.drawRect(2, 2, 16, 16)
        painter.end()
        
        label = QLabel()
        label.setPixmap(pixmap)
        layout.addWidget(label)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        
        return widget
    
    def _on_checkbox_changed(self, file_name: str, checked: bool) -> None:
        """Handle individual checkbox changes."""
        if not self._updating_checkboxes:
            self.selection_state.toggle_file(file_name, checked)
            self.selection_changed.emit()
    
    def _on_external_selection_change(self, selected_files: Set[str]) -> None:
        """Handle selection changes from other sources."""
        self._updating_checkboxes = True
        
        for row in range(self.rowCount()):
            file_name = self.item(row, 2).text()
            checkbox = self.cellWidget(row, 0).findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(file_name in selected_files)
        
        self._updating_checkboxes = False
        self.selection_changed.emit()
    
    def _on_cslow_changed(self, file_name: str, cslow_edit: QWidget) -> None:
        """Handle Cslow value changes."""
        try:
            new_value = float(cslow_edit.text())
            self.cslow_value_changed.emit(file_name, new_value)
        except ValueError:
            logger.warning(f"Invalid Cslow value for {file_name}")
    
    def set_all_checked(self, checked: bool) -> None:
        """Check or uncheck all files at once."""
        self._updating_checkboxes = True
        
        # Collect all filenames
        filenames = set()
        for row in range(self.rowCount()):
            file_name = self.item(row, 2).text()
            if checked:
                filenames.add(file_name)
            
            # Update checkbox UI
            checkbox = self.cellWidget(row, 0).findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(checked)
        
        self._updating_checkboxes = False
        
        # Update selection state once
        self.selection_state.set_files(filenames)
        self.selection_changed.emit()
    
    def get_selected_files(self) -> Set[str]:
        """Get currently selected files from the shared state."""
        return self.selection_state.get_selected_files()