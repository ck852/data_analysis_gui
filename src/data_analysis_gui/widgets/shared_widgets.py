"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

This module provides shared widget components for batch analysis windows.
It includes reusable components for displaying batch analysis results with
consistent behavior and appearance, such as a dynamic plot widget and a
file list widget that maintains selection state across windows.
"""

from typing import Dict, List, Set, Optional, Tuple, Callable
import numpy as np

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QCheckBox, QHeaderView, QLabel,
                                 QRadioButton, QLineEdit)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QPainter, QBrush

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D

from data_analysis_gui.core.models import FileAnalysisResult
from data_analysis_gui.config.logging import get_logger

from data_analysis_gui.config.plot_style import apply_plot_style, format_batch_plot, COLOR_CYCLE

from data_analysis_gui.widgets.custom_toolbar import MinimalNavigationToolbar

logger = get_logger(__name__)


class FileSelectionState:
    """
    FileSelectionState manages the selection state of files across multiple windows.

    It maintains a set of selected files and notifies registered observers
    whenever the selection changes, enabling synchronized selection between
    different UI components.

    Args:
        initial_files (Optional[Set[str]]): Set of filenames to select initially.
    """

    def __init__(self, initial_files: Optional[Set[str]] = None):
        """
        Initialize the selection state.

        Args:
            initial_files (Optional[Set[str]]): Set of filenames to select initially.
        """
        self._selected_files: Set[str] = (
            initial_files.copy() if initial_files else set()
        )
        self._observers: List[Callable[[Set[str]], None]] = []

    def toggle_file(self, filename: str, selected: bool) -> None:
        """
        Toggle the selection state for a file.

        Args:
            filename (str): The file to toggle.
            selected (bool): True to select, False to deselect.
        """
        if selected:
            self._selected_files.add(filename)
        else:
            self._selected_files.discard(filename)
        self._notify_observers()

    def set_files(self, filenames: Set[str]) -> None:
        """
        Set the complete selection state.

        Args:
            filenames (Set[str]): The set of files to select.
        """
        self._selected_files = filenames.copy()
        self._notify_observers()

    def is_selected(self, filename: str) -> bool:
        """
        Check if a file is selected.

        Args:
            filename (str): The file to check.

        Returns:
            bool: True if selected, False otherwise.
        """
        return filename in self._selected_files

    def get_selected_files(self) -> Set[str]:
        """
        Get a copy of the currently selected files.

        Returns:
            Set[str]: The set of selected files.
        """
        return self._selected_files.copy()

    def add_observer(self, callback: Callable[[Set[str]], None]) -> None:
        """
        Add an observer to be notified of selection changes.

        Args:
            callback (Callable[[Set[str]], None]): Function to call on selection change.
        """
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[Set[str]], None]) -> None:
        """
        Remove an observer.

        Args:
            callback (Callable[[Set[str]], None]): Observer to remove.
        """
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify_observers(self) -> None:
        """
        Notify all observers of a selection change.
        """
        selected = self.get_selected_files()
        for observer in self._observers:
            try:
                observer(selected)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")


class DynamicBatchPlotWidget(QWidget):
    """
    DynamicBatchPlotWidget is a reusable plot widget for batch analysis results.

    It maintains a persistent matplotlib figure and updates only the data or
    visibility of plot lines, avoiding flicker from complete redraws. Designed
    for batch results and current density displays.

    Signals:
        plot_updated: Emitted when the plot is updated.

    Args:
        parent: Optional parent widget.
    """

    # Signals
    plot_updated = Signal()

    def __init__(self, parent=None):
        """
        Initialize the plot widget with modern styling.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)

        # Apply global plot style
        apply_plot_style()

        # Plot components (created lazily)
        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvas] = None
        self.ax = None
        self.toolbar: Optional[MinimalNavigationToolbar] = None

        # Data management
        self.line_objects: Dict[str, Dict[str, Line2D]] = {}
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
        self.layout.setSpacing(0)

        # Initialize with styled empty message
        self.empty_label = QLabel("No data to display")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet(
            """
            QLabel {
                color: #808080;
                font-size: 12px;
                font-style: italic;
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 20px;
            }
        """
        )
        self.layout.addWidget(self.empty_label)

    def initialize_plot(self, x_label: str, y_label: str, title: str = "") -> None:
        """
        Initialize the plot with axis labels and title.

        Args:
            x_label (str): Label for x-axis.
            y_label (str): Label for y-axis.
            title (str): Plot title (optional).
        """
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

        if not self.plot_initialized:
            self._create_plot_components()

    def _create_plot_components(self) -> None:
        """
        Create matplotlib figure, canvas, and toolbar with modern styling.
        """
        # Remove empty label
        if self.empty_label:
            self.empty_label.setParent(None)
            self.empty_label = None

        # Create figure with modern style
        self.figure = Figure(figsize=(12, 8), facecolor="#FAFAFA")
        self.ax = self.figure.add_subplot(111)

        # Apply batch plot formatting
        format_batch_plot(self.ax, self.x_label, self.y_label)

        # Create canvas
        self.canvas = FigureCanvas(self.figure)

        # Create minimal toolbar for batch plots
        self.toolbar = MinimalNavigationToolbar(self.canvas, self)

        # Add to layout with proper spacing
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        self.plot_initialized = True
        logger.debug("Plot components created")

    def set_data(
        self,
        results: List[FileAnalysisResult],
        use_dual_range: bool = False,
        color_mapping: Optional[Dict[str, Tuple[float, ...]]] = None,
        auto_scale: bool = True,
    ) -> None:  # Add auto_scale parameter
        """
        Set the data to be plotted, with optional auto-scaling.

        Args:
            results (List[FileAnalysisResult]): List of analysis results.
            use_dual_range (bool): Whether to show dual range data.
            color_mapping (Optional[Dict[str, Tuple[float, ...]]]): Pre-defined color mapping.
            auto_scale (bool): Automatically scale axes to fit data (default: True).
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

        # Auto-scale if requested (especially important for current density)
        if auto_scale:
            self._auto_scale_axes()

        # Draw
        self.canvas.draw_idle()
        self.plot_updated.emit()

    def _generate_color_mapping(
        self, results: List[FileAnalysisResult]
    ) -> Dict[str, Tuple[float, ...]]:
        """
        Generate a color mapping for each file using the modern color palette.

        Args:
            results (List[FileAnalysisResult]): List of analysis results.

        Returns:
            Dict[str, Tuple[float, ...]]: Mapping from file name to RGB color tuple.
        """
        color_mapping = {}

        for idx, result in enumerate(results):
            # Use the modern color cycle
            color_hex = COLOR_CYCLE[idx % len(COLOR_CYCLE)]

            # Convert hex to RGB tuple
            if color_hex.startswith("#"):
                color = tuple(int(color_hex[i : i + 2], 16) / 255 for i in (1, 3, 5))
            else:
                import matplotlib.colors as mcolors

                color = mcolors.to_rgb(color_hex)

            color_mapping[result.base_name] = color

        return color_mapping

    def _create_lines_for_result(self, result: FileAnalysisResult) -> None:
        """
        Create line objects for a given analysis result with modern styling.

        Args:
            result (FileAnalysisResult): The analysis result to plot.
        """
        color = self.file_colors.get(result.base_name, (0, 0, 0))

        # Range 1 line with modern styling
        if len(result.x_data) > 0 and len(result.y_data) > 0:
            (line_r1,) = self.ax.plot(
                result.x_data,
                result.y_data,
                "o-",
                label=f"{result.base_name}",
                markersize=4,
                markeredgewidth=0,
                linewidth=1.5,
                alpha=0.85,
                color=color,
                visible=True,
            )

            if result.base_name not in self.line_objects:
                self.line_objects[result.base_name] = {}
            self.line_objects[result.base_name]["range1"] = line_r1

        # Range 2 line with dashed style
        if self.use_dual_range and result.y_data2 is not None:
            if len(result.x_data) > 0 and len(result.y_data2) > 0:
                (line_r2,) = self.ax.plot(
                    result.x_data if result.x_data2 is None else result.x_data2,
                    result.y_data2,
                    "s--",
                    label=f"{result.base_name} (Range 2)",
                    markersize=4,
                    markeredgewidth=0,
                    linewidth=1.5,
                    alpha=0.85,
                    color=color,
                    visible=True,
                )
                self.line_objects[result.base_name]["range2"] = line_r2

    def update_visibility(self, selected_files: Set[str]) -> None:
        """
        Update line visibility based on selected files.

        Args:
            selected_files (Set[str]): Set of filenames to show.
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

        # Re-scale to fit only visible data
        self._auto_scale_axes()

        # Redraw
        self.canvas.draw_idle()
        self.plot_updated.emit()

    def update_line_data(
        self, filename: str, y_data: np.ndarray, y_data2: Optional[np.ndarray] = None
    ) -> None:
        """
        Update Y data for a specific file's lines.

        Args:
            filename (str): Name of file to update.
            y_data (np.ndarray): New Y data for range 1.
            y_data2 (Optional[np.ndarray]): New Y data for range 2 (if applicable).
        """
        if filename not in self.line_objects:
            logger.warning(f"No line objects for file: {filename}")
            return

        lines = self.line_objects[filename]

        # Update range 1
        if "range1" in lines:
            lines["range1"].set_ydata(y_data)

        # Update range 2
        if self.use_dual_range and y_data2 is not None and "range2" in lines:
            lines["range2"].set_ydata(y_data2)

        # Update axis limits if needed
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw
        self.canvas.draw_idle()
        self.plot_updated.emit()

    def _update_plot_appearance(self) -> None:
        """
        Update legend and plot appearance with modern styling.
        """
        # Get visible lines for legend
        visible_lines = []
        visible_labels = []

        for filename, lines_dict in self.line_objects.items():
            for range_key, line in lines_dict.items():
                if line.get_visible():
                    visible_lines.append(line)
                    visible_labels.append(line.get_label())

        # Update legend with modern styling
        if visible_lines:
            legend = self.ax.legend(
                visible_lines,
                visible_labels,
                loc="best",
                frameon=True,
                fancybox=False,
                shadow=False,
                framealpha=0.95,
                edgecolor="#D0D0D0",
                facecolor="white",
                fontsize=self.legend_fontsize,
                borderpad=0.5,
                columnspacing=1.2,
                handlelength=2,
            )

            # Make legend draggable for better user experience
            if legend:
                legend.set_draggable(True)
        else:
            # Remove legend if no lines visible
            legend = self.ax.get_legend()
            if legend:
                legend.remove()

    def clear_plot(self) -> None:
        """
        Clear all plot data and reset the plot.
        """
        if self.plot_initialized:
            for lines_dict in self.line_objects.values():
                for line in lines_dict.values():
                    line.remove()
            self.line_objects.clear()
            self.file_colors.clear()
            self.ax.clear()
            self.canvas.draw_idle()

    def export_figure(self, filepath: str, dpi: int = 300) -> None:
        """
        Export the current figure to a file with high quality.

        Args:
            filepath (str): Path to save the figure.
            dpi (int): Dots per inch for export (default: 300).
        """
        if self.figure:
            # Ensure the figure looks good when exported
            self.figure.savefig(
                filepath,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                pad_inches=0.1,
            )

    def _auto_scale_axes(self):
        """
        Automatically scale axes to fit the visible data with appropriate padding.
        Especially useful for current density plots.
        """
        if not self.ax or not self.line_objects:
            return

        # Collect all visible y-data
        all_y_data = []
        all_x_data = []

        for filename, lines_dict in self.line_objects.items():
            for line in lines_dict.values():
                if line.get_visible():
                    y_data = line.get_ydata()
                    x_data = line.get_xdata()
                    if len(y_data) > 0:
                        all_y_data.extend(y_data)
                        all_x_data.extend(x_data)

        if all_y_data and all_x_data:
            # Calculate data ranges
            y_min, y_max = np.nanmin(all_y_data), np.nanmax(all_y_data)
            x_min, x_max = np.nanmin(all_x_data), np.nanmax(all_x_data)

            # Add smart padding based on data range
            y_range = y_max - y_min
            if y_range > 0:
                # Use 5% padding, but ensure minimum padding for very small ranges
                y_padding = max(y_range * 0.05, abs(y_max) * 0.01)
            else:
                # If all values are the same, add 10% of the value as padding
                y_padding = abs(y_max) * 0.1 if y_max != 0 else 1.0

            x_range = x_max - x_min
            x_padding = x_range * 0.02 if x_range > 0 else 1.0

            # Set the limits
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)

        # Refresh the canvas
        self.canvas.draw_idle()

    def auto_scale_to_data(self):
        """
        Automatically scale axes to fit the visible data.
        Call after set_data() or update_visibility() for current density plots.
        """
        if not self.ax or not self.line_objects:
            return

        # Collect all visible y-data
        all_y_data = []
        all_x_data = []

        for filename, lines_dict in self.line_objects.items():
            for line in lines_dict.values():
                if line.get_visible():
                    y_data = line.get_ydata()
                    x_data = line.get_xdata()
                    if len(y_data) > 0:
                        # Filter out NaN values
                        valid_mask = ~np.isnan(y_data)
                        if np.any(valid_mask):
                            all_y_data.extend(y_data[valid_mask])
                            all_x_data.extend(x_data[valid_mask])

        if all_y_data and all_x_data:
            # Calculate data ranges
            y_min, y_max = np.min(all_y_data), np.max(all_y_data)
            x_min, x_max = np.min(all_x_data), np.max(all_x_data)

            # Add smart padding based on data range
            y_range = y_max - y_min
            if y_range > 0:
                # Use 5% padding for y-axis
                y_padding = y_range * 0.05
            else:
                # If all values are the same, add 10% of the value as padding
                y_padding = abs(y_max) * 0.1 if y_max != 0 else 1.0

            x_range = x_max - x_min
            x_padding = x_range * 0.02 if x_range > 0 else 1.0

            # Set the limits
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)

            # Refresh the canvas
            self.canvas.draw_idle()


class BatchFileListWidget(QTableWidget):
    """
    BatchFileListWidget displays a list of files with checkboxes and color indicators,
    maintaining selection state across windows.

    It supports optional additional columns (such as Cslow values) and uses a
    FileSelectionState object for consistent selection management.

    Signals:
        selection_changed: Emitted when file selection changes.
        cslow_value_changed: Emitted when a Cslow value is edited.

    Args:
        selection_state (Optional[FileSelectionState]): Shared selection state object.
        show_cslow (bool): Whether to show Cslow column.
        parent: Optional parent widget.
    """

    # Signals
    selection_changed = Signal()
    cslow_value_changed = Signal(str, float)  # filename, new_value

    def __init__(
        self,
        selection_state: Optional[FileSelectionState] = None,
        show_cslow: bool = False,
        parent=None,
    ):
        """
        Initialize the file list widget.

        Args:
            selection_state (Optional[FileSelectionState]): Shared selection state object.
            show_cslow (bool): Whether to show Cslow column.
            parent: Optional parent widget.
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
        """
        Set up table structure, columns, and appearance.
        """
        # Column setup
        if self.show_cslow:
            self.setColumnCount(4)
            self.setHorizontalHeaderLabels(["", "Color", "File", "Cslow (pF)"])
        else:
            self.setColumnCount(3)
            self.setHorizontalHeaderLabels(["", "Color", "File"])

        # Column sizing
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.setColumnWidth(0, 30)  # Checkbox
        self.setColumnWidth(1, 40)  # Color

        if self.show_cslow:
            self.horizontalHeader().setSectionResizeMode(
                3, QHeaderView.ResizeMode.Fixed
            )
            self.setColumnWidth(3, 100)

        # Appearance
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.verticalHeader().setVisible(False)

        # Make file column non-editable
        self.setEditTriggers(QTableWidget.NoEditTriggers)

    def add_file(
        self,
        file_name: str,
        color: Tuple[float, ...],
        cslow_val: Optional[float] = None,
    ) -> None:
        """
        Add a file to the list.

        Args:
            file_name (str): Name of the file.
            color (Tuple[float, ...]): RGB color tuple.
            cslow_val (Optional[float]): Cslow value (if show_cslow is True).
        """
        row = self.rowCount()
        self.insertRow(row)

        # Checkbox
        checkbox = QCheckBox()
        checkbox.setChecked(self.selection_state.is_selected(file_name))
        checkbox.stateChanged.connect(
            lambda: self._on_checkbox_changed(file_name, checkbox.isChecked())
        )

        checkbox_widget = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.setCellWidget(row, 0, checkbox_widget)

        # Color indicator
        self.setCellWidget(row, 1, self._create_color_indicator(color))

        # File name
        file_item = QTableWidgetItem(file_name)
        file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
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
        """
        Create a colored square widget for the file color indicator.

        Args:
            color (Tuple[float, ...]): RGB color tuple.

        Returns:
            QWidget: The color indicator widget.
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)

        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Convert to QColor
        qcolor = QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

        painter.setBrush(QBrush(qcolor))
        painter.setPen(Qt.GlobalColor.black)
        painter.drawRect(2, 2, 16, 16)
        painter.end()

        label = QLabel()
        label.setPixmap(pixmap)
        layout.addWidget(label)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        return widget

    def _on_checkbox_changed(self, file_name: str, checked: bool) -> None:
        """
        Handle individual checkbox changes.

        Args:
            file_name (str): Name of the file.
            checked (bool): Checkbox state.
        """
        if not self._updating_checkboxes:
            self.selection_state.toggle_file(file_name, checked)
            self.selection_changed.emit()

    def _on_external_selection_change(self, selected_files: Set[str]) -> None:
        """
        Handle selection changes from other sources (e.g., other windows).

        Args:
            selected_files (Set[str]): Set of selected filenames.
        """
        self._updating_checkboxes = True

        for row in range(self.rowCount()):
            file_name = self.item(row, 2).text()
            checkbox = self.cellWidget(row, 0).findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(file_name in selected_files)

        self._updating_checkboxes = False
        self.selection_changed.emit()

    def _on_cslow_changed(self, file_name: str, cslow_edit: QWidget) -> None:
        """
        Handle Cslow value changes.

        Args:
            file_name (str): Name of the file.
            cslow_edit (QWidget): The line edit widget for Cslow value.
        """
        try:
            new_value = float(cslow_edit.text())
            self.cslow_value_changed.emit(file_name, new_value)
        except ValueError:
            logger.warning(f"Invalid Cslow value for {file_name}")

    def set_all_checked(self, checked: bool) -> None:
        """
        Check or uncheck all files at once.

        Args:
            checked (bool): True to check all, False to uncheck all.
        """
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
        """
        Get currently selected files from the shared state.

        Returns:
            Set[str]: The set of selected files.
        """
        return self.selection_state.get_selected_files()

class SweepSelectionWidget(QWidget):
    """
    Widget for selecting which sweeps to analyze.
    """
    
    def __init__(self, sweep_names: List[str], parent=None):
        super().__init__(parent)
        
        self.sweep_names = sweep_names
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI with selection mode and table."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Selection mode radio buttons
        mode_layout = QHBoxLayout()
        self.table_mode_radio = QRadioButton("Select from table")
        self.range_mode_radio = QRadioButton("Enter range:")
        self.table_mode_radio.setChecked(True)  # Default to table mode
        
        self.range_input = QLineEdit()
        self.range_input.setPlaceholderText("e.g., 3-15")
        self.range_input.setMaximumWidth(100)
        self.range_input.setEnabled(False)  # Disabled by default
        
        mode_layout.addWidget(self.table_mode_radio)
        mode_layout.addWidget(self.range_mode_radio)
        mode_layout.addWidget(self.range_input)
        mode_layout.addStretch()
        
        layout.addLayout(mode_layout)
        
        # Table widget
        self.table = QTableWidget()
        self._init_table()
        layout.addWidget(self.table)
        
        # Connect mode change signals
        self.table_mode_radio.toggled.connect(self._on_mode_changed)
        
    def _init_table(self):
        """Initialize the table with sweep selection."""
        self.table.setRowCount(len(self.sweep_names))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Select", "Sweep"])
        
        # Configure headers
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        # Hide vertical header
        self.table.verticalHeader().setVisible(False)
        
        # Set alternating row colors
        self.table.setAlternatingRowColors(True)
        
        # Populate rows
        for i, sweep_name in enumerate(self.sweep_names):
            # Checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            self.table.setCellWidget(i, 0, checkbox)
            
            # Sweep name
            item = QTableWidgetItem(f"Sweep {sweep_name}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 1, item)
            
        self.table.setMaximumHeight(600)
        
        # Connect signal to make entire row clickable
        self.table.cellClicked.connect(self._on_cell_clicked)
    
    def _on_mode_changed(self, table_mode_checked: bool):
        """Enable/disable table and range input based on mode."""
        self.table.setEnabled(table_mode_checked)
        self.range_input.setEnabled(not table_mode_checked)
        
    def _on_cell_clicked(self, row: int, column: int):
        """Toggle checkbox when any cell in the row is clicked."""
        checkbox = self.table.cellWidget(row, 0)
        if checkbox:
            checkbox.setChecked(not checkbox.isChecked())
            
    def select_all(self, checked: bool = True):
        """Select or deselect all sweeps in table."""
        for i in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(i, 0)
            if checkbox:
                checkbox.setChecked(checked)
                
    def get_selected_sweeps(self) -> List[str]:
        """Get list of selected sweep names based on current mode."""
        if self.table_mode_radio.isChecked():
            # Table mode - return checked sweeps
            selected = []
            for i in range(self.table.rowCount()):
                checkbox = self.table.cellWidget(i, 0)
                if checkbox and checkbox.isChecked():
                    selected.append(self.sweep_names[i])
            return selected
        else:
            # Range mode - parse range input
            return self._parse_range_input()
    
    def _parse_range_input(self) -> List[str]:
        """Parse range input like '3-15' and return matching sweep names."""
        range_text = self.range_input.text().strip()
        if not range_text:
            return []
        
        try:
            if '-' not in range_text:
                raise ValueError("Invalid format")
            
            start_str, end_str = range_text.split('-', 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            
            if start > end:
                raise ValueError("Start must be <= end")
            
            # Find sweeps in range
            selected = []
            for sweep_name in self.sweep_names:
                try:
                    sweep_num = int(sweep_name)
                    if start <= sweep_num <= end:
                        selected.append(sweep_name)
                except ValueError:
                    # Skip non-numeric sweep names
                    continue
            
            return selected
            
        except (ValueError, AttributeError):
            return []