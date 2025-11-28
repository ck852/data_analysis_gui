"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Shared widget components for batch analysis and current density windows.
DynamicBatchPlotWidget and BatchFileListWidget are designed specifically
for these contexts and should not be used elsewhere.
"""

from typing import Dict, List, Set, Optional, Tuple, Callable
import numpy as np

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QCheckBox, QHeaderView, QLabel,
                                )
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
    Shared selection state that keeps multiple windows in sync.
    
    Windows register as observers and get notified when selection changes,
    preventing the need for direct coupling between UI components.
    """

    def __init__(self, initial_files: Optional[Set[str]] = None):
        self._selected_files: Set[str] = (
            initial_files.copy() if initial_files else set()
        )
        self._observers: List[Callable[[Set[str]], None]] = []

    def toggle_file(self, filename: str, selected: bool) -> None:
        if selected:
            self._selected_files.add(filename)
        else:
            self._selected_files.discard(filename)
        self._notify_observers()

    def set_files(self, filenames: Set[str]) -> None:
        self._selected_files = filenames.copy()
        self._notify_observers()

    def is_selected(self, filename: str) -> bool:
        return filename in self._selected_files

    def get_selected_files(self) -> Set[str]:
        return self._selected_files.copy()

    def add_observer(self, callback: Callable[[Set[str]], None]) -> None:
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[Set[str]], None]) -> None:
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify_observers(self) -> None:
        selected = self.get_selected_files()
        for observer in self._observers:
            try:
                observer(selected)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")


class DynamicBatchPlotWidget(QWidget):
    """
    Plot widget that updates line visibility rather than redrawing entirely.
    
    Maintains persistent matplotlib objects and toggles their visibility,
    which avoids flicker when users check/uncheck files in the list.
    """

    plot_updated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        apply_plot_style()

        # Plot components (created lazily on first data)
        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvas] = None
        self.ax = None
        self.toolbar: Optional[MinimalNavigationToolbar] = None

        # Line objects keyed by filename, then by range
        self.line_objects: Dict[str, Dict[str, Line2D]] = {}
        self.file_colors: Dict[str, Tuple[float, ...]] = {}
        self.voltage_annotations: Dict[str, Tuple[int, int]] = {}
        self.plot_initialized = False

        self.use_dual_range = False
        self.x_label = "X"
        self.y_label = "Y"
        self.title = ""
        self.legend_fontsize = 8

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

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
        """Set axis labels and create matplotlib components if not already done."""
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

        if not self.plot_initialized:
            self._create_plot_components()

    def _create_plot_components(self) -> None:
        if self.empty_label:
            self.empty_label.setParent(None)
            self.empty_label = None

        self.figure = Figure(figsize=(12, 8), facecolor="#FAFAFA")
        self.ax = self.figure.add_subplot(111)

        format_batch_plot(self.ax, self.x_label, self.y_label)

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = MinimalNavigationToolbar(self.canvas, self)

        self.layout.addWidget(self.toolbar, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.canvas)

        self.plot_initialized = True
        logger.debug("Plot components created")

    def set_data(
        self,
        results: List[FileAnalysisResult],
        use_dual_range: bool = False,
        color_mapping: Optional[Dict[str, Tuple[float, ...]]] = None,
        voltage_annotations: Optional[Dict[str, Tuple[int, int]]] = None,
        auto_scale: bool = True,
    ) -> None:
        """Populate plot with analysis results. Creates new line objects for each file."""
        if not self.plot_initialized:
            logger.warning("Plot not initialized. Call initialize_plot first.")
            return

        self.use_dual_range = use_dual_range
        self.voltage_annotations = voltage_annotations or {}

        if color_mapping is None:
            color_mapping = self._generate_color_mapping(results)
        self.file_colors = color_mapping

        # Clear existing lines
        for lines_dict in self.line_objects.values():
            for line in lines_dict.values():
                line.remove()
        self.line_objects.clear()

        for result in results:
            self._create_lines_for_result(result)

        self._update_plot_appearance()

        if auto_scale:
            self._auto_scale_axes()

        self.canvas.draw_idle()
        self.plot_updated.emit()

    def _generate_color_mapping(
        self, results: List[FileAnalysisResult]
    ) -> Dict[str, Tuple[float, ...]]:
        color_mapping = {}

        for idx, result in enumerate(results):
            color_hex = COLOR_CYCLE[idx % len(COLOR_CYCLE)]

            if color_hex.startswith("#"):
                color = tuple(int(color_hex[i : i + 2], 16) / 255 for i in (1, 3, 5))
            else:
                import matplotlib.colors as mcolors
                color = mcolors.to_rgb(color_hex)

            color_mapping[result.base_name] = color

        return color_mapping

    def _create_lines_for_result(self, result: FileAnalysisResult) -> None:
        """Create line objects for a result, using voltage annotations in labels if available."""
        color = self.file_colors.get(result.base_name, (0, 0, 0))
        voltages = self.voltage_annotations.get(result.base_name)

        # Range 1
        if len(result.x_data) > 0 and len(result.y_data) > 0:
            if self.use_dual_range and voltages:
                v1 = voltages[0]
                voltage_str = f"+{v1}" if v1 >= 0 else str(v1)
                label = f"{result.base_name} ({voltage_str}mV)"
            else:
                label = f"{result.base_name}"

            (line_r1,) = self.ax.plot(
                result.x_data,
                result.y_data,
                "o-",
                label=label,
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

        # Range 2 (dashed)
        if self.use_dual_range and result.y_data2 is not None:
            if len(result.x_data) > 0 and len(result.y_data2) > 0:
                if voltages and len(voltages) > 1:
                    v2 = voltages[1]
                    voltage_str = f"+{v2}" if v2 >= 0 else str(v2)
                    label = f"{result.base_name} ({voltage_str}mV)"
                else:
                    label = f"{result.base_name} (Range 2)"

                (line_r2,) = self.ax.plot(
                    result.x_data if result.x_data2 is None else result.x_data2,
                    result.y_data2,
                    "s--",
                    label=label,
                    markersize=4,
                    markeredgewidth=0,
                    linewidth=1.5,
                    alpha=0.85,
                    color=color,
                    visible=True,
                )
                self.line_objects[result.base_name]["range2"] = line_r2

    def update_visibility(self, selected_files: Set[str]) -> None:
        """Toggle line visibility based on which files are currently selected."""
        if not self.plot_initialized:
            return

        for filename, lines_dict in self.line_objects.items():
            visible = filename in selected_files
            for line in lines_dict.values():
                line.set_visible(visible)

        self._update_plot_appearance()
        self._auto_scale_axes()

        self.canvas.draw_idle()
        self.plot_updated.emit()

    def update_line_data(
        self, filename: str, y_data: np.ndarray, y_data2: Optional[np.ndarray] = None
    ) -> None:
        """Update Y data for existing lines (used when recalculating, e.g., current density)."""
        if filename not in self.line_objects:
            logger.warning(f"No line objects for file: {filename}")
            return

        lines = self.line_objects[filename]

        if "range1" in lines:
            lines["range1"].set_ydata(y_data)

        if self.use_dual_range and y_data2 is not None and "range2" in lines:
            lines["range2"].set_ydata(y_data2)

        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw_idle()
        self.plot_updated.emit()

    def _update_plot_appearance(self) -> None:
        """Rebuild legend to show only currently visible lines."""
        visible_lines = []
        visible_labels = []

        for lines_dict in self.line_objects.values():
            for line in lines_dict.values():
                if line.get_visible():
                    visible_lines.append(line)
                    visible_labels.append(line.get_label())

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

            if legend:
                legend.set_draggable(True)
        else:
            legend = self.ax.get_legend()
            if legend:
                legend.remove()

    def clear_plot(self) -> None:
        if self.plot_initialized:
            for lines_dict in self.line_objects.values():
                for line in lines_dict.values():
                    line.remove()
            self.line_objects.clear()
            self.file_colors.clear()
            self.ax.clear()
            self.canvas.draw_idle()

    def export_figure(self, filepath: str, dpi: int = 300) -> None:
        if self.figure:
            self.figure.savefig(
                filepath,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                pad_inches=0.1,
            )

    def _auto_scale_axes(self):
        """Scale axes to fit visible data. Uses nanmin/nanmax to handle NaN gracefully."""
        if not self.ax or not self.line_objects:
            return

        all_y_data = []
        all_x_data = []

        for lines_dict in self.line_objects.values():
            for line in lines_dict.values():
                if line.get_visible():
                    y_data = line.get_ydata()
                    x_data = line.get_xdata()
                    if len(y_data) > 0:
                        all_y_data.extend(y_data)
                        all_x_data.extend(x_data)

        if all_y_data and all_x_data:
            y_min, y_max = np.nanmin(all_y_data), np.nanmax(all_y_data)
            x_min, x_max = np.nanmin(all_x_data), np.nanmax(all_x_data)

            y_range = y_max - y_min
            if y_range > 0:
                y_padding = max(y_range * 0.05, abs(y_max) * 0.01)
            else:
                y_padding = abs(y_max) * 0.1 if y_max != 0 else 1.0

            x_range = x_max - x_min
            x_padding = x_range * 0.02 if x_range > 0 else 1.0

            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)

        self.canvas.draw_idle()


    def auto_scale_to_data(self):
        """
        Public rescaling method for use after Cslow edits in current density dialog.
        
        Differs from _auto_scale_axes by explicitly filtering NaN values before
        computing limits, which is necessary after current density recalculation.
        """
        if not self.ax or not self.line_objects:
            return

        all_y_data = []
        all_x_data = []

        for lines_dict in self.line_objects.values():
            for line in lines_dict.values():
                if line.get_visible():
                    y_data = line.get_ydata()
                    x_data = line.get_xdata()
                    if len(y_data) > 0:
                        valid_mask = ~np.isnan(y_data)
                        if np.any(valid_mask):
                            all_y_data.extend(y_data[valid_mask])
                            all_x_data.extend(x_data[valid_mask])

        if all_y_data and all_x_data:
            y_min, y_max = np.min(all_y_data), np.max(all_y_data)
            x_min, x_max = np.min(all_x_data), np.max(all_x_data)

            y_range = y_max - y_min
            if y_range > 0:
                y_padding = y_range * 0.05
            else:
                y_padding = abs(y_max) * 0.1 if y_max != 0 else 1.0

            x_range = x_max - x_min
            x_padding = x_range * 0.02 if x_range > 0 else 1.0

            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)

            self.canvas.draw_idle()


class BatchFileListWidget(QTableWidget):
    """
    File list with checkboxes and color swatches for batch results.
    
    Syncs with a FileSelectionState so that checking/unchecking files
    here updates other windows showing the same batch results.
    """

    selection_changed = Signal()
    cslow_value_changed = Signal(str, float)  # filename, new_value

    def __init__(
        self,
        selection_state: Optional[FileSelectionState] = None,
        show_cslow: bool = False,
        parent=None,
    ):
        super().__init__(parent)

        self.selection_state = selection_state or FileSelectionState()
        self.show_cslow = show_cslow
        self.file_colors: Dict[str, Tuple[float, ...]] = {}

        self._updating_checkboxes = False

        self._setup_table()
        self.selection_state.add_observer(self._on_external_selection_change)

    def _setup_table(self) -> None:
        """Configure columns: checkbox, color swatch, filename, and optionally Cslow."""
        if self.show_cslow:
            self.setColumnCount(4)
            self.setHorizontalHeaderLabels(["", "Color", "File", "Cslow (pF)"])
        else:
            self.setColumnCount(3)
            self.setHorizontalHeaderLabels(["", "Color", "File"])

        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.setColumnWidth(0, 30)
        self.setColumnWidth(1, 40)

        if self.show_cslow:
            self.horizontalHeader().setSectionResizeMode(
                3, QHeaderView.ResizeMode.Fixed
            )
            self.setColumnWidth(3, 100)

        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

        self.cellClicked.connect(self._on_cell_clicked)

    def _on_cell_clicked(self, row: int, column: int):
        """Toggle checkbox when clicking anywhere in the row."""
        checkbox = self.cellWidget(row, 0).findChild(QCheckBox)
        if checkbox:
            checkbox.setChecked(not checkbox.isChecked())

    def add_file(
        self,
        file_name: str,
        color: Tuple[float, ...],
        cslow_val: Optional[float] = None,
    ) -> None:
        """Add a row for one file. Checkbox state is pulled from selection_state."""
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

        # Color swatch
        self.setCellWidget(row, 1, self._create_color_indicator(color))

        # Filename
        file_item = QTableWidgetItem(file_name)
        file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, 2, file_item)

        # Cslow (for current density dialog)
        if self.show_cslow and cslow_val is not None:
            from data_analysis_gui.widgets.custom_inputs import SelectAllLineEdit

            cslow_edit = SelectAllLineEdit()
            cslow_edit.setText(f"{cslow_val:.2f}")
            cslow_edit.editingFinished.connect(
                lambda: self._on_cslow_changed(file_name, cslow_edit)
            )
            self.setCellWidget(row, 3, cslow_edit)

        self.file_colors[file_name] = color

    def _create_color_indicator(self, color: Tuple[float, ...]) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)

        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

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
        if not self._updating_checkboxes:
            self.selection_state.toggle_file(file_name, checked)
            self.selection_changed.emit()

    def _on_external_selection_change(self, selected_files: Set[str]) -> None:
        """Handle selection changes originating from other windows."""
        self._updating_checkboxes = True

        for row in range(self.rowCount()):
            file_name = self.item(row, 2).text()
            checkbox = self.cellWidget(row, 0).findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(file_name in selected_files)

        self._updating_checkboxes = False
        self.selection_changed.emit()

    def _on_cslow_changed(self, file_name: str, cslow_edit: QWidget) -> None:
        try:
            new_value = float(cslow_edit.text())
            self.cslow_value_changed.emit(file_name, new_value)
        except ValueError:
            logger.warning(f"Invalid Cslow value for {file_name}")

    def set_all_checked(self, checked: bool) -> None:
        """Bulk check/uncheck all files. Updates state once at the end to avoid repeated notifications."""
        self._updating_checkboxes = True

        filenames = set()
        for row in range(self.rowCount()):
            file_name = self.item(row, 2).text()
            if checked:
                filenames.add(file_name)

            checkbox = self.cellWidget(row, 0).findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(checked)

        self._updating_checkboxes = False

        self.selection_state.set_files(filenames)
        self.selection_changed.emit()

    def get_selected_files(self) -> Set[str]:
        return self.selection_state.get_selected_files()