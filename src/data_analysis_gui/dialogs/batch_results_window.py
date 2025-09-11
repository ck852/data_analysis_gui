# batch_results_window.py

from pathlib import Path
import re
from typing import Set
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QMessageBox, QTableWidget, QTableWidgetItem,
                             QCheckBox, QLabel, QSplitter, QHeaderView, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QPainter, QBrush

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from data_analysis_gui.gui_services import FileDialogService
from data_analysis_gui.core.plot_formatter import PlotFormatter
from data_analysis_gui.config.logging import get_logger

from data_analysis_gui.dialogs.current_density_dialog import CurrentDensityDialog
from data_analysis_gui.dialogs.current_density_results_window import CurrentDensityResultsWindow

logger = get_logger(__name__)


class FileListWidget(QTableWidget):
    """Widget to display files with checkboxes and color indicators."""
    
    selection_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.file_colors = {}  # Map file names to colors
        self.selected_files = set()  # Track selected file names
        
        # Configure table
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["", "Color", "File"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.setColumnWidth(0, 30)  # Checkbox column
        self.setColumnWidth(1, 40)  # Color column
        
        # Disable editing
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Style
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.verticalHeader().setVisible(False)
    
    def add_file(self, file_name: str, color: tuple, checked: bool = True):
        """Add a file to the list with color indicator and checkbox."""
        row = self.rowCount()
        self.insertRow(row)
        
        # Checkbox
        checkbox = QCheckBox()
        checkbox.setChecked(checked)
        checkbox.stateChanged.connect(self._on_checkbox_changed)
        
        # Center checkbox in cell
        checkbox_widget = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.setAlignment(Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        
        self.setCellWidget(row, 0, checkbox_widget)
        
        # Color indicator
        color_widget = self._create_color_indicator(color)
        self.setCellWidget(row, 1, color_widget)
        
        # File name
        self.setItem(row, 2, QTableWidgetItem(file_name))
        
        # Store color and selection state
        self.file_colors[file_name] = color
        if checked:
            self.selected_files.add(file_name)
    
    def _create_color_indicator(self, color: tuple) -> QWidget:
        """Create a colored square widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Create colored pixmap
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Convert matplotlib color to Qt color
        if len(color) == 3:
            qcolor = QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        else:
            qcolor = QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*255))
        
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
    
    def _on_checkbox_changed(self):
        """Handle checkbox state changes."""
        # Update selected files set
        self.selected_files.clear()
        for row in range(self.rowCount()):
            checkbox_widget = self.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)
            if checkbox and checkbox.isChecked():
                file_name = self.item(row, 2).text()
                self.selected_files.add(file_name)
        
        self.selection_changed.emit()
    
    def get_selected_files(self) -> Set[str]:
        """Get the set of selected file names."""
        return self.selected_files.copy()
    
    def set_all_checked(self, checked: bool):
        """Check or uncheck all files."""
        for row in range(self.rowCount()):
            checkbox_widget = self.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(checked)


class BatchResultsWindow(QMainWindow):
    """Window for displaying batch analysis results with file selection."""
    
    def __init__(self, parent, batch_result, batch_service, plot_service, data_service):
        super().__init__(parent)
        self.batch_result = batch_result
        self.batch_service = batch_service
        self.plot_service = plot_service
        # New data layer uses DataManager with export_to_csv(...)
        self.export_service = data_service
        # Back-compat alias in case other methods still reference data_service
        self.data_service = data_service
        self.file_dialog_service = FileDialogService()
        
        # Use PlotFormatter for consistent formatting
        self.plot_formatter = PlotFormatter()
        
        self.figure = None
        self.canvas = None
        self.file_colors = {}  # Store color mapping
        
        self.setWindowTitle("Batch Analysis Results")
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        w = max(900, min(int(avail.width() * 0.90), 1200))
        h = max(600, min(int(avail.height() * 0.90), 700))
        self.resize(w, h)
        fg = self.frameGeometry()
        fg.moveCenter(avail.center())
        self.move(fg.topLeft())
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI with file list and plot."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Create splitter for file list and plot
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: File list with controls
        left_panel = self._create_file_list_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Plot
        plot_widget = self._create_plot_widget()
        splitter.addWidget(plot_widget)
        
        # Set initial splitter sizes (30% list, 70% plot)
        splitter.setSizes([360, 840])
        
        main_layout.addWidget(splitter)
        
        # Export controls at bottom
        self._add_export_controls(main_layout)
        
        # Initial plot
        self._update_plot()
    
    def _create_file_list_panel(self) -> QWidget:
        """Create the file list panel with controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Files:")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        # File list widget
        self.file_list = FileListWidget()
        self.file_list.selection_changed.connect(self._on_selection_changed)
        layout.addWidget(self.file_list)
        
        # Selection controls
        controls_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        select_all_btn.clicked.connect(lambda: self.file_list.set_all_checked(True))
        select_none_btn.clicked.connect(lambda: self.file_list.set_all_checked(False))
        controls_layout.addWidget(select_all_btn)
        controls_layout.addWidget(select_none_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Summary label
        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)
        
        return panel
    
    def _create_plot_widget(self) -> QWidget:
        """Create plot widget container."""
        widget = QWidget()
        self.plot_layout = QVBoxLayout(widget)
        return widget
    
    def _sort_results(self, results):
        """Sort results numerically by file name."""
        def extract_number(file_name):
            """Extract number from filename for sorting."""
            # Look for pattern like xxx_001 or xxx_1
            match = re.search(r'_(\d+)', file_name)
            if match:
                return int(match.group(1))
            # Try to find any number in the name
            numbers = re.findall(r'\d+', file_name)
            if numbers:
                return int(numbers[-1])  # Use last number found
            return 0
        
        return sorted(results, key=lambda r: extract_number(r.base_name))
    
    def _update_plot(self):
        """Update the plot based on selected files."""
        # Clear existing plot
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # Get selected files
        selected_files = self.file_list.get_selected_files()
        
        # Filter results to only selected files
        filtered_results = [
            r for r in self.batch_result.successful_results
            if r.base_name in selected_files
        ]
        
        if not filtered_results:
            # Show message if no files selected
            label = QLabel("No files selected for display")
            label.setAlignment(Qt.AlignCenter)
            self.plot_layout.addWidget(label)
            self.figure = None
            self.canvas = None
            self._update_summary()
            return
        
        # Sort filtered results
        filtered_results = self._sort_results(filtered_results)
        
        # Create figure
        self.figure, ax = plt.subplots(figsize=(12, 8))
        
        # Get color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        # Plot filtered results
        plot_count = 0
        for idx, result in enumerate(filtered_results):
            if len(result.x_data) > 0 and len(result.y_data) > 0:
                # Use consistent color for both ranges
                color = colors[idx % len(colors)]
                
                # Plot Range 1
                ax.plot(result.x_data, result.y_data,
                       'o-', label=f"{result.base_name} (Range 1)",
                       markersize=4, alpha=0.7, color=color)
                plot_count += 1
                
                # Plot Range 2 if applicable
                if self.batch_result.parameters.use_dual_range and result.y_data2 is not None:
                    ax.plot(result.x_data, result.y_data2,
                           's--', label=f"{result.base_name} (Range 2)",
                           markersize=4, alpha=0.7, color=color)
        
        # Configure plot
        ax.set_xlabel(self._get_x_label())
        ax.set_ylabel(self._get_y_label())
        ax.grid(True, alpha=0.3)
        
        if plot_count > 0:
            ax.legend(loc='best', fontsize=8)
        
        self.figure.tight_layout()
        
        # Create canvas and toolbar
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        
        self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(self.canvas)
        
        # Update summary
        self._update_summary()
    
    def _populate_file_list(self):
        """Populate the file list with sorted results and colors."""
        # Sort all successful results
        sorted_results = self._sort_results(self.batch_result.successful_results)
        
        # Get color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        # Clear and populate file list
        self.file_list.setRowCount(0)
        self.file_colors.clear()
        
        for idx, result in enumerate(sorted_results):
            # Get color for this file
            color_str = colors[idx % len(colors)]
            # Convert hex color to RGB tuple
            if color_str.startswith('#'):
                color = tuple(int(color_str[i:i+2], 16)/255 for i in (1, 3, 5))
            else:
                # Handle named colors or other formats
                import matplotlib.colors as mcolors
                color = mcolors.to_rgb(color_str)
            
            self.file_colors[result.base_name] = color
            self.file_list.add_file(result.base_name, color, checked=True)
    
    def _on_selection_changed(self):
        """Handle file selection changes."""
        self._update_plot()
    
    def _update_summary(self):
        """Update the summary label."""
        selected = len(self.file_list.get_selected_files())
        total = len(self.batch_result.successful_results)
        self.summary_label.setText(f"{selected} of {total} files selected")
    
    def _get_x_label(self):
        """Get X-axis label using PlotFormatter logic."""
        data, label = self.plot_formatter._extract_axis_data(
            [], self.batch_result.parameters.x_axis, 1
        )
        return label
    
    def _get_y_label(self):
        """Get Y-axis label using PlotFormatter logic."""
        data, label = self.plot_formatter._extract_axis_data(
            [], self.batch_result.parameters.y_axis, 1
        )
        return label
    
    def _update_title_with_results(self):
        """Update window title with result summary."""
        success = len(self.batch_result.successful_results)
        failed = len(self.batch_result.failed_results)
        rate = self.batch_result.success_rate
        
        title = f"Batch Results - {success} files ({rate:.0f}% success)"
        if failed:
            title += f", {failed} failed"
        
        self.setWindowTitle(title)
    
    def _add_export_controls(self, layout):
        """Add export controls."""
        button_layout = QHBoxLayout()
        
        # Create buttons
        export_csvs_btn = QPushButton("Export Individual CSVs...")
        export_plot_btn = QPushButton("Export Plot...")
        
        # IV-specific exports if applicable
        if self._is_iv_analysis():
            export_iv_summary_btn = QPushButton("Export IV Summary...")
            button_layout.addWidget(export_iv_summary_btn)
            export_iv_summary_btn.clicked.connect(self._export_iv_summary)
            
            # Add Current Density Analysis button
            current_density_btn = QPushButton("Current Density Analysis...")
            current_density_btn.setStyleSheet("QPushButton { background-color: #4682b4; }")
            button_layout.addWidget(current_density_btn)
            current_density_btn.clicked.connect(self._open_current_density_analysis)
        
        button_layout.addWidget(export_csvs_btn)
        button_layout.addWidget(export_plot_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Connect signals
        export_csvs_btn.clicked.connect(self._export_individual_csvs)
        export_plot_btn.clicked.connect(self._export_plot)
        
        # Populate file list after UI is created
        self._populate_file_list()
        self._update_summary()
    
    def _is_iv_analysis(self):
        """Check if this is an IV analysis."""
        params = self.batch_result.parameters
        return (
            params.x_axis.channel == "Voltage"
            and params.y_axis.channel == "Current"
            and {params.x_axis.measure, params.y_axis.measure} <= {"Average", "Current"}
        )
    
    def _get_filtered_results(self):
        """Get results filtered by current selection."""
        selected_files = self.file_list.get_selected_files()
        filtered = [
            r for r in self.batch_result.successful_results
            if r.base_name in selected_files
        ]
        return self._sort_results(filtered)
    
    def _export_iv_summary(self):
        """Export IV summary for selected files only."""
        from data_analysis_gui.core.iv_analysis import IVAnalysisService, IVSummaryExporter
        
        # Get filtered results
        filtered_results = self._get_filtered_results()
        
        if not filtered_results:
            QMessageBox.warning(self, "No Data", "No files selected for export.")
            return
        
        # Prepare IV data for selected files only
        batch_data = {
            r.base_name: {
                'x_values': r.x_data.tolist(),
                'y_values': r.y_data.tolist(),
                'x_values2': r.x_data2.tolist() if r.x_data2 is not None else None,
                'y_values2': r.y_data2.tolist() if r.y_data2 is not None else None
            }
            for r in filtered_results
        }
        
        iv_data_r1, mapping, iv_data_r2 = IVAnalysisService.prepare_iv_data(
            batch_data, self.batch_result.parameters
        )
        
        # Get export path
        file_path = self.file_dialog_service.get_export_path(
            self, "IV_Summary.csv",
            file_types="CSV files (*.csv)"
        )
        
        if file_path:
            try:
                # Prepare table for Range 1
                selected_set = set(r.base_name for r in filtered_results)
                table = IVSummaryExporter.prepare_summary_table(
                    iv_data_r1, mapping, selected_set
                )
                
                # New API: DataManager.export_to_csv(table_dict, path)
                result = self.export_service.export_to_csv(table, file_path)
                
                if result.success:
                    QMessageBox.information(
                        self, "Export Complete",
                        f"Exported IV summary with {len(filtered_results)} files"
                    )
                else:
                    QMessageBox.warning(self, "Export Failed", result.error_message)
                    
            except Exception as e:
                logger.error(f"IV summary export failed: {e}", exc_info=True)
                QMessageBox.critical(self, "Export Failed", str(e))
    
    def _export_individual_csvs(self):
        """Export individual CSVs for selected files only."""
        filtered_results = self._get_filtered_results()
        
        if not filtered_results:
            QMessageBox.warning(self, "No Data", "No files selected for export.")
            return
        
        output_dir = self.file_dialog_service.get_directory(
            self, "Select Output Directory"
        )
        
        if output_dir:
            try:
                # Create a temporary BatchAnalysisResult with only selected files
                from data_analysis_gui.core.models import BatchAnalysisResult
                filtered_batch = BatchAnalysisResult(
                    successful_results=filtered_results,
                    failed_results=[],
                    parameters=self.batch_result.parameters,
                    start_time=self.batch_result.start_time,
                    end_time=self.batch_result.end_time
                )
                
                # New API name on BatchProcessor is export_results(...)
                result = self.batch_service.export_results(filtered_batch, output_dir)

                # BatchExportResult no longer has success_count; compute it
                success_count = sum(1 for r in result.export_results if r.success)
                
                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {success_count} files\n"
                    f"Total: {result.total_records} records"
                )
            except Exception as e:
                logger.error(f"Export failed: {e}", exc_info=True)
                QMessageBox.critical(self, "Export Failed", str(e))
    
    def _export_plot(self):
        """Export current plot (with selected files only)."""
        if not self.figure:
            QMessageBox.warning(self, "No Plot", "No plot to export.")
            return
        
        file_path = self.file_dialog_service.get_export_path(
            self, "batch_plot.png",
            file_types="PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )
        
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(
                    self, "Export Complete",
                    f"Plot saved to {Path(file_path).name}"
                )
            except Exception as e:
                logger.error(f"Failed to export plot: {e}")
                QMessageBox.critical(self, "Export Failed", str(e))

    
    def _build_combined_export_table(self, results=None):
        """Build export table using PlotFormatter patterns."""
        import numpy as np
        
        # Use provided results or get filtered results
        if results is None:
            results = self._get_filtered_results()
        
        if not results:
            return {
                'headers': [],
                'data': np.array([[]]),
                'format_spec': '%.6f'
            }
        
        # Determine if all files have the same x_data
        x_arrays_equal = all(
            np.array_equal(results[0].x_data, r.x_data) 
            for r in results[1:]
        )
        
        # Find maximum data points for padding
        max_points = max(len(r.x_data) for r in results)
        
        # Build headers and data columns
        headers = []
        data_columns = []
        
        # Get axis labels
        x_label = self._get_x_label()
        
        # Add data based on X-value consistency
        if x_arrays_equal:
            # All files share same X values - add once
            headers.append(x_label)
            x_col = np.full(max_points, np.nan)
            x_col[:len(results[0].x_data)] = results[0].x_data
            data_columns.append(x_col)
            
            # Add Y data for each file
            for result in results:
                # Range 1 Y data
                headers.append(result.base_name)
                y_col = np.full(max_points, np.nan)
                y_col[:len(result.y_data)] = result.y_data
                data_columns.append(y_col)
                
                # Range 2 Y data if dual range
                if self.batch_result.parameters.use_dual_range and result.y_data2 is not None:
                    headers.append(f"{result.base_name} (Range 2)")
                    y2_col = np.full(max_points, np.nan)
                    y2_col[:len(result.y_data2)] = result.y_data2
                    data_columns.append(y2_col)
        else:
            # Different X values per file - interleave X,Y pairs
            for result in results:
                # Add X column for this file
                headers.append(f"{x_label} ({result.base_name})")
                x_col = np.full(max_points, np.nan)
                x_col[:len(result.x_data)] = result.x_data
                data_columns.append(x_col)
                
                # Add Y column for this file
                headers.append(result.base_name)
                y_col = np.full(max_points, np.nan)
                y_col[:len(result.y_data)] = result.y_data
                data_columns.append(y_col)
                
                # Add Range 2 if applicable
                if self.batch_result.parameters.use_dual_range and result.y_data2 is not None:
                    headers.append(f"{result.base_name} (Range 2)")
                    y2_col = np.full(max_points, np.nan)
                    y2_col[:len(result.y_data2)] = result.y_data2
                    data_columns.append(y2_col)
        
        # Stack columns into 2D array
        if data_columns:
            data = np.column_stack(data_columns)
        else:
            data = np.array([[]])
        
        return {
            'headers': headers,
            'data': data,
            'format_spec': '%.6f'
        }

    def _open_current_density_analysis(self):
        """Open current density analysis dialog."""
        # Create and show dialog
        dialog = CurrentDensityDialog(self, self.batch_result)
        
        if dialog.exec_():
            # Get Cslow values
            cslow_mapping = dialog.get_cslow_mapping()
            
            if not cslow_mapping:
                QMessageBox.warning(
                    self, 
                    "No Data", 
                    "No Cslow values were entered."
                )
                return
            
            # Create and show current density window
            cd_window = CurrentDensityResultsWindow(
                self,
                self.batch_result,
                cslow_mapping,
                self.data_service,
                self.batch_service
            )
            cd_window.show()