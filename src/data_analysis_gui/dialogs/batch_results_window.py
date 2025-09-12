# batch_results_window.py

from pathlib import Path
import re
from typing import Set
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QMessageBox, QTableWidget, QTableWidgetItem,
                             QCheckBox, QLabel, QSplitter, QHeaderView, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QPainter, QBrush

from data_analysis_gui.gui_services import FileDialogService
from data_analysis_gui.core.plot_formatter import PlotFormatter
from data_analysis_gui.config.logging import get_logger

from data_analysis_gui.dialogs.current_density_dialog import CurrentDensityDialog
from data_analysis_gui.dialogs.current_density_results_window import CurrentDensityResultsWindow

from data_analysis_gui.widgets.shared_widgets import (
    DynamicBatchPlotWidget, BatchFileListWidget, FileSelectionState
)

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
        
        # Initialize selection state if not present
        if batch_result.selected_files is None:
            # Create mutable copy of batch_result with selection state
            from dataclasses import replace
            batch_result = replace(
                batch_result,
                selected_files={r.base_name for r in batch_result.successful_results}
            )
        
        self.batch_result = batch_result
        self.batch_service = batch_service
        self.plot_service = plot_service
        self.data_service = data_service
        self.file_dialog_service = FileDialogService()
        
        # Create selection state object
        self.selection_state = FileSelectionState(self.batch_result.selected_files)
        
        # Use PlotFormatter for consistent formatting
        self.plot_formatter = PlotFormatter()
        
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
        self.plot_widget = DynamicBatchPlotWidget()
        self.plot_widget.initialize_plot(
            x_label=self._get_x_label(),
            y_label=self._get_y_label(),
            title=""
        )
        splitter.addWidget(self.plot_widget)
        
        # Set initial splitter sizes (30% list, 70% plot)
        splitter.setSizes([360, 840])
        
        main_layout.addWidget(splitter)
        
        # Export controls at bottom
        self._add_export_controls(main_layout)
        
        # Populate and initial plot
        self._populate_file_list()
        self._update_plot()
    
    def _create_file_list_panel(self) -> QWidget:
        """Create the file list panel with controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Files:")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        # File list widget - use the new shared widget
        self.file_list = BatchFileListWidget(self.selection_state, show_cslow=False)
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
    
    def _sort_results(self, results):
        """Sort results numerically by file name."""
        def extract_number(file_name):
            """Extract number from filename for sorting."""
            match = re.search(r'_(\d+)', file_name)
            if match:
                return int(match.group(1))
            numbers = re.findall(r'\d+', file_name)
            if numbers:
                return int(numbers[-1])
            return 0
        
        return sorted(results, key=lambda r: extract_number(r.base_name))
    
    def _populate_file_list(self):
        """Populate the file list with sorted results and colors."""
        sorted_results = self._sort_results(self.batch_result.successful_results)
        
        # Generate color mapping
        color_mapping = self.plot_widget._generate_color_mapping(sorted_results)
        
        # Clear and populate file list
        self.file_list.setRowCount(0)
        
        for result in sorted_results:
            color = color_mapping[result.base_name]
            self.file_list.add_file(result.base_name, color)
    
    def _update_plot(self):
        """Update the plot based on selected files."""
        # Get all results sorted
        sorted_results = self._sort_results(self.batch_result.successful_results)
        
        # Set data in plot widget
        self.plot_widget.set_data(
            sorted_results,
            use_dual_range=self.batch_result.parameters.use_dual_range
        )
        
        # Update visibility based on selection
        self.plot_widget.update_visibility(self.selection_state.get_selected_files())
        
        # Update summary
        self._update_summary()
    
    def _on_selection_changed(self):
        """Handle file selection changes."""
        # Update plot visibility
        self.plot_widget.update_visibility(self.selection_state.get_selected_files())
        self._update_summary()
    
    def _update_summary(self):
        """Update the summary label."""
        selected = len(self.selection_state.get_selected_files())
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
        selected_files = self.selection_state.get_selected_files()
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
                
                result = self.data_service.export_to_csv(table, file_path)
                
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
                from data_analysis_gui.core.models import BatchAnalysisResult
                from dataclasses import replace
                
                # Create a BatchAnalysisResult with current selection state
                filtered_batch = replace(
                    self.batch_result,
                    successful_results=filtered_results,
                    failed_results=[],
                    selected_files=self.selection_state.get_selected_files()
                )
                
                result = self.batch_service.export_results(filtered_batch, output_dir)
                
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
        if not self.plot_widget.figure:
            QMessageBox.warning(self, "No Plot", "No plot to export.")
            return
        
        file_path = self.file_dialog_service.get_export_path(
            self, "batch_plot.png",
            file_types="PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )
        
        if file_path:
            try:
                self.plot_widget.export_figure(file_path)
                QMessageBox.information(
                    self, "Export Complete",
                    f"Plot saved to {Path(file_path).name}"
                )
            except Exception as e:
                logger.error(f"Failed to export plot: {e}")
                QMessageBox.critical(self, "Export Failed", str(e))
    
    def _open_current_density_analysis(self):
        """Open current density analysis dialog."""
        # Pass the batch_result with selection state
        from dataclasses import replace
        batch_with_selection = replace(
            self.batch_result,
            selected_files=self.selection_state.get_selected_files()
        )
        
        # Create and show dialog
        dialog = CurrentDensityDialog(self, batch_with_selection)
        
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
                batch_with_selection,  # Pass batch result with selection state
                cslow_mapping,
                self.data_service,
                self.batch_service
            )
            cd_window.show()