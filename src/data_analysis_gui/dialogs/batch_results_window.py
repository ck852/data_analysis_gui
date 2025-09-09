# dialogs/batch_results_window.py

from pathlib import Path
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QMessageBox)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from data_analysis_gui.gui_services import FileDialogService
from data_analysis_gui.core.models import ExportResult
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class BatchResultsWindow(QMainWindow):
    """Window for displaying batch analysis results plot."""
    
    def __init__(self, parent, batch_result, batch_service, plot_service, export_service):
        super().__init__(parent)
        self.batch_result = batch_result
        self.batch_service = batch_service
        self.plot_service = plot_service
        self.export_service = export_service
        self.file_dialog_service = FileDialogService()
        
        # Store the figure for export
        self.figure = None
        
        self.setWindowTitle("Batch Analysis Results")
        self.setGeometry(150, 150, 1000, 700)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI - simplified to just show plot."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Create the plot
        plot_widget = self._create_plot_widget()
        layout.addWidget(plot_widget)
        
        # Export buttons
        button_layout = QHBoxLayout()
        export_csvs_btn = QPushButton("Export Individual CSVs...")
        export_plot_btn = QPushButton("Export Plot...")
        export_summary_btn = QPushButton("Export Combined CSV...")
        
        button_layout.addWidget(export_csvs_btn)
        button_layout.addWidget(export_plot_btn)
        button_layout.addWidget(export_summary_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Connect signals
        export_csvs_btn.clicked.connect(self._export_individual_csvs)
        export_plot_btn.clicked.connect(self._export_plot)
        export_summary_btn.clicked.connect(self._export_combined_csv)
    
    def _create_plot_widget(self):
        """Create the plot widget using PlotService."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        try:
            # Use PlotService to build the figure
            self.figure, plot_count = self.plot_service.build_batch_figure(
                self.batch_result,
                self.batch_result.parameters,
                self._get_axis_label(self.batch_result.parameters.x_axis),
                self._get_axis_label(self.batch_result.parameters.y_axis)
            )
            
            # Create canvas and toolbar
            canvas = FigureCanvas(self.figure)
            toolbar = NavigationToolbar(canvas, widget)
            
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            # Show status in window title
            success = len(self.batch_result.successful_results)
            failed = len(self.batch_result.failed_results)
            self.setWindowTitle(
                f"Batch Analysis Results - {success} files"
                f"{f', {failed} failed' if failed else ''}"
            )
            
        except Exception as e:
            logger.error(f"Failed to create plot: {e}", exc_info=True)
            from PyQt5.QtWidgets import QLabel
            layout.addWidget(QLabel(f"Failed to create plot: {str(e)}"))
        
        return widget
    
    def _get_axis_label(self, axis_config):
        """Generate axis label from AxisConfig."""
        if axis_config.measure == "Time":
            return "Time (s)"
        
        unit = "mV" if axis_config.channel == "Voltage" else "pA"
        
        if axis_config.measure == "Average":
            return f"Average {axis_config.channel} ({unit})"
        else:  # Peak
            peak_type = axis_config.peak_type or "Absolute"
            return f"{peak_type} Peak {axis_config.channel} ({unit})"
    
    def _export_individual_csvs(self):
        """Export each file's results to separate CSV files."""
        if not self.batch_result.successful_results:
            QMessageBox.warning(self, "No Data", "No successful results to export.")
            return
        
        # Get output directory
        output_dir = self.file_dialog_service.get_directory(
            self, "Select Output Directory for CSV Files"
        )
        
        if not output_dir:
            return
        
        try:
            # Use BatchService's export method
            export_result = self.batch_service.export_batch_results(
                self.batch_result, output_dir
            )
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {export_result.success_count} files\n"
                f"Total records: {export_result.total_records}"
            )
            
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", str(e))
    
    def _export_plot(self):
        """Export the plot image."""
        if not self.figure:
            QMessageBox.warning(self, "No Plot", "No plot to export.")
            return
        
        file_path = self.file_dialog_service.get_export_path(
            self,
            "batch_analysis_plot.png",
            file_types="PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )
        
        if not file_path:
            return
        
        try:
            # Save the existing figure
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(
                self, "Export Complete", 
                f"Plot saved to {Path(file_path).name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to export plot: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", str(e))
    
    def _export_combined_csv(self):
        """Export all results combined in one CSV file."""
        if not self.batch_result.successful_results:
            QMessageBox.warning(self, "No Data", "No successful results to export.")
            return
        
        file_path = self.file_dialog_service.get_export_path(
            self,
            "batch_combined.csv",
            file_types="CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Build combined export table
            export_table = self._build_combined_table()
            
            # Use ExportService to save
            result = self.export_service.export_analysis_data(
                export_table, file_path
            )
            
            if result.success:
                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {result.records_exported} records"
                )
            else:
                QMessageBox.warning(self, "Export Failed", result.error_message)
                
        except Exception as e:
            logger.error(f"Failed to export combined CSV: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", str(e))
    
    def _build_combined_table(self):
        """Build a combined export table from all successful results."""
        import numpy as np
        
        successful = self.batch_result.successful_results
        if not successful:
            return {'headers': [], 'data': np.array([[]]), 'format_spec': '%.6f'}
        
        # Find max data points
        max_points = max(len(r.x_data) for r in successful)
        
        # Build headers
        headers = []
        data_columns = []
        
        # Add data from each file
        for result in successful:
            # Add X column (only for first file to avoid duplicates if all X are same)
            if not headers:  # First file
                headers.append(self._get_axis_label(self.batch_result.parameters.x_axis))
                # Pad with NaN if needed
                x_col = np.full(max_points, np.nan)
                x_col[:len(result.x_data)] = result.x_data
                data_columns.append(x_col)
            
            # Add Y column
            headers.append(f"{result.base_name}")
            y_col = np.full(max_points, np.nan)
            y_col[:len(result.y_data)] = result.y_data
            data_columns.append(y_col)
            
            # Add Y2 if dual range
            if self.batch_result.parameters.use_dual_range and result.y_data2 is not None:
                headers.append(f"{result.base_name} (Range 2)")
                y2_col = np.full(max_points, np.nan)
                y2_col[:len(result.y_data2)] = result.y_data2
                data_columns.append(y2_col)
        
        # Combine into array
        data = np.column_stack(data_columns)
        
        return {
            'headers': headers,
            'data': data,
            'format_spec': '%.6f'
        }