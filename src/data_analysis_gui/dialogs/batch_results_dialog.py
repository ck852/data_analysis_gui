# src/data_analysis_gui/dialogs/batch_results_dialog.py
"""
GUI dialog for displaying batch analysis results.
This is a thin wrapper around the core batch_results module,
handling only GUI-specific interactions.
"""

import os
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QVBoxLayout, QWidget,
                             QPushButton, QCheckBox, QFileDialog, QMessageBox,
                             QGroupBox, QLabel)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Core imports - all data processing is delegated to these
from data_analysis_gui.core.batch_results import (
    BatchResultsData, 
    BatchResultsExporter,
    BatchResultsAnalyzer
)

# CRITICAL: Use the SAME export functions that tests validate
from data_analysis_gui.core.exporter import write_tables, write_single_table
from data_analysis_gui.services.export_service import ExportService

# Dialog imports
from .current_density_iv_dialog import CurrentDensityIVDialog


class BatchResultDialog(QDialog):
    """Dialog for displaying batch analysis results"""
    
    def __init__(self, parent, batch_result, batch_fig, iv_data=None, 
                 iv_file_mapping=None, x_label=None, y_label=None, 
                 destination_folder=None):
        super().__init__(parent)
        
        # Store the batch_result object - this is what gets exported
        self.batch_result = batch_result
        
        # Extract batch_data from batch_result for display purposes
        batch_data = self._extract_batch_data_from_result(batch_result)
        
        # Create core data structure for UI state management
        self.results_data = BatchResultsData(
            batch_data=batch_data,
            iv_data=iv_data,
            iv_file_mapping=iv_file_mapping or {},
            x_label=x_label,
            y_label=y_label,
            destination_folder=destination_folder
        )
        
        # Create core components
        self.exporter = BatchResultsExporter(self.results_data)
        self.analyzer = BatchResultsAnalyzer()
        
        # GUI state
        self.batch_fig = batch_fig
        self.batch_checkboxes = {}
        self.batch_plot_lines = {}
        
        self.setWindowTitle("Batch Analysis Results")
        self.setGeometry(200, 200, 1200, 800)
        self.init_ui()
    
    def _extract_batch_data_from_result(self, batch_result):
        """Extract display data from BatchResult object"""
        batch_data = {}
        
        if batch_result and hasattr(batch_result, 'successful_results'):
            for result in batch_result.successful_results:
                base_name = result.base_name
                
                # Extract data for display
                if hasattr(result, 'export_table'):
                    # Assuming export_table has the x and y values
                    batch_data[base_name] = {
                        'x_values': result.export_table.get('x_values', []),
                        'y_values': result.export_table.get('y_values', []),
                        'y_values2': result.export_table.get('y_values2', [])
                    }
        
        return batch_data
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QHBoxLayout(self)
        
        # Left panel for file visibility
        left_panel = self._create_left_panel()
        layout.addWidget(left_panel)
        
        # Right panel for plot
        right_panel = self._create_right_panel()
        layout.addWidget(right_panel)
        
        self.canvas.draw()
    
    def _create_left_panel(self):
        """Create the left control panel"""
        left_panel = QWidget()
        left_panel.setMaximumWidth(250)
        left_layout = QVBoxLayout(left_panel)
        
        # File visibility group
        file_group = self._create_file_visibility_group()
        left_layout.addWidget(file_group)
        
        # Buttons
        button_layout = self._create_button_layout()
        left_layout.addLayout(button_layout)
        
        # Statistics display
        stats_group = self._create_statistics_group()
        left_layout.addWidget(stats_group)
        
        left_layout.addStretch()
        
        return left_panel
    
    def _create_file_visibility_group(self):
        """Create the file visibility control group"""
        file_group = QGroupBox("Show/Hide Files")
        file_layout = QVBoxLayout(file_group)
        
        # Get plot lines from the figure
        ax = self.batch_fig.get_axes()[0]
        all_plot_lines = ax.get_lines()
        
        # Create a mapping of files to their lines based on line labels
        file_to_lines = {}
        for line in all_plot_lines:
            label = line.get_label()
            if label and not label.startswith('_'):  # Skip matplotlib internal labels
                # Extract base_name from label like "250514_001 (Range 1)" or "250514_001 (Range 2)"
                base_name = label.split(' (Range')[0].strip() if ' (Range' in label else label
                
                # Add this line to the appropriate file's list
                if base_name in self.results_data.batch_data:
                    if base_name not in file_to_lines:
                        file_to_lines[base_name] = []
                    file_to_lines[base_name].append(line)
        
        # Fallback: if no labels matched, distribute lines sequentially
        if not file_to_lines:
            sorted_files = sorted(self.results_data.batch_data.keys())
            line_idx = 0
            
            for file_name in sorted_files:
                file_data = self.results_data.batch_data[file_name]
                lines_for_file = []
                
                # Single range: 1 line per file
                # Dual range: 2 lines per file
                if 'y_values' in file_data and len(file_data['y_values']) > 0:
                    if line_idx < len(all_plot_lines):
                        lines_for_file.append(all_plot_lines[line_idx])
                        line_idx += 1
                    
                    # Check for dual range
                    if 'y_values2' in file_data and len(file_data.get('y_values2', [])) > 0:
                        if line_idx < len(all_plot_lines):
                            lines_for_file.append(all_plot_lines[line_idx])
                            line_idx += 1
                
                if lines_for_file:
                    file_to_lines[file_name] = lines_for_file
        
        # Create checkbox for each file with color swatch
        for file_name in sorted(self.results_data.batch_data.keys()):
            entry_layout = QHBoxLayout()
            entry_layout.setSpacing(5)
            
            # Get the primary color from the first line for this file
            lines_for_file = file_to_lines.get(file_name, [])
            color = None
            
            if lines_for_file:
                # Use the color from the first line (Range 1)
                color = lines_for_file[0].get_color()
            
            # Create color swatch
            color_swatch = QLabel()
            color_swatch.setMinimumSize(15, 15)
            color_swatch.setMaximumSize(15, 15)
            
            if color:
                # Matplotlib colors are already in a format Qt can use
                color_str = str(color)
            else:
                # Fallback color if something went wrong
                color_str = '#808080'
            
            color_swatch.setStyleSheet(
                f"background-color: {color_str}; "
                f"border: 1px solid #333; "
                f"border-radius: 2px;"
            )
            
            # Create checkbox
            checkbox = QCheckBox(file_name)
            checkbox.setChecked(file_name in self.results_data.included_files)
            checkbox.stateChanged.connect(
                lambda state, f=file_name: self._update_file_visibility(f, state)
            )
            self.batch_checkboxes[file_name] = checkbox
            
            # Store the lines for this file (for show/hide functionality)
            self.batch_plot_lines[file_name] = lines_for_file
            
            # Add widgets to layout
            entry_layout.addWidget(color_swatch)
            entry_layout.addWidget(checkbox)
            entry_layout.addStretch()
            
            file_layout.addLayout(entry_layout)
        
        return file_group
    
    def _create_button_layout(self):
        """Create the button layout"""
        button_layout = QVBoxLayout()
        
        # Current Density I-V button (if IV data available)
        if self.results_data.iv_data:
            iv_btn = QPushButton("Current Density I-V")
            iv_btn.clicked.connect(self._generate_current_density_iv)
            button_layout.addWidget(iv_btn)
        
        # Export plot image button
        export_img_btn = QPushButton("Export Plot Image")
        export_img_btn.clicked.connect(self._export_plot_image)
        button_layout.addWidget(export_img_btn)
        
        # Export all data button
        export_all_btn = QPushButton("Export All Data to CSV")
        export_all_btn.clicked.connect(self._export_all_data)
        button_layout.addWidget(export_all_btn)
        
        # Export individual files button
        export_individual_btn = QPushButton("Export Individual Files")
        export_individual_btn.clicked.connect(self._export_individual_files)
        button_layout.addWidget(export_individual_btn)
        
        return button_layout
    
    def _create_statistics_group(self):
        """Create statistics display group"""
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("Calculating...")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        # Update statistics
        self._update_statistics()
        
        return stats_group
    
    def _create_right_panel(self):
        """Create the right panel for plot display"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create canvas for the existing figure
        self.canvas = FigureCanvas(self.batch_fig)
        toolbar = NavigationToolbar(self.canvas, right_panel)
        
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.canvas)
        
        return right_panel
    
    def _update_file_visibility(self, file_name: str, state: int):
        """Update file visibility in both data and plot"""
        is_visible = state == 2  # Qt.Checked
        
        # Update core data
        self.results_data.toggle_file_inclusion(file_name, is_visible)
        
        # Update plot lines
        if file_name in self.batch_plot_lines:
            for line in self.batch_plot_lines[file_name]:
                line.set_visible(is_visible)
            self.canvas.draw()
        
        # Update statistics
        self._update_statistics()
    
    def _update_statistics(self):
        """Update the statistics display"""
        stats = self.analyzer.calculate_statistics(self.results_data)
        
        if stats:
            stats_text = f"""Files: {stats['num_files']}
Mean: {stats['y_mean']:.4f}
Std: {stats['y_std']:.4f}
Min: {stats['y_min']:.4f}
Max: {stats['y_max']:.4f}
Median: {stats['y_median']:.4f}"""
            
            if 'x_range' in stats:
                stats_text += f"\nX Range: {stats['x_range']:.4f}"
        else:
            stats_text = "No data available"
        
        self.stats_label.setText(stats_text)
    
    def _generate_current_density_iv(self):
        """Generate Current Density I-V analysis"""
        # Get all data (not filtered)
        cd_data = self.exporter.prepare_current_density_data()
        
        if not cd_data or not cd_data.get('iv_data'):
            QMessageBox.warning(self, "No Data", 
                            "No IV data available.")
            return
        
        # Create included files dict with actual inclusion state
        included_files = {
            file_name: file_name in self.results_data.included_files
            for file_name in self.results_data.batch_data.keys()
        }
        
        dialog = CurrentDensityIVDialog(
            self, 
            cd_data['iv_data'],
            cd_data['iv_file_mapping'],
            included_files=included_files,
            destination_folder=cd_data['destination_folder']
        )
        dialog.exec()
    
    def _export_plot_image(self):
        """Export plot as image using centralized service"""
        result = ExportService.export_plot_image(
            figure=self.batch_fig,
            parent=self,
            default_path="batch_plot.png",
            title="Export Batch Plot"
        )
    
    def _export_all_data(self):
        """Export all included data to a single CSV file"""
        # This should also potentially use a core function
        # But keeping existing implementation for now
        default_path = ExportService.get_suggested_filename(
            base_name="Summary IV",
            destination_folder=self.results_data.destination_folder
        )
        
        export_data = self.exporter.prepare_combined_export_data()
        
        if not export_data:
            QMessageBox.warning(self, "Export Error", "No data to export")
            return
        
        result = ExportService.export_data_to_csv(
            data=export_data['data'],
            headers=export_data['headers'],
            parent=self,
            default_path=default_path,
            title="Export All Data to CSV"
        )
    
    def _export_individual_files(self):
        """
        Export individual files using the SAME function that tests validate.
        This calls write_tables() directly from core.exporter.
        """
        # Get output folder from user
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self.results_data.destination_folder or ""
        )
        
        if not folder:
            return
        
        # CRITICAL: Call the SAME function that tests use
        # This is write_tables() from data_analysis_gui.core.exporter
        try:
            export_outcomes = write_tables(self.batch_result, folder)
            
            # Count successes and failures
            successful = sum(1 for outcome in export_outcomes if outcome.success)
            failed = sum(1 for outcome in export_outcomes if not outcome.success)
            
            # Show results to user
            if failed == 0:
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Successfully exported {successful} file(s) to:\n{folder}"
                )
            elif successful > 0:
                QMessageBox.warning(
                    self,
                    "Export Partially Complete",
                    f"Exported {successful} file(s), {failed} failed.\nLocation: {folder}"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export all {failed} file(s)."
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred during export:\n{str(e)}"
            )