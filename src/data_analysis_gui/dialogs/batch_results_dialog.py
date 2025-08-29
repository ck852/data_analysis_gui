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

# Dialog imports
from .current_density_iv_dialog import CurrentDensityIVDialog


class BatchResultDialog(QDialog):
    """Dialog for displaying batch analysis results"""
    
    def __init__(self, parent, batch_data, batch_fig, iv_data=None, 
                 iv_file_mapping=None, x_label=None, y_label=None, 
                 destination_folder=None):
        super().__init__(parent)
        
        # Create core data structure
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
        line_idx_counter = 0
        
        # Create checkbox for each file
        for file_name in self.results_data.batch_data.keys():
            entry_layout = QHBoxLayout()
            
            # Color swatch
            color_swatch = QLabel()
            color_swatch.setMinimumSize(20, 20)
            color_swatch.setMaximumSize(20, 20)
            
            # Checkbox
            checkbox = QCheckBox(file_name)
            checkbox.setChecked(file_name in self.results_data.included_files)
            checkbox.stateChanged.connect(
                lambda state, f=file_name: self._update_file_visibility(f, state)
            )
            self.batch_checkboxes[file_name] = checkbox
            
            # Map to plot lines
            lines_for_file = []
            file_data = self.results_data.batch_data[file_name]
            
            if 'y_values' in file_data and len(file_data['y_values']) > 0:
                if line_idx_counter < len(all_plot_lines):
                    line = all_plot_lines[line_idx_counter]
                    lines_for_file.append(line)
                    color = line.get_color()
                    color_swatch.setStyleSheet(
                        f"background-color: {color}; border: 1px solid black;"
                    )
                    line_idx_counter += 1
            
            if 'y_values2' in file_data and len(file_data.get('y_values2', [])) > 0:
                if line_idx_counter < len(all_plot_lines):
                    lines_for_file.append(all_plot_lines[line_idx_counter])
                    line_idx_counter += 1
            
            entry_layout.addWidget(color_swatch)
            entry_layout.addWidget(checkbox)
            file_layout.addLayout(entry_layout)
            
            self.batch_plot_lines[file_name] = lines_for_file
        
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
        # Get filtered data from core
        cd_data = self.exporter.prepare_current_density_data()
        
        if not cd_data or not cd_data.get('iv_data'):
            QMessageBox.warning(self, "No Data", 
                               "No IV data available for included files.")
            return
        
        # Create included files dict for dialog
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
        """Export plot as image"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "", "PNG files (*.png)"
        )
        
        if file_path:
            try:
                self.batch_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(
                    self, "Export Successful", 
                    f"Plot saved to {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Failed",
                    f"Failed to save plot: {str(e)}"
                )
    
    def _export_all_data(self):
        """Export all included data to a single CSV file"""
        # Get default path
        if self.results_data.destination_folder:
            default_path = os.path.join(
                self.results_data.destination_folder, 
                "Summary IV.csv"
            )
        else:
            default_path = "Summary IV.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export All Data to CSV", 
            default_path, 
            "CSV files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            # Use core exporter
            self.exporter.export_all_data_to_csv(file_path)
            QMessageBox.information(
                self, "Export Successful",
                f"All data successfully saved to {file_path}"
            )
        except ValueError as e:
            QMessageBox.warning(self, "Export Error", str(e))
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"An error occurred while saving the file:\n{e}"
            )
    
    def _export_individual_files(self):
        """Export each included file to a separate CSV"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
            self.results_data.destination_folder or ""
        )
        
        if not folder:
            return
        
        try:
            results = self.exporter.export_individual_files(folder)
            
            successful = sum(1 for _, success, _ in results if success)
            failed = len(results) - successful
            
            message = f"Exported {successful} files successfully."
            if failed > 0:
                message += f"\n{failed} files failed to export."
            
            QMessageBox.information(self, "Export Complete", message)
            
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"An error occurred during export:\n{e}"
            )