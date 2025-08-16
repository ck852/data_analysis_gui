import os
import numpy as np
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QVBoxLayout, QWidget,
                             QPushButton, QCheckBox, QFileDialog, QMessageBox,
                             QGroupBox, QLabel)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Internal imports
from .current_density_iv_dialog import CurrentDensityIVDialog
from utils import export_to_csv


class BatchResultDialog(QDialog):
    """Dialog for displaying batch analysis results"""
    def __init__(self, parent, batch_data, batch_fig, iv_data=None, iv_file_mapping=None, x_label=None, y_label=None, destination_folder=None):
        super().__init__(parent)
        self.batch_data = batch_data
        self.batch_fig = batch_fig
        self.iv_data = iv_data
        self.iv_file_mapping = iv_file_mapping or {}
        self.batch_checkboxes = {}
        self.x_label = x_label
        self.y_label = y_label
        self.destination_folder = destination_folder

        self.setWindowTitle("Batch Analysis Results")
        self.setGeometry(200, 200, 1200, 800)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # Left panel for file visibility
        left_panel = QWidget()
        left_panel.setMaximumWidth(250)
        left_layout = QVBoxLayout(left_panel)

        # File visibility group
        file_group = QGroupBox("Show/Hide Files")
        file_layout = QVBoxLayout(file_group)

        # Get all plot lines from the figure's axes
        self.batch_plot_lines = {}
        ax = self.batch_fig.get_axes()[0]
        all_plot_lines = ax.get_lines()
        line_idx_counter = 0

        # Iterate through each file's data to link it to its plot lines
        for file_name, file_data in self.batch_data.items():
            entry_layout = QHBoxLayout()
            
            color_swatch = QLabel()
            color_swatch.setMinimumSize(20, 20)
            color_swatch.setMaximumSize(20, 20)

            checkbox = QCheckBox(file_name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, f=file_name: self.update_plot_visibility(f))
            self.batch_checkboxes[file_name] = checkbox

            lines_for_file = []

            if 'y_values' in file_data and len(file_data['y_values']) > 0 and line_idx_counter < len(all_plot_lines):
                line = all_plot_lines[line_idx_counter]
                lines_for_file.append(line)
                color = line.get_color()
                color_swatch.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
                line_idx_counter += 1

            if 'y_values2' in file_data and len(file_data['y_values2']) > 0 and line_idx_counter < len(all_plot_lines):
                lines_for_file.append(all_plot_lines[line_idx_counter])
                line_idx_counter += 1
            
            entry_layout.addWidget(color_swatch)
            entry_layout.addWidget(checkbox)
            file_layout.addLayout(entry_layout)

            self.batch_plot_lines[file_name] = lines_for_file

        left_layout.addWidget(file_group)

        # Buttons
        button_layout = QVBoxLayout()

        if self.iv_data:
            iv_btn = QPushButton("Current Density I-V")
            iv_btn.clicked.connect(self.generate_current_density_iv)
            button_layout.addWidget(iv_btn)

        export_btn = QPushButton("Export Plot Image")
        export_btn.clicked.connect(self.export_plot_image)
        button_layout.addWidget(export_btn)
        
        export_all_btn = QPushButton("Export All Data to CSV")
        export_all_btn.clicked.connect(self.export_all_data)
        button_layout.addWidget(export_all_btn)

        left_layout.addLayout(button_layout)
        left_layout.addStretch()

        layout.addWidget(left_panel)

        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create canvas for the existing figure
        self.canvas = FigureCanvas(self.batch_fig)
        toolbar = NavigationToolbar(self.canvas, right_panel)

        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.canvas)

        layout.addWidget(right_panel)

        self.canvas.draw()

    def update_plot_visibility(self, file_name):
        """Update plot line visibility"""
        if file_name in self.batch_plot_lines:
            is_visible = self.batch_checkboxes[file_name].isChecked()
            for line in self.batch_plot_lines[file_name]:
                line.set_visible(is_visible)
            self.canvas.draw()

    def generate_current_density_iv(self):
        """Generate Current Density I-V analysis"""
        # Get the current state of included files
        included_files = {file_name: checkbox.isChecked() for file_name, checkbox in self.batch_checkboxes.items()}

        dialog = CurrentDensityIVDialog(self, self.iv_data, self.iv_file_mapping, included_files=included_files)
        dialog.exec()

    def export_plot_image(self):
        """Export plot as image"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Plot", "", "PNG files (*.png)")
        if file_path:
            self.batch_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", f"Plot saved to {file_path}")
            
    def export_all_data(self):
        """
        Exports the batch analysis data for all included files into a single CSV file.
        """
        if self.destination_folder:
            default_path = os.path.join(self.destination_folder, "Summary IV.csv")
        else:
            default_path = "Summary IV.csv"

        file_path, _ = QFileDialog.getSaveFileName(self, "Export All Data to CSV", default_path, "CSV files (*.csv)")
        
        if not file_path:
            return

        included_files = []
        for file_name, checkbox in self.batch_checkboxes.items():
            if checkbox.isChecked():
                included_files.append(file_name)

        if not included_files:
            QMessageBox.warning(self, "Export Error", "No files are included for export.")
            return

        first_file_name = included_files[0]
        x_values = self.batch_data[first_file_name].get('x_values', [])

        header = [self.x_label] + included_files

        data_to_export = [x_values]
        for file_name in included_files:
            y_values = self.batch_data[file_name].get('y_values', [])
            if len(y_values) < len(x_values):
                y_values.extend([np.nan] * (len(x_values) - len(y_values)))
            data_to_export.append(y_values)

        try:
            export_array = np.array(data_to_export).T
            export_to_csv(file_path, export_array, ','.join(header), '%.6f')
            QMessageBox.information(self, "Export Successful", f"All data successfully saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred while saving the file:\n{e}")