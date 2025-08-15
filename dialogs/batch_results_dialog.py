from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QVBoxLayout, QWidget,
                             QPushButton, QCheckBox, QFileDialog, QMessageBox,
                             QGroupBox)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Internal imports
from .current_density_iv_dialog import CurrentDensityIVDialog


class BatchResultDialog(QDialog):
    """Dialog for displaying batch analysis results"""
    def __init__(self, parent, batch_data, batch_fig, iv_data=None, iv_file_mapping=None):
        super().__init__(parent)
        self.batch_data = batch_data
        self.batch_fig = batch_fig
        self.iv_data = iv_data
        self.iv_file_mapping = iv_file_mapping or {}
        self.batch_checkboxes = {}

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
            # Create a checkbox for the file
            checkbox = QCheckBox(file_name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, f=file_name: self.update_plot_visibility(f))
            file_layout.addWidget(checkbox)
            self.batch_checkboxes[file_name] = checkbox

            # This list will store all plot lines associated with the current file
            lines_for_file = []

            # Check for the first data range (Range 1)
            # If data exists for it, associate the next plot line with this file
            if 'y_values' in file_data and len(file_data['y_values']) > 0 and line_idx_counter < len(all_plot_lines):
                lines_for_file.append(all_plot_lines[line_idx_counter])
                line_idx_counter += 1

            # Check for the second data range (Range 2 from dual analysis)
            # If data exists, associate the next plot line with this file as well
            if 'y_values2' in file_data and len(file_data['y_values2']) > 0 and line_idx_counter < len(all_plot_lines):
                lines_for_file.append(all_plot_lines[line_idx_counter])
                line_idx_counter += 1

            # Map the file name to its collected plot lines (either one or two)
            self.batch_plot_lines[file_name] = lines_for_file

        left_layout.addWidget(file_group)

        # Buttons
        button_layout = QVBoxLayout()

        if self.iv_data:
            iv_btn = QPushButton("Current Density I-V Analysis")
            iv_btn.clicked.connect(self.generate_current_density_iv)
            button_layout.addWidget(iv_btn)

        export_btn = QPushButton("Export Plot Image")
        export_btn.clicked.connect(self.export_plot_image)
        button_layout.addWidget(export_btn)

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
        dialog = CurrentDensityIVDialog(self, self.iv_data, self.iv_file_mapping)
        dialog.exec()

    def export_plot_image(self):
        """Export plot as image"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Plot", "", "PNG files (*.png)")
        if file_path:
            self.batch_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", f"Plot saved to {file_path}")