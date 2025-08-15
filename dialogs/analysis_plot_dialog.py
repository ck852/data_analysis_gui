import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QMessageBox)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Internal imports
from utils import export_to_csv


class AnalysisPlotDialog(QDialog):
    """Dialog for displaying analysis plot in a separate window"""
    def __init__(self, parent, plot_data, x_label, y_label, title):
        super().__init__(parent)
        self.plot_data = plot_data
        self.x_label = x_label
        self.y_label = y_label
        self.plot_title = title

        self.setWindowTitle("Analysis Plot")
        self.setGeometry(200, 200, 800, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create plot
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Create toolbar
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        # Create the plot
        self.create_plot()

        # Add export button
        button_layout = QHBoxLayout()

        export_img_btn = QPushButton("Export Plot Image")
        export_img_btn.clicked.connect(self.export_plot_image)
        button_layout.addWidget(export_img_btn)

        export_data_btn = QPushButton("Export Data")
        export_data_btn.clicked.connect(self.export_data)
        button_layout.addWidget(export_data_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def create_plot(self):
        """Create the analysis plot"""
        x_data = self.plot_data['x_data']
        y_data = self.plot_data['y_data']
        sweep_indices = self.plot_data['sweep_indices']
        use_dual_range = self.plot_data.get('use_dual_range', False)

        if len(x_data) > 0 and len(y_data) > 0:
            # Create scatter plot with connecting lines for Range 1
            self.ax.plot(x_data, y_data, 'o-', linewidth=2, markersize=6, label="Range 1")

            # Add sweep labels to the points (only for the first series to avoid clutter)
            for i, sweep_idx in enumerate(sweep_indices):
                if i < len(x_data) and i < len(y_data):
                    self.ax.annotate(f"{sweep_idx}",
                                   (x_data[i], y_data[i]),
                                   textcoords="offset points",
                                   xytext=(0, 5),
                                   ha='center')

        # Plot Range 2 if available
        if use_dual_range:
            y_data2 = self.plot_data.get('y_data2', [])
            if len(x_data) > 0 and len(y_data2) > 0:
                self.ax.plot(x_data, y_data2, 's--', linewidth=2, markersize=6, label="Range 2")

        # Format plot
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.plot_title)
        self.ax.grid(True, alpha=0.3)

        # Add legend if needed
        if use_dual_range:
            self.ax.legend()

        # Autoscale to fit data
        self.ax.relim()
        self.ax.autoscale_view()

        # Add padding to both axes
        if len(x_data) > 0 and len(y_data) > 0:
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()

            x_range = x_max - x_min
            y_range = y_max - y_min

            x_padding = x_range * 0.05 if x_range > 0 else 0.1
            y_padding = y_range * 0.05 if y_range > 0 else 0.1

            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)

        self.figure.tight_layout()
        self.canvas.draw()

    def export_plot_image(self):
        """Export plot as image"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Plot", "", "PNG files (*.png)")
        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", f"Plot saved to {file_path}")

    def export_data(self):
        """Export data as CSV"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "CSV files (*.csv)")
        if file_path:
            x_data = self.plot_data['x_data']
            y_data = self.plot_data['y_data']
            use_dual_range = self.plot_data.get('use_dual_range', False)

            if use_dual_range:
                y_data2 = self.plot_data.get('y_data2', [])
                export_data = np.column_stack((x_data, y_data, y_data2))

                # Use the descriptive labels passed from the main window
                y_label_r1 = self.plot_data.get('y_label_r1', f"{self.y_label} (Range 1)")
                y_label_r2 = self.plot_data.get('y_label_r2', f"{self.y_label} (Range 2)")
                header = f"{self.x_label},{y_label_r1},{y_label_r2}"
            else:
                export_data = np.column_stack((x_data, y_data))
                header = f"{self.x_label},{self.y_label}"

            export_to_csv(file_path, export_data, header, '%.6f')
            QMessageBox.information(self, "Export Successful", f"Data saved to {file_path}")