# dialogs/batch_results_window.py

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QPushButton,
                             QSplitter, QTabWidget)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class BatchServicesWindow(QMainWindow):
    """Window for displaying batch analysis results with plot."""
    
    def __init__(self, parent, batch_result, batch_service, plot_service):
        super().__init__(parent)
        self.batch_result = batch_result
        self.batch_service = batch_service
        self.plot_service = plot_service
        
        self.setWindowTitle("Batch Analysis Results")
        self.setGeometry(150, 150, 1200, 800)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Plot tab
        plot_widget = self.create_plot_widget()
        tabs.addTab(plot_widget, "Combined Plot")
        
        # Results table tab
        table_widget = self.create_results_table()
        tabs.addTab(table_widget, f"Results ({self.batch_result.total_files} files)")
        
        # Export buttons
        button_layout = QHBoxLayout()
        export_all_btn = QPushButton("Export All CSVs...")
        export_plot_btn = QPushButton("Export Plot...")
        button_layout.addWidget(export_all_btn)
        button_layout.addWidget(export_plot_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Connect signals
        export_all_btn.clicked.connect(self.export_all_csvs)
        export_plot_btn.clicked.connect(self.export_plot)
    
    def create_plot_widget(self):
        """Create the plot widget with all batch results."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Build the plot using plot_service
        figure, plot_count = self.plot_service.build_batch_figure(
            self.batch_result,
            self.batch_result.parameters,
            self.get_x_label(),
            self.get_y_label()
        )
        
        # Create canvas and toolbar
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, widget)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        return widget
    
    def create_results_table(self):
        """Create table showing analysis results."""
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "File", "Status", "Data Points", "Processing Time", "Error"
        ])
        
        all_results = self.batch_result.successful_results + self.batch_result.failed_results
        table.setRowCount(len(all_results))
        
        for i, result in enumerate(all_results):
            table.setItem(i, 0, QTableWidgetItem(result.base_name))
            table.setItem(i, 1, QTableWidgetItem("✓" if result.success else "✗"))
            table.setItem(i, 2, QTableWidgetItem(str(len(result.x_data))))
            table.setItem(i, 3, QTableWidgetItem(f"{result.processing_time:.2f}s"))
            table.setItem(i, 4, QTableWidgetItem(result.error_message or ""))
        
        return table