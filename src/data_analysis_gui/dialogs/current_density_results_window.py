"""
Window for displaying current density analysis results.

Author: Data Analysis GUI Contributors
License: MIT
"""

from pathlib import Path
from typing import Dict, Set, Optional, List
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QMessageBox, QTableWidget, QTableWidgetItem,
                             QCheckBox, QLabel, QSplitter, QHeaderView, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np

from data_analysis_gui.gui_services import FileDialogService
from data_analysis_gui.services.current_density_service import CurrentDensityService
from data_analysis_gui.core.models import BatchAnalysisResult
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class CurrentDensityResultsWindow(QMainWindow):
    """Window for displaying current density results with file selection."""
    
    def __init__(self, parent, batch_result: BatchAnalysisResult, 
                 cslow_mapping: Dict[str, float], data_service):
        super().__init__(parent)
        self.batch_result = batch_result
        self.cslow_mapping = cslow_mapping
        self.data_service = data_service
        self.file_dialog_service = FileDialogService()
        self.cd_service = CurrentDensityService()
        
        # State
        self.selected_files: Set[str] = set(cslow_mapping.keys())
        self.current_density_results = {}  # Store calculated results
        self.y_unit = "pA/pF"  # Default unit
        
        # UI elements
        self.figure = None
        self.canvas = None
        
        self.setWindowTitle("Current Density Analysis Results")
        self.setGeometry(175, 175, 1200, 700)
        
        # Calculate current density for all files
        self._calculate_all_current_densities()
        
        self.init_ui()
    
    def _calculate_all_current_densities(self):
        """Pre-calculate current density for all files with Cslow values."""
        for result in self.batch_result.successful_results:
            if result.base_name in self.cslow_mapping:
                cslow = self.cslow_mapping[result.base_name]
                
                # Calculate current density for Range 1
                cd_range1 = self.cd_service.calculate_current_density(
                    result.y_data, cslow
                )
                
                cd_data = {
                    'x_data': result.x_data,
                    'cd_range1': cd_range1,
                    'cslow': cslow
                }
                
                # Calculate for Range 2 if applicable
                if self.batch_result.parameters.use_dual_range and result.y_data2 is not None:
                    cd_data['cd_range2'] = self.cd_service.calculate_current_density(
                        result.y_data2, cslow
                    )
                
                self.current_density_results[result.base_name] = cd_data
    
    def init_ui(self):
        """Initialize the UI."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: File list
        left_panel = self._create_file_list_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Plot
        plot_widget = self._create_plot_widget()
        splitter.addWidget(plot_widget)
        
        splitter.setSizes([360, 840])
        main_layout.addWidget(splitter)
        
        # Export controls
        self._add_export_controls(main_layout)
        
        # Initial plot
        self._update_plot()
    
    def _create_file_list_panel(self) -> QWidget:
        """Create the file list panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Files with Cslow Values:")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        # Table for files
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(3)
        self.file_table.setHorizontalHeaderLabels(["", "File", "Cslow (pF)"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.file_table.setColumnWidth(0, 30)
        self.file_table.setColumnWidth(2, 100)
        
        layout.addWidget(self.file_table)
        
        # Selection controls
        controls_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        select_all_btn.clicked.connect(self._select_all)
        select_none_btn.clicked.connect(self._select_none)
        controls_layout.addWidget(select_all_btn)
        controls_layout.addWidget(select_none_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Summary - create before populating table
        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)
        
        # Populate table after creating all widgets
        self._populate_file_table()
        
        return panel
    
    def _create_plot_widget(self) -> QWidget:
        """Create plot widget with unit selector."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Unit selector
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel("Y-axis unit:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["pA/pF", "µA/cm²", "mA/cm²"])
        self.unit_combo.currentTextChanged.connect(self._on_unit_changed)
        unit_layout.addWidget(self.unit_combo)
        unit_layout.addStretch()
        layout.addLayout(unit_layout)
        
        # Plot area
        self.plot_layout = QVBoxLayout()
        layout.addLayout(self.plot_layout)
        
        return widget
    
    def _populate_file_table(self):
        """Populate the file table."""
        sorted_files = sorted(self.current_density_results.keys(), 
                            key=lambda x: self._extract_number(x))
        
        self.file_table.setRowCount(len(sorted_files))
        
        for row, filename in enumerate(sorted_files):
            # Checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(filename in self.selected_files)
            checkbox.stateChanged.connect(self._on_selection_changed)
            
            # Center checkbox
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            
            self.file_table.setCellWidget(row, 0, checkbox_widget)
            
            # Filename
            self.file_table.setItem(row, 1, QTableWidgetItem(filename))
            
            # Cslow value
            cslow = self.cslow_mapping.get(filename, 0)
            cslow_item = QTableWidgetItem(f"{cslow:.2f}")
            cslow_item.setTextAlignment(Qt.AlignCenter)
            self.file_table.setItem(row, 2, cslow_item)
        
        self._update_summary()
    
    def _extract_number(self, filename: str) -> int:
        """Extract number from filename for sorting."""
        import re
        match = re.search(r'_(\d+)', filename)
        if match:
            return int(match.group(1))
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])
        return 0
    
    def _on_selection_changed(self):
        """Handle selection changes."""
        self._update_selected_files()
        self._update_plot()
        self._update_summary()
    
    def _on_unit_changed(self):
        """Handle unit change."""
        self.y_unit = self.unit_combo.currentText()
        self._update_plot()
    
    def _update_selected_files(self):
        """Update the set of selected files."""
        self.selected_files.clear()
        
        for row in range(self.file_table.rowCount()):
            checkbox_widget = self.file_table.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)
            if checkbox and checkbox.isChecked():
                filename = self.file_table.item(row, 1).text()
                self.selected_files.add(filename)
    
    def _select_all(self):
        """Select all files."""
        for row in range(self.file_table.rowCount()):
            checkbox_widget = self.file_table.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(True)
    
    def _select_none(self):
        """Deselect all files."""
        for row in range(self.file_table.rowCount()):
            checkbox_widget = self.file_table.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(False)
    
    def _update_summary(self):
        """Update the summary label."""
        total = len(self.current_density_results)
        selected = len(self.selected_files)
        self.summary_label.setText(f"{selected} of {total} files selected")
    
    def _get_display_data(self, cd_values: np.ndarray) -> np.ndarray:
        """Convert current density values to selected unit."""
        if self.y_unit == "pA/pF":
            return cd_values
        elif self.y_unit == "µA/cm²":
            # 1 pA/pF = 10 µA/cm²
            return cd_values * 10
        elif self.y_unit == "mA/cm²":
            # 1 pA/pF = 0.01 mA/cm²
            return cd_values * 0.01
        return cd_values
    
    def _update_plot(self):
        """Update the plot."""
        # Clear existing plot
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        if not self.selected_files:
            label = QLabel("No files selected for display")
            label.setAlignment(Qt.AlignCenter)
            self.plot_layout.addWidget(label)
            self.figure = None
            self.canvas = None
            return
        
        # Create figure
        self.figure, ax = plt.subplots(figsize=(10, 6))
        
        # Sort selected files
        sorted_files = sorted(self.selected_files, key=self._extract_number)
        
        # Plot each file
        for filename in sorted_files:
            if filename in self.current_density_results:
                data = self.current_density_results[filename]
                x_data = data['x_data']
                
                # Convert and plot Range 1
                cd_display = self._get_display_data(data['cd_range1'])
                ax.plot(x_data, cd_display, 'o-', label=filename, 
                       markersize=4, alpha=0.8)
                
                # Plot Range 2 if available
                if 'cd_range2' in data:
                    cd2_display = self._get_display_data(data['cd_range2'])
                    ax.plot(x_data, cd2_display, 's--', 
                           label=f"{filename} (Range 2)",
                           markersize=4, alpha=0.8)
        
        # Configure plot
        ax.set_xlabel("Voltage (mV)")
        ax.set_ylabel(f"Current Density ({self.y_unit})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        self.figure.tight_layout()
        
        # Create canvas and toolbar
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        
        self.plot_layout.addWidget(toolbar)
        self.plot_layout.addWidget(self.canvas)
    
    def _add_export_controls(self, layout):
        """Add export controls."""
        button_layout = QHBoxLayout()
        
        export_csvs_btn = QPushButton("Export Individual CSVs...")
        export_summary_btn = QPushButton("Export Summary CSV...")
        export_plot_btn = QPushButton("Export Plot...")
        
        button_layout.addWidget(export_csvs_btn)
        button_layout.addWidget(export_summary_btn)
        button_layout.addWidget(export_plot_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Connect signals
        export_csvs_btn.clicked.connect(self._export_individual_csvs)
        export_summary_btn.clicked.connect(self._export_summary_csv)
        export_plot_btn.clicked.connect(self._export_plot)
    
    def _export_individual_csvs(self):
        """Export individual CSV files for selected files."""
        if not self.selected_files:
            QMessageBox.warning(self, "No Selection", 
                              "Please select files to export.")
            return
        
        output_dir = self.file_dialog_service.get_directory(
            self, "Select Output Directory"
        )
        
        if not output_dir:
            return
        
        success_count = 0
        
        for filename in self.selected_files:
            if filename not in self.current_density_results:
                continue
            
            data = self.current_density_results[filename]
            
            # Prepare export data
            export_data = self.cd_service.prepare_export_data(
                voltages=data['x_data'],
                current_density=self._get_display_data(data['cd_range1']),
                cslow_pf=data['cslow'],
                y_unit=self.y_unit
            )
            
            # Export
            output_path = Path(output_dir) / f"{filename}_current_density.csv"
            result = self.data_service.export_to_csv(export_data, str(output_path))
            
            if result.success:
                success_count += 1
        
        QMessageBox.information(
            self, "Export Complete",
            f"Exported {success_count} files to {Path(output_dir).name}"
        )
    
    def _export_summary_csv(self):
        """Export summary CSV."""
        if not self.selected_files:
            QMessageBox.warning(self, "No Selection", 
                              "Please select files to export.")
            return
        
        # Prepare data for summary export
        voltage_data = {}  # voltage -> list of current densities
        file_mapping = {}  # recording_id -> filename
        
        # Get all unique voltages
        all_voltages = set()
        for filename in self.selected_files:
            if filename in self.current_density_results:
                data = self.current_density_results[filename]
                all_voltages.update(data['x_data'])
        
        # Initialize voltage data
        for v in sorted(all_voltages):
            voltage_data[round(v, 1)] = []
        
        # Build data structure matching IV analysis format
        sorted_files = sorted(self.selected_files, key=self._extract_number)
        for idx, filename in enumerate(sorted_files):
            if filename not in self.current_density_results:
                continue
            
            data = self.current_density_results[filename]
            recording_id = f"Recording {idx + 1}"
            file_mapping[recording_id] = filename
            
            # Add current density values
            for v, cd in zip(data['x_data'], data['cd_range1']):
                rounded_v = round(v, 1)
                if rounded_v in voltage_data:
                    cd_display = self._get_display_data(np.array([cd]))[0]
                    voltage_data[rounded_v].append(cd_display)
        
        # Get filename
        suggested = f"Current_Density_Summary_{self.y_unit.replace('/', '_')}.csv"
        file_path = self.file_dialog_service.get_export_path(
            self, suggested, file_types="CSV files (*.csv)"
        )
        
        if not file_path:
            return
        
        # Prepare export
        export_data = self.cd_service.prepare_summary_export(
            voltage_data=voltage_data,
            file_mapping=file_mapping,
            cslow_mapping=self.cslow_mapping,
            included_files=self.selected_files,
            y_unit=self.y_unit
        )
        
        # Export
        result = self.data_service.export_to_csv(export_data, file_path)
        
        if result.success:
            QMessageBox.information(
                self, "Export Complete",
                f"Exported summary with {len(self.selected_files)} files"
            )
        else:
            QMessageBox.warning(self, "Export Failed", result.error_message)
    
    def _export_plot(self):
        """Export the current plot."""
        if not self.figure:
            QMessageBox.warning(self, "No Plot", "No plot to export.")
            return
        
        file_path = self.file_dialog_service.get_export_path(
            self, "current_density_plot.png",
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