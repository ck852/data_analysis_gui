# src/data_analysis_gui/dialogs/current_density_iv_dialog.py

import subprocess
import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QVBoxLayout, QWidget,
                             QPushButton, QCheckBox, QFileDialog, QMessageBox,
                             QGroupBox, QGridLayout, QLabel, QAbstractSpinBox)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Internal imports
from data_analysis_gui.widgets import SelectAllSpinBox
from data_analysis_gui.config import DEFAULT_SETTINGS
from data_analysis_gui.utils import calculate_current_density, calculate_sem
from data_analysis_gui.services.export_service import ExportService
from data_analysis_gui.core.current_density_exporter import CurrentDensityExporter


class CurrentDensityIVDialog(QDialog):
    """Dialog for displaying Current Density I-V curves"""
    def __init__(self, parent, iv_data, iv_file_mapping=None, included_files=None, destination_folder=None):
        super().__init__(parent)
        self.iv_data = iv_data
        self.iv_file_mapping = iv_file_mapping or {}
        self.file_data = {}
        self.checkboxes = {}
        self.cslow_entries = {}
        self.included_files = included_files # Store the included files dictionary
        self.destination_folder = destination_folder
        self.cd_analysis_folder = None
        if self.destination_folder:
            self.cd_analysis_folder = os.path.join(self.destination_folder, "Current Density Analysis")

        self.setWindowTitle("Current Density I-V")
        self.setGeometry(200, 200, 1200, 800)
        self.init_ui()
        self.ask_confirm_apply_all = True

    def init_ui(self):
        layout = QHBoxLayout(self)

        # Left panel for file selection and Cslow input
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # File settings group
        file_group = QGroupBox("File Settings")
        file_layout = QGridLayout(file_group)

        # Headers
        file_layout.addWidget(QLabel("File"), 0, 0)
        file_layout.addWidget(QLabel("Include"), 0, 1)
        file_layout.addWidget(QLabel("Cslow (pF)"), 0, 2)

        # Initialize file data based on iv_file_mapping
        if self.iv_file_mapping:
            # Use the actual file mapping provided
            for recording_id in sorted(self.iv_file_mapping.keys(), 
                                    key=lambda x: int(x.split()[-1])):
                file_name = self.iv_file_mapping[recording_id]
                # Extract index from "Recording X" format
                idx = int(recording_id.split()[-1]) - 1
                self.file_data[recording_id] = {
                    'data': {}, 
                    'included': self.included_files.get(file_name, True) if self.included_files else True,
                    'cslow': DEFAULT_SETTINGS['cslow_default']
                }
                
            # Populate data - using the actual indices from the mapping
            for voltage in sorted(self.iv_data.keys()):
                current_values = self.iv_data[voltage]
                for recording_id in self.file_data.keys():
                    idx = int(recording_id.split()[-1]) - 1
                    if idx < len(current_values):
                        self.file_data[recording_id]['data'][voltage] = current_values[idx]
        else:
            # Fallback for when no mapping is provided
            num_recordings = max([len(current_values) for current_values in self.iv_data.values()])
            for i in range(num_recordings):
                file_id = f"Recording {i+1}"
                self.file_data[file_id] = {'data': {}, 'included': True, 'cslow': DEFAULT_SETTINGS['cslow_default']}
                
            for voltage in sorted(self.iv_data.keys()):
                current_values = self.iv_data[voltage]
                for i, current in enumerate(current_values):
                    if i < num_recordings:
                        file_id = f"Recording {i+1}"
                        if file_id in self.file_data:
                            self.file_data[file_id]['data'][voltage] = current

        # Create controls for each file
        row = 1
        for file_id in self.file_data:
            display_name = file_id
            base_file_name = None
            if file_id in self.iv_file_mapping:
                base_file_name = self.iv_file_mapping[file_id]
                display_name = f"{base_file_name}"

            # File label
            file_label = QLabel(display_name)
            file_label.setMaximumWidth(200)
            file_layout.addWidget(file_label, row, 0)

            # Checkbox
            checkbox = QCheckBox()
            # Use the file_data included state which was set from included_files
            checkbox.setChecked(self.file_data[file_id]['included'])
            checkbox.stateChanged.connect(
                lambda state, fid=file_id: self.update_file_inclusion(fid, state)
            )
            file_layout.addWidget(checkbox, row, 1)
            self.checkboxes[file_id] = checkbox

            # Cslow entry with default value, no arrow buttons
            cslow_entry = SelectAllSpinBox()
            cslow_entry.setRange(0.1, 1000.0)
            cslow_entry.setValue(DEFAULT_SETTINGS['cslow_default'])
            cslow_entry.setSingleStep(0.1)
            cslow_entry.setDecimals(2)
            cslow_entry.setButtonSymbols(QAbstractSpinBox.NoButtons)
            file_layout.addWidget(cslow_entry, row, 2)
            self.cslow_entries[file_id] = cslow_entry

            row += 1

        left_layout.addWidget(file_group)

        # Apply to all group
        apply_group = QGroupBox("Apply to All")
        apply_layout = QHBoxLayout(apply_group)

        apply_layout.addWidget(QLabel("Cslow (pF):"))
        self.apply_all_spin = SelectAllSpinBox()
        self.apply_all_spin.setRange(0.1, 1000.0)
        self.apply_all_spin.setValue(DEFAULT_SETTINGS['cslow_default'])
        self.apply_all_spin.setDecimals(2)
        self.apply_all_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        # Prevent Enter key from triggering apply
        self.apply_all_spin.setKeyboardTracking(False)
        apply_layout.addWidget(self.apply_all_spin)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_to_all)
        apply_btn.setAutoDefault(False)
        apply_btn.setDefault(False)
        apply_layout.addWidget(apply_btn)

        left_layout.addWidget(apply_group)

        # Buttons
        button_layout = QVBoxLayout()

        update_btn = QPushButton("Generate Current Density IV")
        update_btn.clicked.connect(self.update_cd_plot)
        update_btn.setAutoDefault(False)
        button_layout.addWidget(update_btn)

        self.export_img_btn = QPushButton("Export Plot Image")
        self.export_img_btn.clicked.connect(self.export_plot_image)
        self.export_img_btn.setAutoDefault(False)
        self.export_img_btn.setEnabled(False)
        button_layout.addWidget(self.export_img_btn)

        self.export_individual_btn = QPushButton("Export Individual Files")
        self.export_individual_btn.clicked.connect(self.export_individual_files)
        self.export_individual_btn.setAutoDefault(False)
        self.export_individual_btn.setEnabled(False)
        button_layout.addWidget(self.export_individual_btn)

        self.export_all_btn = QPushButton("Export All Data to CSV")
        self.export_all_btn.clicked.connect(self.export_all_data)
        self.export_all_btn.setAutoDefault(False)
        self.export_all_btn.setEnabled(False)
        button_layout.addWidget(self.export_all_btn)

        self.open_dest_folder_btn = QPushButton("Open Destination Folder")
        self.open_dest_folder_btn.clicked.connect(self.open_destination_folder)
        self.open_dest_folder_btn.setAutoDefault(False)
        self.open_dest_folder_btn.setEnabled(False)
        button_layout.addWidget(self.open_dest_folder_btn)

        left_layout.addLayout(button_layout)
        left_layout.addStretch()

        layout.addWidget(left_panel)

        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create plot
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        toolbar = NavigationToolbar(self.canvas, right_panel)
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.canvas)

        layout.addWidget(right_panel)

        # Initial plot (show raw current data)
        self.show_initial_plot()

    def update_file_inclusion(self, file_id, state):
        """Update file inclusion state when checkbox is toggled"""
        self.file_data[file_id]['included'] = (state == 2)  # Qt.Checked

    def show_initial_plot(self):
            """Show an empty plot initially."""
            self.ax.clear()
            self.ax.set_xlabel("Voltage (mV)")
            self.ax.set_ylabel("Current Density (pA/pF)")
            self.ax.set_title("Current Density I-V Relationship")
            self.ax.grid(True, alpha=0.3)
            self.figure.tight_layout()
            self.canvas.draw()

    def apply_to_all(self):
        """Apply Cslow value to all entries (with confirmation)"""
        value = self.apply_all_spin.value()

        if self.ask_confirm_apply_all:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Apply to all?")
            box.setText(
                f"Apply {value:.2f} pF to ALL {len(self.cslow_entries)} recordings?\n"
                "This will overwrite each file's Cslow."
            )
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setDefaultButton(QMessageBox.No)

            #Implement the following after establishing save between sessions
            # dont_ask = QCheckBox("Don't ask me again")
            # box.setCheckBox(dont_ask)

            # if box.exec_() != QMessageBox.Yes:
            #     return
            # if dont_ask.isChecked():
            #     self.ask_confirm_apply_all = False

        # proceed with the update
        for file_id, spin in self.cslow_entries.items():
            spin.setValue(value)
            self.file_data[file_id]['cslow'] = value

    def update_cd_plot(self):
        """Update plot to show current density"""
        self.ax.clear()

        # Update cslow values from entries
        for file_id in self.file_data:
            self.file_data[file_id]['cslow'] = self.cslow_entries[file_id].value()

        # Collect data from included files
        voltage_cd_pairs = {}

        for file_id, file_info in self.file_data.items():
            if not file_info['included']:  # Check the included state in file_data
                continue

            cslow = file_info['cslow']
            if cslow <= 0:
                continue

            for voltage, current in file_info['data'].items():
                # Calculate current density
                current_density = calculate_current_density(current, cslow)

                if voltage not in voltage_cd_pairs:
                    voltage_cd_pairs[voltage] = []
                voltage_cd_pairs[voltage].append(current_density)

        # Calculate average and SEM
        voltages = sorted(voltage_cd_pairs.keys())
        cd_means = []
        cd_sems = []

        for voltage in voltages:
            cd_values = voltage_cd_pairs[voltage]
            mean_cd = np.mean(cd_values)
            sem = calculate_sem(cd_values)
            cd_means.append(mean_cd)
            cd_sems.append(sem)

        # Store for export
        self.export_voltages = np.array(voltages)
        self.export_currents = np.array(cd_means)
        self.export_sems = np.array(cd_sems)

        # Plot current density data
        self.ax.errorbar(voltages, cd_means, yerr=cd_sems,
                        fmt='o-', capsize=5, linewidth=2, markersize=8)

        self.ax.set_xlabel("Voltage (mV)")
        self.ax.set_ylabel("Current Density (pA/pF)")
        self.ax.set_title("Current Density I-V Relationship")
        self.ax.grid(True, alpha=0.3)

        # Add count
        included_files = sum(1 for file_id in self.file_data if self.checkboxes[file_id].isChecked())
        self.ax.text(0.95, 0.05, f"n = {included_files} recordings",
                     transform=self.ax.transAxes,
                     horizontalalignment='right',
                     verticalalignment='bottom',
                     bbox=dict(facecolor='white', alpha=0.7))

        self.figure.tight_layout()
        self.canvas.draw()
        
        # Enable export buttons
        self.export_img_btn.setEnabled(True)
        self.export_individual_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)
        self.open_dest_folder_btn.setEnabled(True)

    def _get_included_files(self):
        """Returns a list of file_id for all checked files."""
        return [file_id for file_id, checkbox in self.checkboxes.items() if checkbox.isChecked()]

    def export_plot_image(self):
        """Export plot as image using centralized service"""
        default_path = ExportService.get_suggested_filename(
            base_name="current_density_plot",
            extension="png",
            destination_folder=self.cd_analysis_folder
        )
        
        result = ExportService.export_plot_image(
            figure=self.figure,
            parent=self,
            default_path=default_path,
            title="Export Current Density Plot"
        )
    
    def export_individual_files(self):
        """Export individual files using centralized service"""
        folder_path = self.cd_analysis_folder or ExportService.select_export_folder(
            parent=self,
            title="Select Destination Folder"
        )
        
        if not folder_path:
            return

        exporter = CurrentDensityExporter(self.file_data, self.iv_file_mapping, self._get_included_files())
        files_data = exporter.prepare_individual_files_data()

        if not files_data:
            QMessageBox.warning(self, "Export Error", "No files are included for export.")
            return

        # Use centralized service
        results = ExportService.export_multiple_files(
            files_data=files_data,
            output_folder=folder_path,
            parent=self,
            show_summary=True
        )
    
    def export_all_data(self):
        """Export all data to a single CSV using centralized service"""
        default_path = ExportService.get_suggested_filename(
            base_name="Current_Density_Summary",
            destination_folder=self.cd_analysis_folder
        )
        
        exporter = CurrentDensityExporter(self.file_data, self.iv_file_mapping, self._get_included_files())
        summary_data = exporter.prepare_summary_data()

        if not summary_data:
            QMessageBox.warning(self, "Export Error", "No files are included for export.")
            return
        
        # Use centralized service
        result = ExportService.export_data_to_csv(
            data=summary_data['data'],
            headers=summary_data['headers'],
            parent=self,
            default_path=default_path,
            title="Export All Data to CSV"
        )
    
    def open_destination_folder(self):
        """Open the destination folder in the file explorer."""
        if self.cd_analysis_folder and os.path.isdir(self.cd_analysis_folder):
            try:
                if sys.platform == "win32":
                    os.startfile(self.cd_analysis_folder)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", self.cd_analysis_folder])
                else:
                    subprocess.Popen(["xdg-open", self.cd_analysis_folder])
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not open folder: {e}")
        else:
            QMessageBox.warning(self, "Warning", "Destination folder not found.")