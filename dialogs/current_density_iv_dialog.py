# ck852/data-analysis-gui/ck852-Data-Analysis-GUI-1c45529b512f622436a8ba3f47b384da73789119/dialogs/current_density_iv_dialog.py

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
from widgets import SelectAllSpinBox
from config import DEFAULT_SETTINGS
from utils import export_to_csv, calculate_current_density, calculate_sem


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

            # Initialize file data
            num_recordings = len(self.iv_file_mapping) if self.iv_file_mapping else max([len(current_values) for current_values in self.iv_data.values()])

            for i in range(num_recordings):
                file_id = f"Recording {i+1}"
                self.file_data[file_id] = {'data': {}, 'included': True, 'cslow': DEFAULT_SETTINGS['cslow_default']}

            # Populate data
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
                # Set the checkbox state based on the passed dictionary
                if self.included_files and base_file_name in self.included_files:
                    checkbox.setChecked(self.included_files[base_file_name])
                else:
                    checkbox.setChecked(True) # Default to checked if not found
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

    def show_initial_plot(self):
        """Show initial plot with raw current data (not density)"""
        self.ax.clear()

        # Collect data from included files
        voltage_current_pairs = {}

        for file_id, file_info in self.file_data.items():
            for voltage, current in file_info['data'].items():
                if voltage not in voltage_current_pairs:
                    voltage_current_pairs[voltage] = []
                voltage_current_pairs[voltage].append(current)

        # Calculate average and SEM
        voltages = sorted(voltage_current_pairs.keys())
        current_means = []
        current_sems = []

        for voltage in voltages:
            current_values = voltage_current_pairs[voltage]
            mean_current = np.mean(current_values)
            sem = calculate_sem(current_values)
            current_means.append(mean_current)
            current_sems.append(sem)

        # Plot raw current data
        self.ax.errorbar(voltages, current_means, yerr=current_sems,
                        fmt='o-', capsize=5, linewidth=2, markersize=8)

        self.ax.set_xlabel("Voltage (mV)")
        self.ax.set_ylabel("Current (pA)")
        self.ax.set_title("Average I-V Relationship")
        self.ax.grid(True, alpha=0.3)

        # Add count
        self.ax.text(0.95, 0.05, f"n = {len(self.file_data)} recordings",
                     transform=self.ax.transAxes,
                     horizontalalignment='right',
                     verticalalignment='bottom',
                     bbox=dict(facecolor='white', alpha=0.7))

        self.figure.tight_layout()
        self.canvas.draw()

    def apply_to_all(self):
        """Apply Cslow value to all entries - only called when Apply button is clicked"""
        value = self.apply_all_spin.value()
        for file_id in self.file_data:
            self.cslow_entries[file_id].setValue(value)
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
            if not self.checkboxes[file_id].isChecked():
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

    def export_plot_image(self):
        """Export plot as image"""
        if self.cd_analysis_folder:
            os.makedirs(self.cd_analysis_folder, exist_ok=True)
            default_path = os.path.join(self.cd_analysis_folder, "current_density_plot.png")
        else:
            default_path = "current_density_plot.png"
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Plot", default_path, "PNG files (*.png)")
        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", f"Plot saved to {file_path}")

    def export_individual_files(self):
        """Export individual files"""
        if self.cd_analysis_folder:
            folder_path = self.cd_analysis_folder
        else:
            folder_path = QFileDialog.getExistingDirectory(self, "Select Destination Folder")
        
        if not folder_path:
            return
            
        os.makedirs(folder_path, exist_ok=True)
        files_exported = 0

        for file_id, file_info in self.file_data.items():
            if not self.checkboxes[file_id].isChecked():
                continue

            cslow = file_info['cslow']
            if cslow <= 0:
                continue

            # Get filename
            if file_id in self.iv_file_mapping:
                file_basename = self.iv_file_mapping[file_id]
            else:
                file_basename = file_id.replace(" ", "_")

            export_path = os.path.join(folder_path, f"{file_basename}_CD.csv")

            # Collect data
            voltages = []
            current_densities = []

            for voltage, current in file_info['data'].items():
                voltages.append(voltage)
                current_densities.append(calculate_current_density(current, cslow))

            # Sort and export
            sorted_indices = np.argsort(voltages)
            sorted_voltages = np.array(voltages)[sorted_indices]
            sorted_currents = np.array(current_densities)[sorted_indices]

            header = f"Voltage (mV),Current Density (pA/pF),Cslow = {cslow:.2f} pF"
            export_data = np.column_stack((sorted_voltages, sorted_currents))
            export_to_csv(export_path, export_data, header, '%.6f')

            files_exported += 1

        QMessageBox.information(self, "Export Successful",
                               f"{files_exported} files exported to {folder_path}")

    def export_all_data(self):
        """
        Exports the current density data for all included files into a single CSV file.
        The first column contains the voltage, and subsequent columns contain the
        current density for each file.
        """
        if self.cd_analysis_folder:
            os.makedirs(self.cd_analysis_folder, exist_ok=True)
            default_path = os.path.join(self.cd_analysis_folder, "Current_Density_Summary.csv")
        else:
            default_path = "Current_Density_Summary.csv"
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Export All Data to CSV", default_path, "CSV files (*.csv)")
        if not file_path:
            return

        # 1. Identify which files are included for the export
        included_files = []
        for file_id, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                cslow = self.cslow_entries[file_id].value()
                if cslow > 0:
                    # Use the descriptive filename from the mapping, or the recording ID as a fallback
                    file_name = self.iv_file_mapping.get(file_id, file_id)
                    included_files.append({
                        'id': file_id,
                        'name': file_name,
                        'cslow': cslow
                    })

        if not included_files:
            QMessageBox.warning(self, "Export Error", "No files are included for export.")
            return

        # 2. Assume voltages are consistent and get them from the first included file
        try:
            first_file_id = included_files[0]['id']
            voltages = sorted(self.file_data[first_file_id]['data'].keys())
        except IndexError:
            QMessageBox.warning(self, "Export Error", "Could not retrieve voltage data.")
            return

        # 3. Create the header row for the CSV file
        header = ["Voltage (mV)"] + [f['name'] for f in included_files]

        # 4. Prepare the data for export, starting with the voltage column
        data_to_export = [voltages]

        # 5. Calculate and add a current density column for each file
        for file_info in included_files:
            file_id = file_info['id']
            cslow = file_info['cslow']
            raw_data = self.file_data[file_id]['data']

            # Ensure current densities align with the sorted voltages
            current_densities = [calculate_current_density(raw_data.get(v, np.nan), cslow) for v in voltages]
            data_to_export.append(current_densities)

        # 6. Transpose the data so that each list becomes a column, and save to CSV
        try:
            # Convert list of lists to a NumPy array and transpose it
            export_array = np.array(data_to_export).T

            # Save the transposed array to a CSV file
            export_to_csv(file_path, export_array, ','.join(header), '%.6f')
            QMessageBox.information(self, "Export Successful", f"All data successfully saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred while saving the file:\n{e}")