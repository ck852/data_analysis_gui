"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Dialog for extracting sweep data to CSV format. Handles sweep selection,
channel filtering (voltage/current/both), time range specification, and
batch processing across multiple files.
"""

from typing import List
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QRadioButton, QCheckBox, QFormLayout, QWidget,
    QMessageBox, QApplication
)

import numpy as np

from data_analysis_gui.config.themes import (
    apply_modern_theme, create_styled_button, style_group_box
)
from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.data_extractor import DataExtractor
from data_analysis_gui.widgets.custom_inputs import NumericLineEdit
from data_analysis_gui.widgets.sweep_select_list import SweepSelectionWidget
from data_analysis_gui.gui_services import FileDialogService, ClipboardService

from data_analysis_gui.config.session_settings import save_extract_sweeps_settings, load_extract_sweeps_settings

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class SweepExtractorDialog(QDialog):
    """
    Dialog for extracting selected sweeps to CSV format.
    
    Allows sweep selection, channel choice (voltage/current/both), and optional
    time range specification. Exports with proper headers and units.
    """
    
    def __init__(self, parent, dataset: ElectrophysiologyDataset, file_path: str, 
                default_start: float = 0.0, default_end: float = None):
        super().__init__(parent)
        
        self.dataset = dataset
        self.file_path = file_path
        self.data_extractor = DataExtractor()
        
        self.default_start = default_start
        self.default_end = default_end
        
        if hasattr(parent, 'file_dialog_service'):
            self.file_dialog_service = parent.file_dialog_service
        else:
            self.file_dialog_service = FileDialogService()
        
        self.sweep_names = sorted(dataset.sweeps(), 
                                key=lambda x: int(x) if x.isdigit() else 0)
        
        channel_config = dataset.metadata.get('channel_config', {})
        self.voltage_units = channel_config.get('voltage_units', 'mV')
        self.current_units = channel_config.get('current_units', 'pA')
        
        self.setWindowTitle("Sweep Extractor")
        self.setModal(True)
        
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.resize(int(420), int(avail.height() * 0.9))
        
        self._init_ui()
        self._connect_signals()
        self._load_channel_mode()
        
        apply_modern_theme(self)
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self._create_file_info_section(layout)
        self._create_sweep_selection_section(layout)
        
        horizontal_section = QHBoxLayout()
        horizontal_section.setSpacing(15)
        
        self._create_channel_selection_section(horizontal_section)
        self._create_time_range_section(horizontal_section)
        
        layout.addLayout(horizontal_section)
        self._create_action_buttons(layout)
        
    def _create_file_info_section(self, layout):
        file_group = QGroupBox("Source File")
        style_group_box(file_group)
        file_layout = QVBoxLayout(file_group)
        
        file_name = Path(self.file_path).name
        file_label = QLabel(f"File: {file_name}")
        file_layout.addWidget(file_label)
        
        sweep_count = self.dataset.sweep_count()
        count_label = QLabel(f"Total Sweeps: {sweep_count}")
        file_layout.addWidget(count_label)
        
        layout.addWidget(file_group)
        
    def _create_sweep_selection_section(self, layout):
        sweep_group = QGroupBox(f"Select Sweeps to Extract")
        style_group_box(sweep_group)
        sweep_layout = QVBoxLayout(sweep_group)
        
        self.sweep_selection = SweepSelectionWidget(self.sweep_names)
        self.sweep_selection.select_all(True)
        sweep_layout.addWidget(self.sweep_selection)
        
        button_row = QHBoxLayout()
        select_all_btn = create_styled_button("Select All", "secondary")
        select_none_btn = create_styled_button("Select None", "secondary")
        
        select_all_btn.clicked.connect(lambda: self.sweep_selection.select_all(True))
        select_none_btn.clicked.connect(lambda: self.sweep_selection.select_all(False))
        
        button_row.addWidget(select_all_btn)
        button_row.addWidget(select_none_btn)
        button_row.addStretch()
        sweep_layout.addLayout(button_row)
        
        layout.addWidget(sweep_group)
        
    def _create_channel_selection_section(self, layout):
        channel_group = QGroupBox("Channel to Extract")
        style_group_box(channel_group)
        channel_layout = QVBoxLayout(channel_group)
        
        self.voltage_radio = QRadioButton("Voltage")
        self.current_radio = QRadioButton("Current")
        self.both_radio = QRadioButton("Both Channels")
        
        self.voltage_radio.setChecked(True)
        
        channel_layout.addWidget(self.voltage_radio)
        channel_layout.addWidget(self.current_radio)
        channel_layout.addWidget(self.both_radio)
        
        layout.addWidget(channel_group)
        

    def _create_time_range_section(self, layout):
        """
        Time range controls with 'use full trace' checkbox.
        
        Clicking spinboxes auto-unchecks the checkbox for convenience.
        """
        time_group = QGroupBox("Analysis Time Range")
        style_group_box(time_group)
        time_layout = QVBoxLayout(time_group)
        
        self.full_trace_checkbox = QCheckBox("Use full trace")
        self.full_trace_checkbox.setChecked(True)
        time_layout.addWidget(self.full_trace_checkbox)
        
        range_widget = QWidget()
        range_layout = QFormLayout(range_widget)
        range_layout.setSpacing(8)
        
        max_time = self.dataset.get_max_sweep_time()
        end_value = self.default_end if self.default_end is not None else max_time
        
        self.start_spinbox = NumericLineEdit()
        self.start_spinbox.setRange(0.0, max_time)
        self.start_spinbox.setDecimals(2)
        self.start_spinbox.setValue(self.default_start)
        self.start_spinbox.setMinimumWidth(80)
        self.start_spinbox.setMaximumWidth(100)
        
        self.end_spinbox = NumericLineEdit()
        self.end_spinbox.setRange(0.0, max_time)
        self.end_spinbox.setDecimals(2)
        self.end_spinbox.setValue(end_value)
        self.end_spinbox.setMinimumWidth(80)
        self.end_spinbox.setMaximumWidth(100)
        
        range_layout.addRow("Start (ms):", self.start_spinbox)
        range_layout.addRow("End (ms):", self.end_spinbox)
        
        time_layout.addWidget(range_widget)
        
        # Set initial disabled state after widgets are added
        self.start_spinbox.setEnabled(False)
        self.end_spinbox.setEnabled(False)
        
        layout.addWidget(time_group)
        
    def _create_action_buttons(self, layout):
        button_layout = QHBoxLayout()
        
        self.export_btn = create_styled_button("Export to CSV...", "primary")
        self.export_btn.setMinimumHeight(40)
        
        self.batch_extract_btn = create_styled_button("Batch Extract...", "accent")
        self.batch_extract_btn.setMinimumHeight(40)

        self.copy_btn = create_styled_button("Copy Data", "secondary")
        self.copy_btn.setMinimumHeight(40)
        
        self.close_btn = create_styled_button("Close", "secondary")
        
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.batch_extract_btn)
        button_layout.addWidget(self.copy_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def _connect_signals(self):
        self.full_trace_checkbox.toggled.connect(self._on_full_trace_toggled)
        
        self.start_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.end_spinbox.valueChanged.connect(self._on_spinbox_changed)
        
        self.voltage_radio.toggled.connect(self._save_channel_mode)
        self.current_radio.toggled.connect(self._save_channel_mode)
        self.both_radio.toggled.connect(self._save_channel_mode)
        
        self.export_btn.clicked.connect(self._export_sweeps)
        self.batch_extract_btn.clicked.connect(self._batch_extract_sweeps)
        self.copy_btn.clicked.connect(self._copy_sweeps_to_clipboard)
        self.close_btn.clicked.connect(self.close)

    def _load_channel_mode(self):
        """Restore last-used channel mode from session settings."""
        
        settings = load_extract_sweeps_settings()
        if settings and 'channel_mode' in settings:
            mode = settings['channel_mode']
            logger.debug(f"Loading saved channel mode: {mode}")
            
            # Block signals to prevent triggering save while loading
            self.voltage_radio.blockSignals(True)
            self.current_radio.blockSignals(True)
            self.both_radio.blockSignals(True)
            
            if mode == 'voltage':
                self.voltage_radio.setChecked(True)
            elif mode == 'current':
                self.current_radio.setChecked(True)
            elif mode == 'both':
                self.both_radio.setChecked(True)
            
            self.voltage_radio.blockSignals(False)
            self.current_radio.blockSignals(False)
            self.both_radio.blockSignals(False)


    def _save_channel_mode(self):
        """Persist current channel mode to session settings."""
        
        sender = self.sender()
        if sender and isinstance(sender, QRadioButton) and not sender.isChecked():
            return
        
        mode = self._get_selected_channel_mode()
        
        settings = {
            'channel_mode': mode
        }
        
        success = save_extract_sweeps_settings(settings)
        if success:
            logger.debug(f"Saved channel mode: {mode}")
        else:
            logger.warning(f"Failed to save channel mode preference: {mode}")

    def _copy_sweeps_to_clipboard(self):
        """Export sweep data to system clipboard as TSV for paste into Excel/Prism."""
        self._save_channel_mode()
        
        selected_sweeps, invalid_sweeps = self.sweep_selection.get_selected_sweeps()
        
        if invalid_sweeps:
            QMessageBox.warning(
                self, "Invalid Sweeps",
                f"Sweep(s) {', '.join(invalid_sweeps)} not found in file.\n"
                f"Proceeding with valid sweeps only."
            )
        
        if not selected_sweeps:
            QMessageBox.warning(
                self, "No Sweeps Selected",
                "Please select at least one sweep to copy."
            )
            return
        
        if self.full_trace_checkbox.isChecked():
            start_time = 0.0
            end_time = self.dataset.get_max_sweep_time()
        else:
            start_time = self.start_spinbox.value()
            end_time = self.end_spinbox.value()
            
            if start_time >= end_time:
                QMessageBox.warning(
                    self, "Invalid Time Range",
                    "Start time must be less than end time."
                )
                return
        
        channel_mode = self._get_selected_channel_mode()
        
        try:
            all_data = {}
            reference_time = None
            
            for sweep_idx in selected_sweeps:
                try:
                    sweep_data = self.data_extractor.extract_sweep_data(self.dataset, sweep_idx)
                    time_ms = sweep_data['time_ms']
                    voltage = sweep_data['voltage']
                    current = sweep_data['current']
                    
                    mask = (time_ms >= start_time) & (time_ms <= end_time)
                    filtered_time = time_ms[mask]
                    filtered_voltage = voltage[mask]
                    filtered_current = current[mask]
                    
                    if reference_time is None:
                        reference_time = filtered_time
                    
                    all_data[sweep_idx] = {
                        'time': filtered_time,
                        'voltage': filtered_voltage,
                        'current': filtered_current
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not extract sweep {sweep_idx}: {e}")
                    all_data[sweep_idx] = {
                        'time': reference_time if reference_time is not None else np.array([]),
                        'voltage': np.full_like(reference_time, np.nan) if reference_time is not None else np.array([]),
                        'current': np.full_like(reference_time, np.nan) if reference_time is not None else np.array([])
                    }
            
            if reference_time is None or len(reference_time) == 0:
                raise ValueError("No valid data extracted from selected sweeps")
            
            headers, data_array = self._build_csv_arrays(all_data, selected_sweeps, channel_mode, reference_time)
            
            export_data = {
                'headers': headers,
                'data': data_array.tolist()
            }
            
            success = ClipboardService.copy_data_to_clipboard(export_data)
            
            if success:
                logger.info(f"Sweep data copied to clipboard: {len(selected_sweeps)} sweeps")
                
        except Exception as e:
            logger.error(f"Error copying sweep data: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Copy Error",
                f"Failed to copy data:\n{str(e)}"
            )

    def _on_full_trace_toggled(self, checked: bool):
        """Enable/disable time range spinboxes based on checkbox state."""
        enabled = not checked
        self.start_spinbox.setEnabled(enabled)
        self.end_spinbox.setEnabled(enabled)

    def _on_spinbox_changed(self):
        """Auto-uncheck 'use full trace' when user edits time range directly."""
        if self.full_trace_checkbox.isChecked():
            self.full_trace_checkbox.blockSignals(True)
            self.full_trace_checkbox.setChecked(False)
            self.full_trace_checkbox.blockSignals(False)
            
            self.start_spinbox.setEnabled(True)
            self.end_spinbox.setEnabled(True)
        
    def _get_selected_channel_mode(self) -> str:
        """Return 'voltage', 'current', or 'both' based on radio selection."""
        if self.voltage_radio.isChecked():
            return 'voltage'
        elif self.current_radio.isChecked():
            return 'current'
        else:
            return 'both'
            
    def _export_sweeps(self):
        """Handle single-file CSV export workflow."""
        self._save_channel_mode()

        selected_sweeps, invalid_sweeps = self.sweep_selection.get_selected_sweeps()
        
        if invalid_sweeps:
            QMessageBox.warning(
                self, "Invalid Sweeps",
                f"Sweep(s) {', '.join(invalid_sweeps)} not found in file.\n"
                f"Proceeding with valid sweeps only."
            )
        
        if not selected_sweeps:
            QMessageBox.warning(
                self, "No Sweeps Selected",
                "Please select at least one sweep to export."
            )
            return
        
        if self.full_trace_checkbox.isChecked():
            start_time = 0.0
            end_time = self.dataset.get_max_sweep_time()
        else:
            start_time = self.start_spinbox.value()
            end_time = self.end_spinbox.value()
            
            if start_time >= end_time:
                QMessageBox.warning(
                    self, "Invalid Time Range",
                    "Start time must be less than end time."
                )
                return
        
        channel_mode = self._get_selected_channel_mode()
        
        base_name = Path(self.file_path).stem
        suggested_name = f"{base_name}_sweeps.csv"
        
        file_path = self.file_dialog_service.get_export_path(
            parent=self,
            suggested_name=suggested_name,
            default_directory=None,
            file_types="CSV files (*.csv)",
            dialog_type="export"
        )
        
        if not file_path:
            return
            
        try:
            self._perform_export(selected_sweeps, channel_mode, start_time, end_time, file_path)
            
            if hasattr(self.parent(), '_auto_save_settings'):
                try:
                    self.parent()._auto_save_settings()
                except Exception as e:
                    logger.warning(f"Could not auto-save settings: {e}")
                    
        except Exception as e:
            logger.error(f"Export failed: {e}")
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export sweeps:\n{str(e)}"
            )
    
    def _batch_extract_sweeps(self):
        """Launch batch extraction dialog with current settings as template."""
        self._save_channel_mode()
        
        selected_sweeps, invalid_sweeps = self.sweep_selection.get_selected_sweeps()
        
        if invalid_sweeps:
            QMessageBox.warning(
                self, "Invalid Sweeps",
                f"Sweep(s) {', '.join(invalid_sweeps)} not found in file.\n"
                f"Please adjust your selection."
            )
            return
        
        if not selected_sweeps:
            QMessageBox.warning(
                self, "No Sweeps Selected",
                "Please select at least one sweep to extract."
            )
            return
        
        channel_mode = self._get_selected_channel_mode()
        
        if self.full_trace_checkbox.isChecked():
            time_range = (0.0, self.dataset.get_max_sweep_time())
        else:
            start_time = self.start_spinbox.value()
            end_time = self.end_spinbox.value()
            
            if start_time >= end_time:
                QMessageBox.warning(
                    self, "Invalid Time Range",
                    "Start time must be less than end time."
                )
                return
            
            time_range = (start_time, end_time)
        
        file_types = (
            "WCP files (*.wcp);;"
            "ABF files (*.abf);;"
            "Data files (*.wcp *.abf);;"
            "All files (*.*)"
        )
        
        file_paths = self.file_dialog_service.get_import_paths(
            self,
            "Select Files for Batch Extraction",
            default_directory=None,
            file_types=file_types,
            dialog_type="batch_import"
        )
        
        if not file_paths:
            return
        
        from data_analysis_gui.dialogs.batch_sweep_extractor_dialog import BatchSweepExtractorDialog
        
        dialog = BatchSweepExtractorDialog(
            parent=self,
            initial_files=file_paths,
            sweep_indices=selected_sweeps,
            channel_mode=channel_mode,
            time_range=time_range
        )
        dialog.exec()
            
    def _perform_export(self, selected_sweeps: List[str], channel_mode: str,
                       start_time: float, end_time: float, file_path: str):
        """
        Execute data extraction and write CSV file.
        
        Failed sweeps are filled with NaN rather than aborting the entire export.
        First valid sweep's time array becomes the reference for all subsequent sweeps.
        """
        all_data = {}
        reference_time = None
        
        for sweep_idx in selected_sweeps:
            try:
                sweep_data = self.data_extractor.extract_sweep_data(self.dataset, sweep_idx)
                time_ms = sweep_data['time_ms']
                voltage = sweep_data['voltage']
                current = sweep_data['current']
                
                mask = (time_ms >= start_time) & (time_ms <= end_time)
                filtered_time = time_ms[mask]
                filtered_voltage = voltage[mask]
                filtered_current = current[mask]
                
                if reference_time is None:
                    reference_time = filtered_time
                
                all_data[sweep_idx] = {
                    'time': filtered_time,
                    'voltage': filtered_voltage,
                    'current': filtered_current
                }
                
            except Exception as e:
                logger.warning(f"Could not extract sweep {sweep_idx}: {e}")
                all_data[sweep_idx] = {
                    'time': reference_time if reference_time is not None else np.array([]),
                    'voltage': np.full_like(reference_time, np.nan) if reference_time is not None else np.array([]),
                    'current': np.full_like(reference_time, np.nan) if reference_time is not None else np.array([])
                }
        
        if reference_time is None or len(reference_time) == 0:
            raise ValueError("No valid data extracted from selected sweeps")
        
        headers, data_array = self._build_csv_arrays(all_data, selected_sweeps, channel_mode, reference_time)
        
        header_str = ",".join(headers)
        np.savetxt(
            file_path,
            data_array,
            delimiter=",",
            fmt="%.6f",
            header=header_str,
            comments="",
            encoding="utf-8"
        )
        
        num_records = len(data_array)
        QMessageBox.information(
            self, "Export Complete",
            f"Successfully exported {len(selected_sweeps)} sweeps "
            f"with {num_records} data points to:\n{Path(file_path).name}"
        )
        
        logger.info(f"Exported {len(selected_sweeps)} sweeps to {file_path}")
        
    def _build_csv_arrays(self, all_data: dict, selected_sweeps: List[str],
                        channel_mode: str, reference_time: np.ndarray):
        """
        Construct headers and data array for CSV output.
        
        When channel_mode='both', all voltage columns are grouped together
        followed by all current columns, rather than interleaving V/I pairs.
        """
        headers = ["Time (ms)"]
        columns = [reference_time]
        
        if channel_mode == 'voltage':
            for sweep_idx in selected_sweeps:
                sweep_data = all_data[sweep_idx]
                headers.append(f"Sweep {sweep_idx} Voltage ({self.voltage_units})")
                columns.append(sweep_data['voltage'])
                
        elif channel_mode == 'current':
            for sweep_idx in selected_sweeps:
                sweep_data = all_data[sweep_idx]
                headers.append(f"Sweep {sweep_idx} Current ({self.current_units})")
                columns.append(sweep_data['current'])
                
        else:  # both
            for sweep_idx in selected_sweeps:
                sweep_data = all_data[sweep_idx]
                headers.append(f"Sweep {sweep_idx} Voltage ({self.voltage_units})")
                columns.append(sweep_data['voltage'])
            
            for sweep_idx in selected_sweeps:
                sweep_data = all_data[sweep_idx]
                headers.append(f"Sweep {sweep_idx} Current ({self.current_units})")
                columns.append(sweep_data['current'])
        
        data_array = np.column_stack(columns)
        
        return headers, data_array