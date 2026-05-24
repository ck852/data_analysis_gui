"""
Batch Sweep Extractor Dialog 

PatchBatch Electrophysiology Data Analysis Tool

This module provides a dialog for batch extracting selected sweeps from multiple
data files to a combined CSV format. Users can select multiple files, and the dialog
will extract the same sweeps from each file, combining them into a single output.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import List, Tuple, Dict, Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QListWidget, QMessageBox, QAbstractItemView, QFormLayout,
    QRadioButton, QCheckBox, QWidget, QApplication
)

import numpy as np

from data_analysis_gui.config.themes import (
    apply_modern_theme, create_styled_button, style_button, style_group_box,
    style_list_widget, style_label, get_file_count_color
)
from data_analysis_gui.services.data_manager import DataManager
from data_analysis_gui.core.data_extractor import DataExtractor
from data_analysis_gui.gui_services import FileDialogService, ClipboardService
from data_analysis_gui.widgets.custom_inputs import NumericLineEdit
from data_analysis_gui.widgets.sweep_select_list import SweepSelectionWidget
from data_analysis_gui.config.logging import get_logger
from data_analysis_gui.core.file_sort import filename_sort_key, clean_filename

logger = get_logger(__name__)


class BatchSweepExtractorDialog(QDialog):
    """
    Dialog for batch extracting sweeps from multiple files.
    
    Allows users to select multiple files and extract the same sweeps with the same
    parameters, combining all results into a single CSV output.
    """
    
    def __init__(self, parent, initial_files: List[str], sweep_indices: List[str],
                channel_mode: str, time_range: Tuple[float, float],
                sweep_names: List[str], max_time: float):

        super().__init__(parent)
        
        # Initial extraction parameters (used to populate editable widgets;
        # the live values come from the widgets at extraction time)
        self.initial_sweep_indices = sweep_indices
        self.initial_channel_mode = channel_mode
        self.initial_time_range = time_range
        
        # Reference data from the parent's loaded file
        self.sweep_names = sweep_names
        self.max_time = max_time
        
        # Initialize file list
        self.file_paths = list(initial_files)
        
        # Services
        self.data_manager = DataManager()
        self.data_extractor = DataExtractor()
        
        # Get file dialog service from parent
        if hasattr(parent, 'file_dialog_service'):
            self.file_dialog_service = parent.file_dialog_service
        else:
            self.file_dialog_service = FileDialogService()
        
        # Initialize batch directory from parent's import_data directory
        self._initialize_batch_directory()
        
        # State
        self.extraction_result: Optional[Dict] = None
        self.had_missing_data = False
        self.had_time_mismatch = False
        
        self.setWindowTitle("Batch Sweep Extraction")
        self.setModal(True)
        
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.resize(int(900), int(avail.height() * 0.7))
        
        self._init_ui()
        
        apply_modern_theme(self)
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Top section: file list and sweep selection side-by-side
        top_section = QHBoxLayout()
        top_section.setSpacing(15)
        self._create_file_list_section(top_section)
        self._create_sweep_selection_section(top_section)
        layout.addLayout(top_section)
        
        # Middle section: channel mode and time range side-by-side
        params_section = QHBoxLayout()
        params_section.setSpacing(15)
        self._create_channel_selection_section(params_section)
        self._create_time_range_section(params_section)
        layout.addLayout(params_section)
        
        # Status section
        self._create_status_section(layout)
        
        # Action buttons
        self._create_action_buttons(layout)
        
        # Wire up signals to mark parameters as stale after extraction
        self._connect_stale_signals()

    def _initialize_batch_directory(self):
        """
        Initialize batch_sweep_extract directory from parent's current import_data directory.
        
        This ensures that the batch dialog always starts with the same directory
        that was last used in MainWindow for opening data files. Each time the dialog
        is opened, it syncs with the current MainWindow directory, but within the dialog
        session, Add Files and Export CSV will remember their own locations.
        """
        dirs = self.file_dialog_service.get_last_directories()
        
        # Always sync batch_sweep_extract with current import_data when dialog opens
        if 'import_data' in dirs and dirs['import_data']:
            logger.debug(f"Initializing batch directory from import_data: {dirs['import_data']}")
            dirs['batch_import'] = dirs['import_data']
            self.file_dialog_service.set_last_directories(dirs)
        else:
            logger.debug("No import_data directory found, batch dialog will use default") 

    def _create_file_list_section(self, layout):
        """Create the file list management section."""
        file_group = QGroupBox("Files to Process")
        style_group_box(file_group)
        file_layout = QVBoxLayout(file_group)
        
        # File list widget
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        style_list_widget(self.file_list)
        file_layout.addWidget(self.file_list)
        
        # File count label
        self.file_count_label = QLabel()
        style_label(self.file_count_label, "muted")
        file_layout.addWidget(self.file_count_label)
        
        # File management buttons
        button_row = QHBoxLayout()
        self.add_files_btn = create_styled_button("Add Files...", "secondary")
        self.remove_selected_btn = create_styled_button("Remove Selected", "secondary")
        self.clear_all_btn = create_styled_button("Clear All", "secondary")
        
        button_row.addWidget(self.add_files_btn)
        button_row.addWidget(self.remove_selected_btn)
        button_row.addWidget(self.clear_all_btn)
        button_row.addStretch()
        file_layout.addLayout(button_row)
        
        layout.addWidget(file_group)
        
        # Populate initial files
        self._update_file_list()
        
        # Connect signals
        self.add_files_btn.clicked.connect(self._add_files)
        self.remove_selected_btn.clicked.connect(self._remove_selected)
        self.clear_all_btn.clicked.connect(self._clear_files)
        
    def _create_sweep_selection_section(self, layout):
        """Editable sweep selection, pre-populated from the parent dialog's choices."""
        sweep_group = QGroupBox("Select Sweeps to Extract")
        style_group_box(sweep_group)
        sweep_layout = QVBoxLayout(sweep_group)
        
        self.sweep_selection = SweepSelectionWidget(self.sweep_names)
        sweep_layout.addWidget(self.sweep_selection)
        
        # Apply initial selection: uncheck all, then check only the incoming sweeps
        initial_set = set(self.initial_sweep_indices)
        self.sweep_selection.select_all(False)
        for i, sweep_name in enumerate(self.sweep_selection.sweep_names):
            if sweep_name in initial_set:
                cb = self.sweep_selection.table.cellWidget(i, 0)
                if cb:
                    cb.setChecked(True)
        
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
        """Editable channel mode (voltage / current / both)."""
        channel_group = QGroupBox("Channel to Extract")
        style_group_box(channel_group)
        channel_layout = QVBoxLayout(channel_group)
        
        self.voltage_radio = QRadioButton("Voltage")
        self.current_radio = QRadioButton("Current")
        self.both_radio = QRadioButton("Both Channels")
        
        if self.initial_channel_mode == 'voltage':
            self.voltage_radio.setChecked(True)
        elif self.initial_channel_mode == 'current':
            self.current_radio.setChecked(True)
        else:
            self.both_radio.setChecked(True)
        
        channel_layout.addWidget(self.voltage_radio)
        channel_layout.addWidget(self.current_radio)
        channel_layout.addWidget(self.both_radio)
        
        layout.addWidget(channel_group)
    
    def _create_time_range_section(self, layout):
        """Editable time range with 'use full trace' checkbox."""
        time_group = QGroupBox("Analysis Time Range")
        style_group_box(time_group)
        time_layout = QVBoxLayout(time_group)
        
        start_ms, end_ms = self.initial_time_range
        # Detect whether the incoming range covers the full trace
        is_full_trace = (start_ms == 0.0 and abs(end_ms - self.max_time) < 1e-9)
        
        self.full_trace_checkbox = QCheckBox("Use full trace")
        self.full_trace_checkbox.setChecked(is_full_trace)
        time_layout.addWidget(self.full_trace_checkbox)
        
        range_widget = QWidget()
        range_layout = QFormLayout(range_widget)
        range_layout.setSpacing(8)
        
        self.start_spinbox = NumericLineEdit()
        self.start_spinbox.setRange(0.0, self.max_time)
        self.start_spinbox.setDecimals(2)
        self.start_spinbox.setValue(start_ms)
        self.start_spinbox.setMinimumWidth(80)
        self.start_spinbox.setMaximumWidth(100)
        
        self.end_spinbox = NumericLineEdit()
        self.end_spinbox.setRange(0.0, self.max_time)
        self.end_spinbox.setDecimals(2)
        self.end_spinbox.setValue(end_ms)
        self.end_spinbox.setMinimumWidth(80)
        self.end_spinbox.setMaximumWidth(100)
        
        range_layout.addRow("Start (ms):", self.start_spinbox)
        range_layout.addRow("End (ms):", self.end_spinbox)
        time_layout.addWidget(range_widget)
        
        # Disable spinboxes if full trace is checked
        self.start_spinbox.setEnabled(not is_full_trace)
        self.end_spinbox.setEnabled(not is_full_trace)
        
        # Toggle spinbox enable state with checkbox
        self.full_trace_checkbox.toggled.connect(self._on_full_trace_toggled)
        # Auto-uncheck full trace if user edits spinboxes
        self.start_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.end_spinbox.valueChanged.connect(self._on_spinbox_changed)
        
        layout.addWidget(time_group)
    
    def _on_full_trace_toggled(self, checked: bool):
        """Enable/disable time range spinboxes based on checkbox state."""
        self.start_spinbox.setEnabled(not checked)
        self.end_spinbox.setEnabled(not checked)
    
    def _on_spinbox_changed(self):
        """Auto-uncheck 'use full trace' when user edits time range directly."""
        if self.full_trace_checkbox.isChecked():
            self.full_trace_checkbox.blockSignals(True)
            self.full_trace_checkbox.setChecked(False)
            self.full_trace_checkbox.blockSignals(False)
            self.start_spinbox.setEnabled(True)
            self.end_spinbox.setEnabled(True)
    
    def _connect_stale_signals(self):
        """
        Connect every editable parameter widget to _mark_parameters_stale.
        
        After a successful extraction, any subsequent edit will turn the Extract
        button red to signal that the held extraction_result is out of date.
        """
        # Sweep selection: connect each checkbox in the table
        for i in range(self.sweep_selection.table.rowCount()):
            cb = self.sweep_selection.table.cellWidget(i, 0)
            if cb:
                cb.stateChanged.connect(self._mark_parameters_stale)
        # Sweep selection: range input text changes + mode radios
        self.sweep_selection.range_input.textChanged.connect(self._mark_parameters_stale)
        self.sweep_selection.table_mode_radio.toggled.connect(self._mark_parameters_stale)
        self.sweep_selection.range_mode_radio.toggled.connect(self._mark_parameters_stale)
        
        # Channel mode
        self.voltage_radio.toggled.connect(self._mark_parameters_stale)
        self.current_radio.toggled.connect(self._mark_parameters_stale)
        self.both_radio.toggled.connect(self._mark_parameters_stale)
        
        # Time range
        self.full_trace_checkbox.toggled.connect(self._mark_parameters_stale)
        self.start_spinbox.valueChanged.connect(self._mark_parameters_stale)
        self.end_spinbox.valueChanged.connect(self._mark_parameters_stale)
    
    def _mark_parameters_stale(self):
        """
        Mark the held extraction as stale by recoloring the Extract button red.
        
        Only applies if at least one extraction has succeeded; otherwise the
        button stays primary (blue).
        """
        if self.extraction_result is not None:
            style_button(self.extract_btn, "danger")
    
    def _get_selected_channel_mode(self) -> str:
        """Return 'voltage', 'current', or 'both' based on radio selection."""
        if self.voltage_radio.isChecked():
            return 'voltage'
        elif self.current_radio.isChecked():
            return 'current'
        else:
            return 'both'
        
    def _create_status_section(self, layout):

        status_group = QGroupBox("Status")
        style_group_box(status_group)
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready to extract")
        style_label(self.status_label, "muted")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_group)
        
    def _create_action_buttons(self, layout):
        button_layout = QHBoxLayout()
        
        # Perform extraction (primary action)
        self.extract_btn = create_styled_button("Extract", "primary")
        self.extract_btn.setMinimumHeight(40)
        
        # Export and copy (disabled until extraction complete)
        self.export_btn = create_styled_button("Export CSV...", "secondary")
        self.export_btn.setEnabled(False)
        
        self.copy_btn = create_styled_button("Copy Data", "secondary")
        self.copy_btn.setEnabled(False)
        
        # Copy file names button (disabled until extraction complete)
        self.copy_filenames_btn = create_styled_button("Copy File Names", "secondary")
        self.copy_filenames_btn.setEnabled(False)
        
        # Close button
        self.close_btn = create_styled_button("Close", "secondary")
        
        button_layout.addWidget(self.extract_btn)
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.copy_btn)
        button_layout.addWidget(self.copy_filenames_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Connect signals
        self.extract_btn.clicked.connect(self._perform_extraction)
        self.export_btn.clicked.connect(self._export_to_csv)
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        self.copy_filenames_btn.clicked.connect(self._copy_file_names_to_clipboard)
        self.close_btn.clicked.connect(self.close)

        
    def _update_file_list(self):
        self.file_list.clear()
        for file_path in self.file_paths:
            self.file_list.addItem(Path(file_path).name)
        
        # Update count with color
        count = len(self.file_paths)
        self.file_count_label.setText(
            f"{count} file{'s' if count != 1 else ''} selected"
        )
        color = get_file_count_color(count)
        self.file_count_label.setStyleSheet(f"color: {color};")
        
        # Update extract button state (if it exists yet)
        if hasattr(self, 'extract_btn'):
            self.extract_btn.setEnabled(count > 0)
        
    def _add_files(self):
        file_types = (
            "WCP files (*.wcp);;"
            "ABF files (*.abf);;"
            "Data files (*.wcp *.abf);;"
            "All files (*.*)"
        )
        
        new_files = self.file_dialog_service.get_import_paths(
            self,
            "Select Files for Batch Extraction",
            default_directory=None,
            file_types=file_types,
            dialog_type="batch_import"
        )
        
        if not new_files:
            return
        
        # Validate file formats
        if not self._validate_file_formats(new_files):
            return
        
        # Add new files (avoid duplicates)
        for file_path in new_files:
            if file_path not in self.file_paths:
                self.file_paths.append(file_path)
        
        self._update_file_list()
        
        # Trigger auto-save on parent if available
        if hasattr(self.parent(), '_auto_save_settings'):
            try:
                self.parent()._auto_save_settings()
            except Exception as e:
                logger.warning(f"Failed to auto-save settings: {e}")
                
    def _remove_selected(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        for item in reversed(selected_items):
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            del self.file_paths[row]
        
        self._update_file_list()
        
    def _clear_files(self):
        """Clear all files from the list."""
        self.file_list.clear()
        self.file_paths.clear()
        self._update_file_list()
        
    def _validate_file_formats(self, new_files: List[str]) -> bool:

        all_files = self.file_paths + new_files
        extensions = set(Path(fp).suffix.lower() for fp in all_files)
        
        if len(extensions) > 1:
            QMessageBox.warning(
                self, "Mixed Formats",
                f"All files must have the same format. Found: {extensions}"
            )
            return False
        return True
        
    def _perform_extraction(self):
        """Process all files and build combined data structure."""
        if not self.file_paths:
            QMessageBox.warning(self, "No Files", "Please add files to extract.")
            return
        
        # Refresh parameters from the editable widgets
        selected_sweeps, invalid_sweeps = self.sweep_selection.get_selected_sweeps()
        if invalid_sweeps:
            QMessageBox.warning(
                self, "Invalid Sweeps",
                f"Sweep(s) {', '.join(invalid_sweeps)} not present in the reference file.\n"
                f"Proceeding with valid sweeps only."
            )
        if not selected_sweeps:
            QMessageBox.warning(
                self, "No Sweeps Selected",
                "Please select at least one sweep to extract."
            )
            return
        
        if self.full_trace_checkbox.isChecked():
            start_time = 0.0
            end_time = self.max_time
        else:
            start_time = self.start_spinbox.value()
            end_time = self.end_spinbox.value()
            if start_time >= end_time:
                QMessageBox.warning(
                    self, "Invalid Time Range",
                    "Start time must be less than end time."
                )
                return
        
        self.sweep_indices = selected_sweeps
        self.channel_mode = self._get_selected_channel_mode()
        self.time_range = (start_time, end_time)
        
        # Reset state
        self.extraction_result = None
        self.had_missing_data = False
        self.had_time_mismatch = False
        
        # Update status
        style_label(self.status_label, "info")
        self.status_label.setText("Extracting data...")
        
        try:
            all_data = []
            reference_time = None
            reference_units = None
            
            # Sort files using same logic as batch results
            sorted_files = self._sort_files(self.file_paths)
            
            # Process each file
            for file_path in sorted_files:
                try:
                    # Load dataset
                    dataset = self.data_manager.load_dataset(file_path)
                    
                    # First successful file sets reference
                    if reference_time is None:
                        reference_time = self._extract_reference_time(dataset)
                        channel_config = dataset.metadata.get('channel_config', {})
                        reference_units = {
                            'voltage': channel_config.get('voltage_units', 'mV'),
                            'current': channel_config.get('current_units', 'pA')
                        }
                    
                    # Extract sweeps from this file
                    file_data = self._extract_file_sweeps(
                        dataset, file_path, reference_time
                    )
                    all_data.append(file_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {Path(file_path).name}: {e}")
                    # Create NaN placeholders for this file
                    file_data = self._create_nan_file_data(file_path, reference_time)
                    all_data.append(file_data)
                    self.had_missing_data = True
            
            if reference_time is None or len(reference_time) == 0:
                raise ValueError("No valid data extracted from any file")
            
            # Build final output structure
            self.extraction_result = self._build_output_structure(
                all_data, reference_time, reference_units
            )
            
            # Update UI
            self._update_status_after_extraction(len(sorted_files), all_data)
            self._enable_export_buttons()
            
            # Reset extract button to primary (extraction now reflects current params)
            style_button(self.extract_btn, "primary")
            
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}", exc_info=True)
            style_label(self.status_label, "error")
            self.status_label.setText(f"Extraction failed: {str(e)}")
            
    def _extract_reference_time(self, dataset) -> np.ndarray:

        start_ms, end_ms = self.time_range
        
        for sweep_idx in self.sweep_indices:
            try:
                sweep_data = self.data_extractor.extract_sweep_data(dataset, sweep_idx)
                time_ms = sweep_data['time_ms']
                
                # Apply time range filter
                mask = (time_ms >= start_ms) & (time_ms <= end_ms)
                filtered_time = time_ms[mask]
                
                if len(filtered_time) > 0:
                    return filtered_time
                    
            except Exception as e:
                logger.warning(f"Could not extract reference time from sweep {sweep_idx}: {e}")
                continue
        
        raise ValueError("Could not extract valid reference time array")
        
    def _extract_file_sweeps(self, dataset, file_path: str, 
                            reference_time: np.ndarray) -> Dict:

        start_ms, end_ms = self.time_range
        base_name = clean_filename(file_path)
        
        file_data = {
            'file_path': file_path,
            'base_name': base_name,
            'sweeps': {}
        }
        
        for sweep_idx in self.sweep_indices:
            try:
                # Extract sweep data
                sweep_data = self.data_extractor.extract_sweep_data(dataset, sweep_idx)
                time_ms = sweep_data['time_ms']
                voltage = sweep_data['voltage']
                current = sweep_data['current']
                
                # Apply time range filter
                mask = (time_ms >= start_ms) & (time_ms <= end_ms)
                filtered_time = time_ms[mask]
                filtered_voltage = voltage[mask]
                filtered_current = current[mask]
                
                # Check if time array matches reference
                if not np.array_equal(filtered_time, reference_time):
                    self.had_time_mismatch = True
                    # Pad with NaN if length mismatch
                    if len(filtered_time) != len(reference_time):
                        filtered_voltage = self._pad_or_truncate(
                            filtered_voltage, len(reference_time)
                        )
                        filtered_current = self._pad_or_truncate(
                            filtered_current, len(reference_time)
                        )
                
                file_data['sweeps'][sweep_idx] = {
                    'voltage': filtered_voltage,
                    'current': filtered_current
                }
                
            except Exception as e:
                logger.warning(f"Could not extract sweep {sweep_idx} from {base_name}: {e}")
                # Store NaN arrays for failed sweep
                file_data['sweeps'][sweep_idx] = {
                    'voltage': np.full(len(reference_time), np.nan),
                    'current': np.full(len(reference_time), np.nan)
                }
                self.had_missing_data = True
        
        return file_data
        
    def _create_nan_file_data(self, file_path: str, reference_time: np.ndarray) -> Dict:

        base_name = clean_filename(file_path)
        
        file_data = {
            'file_path': file_path,
            'base_name': base_name,
            'sweeps': {}
        }
        
        for sweep_idx in self.sweep_indices:
            file_data['sweeps'][sweep_idx] = {
                'voltage': np.full(len(reference_time), np.nan),
                'current': np.full(len(reference_time), np.nan)
            }
        
        return file_data
        
    def _pad_or_truncate(self, array: np.ndarray, target_length: int) -> np.ndarray:

        if len(array) < target_length:
            # Pad with NaN
            return np.pad(array, (0, target_length - len(array)), 
                         constant_values=np.nan)
        elif len(array) > target_length:
            # Truncate
            return array[:target_length]
        else:
            return array
            
    def _build_output_structure(self, all_data: List[Dict], 
                                reference_time: np.ndarray,
                                reference_units: Dict) -> Dict:

        headers = ["Time (ms)"]
        columns = [reference_time]
        
        # Process each file
        for file_data in all_data:
            base_name = file_data['base_name']
            
            # Add voltage columns first (if needed)
            if self.channel_mode in ['voltage', 'both']:
                for sweep_idx in self.sweep_indices:
                    sweep_data = file_data['sweeps'].get(sweep_idx, {})
                    voltage = sweep_data.get('voltage', np.full(len(reference_time), np.nan))
                    headers.append(
                        f"{base_name} Sweep {sweep_idx} Voltage ({reference_units['voltage']})"
                    )
                    columns.append(voltage)
            
            # Add current columns (if needed)
            if self.channel_mode in ['current', 'both']:
                for sweep_idx in self.sweep_indices:
                    sweep_data = file_data['sweeps'].get(sweep_idx, {})
                    current = sweep_data.get('current', np.full(len(reference_time), np.nan))
                    headers.append(
                        f"{base_name} Sweep {sweep_idx} Current ({reference_units['current']})"
                    )
                    columns.append(current)
        
        # Combine into single array
        data_array = np.column_stack(columns)
        
        return {
            'headers': headers,
            'data': data_array
        }
        
    def _update_status_after_extraction(self, num_files: int, all_data: List[Dict]):

        total_sweeps = sum(len(fd['sweeps']) for fd in all_data)
        
        # Build status message
        status_parts = [f"Extraction complete: {num_files} files, {total_sweeps} sweeps"]
        
        if self.had_missing_data:
            status_parts.append(
                "Note: NaNs inserted where data could not be extracted"
            )
        
        if self.had_time_mismatch:
            status_parts.append(
                "Note: Some files had different time arrays than the reference file"
            )
        
        status_text = "\n".join(status_parts)
        
        # Set style based on warnings
        if self.had_missing_data or self.had_time_mismatch:
            style_label(self.status_label, "warning")
        else:
            style_label(self.status_label, "success")
        
        self.status_label.setText(status_text)
        
    def _enable_export_buttons(self):
        """Enable export and copy buttons after successful extraction."""
        self.export_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self.copy_filenames_btn.setEnabled(True)
        
    def _export_to_csv(self):

        if not self.extraction_result:
            return
        
        # Generate suggested filename
        suggested_name = "batch_sweep_extraction.csv"
        
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
            # Export using data manager
            result = self.data_manager.export_to_csv(self.extraction_result, file_path)
            
            if result.success:
                num_records = len(self.extraction_result['data'])
                style_label(self.status_label, "success")
                self.status_label.setText(
                    f"Exported {num_records} data points to {Path(file_path).name}"
                )
                
                # Trigger auto-save on parent
                if hasattr(self.parent(), '_auto_save_settings'):
                    try:
                        self.parent()._auto_save_settings()
                    except Exception as e:
                        logger.warning(f"Failed to auto-save settings: {e}")
            else:
                style_label(self.status_label, "error")
                self.status_label.setText(f"Export failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            style_label(self.status_label, "error")
            self.status_label.setText(f"Export failed: {str(e)}")
            
    def _copy_to_clipboard(self):

        if not self.extraction_result:
            return
        
        try:
            success = ClipboardService.copy_data_to_clipboard(self.extraction_result)
            
            if success:
                style_label(self.status_label, "success")
                self.status_label.setText("Data copied to clipboard")
            else:
                style_label(self.status_label, "error")
                self.status_label.setText("Failed to copy data to clipboard")
                
        except Exception as e:
            logger.error(f"Copy failed: {e}", exc_info=True)
            style_label(self.status_label, "error")
            self.status_label.setText(f"Copy failed: {str(e)}")

    def _copy_file_names_to_clipboard(self):
        """
        Copy all file names to clipboard as a column (one per line).
        
        Copies all files in the extraction list in sorted order,
        using cleaned file names (without extensions and bracketed content).
        """
        if not self.file_paths:
            QMessageBox.warning(self, "No Files", "No files in the list to copy.")
            return
        
        try:
            # Sort files using same logic as display
            sorted_files = self._sort_files(self.file_paths)
            
            # Extract and clean file names
            file_names = [clean_filename(file_path) for file_path in sorted_files]
            
            # Join with newlines to create a column
            text = "\n".join(file_names)
            
            # Copy to clipboard
            success = ClipboardService.copy_to_clipboard(text)
            
            if success:
                logger.info(f"Copied {len(file_names)} file names to clipboard")
                style_label(self.status_label, "success")
                self.status_label.setText(f"Copied {len(file_names)} file names to clipboard")
            else:
                QMessageBox.warning(
                    self, 
                    "Copy Failed", 
                    "Failed to copy file names to clipboard."
                )
                style_label(self.status_label, "error")
                self.status_label.setText("Failed to copy file names")
        
        except Exception as e:
            logger.error(f"Error copying file names: {e}", exc_info=True)
            QMessageBox.critical(self, "Copy Error", f"Copy failed: {str(e)}")
            style_label(self.status_label, "error")
            self.status_label.setText(f"Copy failed: {str(e)}")         
        
    def _sort_files(self, file_paths: List[str]) -> List[str]:
        """Sort file paths in series-aware order (base files before their decimal sub-versions)."""
        return sorted(file_paths, key=lambda p: filename_sort_key(Path(p).stem))