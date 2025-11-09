"""
Leak Subtraction Dialog for PatchBatch

Interactive dialog for performing leak current subtraction on WCP files.
Mimics WinWCP's leak subtraction interface with dual voltage/current plots,
draggable cursors, and group navigation.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

import logging
from typing import Optional, Dict, Set
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QGroupBox, QMessageBox, QCheckBox,
    QComboBox, QApplication
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.services.leak_subtraction_service import LeakSubtractionService
from data_analysis_gui.services.leak_subtraction_utils import (
    is_leak_subtraction_available,
    get_group_summary,
    validate_group_pairing,
    get_sweep_by_group_and_type
)
from data_analysis_gui.widgets.custom_inputs import NumericLineEdit, NoScrollComboBox
from data_analysis_gui.config.themes import (
    apply_modern_theme, create_styled_button, style_label, style_checkbox
)
from data_analysis_gui.config.plot_style import add_zero_axis_lines

logger = logging.getLogger(__name__)


class LeakCursorManager:
    """
    Manages VHold and VTest cursors across dual voltage/current plots.
    
    Simplified cursor manager for exactly 2 cursors that span both subplots.
    
    Note: Cursors are matplotlib Line2D objects that get removed when axes.clear()
    is called. Use recreate_cursors() after clearing to restore them.
    """
    
    def __init__(self, ax_voltage, ax_current, canvas):
        self.ax_v = ax_voltage
        self.ax_c = ax_current
        self.canvas = canvas
        
        self.dragging_cursor = None  # 'vhold' or 'vtest'
        
        # Create cursor lines
        self.vhold_line_v = ax_voltage.axvline(0, color='blue', linewidth=1.5, picker=5, alpha=0.7)
        self.vhold_line_c = ax_current.axvline(0, color='blue', linewidth=1.5, picker=5, alpha=0.7)
        
        self.vtest_line_v = ax_voltage.axvline(400, color='red', linewidth=1.5, picker=5, alpha=0.7)
        self.vtest_line_c = ax_current.axvline(400, color='red', linewidth=1.5, picker=5, alpha=0.7)
        
        # Store for identification
        self.vhold_lines = [self.vhold_line_v, self.vhold_line_c]
        self.vtest_lines = [self.vtest_line_v, self.vtest_line_c]
        
        # Connect events
        self.canvas.mpl_connect('pick_event', self._on_pick)
        self.canvas.mpl_connect('motion_notify_event', self._on_drag)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        
        # Callback for position changes (to update spinboxes)
        self.on_position_changed = None  # Set by dialog: callback(cursor_name, position)
    
    def set_vhold(self, time_ms: float):
        """Update VHold cursor position."""
        for line in self.vhold_lines:
            line.set_xdata([time_ms, time_ms])
        self.canvas.draw_idle()
    
    def set_vtest(self, time_ms: float):
        """Update VTest cursor position."""
        for line in self.vtest_lines:
            line.set_xdata([time_ms, time_ms])
        self.canvas.draw_idle()
    
    def recreate_cursors(self, vhold_pos: float, vtest_pos: float):
        """Recreate cursor lines after axes have been cleared."""
        # Create new cursor lines
        self.vhold_line_v = self.ax_v.axvline(vhold_pos, color='blue', linewidth=1.5, picker=5, alpha=0.7)
        self.vhold_line_c = self.ax_c.axvline(vhold_pos, color='blue', linewidth=1.5, picker=5, alpha=0.7)
        
        self.vtest_line_v = self.ax_v.axvline(vtest_pos, color='red', linewidth=1.5, picker=5, alpha=0.7)
        self.vtest_line_c = self.ax_c.axvline(vtest_pos, color='red', linewidth=1.5, picker=5, alpha=0.7)
        
        # Update stored references
        self.vhold_lines = [self.vhold_line_v, self.vhold_line_c]
        self.vtest_lines = [self.vtest_line_v, self.vtest_line_c]
    
    def _on_pick(self, event):
        """Handle cursor pick event."""
        artist = event.artist
        
        if artist in self.vhold_lines:
            self.dragging_cursor = 'vhold'
        elif artist in self.vtest_lines:
            self.dragging_cursor = 'vtest'
    
    def _on_drag(self, event):
        """Handle cursor drag."""
        if self.dragging_cursor and event.xdata is not None:
            x_pos = float(event.xdata)
            
            if self.dragging_cursor == 'vhold':
                self.set_vhold(x_pos)
                if self.on_position_changed:
                    self.on_position_changed('vhold', x_pos)
            elif self.dragging_cursor == 'vtest':
                self.set_vtest(x_pos)
                if self.on_position_changed:
                    self.on_position_changed('vtest', x_pos)
    
    def _on_release(self, event):
        """Handle mouse release."""
        self.dragging_cursor = None


class LeakSubtractionDialog(QDialog):
    """
    Dialog for interactive leak current subtraction.
    
    Features:
    - Dual plot layout (voltage top, current bottom)
    - Draggable VHold/VTest cursors
    - Group navigation
    - View mode selector (LEAK/TEST/Both)
    - Range selector
    - Sweep rejection
    - Preview and apply functionality
    """
    
    def __init__(self, dataset: ElectrophysiologyDataset, parent=None):
        super().__init__(parent)
        
        self.dataset = dataset
        self.service = LeakSubtractionService()
        self.modified_dataset = None
        
        # Rejected sweeps set (matches MainWindow pattern)
        self.rejected_sweeps: Set[int] = set()
        
        # Current group navigation state
        self.valid_groups = []
        self.current_group_index = 0
        
        # Preview state
        self.preview_active = False
        self.preview_lines = []
        
        # Initialize and validate
        self._validate_and_initialize()
        
        # Build UI
        self._init_ui()
        
        # Load first group
        self._update_plot()
    
    def _validate_and_initialize(self):
        """Validate dataset and initialize group list."""
        # Validate dataset is suitable
        available, msg = is_leak_subtraction_available(self.dataset)
        if not available:
            raise ValueError(f"Dataset not suitable for leak subtraction: {msg}")
        
        # Get valid groups
        groups = get_group_summary(self.dataset, self.rejected_sweeps)
        valid_group_nums, _ = validate_group_pairing(groups)
        
        if not valid_group_nums:
            raise ValueError("No valid LEAK/TEST pairs found")
        
        self.valid_groups = sorted(valid_group_nums)
        self.min_group = self.valid_groups[0]
        self.max_group = self.valid_groups[-1]
        
        # Get channel configuration
        channel_config = self.dataset.metadata.get('channel_config')
        self.voltage_ch = channel_config['voltage_channel']
        self.current_ch = channel_config['current_channel']
        self.current_units = channel_config.get('current_units', 'pA')
        
        logger.info(f"Initialized with {len(self.valid_groups)} valid groups")
    
    def _init_ui(self):
        """Build the dialog UI."""
        self.setWindowTitle("Leak Subtraction")
        self.setModal(True)
        self.resize(1000, 700)
        
        layout = QVBoxLayout()
        
        # Matplotlib figure with 2 subplots
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Create subplots (shared x-axis)
        self.ax_voltage = self.figure.add_subplot(2, 1, 1)
        self.ax_current = self.figure.add_subplot(2, 1, 2, sharex=self.ax_voltage)
        
        self.figure.tight_layout(pad=3.0)
        
        layout.addWidget(self.canvas)
        
        # Initialize cursor manager
        self.cursor_manager = LeakCursorManager(
            self.ax_voltage, self.ax_current, self.canvas
        )
        self.cursor_manager.on_position_changed = self._on_cursor_dragged
        
        # Control panel
        controls = self._create_controls()
        layout.addLayout(controls)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.preview_btn = create_styled_button("Preview Subtraction", "secondary")
        self.preview_btn.clicked.connect(self._preview_subtraction)
        button_layout.addWidget(self.preview_btn)
        
        self.apply_btn = create_styled_button("Apply", "accent")
        self.apply_btn.clicked.connect(self._apply_subtraction)
        button_layout.addWidget(self.apply_btn)
        
        cancel_btn = create_styled_button("Cancel", "secondary")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.resize(int(avail.width() * 0.9), int(avail.height() * 0.9))
        self.move(avail.center() - self.rect().center())

        apply_modern_theme(self)
    
    def _create_controls(self) -> QVBoxLayout:
        """Create the control panel."""
        layout = QVBoxLayout()
        
        # Row 1: Group navigation and view mode
        row1 = QHBoxLayout()
        
        # Group navigation
        group_label = QLabel("Group:")
        style_label(group_label, "normal")
        row1.addWidget(group_label)
        
        self.prev_group_btn = create_styled_button("◀", "secondary")
        self.prev_group_btn.setMaximumWidth(30)
        self.prev_group_btn.clicked.connect(self._prev_group)
        row1.addWidget(self.prev_group_btn)
        
        self.group_display = QLabel(f"{self.valid_groups[0]} / {len(self.valid_groups)}")
        style_label(self.group_display, "normal")
        self.group_display.setMinimumWidth(60)
        self.group_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row1.addWidget(self.group_display)
        
        self.next_group_btn = create_styled_button("▶", "secondary")
        self.next_group_btn.setMaximumWidth(30)
        self.next_group_btn.clicked.connect(self._next_group)
        row1.addWidget(self.next_group_btn)
        
        row1.addSpacing(20)
        
        # View mode
        view_label = QLabel("View:")
        style_label(view_label, "normal")
        row1.addWidget(view_label)
        
        self.view_group = QButtonGroup()
        self.view_leak_radio = QRadioButton("LEAK")
        self.view_test_radio = QRadioButton("TEST")
        self.view_both_radio = QRadioButton("Both")
        self.view_both_radio.setChecked(True)
        
        self.view_group.addButton(self.view_leak_radio, 0)
        self.view_group.addButton(self.view_test_radio, 1)
        self.view_group.addButton(self.view_both_radio, 2)
        
        self.view_leak_radio.toggled.connect(self._update_plot)
        self.view_test_radio.toggled.connect(self._update_plot)
        self.view_both_radio.toggled.connect(self._update_plot)
        
        row1.addWidget(self.view_leak_radio)
        row1.addWidget(self.view_test_radio)
        row1.addWidget(self.view_both_radio)
        
        row1.addStretch()
        layout.addLayout(row1)
        
        # Row 2: Channel selectors
        row2 = QHBoxLayout()
        
        v_label = QLabel("Voltage channel:")
        style_label(v_label, "normal")
        row2.addWidget(v_label)
        
        self.voltage_combo = NoScrollComboBox()
        self.voltage_combo.addItems(self.dataset.metadata.get('channel_labels', []))
        self.voltage_combo.setCurrentIndex(self.voltage_ch)
        self.voltage_combo.currentIndexChanged.connect(self._on_channel_changed)
        row2.addWidget(self.voltage_combo)
        
        row2.addSpacing(20)
        
        i_label = QLabel("Current channel:")
        style_label(i_label, "normal")
        row2.addWidget(i_label)
        
        self.current_combo = NoScrollComboBox()
        self.current_combo.addItems(self.dataset.metadata.get('channel_labels', []))
        self.current_combo.setCurrentIndex(self.current_ch)
        self.current_combo.currentIndexChanged.connect(self._on_channel_changed)
        row2.addWidget(self.current_combo)
        
        row2.addStretch()
        layout.addLayout(row2)
        
        # Row 3: Cursor positions
        row3 = QHBoxLayout()
        
        vh_label = QLabel("VHold:")
        style_label(vh_label, "normal")
        row3.addWidget(vh_label)
        
        self.vhold_spinbox = NumericLineEdit()
        self.vhold_spinbox.setValue(0.05)
        self.vhold_spinbox.setMaximumWidth(100)
        self.vhold_spinbox.valueChanged.connect(self._on_vhold_changed)
        row3.addWidget(self.vhold_spinbox)
        
        vh_ms = QLabel("ms")
        style_label(vh_ms, "normal")
        row3.addWidget(vh_ms)
        
        row3.addSpacing(20)
        
        vt_label = QLabel("VTest:")
        style_label(vt_label, "normal")
        row3.addWidget(vt_label)
        
        self.vtest_spinbox = NumericLineEdit()
        self.vtest_spinbox.setValue(377.6)
        self.vtest_spinbox.setMaximumWidth(100)
        self.vtest_spinbox.valueChanged.connect(self._on_vtest_changed)
        row3.addWidget(self.vtest_spinbox)
        
        vt_ms = QLabel("ms")
        style_label(vt_ms, "normal")
        row3.addWidget(vt_ms)
        
        row3.addStretch()
        layout.addLayout(row3)
        
        # Row 4: Scaling mode
        row4 = QHBoxLayout()
        
        scale_label = QLabel("Scaling:")
        style_label(scale_label, "normal")
        row4.addWidget(scale_label)
        
        self.scaling_group = QButtonGroup()
        self.voltage_scaling_radio = QRadioButton("From Voltage")
        self.fixed_scaling_radio = QRadioButton("Fixed:")
        self.voltage_scaling_radio.setChecked(True)
        
        self.scaling_group.addButton(self.voltage_scaling_radio, 0)
        self.scaling_group.addButton(self.fixed_scaling_radio, 1)
        
        self.voltage_scaling_radio.toggled.connect(self._on_scaling_mode_changed)
        
        row4.addWidget(self.voltage_scaling_radio)
        row4.addWidget(self.fixed_scaling_radio)
        
        self.fixed_scale_input = NumericLineEdit()
        self.fixed_scale_input.setValue(1.0)
        self.fixed_scale_input.setMaximumWidth(80)
        self.fixed_scale_input.setEnabled(False)
        row4.addWidget(self.fixed_scale_input)
        
        row4.addStretch()
        layout.addLayout(row4)
        
        # Row 5: Scale factor display
        row5 = QHBoxLayout()
        
        self.scale_display = QLabel("Calculated Scale Factor: --")
        style_label(self.scale_display, "info")
        row5.addWidget(self.scale_display)
        
        row5.addStretch()
        layout.addLayout(row5)
        
        # Row 6: Range selector
        row6 = QHBoxLayout()
        
        range_label = QLabel("Range:")
        style_label(range_label, "normal")
        row6.addWidget(range_label)
        
        self.start_group_input = NumericLineEdit()
        self.start_group_input.setValue(float(self.min_group))
        self.start_group_input.setMaximumWidth(60)
        row6.addWidget(self.start_group_input)
        
        dash = QLabel("-")
        style_label(dash, "normal")
        row6.addWidget(dash)
        
        self.end_group_input = NumericLineEdit()
        self.end_group_input.setValue(float(self.max_group))
        self.end_group_input.setMaximumWidth(60)
        row6.addWidget(self.end_group_input)
        
        all_label = QLabel("(All groups)")
        style_label(all_label, "muted")
        row6.addWidget(all_label)
        
        row6.addStretch()
        layout.addLayout(row6)
        
        # Row 7: Reject sweep checkbox
        row7 = QHBoxLayout()
        
        self.reject_cb = QCheckBox("Reject Sweep")
        style_checkbox(self.reject_cb)
        self.reject_cb.stateChanged.connect(self._on_reject_toggled)
        row7.addWidget(self.reject_cb)
        
        row7.addStretch()
        layout.addLayout(row7)
        
        return layout
    
    def _on_channel_changed(self):
        """Handle channel selection change."""
        self.voltage_ch = self.voltage_combo.currentIndex()
        self.current_ch = self.current_combo.currentIndex()
        self._update_plot()
    
    def _on_vhold_changed(self):
        """Handle VHold spinbox change."""
        val = self.vhold_spinbox.value()
        self.cursor_manager.set_vhold(val)
        self._update_scale_display()
    
    def _on_vtest_changed(self):
        """Handle VTest spinbox change."""
        val = self.vtest_spinbox.value()
        self.cursor_manager.set_vtest(val)
        self._update_scale_display()
    
    def _on_cursor_dragged(self, cursor_name: str, position: float):
        """Handle cursor drag (update spinbox)."""
        if cursor_name == 'vhold':
            self.vhold_spinbox.blockSignals(True)
            self.vhold_spinbox.setValue(position)
            self.vhold_spinbox.blockSignals(False)
        elif cursor_name == 'vtest':
            self.vtest_spinbox.blockSignals(True)
            self.vtest_spinbox.setValue(position)
            self.vtest_spinbox.blockSignals(False)
        
        self._update_scale_display()
    
    def _on_scaling_mode_changed(self):
        """Handle scaling mode change."""
        is_fixed = self.fixed_scaling_radio.isChecked()
        self.fixed_scale_input.setEnabled(is_fixed)
        self._update_scale_display()
    
    def _on_reject_toggled(self):
        """Handle reject sweep checkbox toggle."""
        current_group = self.valid_groups[self.current_group_index]
        groups = get_group_summary(self.dataset, self.rejected_sweeps)
        
        leak_idx = groups[current_group]['leak'][0]
        test_idx = groups[current_group]['test'][0]
        
        if self.reject_cb.isChecked():
            self.rejected_sweeps.add(int(leak_idx))
            self.rejected_sweeps.add(int(test_idx))
        else:
            self.rejected_sweeps.discard(int(leak_idx))
            self.rejected_sweeps.discard(int(test_idx))
        
        logger.debug(f"Group {current_group} rejection toggled: {self.reject_cb.isChecked()}")
    
    def _prev_group(self):
        """Navigate to previous group."""
        if self.current_group_index > 0:
            self.current_group_index -= 1
            self._update_plot()
    
    def _next_group(self):
        """Navigate to next group."""
        if self.current_group_index < len(self.valid_groups) - 1:
            self.current_group_index += 1
            self._update_plot()
    
    def _update_plot(self):
        """Update the dual plot display."""
        # Clear preview
        self._clear_preview()
        
        # Get current group
        current_group = self.valid_groups[self.current_group_index]
        
        # Update group display
        self.group_display.setText(f"{current_group} / {len(self.valid_groups)}")
        
        # Update reject checkbox
        groups = get_group_summary(self.dataset, self.rejected_sweeps)
        leak_idx = groups[current_group]['leak'][0]
        test_idx = groups[current_group]['test'][0]
        
        is_rejected = int(leak_idx) in self.rejected_sweeps or int(test_idx) in self.rejected_sweeps
        self.reject_cb.blockSignals(True)
        self.reject_cb.setChecked(is_rejected)
        self.reject_cb.blockSignals(False)
        
        # Clear axes
        self.ax_voltage.clear()
        self.ax_current.clear()
        
        # Get view mode
        if self.view_leak_radio.isChecked():
            show_leak, show_test = True, False
        elif self.view_test_radio.isChecked():
            show_leak, show_test = False, True
        else:
            show_leak, show_test = True, True
        
        # Plot LEAK traces
        if show_leak:
            time_ms, leak_data = self.dataset.get_sweep(leak_idx)
            v_leak = leak_data[:, self.voltage_ch]
            i_leak = leak_data[:, self.current_ch]
            
            self.ax_voltage.plot(time_ms, v_leak, color='blue', linewidth=1.5, label='LEAK')
            self.ax_current.plot(time_ms, i_leak, color='blue', linewidth=1.5, label='LEAK')
        
        # Plot TEST traces
        if show_test:
            time_ms, test_data = self.dataset.get_sweep(test_idx)
            v_test = test_data[:, self.voltage_ch]
            i_test = test_data[:, self.current_ch]
            
            self.ax_voltage.plot(time_ms, v_test, color='green', linewidth=1.5, label='TEST')
            self.ax_current.plot(time_ms, i_test, color='green', linewidth=1.5, label='TEST')
        
        # Labels and formatting
        self.ax_voltage.set_ylabel('Voltage (mV)')
        self.ax_voltage.legend(loc='upper right')
        self.ax_voltage.grid(True, alpha=0.3)
        add_zero_axis_lines(self.ax_voltage, alpha=0.4, linewidth=0.8)
        
        self.ax_current.set_ylabel(f'Current ({self.current_units})')
        self.ax_current.set_xlabel('Time (ms)')
        self.ax_current.legend(loc='upper right')
        self.ax_current.grid(True, alpha=0.3)
        add_zero_axis_lines(self.ax_current, alpha=0.4, linewidth=0.8)
        
        # Recreate cursors at current positions (they were cleared with axes)
        vhold = self.vhold_spinbox.value()
        vtest = self.vtest_spinbox.value()
        self.cursor_manager.recreate_cursors(vhold, vtest)
        
        self.canvas.draw()
        
        # Update scale display
        self._update_scale_display()
    
    def _update_scale_display(self):
        """Calculate and display scaling factor."""
        if not self.voltage_scaling_radio.isChecked():
            # Fixed mode
            scale = self.fixed_scale_input.value()
            self.scale_display.setText(f"Scale Factor: {scale:.4f} (Fixed)")
            return
        
        try:
            # Get current group data
            current_group = self.valid_groups[self.current_group_index]
            groups = get_group_summary(self.dataset, self.rejected_sweeps)
            
            leak_idx = groups[current_group]['leak'][0]
            test_idx = groups[current_group]['test'][0]
            
            # Extract sweep data
            time_ms, leak_data = self.dataset.get_sweep(leak_idx)
            _, test_data = self.dataset.get_sweep(test_idx)
            
            v_leak = leak_data[:, self.voltage_ch]
            v_test = test_data[:, self.voltage_ch]
            
            # Convert cursor positions from ms to sample indices
            vhold_ms = self.vhold_spinbox.value()
            vtest_ms = self.vtest_spinbox.value()
            vhold_idx = int(np.argmin(np.abs(time_ms - vhold_ms)))
            vtest_idx = int(np.argmin(np.abs(time_ms - vtest_ms)))
            
            # Apply baseline correction (subtract voltage at VHold cursor)
            v_leak_bc = v_leak - v_leak[vhold_idx]
            v_test_bc = v_test - v_test[vhold_idx]
            
            # Calculate scaling (now with correct arguments)
            leak_scale, details = self.service.calculate_voltage_scaling(
                v_leak_bc, v_test_bc, vhold_idx, vtest_idx
            )
            
            self.scale_display.setText(f"Calculated Scale Factor: {leak_scale:.4f}")
            
        except Exception as e:
            self.scale_display.setText(f"Scale Factor: Error ({str(e)})")
    
    def _preview_subtraction(self):
        """Preview the subtracted current trace."""
        if self.preview_active:
            self._clear_preview()
            return
        
        try:
            # Get current group data
            current_group = self.valid_groups[self.current_group_index]
            groups = get_group_summary(self.dataset, self.rejected_sweeps)
            
            leak_idx = groups[current_group]['leak'][0]
            test_idx = groups[current_group]['test'][0]
            
            # Extract sweep data
            time_ms, leak_data = self.dataset.get_sweep(leak_idx)
            _, test_data = self.dataset.get_sweep(test_idx)
            
            v_leak = leak_data[:, self.voltage_ch]
            i_leak = leak_data[:, self.current_ch]
            v_test = test_data[:, self.voltage_ch]
            i_test = test_data[:, self.current_ch]
            
            # Calculate scaling
            if self.voltage_scaling_radio.isChecked():
                vhold_ms = self.vhold_spinbox.value()
                vtest_ms = self.vtest_spinbox.value()
                
                # Convert to indices
                vhold_idx = int(np.argmin(np.abs(time_ms - vhold_ms)))
                vtest_idx = int(np.argmin(np.abs(time_ms - vtest_ms)))
                
                # Apply baseline correction
                v_leak_bc = v_leak - v_leak[vhold_idx]
                v_test_bc = v_test - v_test[vhold_idx]
                
                # Calculate scaling
                leak_scale, _ = self.service.calculate_voltage_scaling(
                    v_leak_bc, v_test_bc, vhold_idx, vtest_idx
                )
            else:
                leak_scale = self.fixed_scale_input.value()
            
            # Perform subtraction
            i_subtracted = i_test - leak_scale * i_leak
            
            # Plot subtracted trace
            line, = self.ax_current.plot(
                time_ms, i_subtracted, 
                color='red', linewidth=2, label='Subtracted', linestyle='--'
            )
            self.preview_lines.append(line)
            
            self.ax_current.legend(loc='upper right')
            self.canvas.draw()
            
            self.preview_active = True
            self.preview_btn.setText("Clear Preview")
            
        except Exception as e:
            QMessageBox.warning(self, "Preview Failed", f"Could not preview subtraction:\n{str(e)}")
    
    def _clear_preview(self):
        """Clear preview traces."""
        for line in self.preview_lines:
            line.remove()
        self.preview_lines.clear()
        
        self.preview_active = False
        self.preview_btn.setText("Preview Subtraction")
        
        if hasattr(self, 'ax_current'):
            self.ax_current.legend(loc='upper right')
            self.canvas.draw()
    
    def _apply_subtraction(self):
        """Apply leak subtraction to dataset."""
        try:
            # Get range
            start_group = int(self.start_group_input.value())
            end_group = int(self.end_group_input.value())
            
            # Validate range
            if start_group > end_group:
                QMessageBox.warning(
                    self, "Invalid Range",
                    "Start group must be <= end group"
                )
                return
            
            # Get parameters
            vhold_ms = self.vhold_spinbox.value()
            vtest_ms = self.vtest_spinbox.value()
            
            scaling_mode = "voltage" if self.voltage_scaling_radio.isChecked() else "fixed"
            fixed_scale = self.fixed_scale_input.value()
            
            # Perform subtraction
            self.modified_dataset = self.service.perform_leak_subtraction(
                dataset=self.dataset,
                vhold_ms=vhold_ms,
                vtest_ms=vtest_ms,
                start_group=start_group,
                end_group=end_group,
                scaling_mode=scaling_mode,
                fixed_scale=fixed_scale,
                rejected_sweeps=self.rejected_sweeps
            )
            
            # Success message
            QMessageBox.information(
                self, "Success",
                f"Leak subtraction applied successfully!\n"
                f"Original sweeps: {self.dataset.sweep_count()}\n"
                f"Subtracted sweeps: {self.modified_dataset.sweep_count()}"
            )
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Subtraction Failed",
                f"Failed to apply leak subtraction:\n{str(e)}"
            )
    
    def get_modified_dataset(self) -> Optional[ElectrophysiologyDataset]:
        """Get the modified dataset after successful subtraction."""
        return self.modified_dataset