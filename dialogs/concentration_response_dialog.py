import os
import re
import csv
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QCheckBox, QFileDialog, QMessageBox, QGroupBox,
                             QDoubleSpinBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QLineEdit, QLabel, QWidget,
                             QSplitter, QApplication, QAbstractSpinBox)
from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Internal imports
from widgets import NoScrollComboBox, SelectAllLineEdit
from config import ANALYSIS_CONSTANTS, TABLE_HEADERS
from utils import get_next_available_filename, sanitize_filename


class ConcentrationResponseDialog(QDialog):
    """
    Enhanced dialog for analyzing patch-clamp time-series data.
    Supports multiple data columns from a single CSV file.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Stored Data & State ---
        self.data_df = None
        self.filepath = None
        self.filename = None
        self.data_columns = []
        self.results_dfs = {}  # Dictionary to hold a results DataFrame for each data column
        self.dragging_line = None
        self.range_lines = []
        self.range_patches = []
        self.line_to_table_row_map = {}
        self.last_focused_editor = None
        self.pan_active = False
        self.pan_start_pos = None
        self.pan_start_lim = None

        # --- Install event filter to track focus changes ---
        QApplication.instance().installEventFilter(self)

        # --- Window Properties ---
        self.setWindowTitle("Patch-Clamp Concentration-Response Analysis")
        self.setGeometry(25, 50, 1850, 950)

        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(5, 5, 5, 5)

        self.status_label = QLabel("Load a CSV file to begin")
        self.status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; font-size: 9pt; }")
        self.status_label.setMaximumHeight(20)
        main_layout.addWidget(self.status_label)

        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # --- Left Panel ---
        left_panel = QWidget()
        left_panel.setMaximumWidth(550)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(5)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.addWidget(self._create_files_group())
        left_layout.addWidget(self._create_ranges_group())
        left_layout.addWidget(self._create_results_group())
        left_layout.addStretch()
        main_splitter.addWidget(left_panel)

        # --- Right Panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.addWidget(self._create_plot_group())
        main_splitter.addWidget(right_panel)

        main_splitter.setSizes([550, 1300])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        self.add_range_row()

    def eventFilter(self, obj, event):
        """
        Event filter to capture focus-in events and store the
        last focused QLineEdit widget.
        """
        if event.type() == QEvent.FocusIn:
            if isinstance(obj, QLineEdit):
                self.last_focused_editor = obj
        return super().eventFilter(obj, event)

    def _create_files_group(self):
        """Creates the UI group for loading and selecting a single CSV file."""
        group = QGroupBox("File")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(5, 5, 5, 5)

        btn_layout = QHBoxLayout()
        load_btn = QPushButton("📁 Load CSV")
        load_btn.clicked.connect(self.load_file)
        #load_btn.setFixedWidth(110)
        btn_layout.addWidget(load_btn)

        self.file_path_display = QLineEdit("No file loaded")
        self.file_path_display.setReadOnly(True)
        self.file_path_display.setStyleSheet("QLineEdit { color: #666; }")
        btn_layout.addWidget(self.file_path_display)

        layout.addLayout(btn_layout)
        return group

    def _create_ranges_group(self):
        """Creates the UI group for defining analysis ranges."""
        group = QGroupBox("Analysis Ranges (drag boundaries in plot)")
        layout = QVBoxLayout(group)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)

        self.ranges_table = QTableWidget()
        self.ranges_table.setColumnCount(7)
        self.ranges_table.setHorizontalHeaderLabels(TABLE_HEADERS['ranges'])
        self.ranges_table.setMaximumHeight(250)
        self.ranges_table.setMinimumWidth(430)
        self.ranges_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        header = self.ranges_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)

        self.ranges_table.setColumnWidth(2, 75)
        self.ranges_table.setColumnWidth(3, 75)
        self.ranges_table.setColumnWidth(5, 35)

        self.ranges_table.itemChanged.connect(self.on_range_value_changed)
        layout.addWidget(self.ranges_table)

        bottom_layout = QHBoxLayout()
        add_range_btn = QPushButton("➕ Add Range")
        add_range_btn.clicked.connect(lambda: self.add_range_row(is_background=False))
        #add_range_btn.setFixedHeight(22)

        add_bg_range_btn = QPushButton("➕ Add Background Range")
        add_bg_range_btn.clicked.connect(lambda: self.add_range_row(is_background=True))
        #add_bg_range_btn.setFixedHeight(22)

        mu_button = QPushButton("Insert μ")
        #mu_button.setFixedSize(60, 22)
        mu_button.clicked.connect(self.insert_mu_char)

        bottom_layout.addWidget(add_range_btn)
        bottom_layout.addWidget(add_bg_range_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(mu_button)
        layout.addLayout(bottom_layout)

        return group

    def _create_plot_group(self):
        """Creates the UI group for the interactive plot."""
        group = QGroupBox("Data Visualization")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 5, 5, 5)

        self.figure = Figure(figsize=(14, 9), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        self.ax.set_xlabel("Time (s)", fontsize=10)
        self.ax.set_ylabel("Current (pA)", fontsize=10)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self.canvas.mpl_connect('scroll_event', self.on_scroll_zoom)
        self.canvas.mpl_connect('button_press_event', self.on_pan_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_pan_motion)
        self.canvas.mpl_connect('button_release_event', self.on_pan_release)

        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        return group

    def _create_results_group(self):
        """Creates the UI group for running analysis and viewing results."""
        group = QGroupBox("Results")
        layout = QVBoxLayout(group)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)

        btn_layout = QHBoxLayout()
        self.run_analysis_btn = QPushButton("▶ Run Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        #self.run_analysis_btn.setFixedHeight(24)
        btn_layout.addWidget(self.run_analysis_btn)

        self.export_btn = QPushButton("💾 Export CSV(s)")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        #self.export_btn.setFixedHeight(24)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(TABLE_HEADERS['results'])
        self.results_table.setMaximumHeight(250)

        header = self.results_table.horizontalHeader()
        for i in range(6):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        layout.addWidget(self.results_table)

        return group

    def load_file(self):
        """Opens a dialog to select a single CSV file and plots it."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select a Patch-Clamp CSV File", "", "CSV files (*.csv)"
        )

        if filepath:
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            self.file_path_display.setText(self.filename)
            self.status_label.setText(f"Loading: {self.filename}")
            self.process_and_plot_file()

    def process_and_plot_file(self):
        """Loads, validates, and plots data from the selected file."""
        self.ax.clear()
        self.data_columns = []

        if not self.filepath:
            self.ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            self.canvas.draw()
            return

        try:
            df = pd.read_csv(self.filepath)

            if df.shape[1] < 2:
                raise ValueError("CSV must have at least 2 columns (time and data)")

            self.data_df = df
            time_col = df.columns[0]
            self.data_columns = df.columns[1:].tolist()

            # Plot each data column
            color_cycle = plt.get_cmap('viridis')(np.linspace(0, 1, len(self.data_columns)))
            for i, data_col in enumerate(self.data_columns):
                self.ax.plot(df[time_col], df[data_col], lw=0.8, alpha=0.9, label=data_col, color=color_cycle[i])

            self.ax.set_xlabel(f"{time_col}")
            y_label = " and ".join(self.data_columns)
            self.ax.set_ylabel(y_label)
            self.ax.set_title(f"Data: {self.filename}", fontsize=10)
            self.ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            if len(self.data_columns) > 1:
                self.ax.legend()

            self.status_label.setText(f"{self.filename} ({len(df)} pts, {len(self.data_columns)} traces)")

        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Could not plot file: {self.filename}\n\nError: {str(e)}")
            self.status_label.setText(f"Error plotting {self.filename}")
            self.data_df = None
            self.data_columns = []

        self.draw_range_indicators()
        self.canvas.draw()

    def _get_next_range_name(self):
        """Finds the next available 'Range X' name for a new row."""
        existing_names = set()
        for row in range(self.ranges_table.rowCount()):
            name_widget = self.ranges_table.cellWidget(row, 1)
            if name_widget:
                existing_names.add(name_widget.text())

        i = 1
        while True:
            next_name = f"Range {i}"
            if next_name not in existing_names:
                return next_name
            i += 1

    def _get_next_background_name(self):
        """Finds the next available 'Background' or 'Background_X' name."""
        existing_names = set()
        for row in range(self.ranges_table.rowCount()):
            name_widget = self.ranges_table.cellWidget(row, 1)
            if name_widget:
                existing_names.add(name_widget.text())

        if "Background" not in existing_names:
            return "Background"

        i = 2
        while True:
            next_name = f"Background_{i}"
            if next_name not in existing_names:
                return next_name
            i += 1

    def add_range_row(self, is_background=False):
        """Adds a new row to the analysis ranges table with consistent styling."""
        # --- New timing logic: Insert 5s after the latest existing range ---
        all_end_times = [0.0]  # Default if table is empty
        for r in range(self.ranges_table.rowCount()):
            end_spin = self.ranges_table.cellWidget(r, 3)
            if end_spin:
                all_end_times.append(end_spin.value())

        latest_time = max(all_end_times)
        new_start_time = latest_time + 5.0 if self.ranges_table.rowCount() > 0 else 0.0
        new_end_time = new_start_time + 5.0

        row = self.ranges_table.rowCount()
        self.ranges_table.insertRow(row)
        self.ranges_table.setRowHeight(row, 24)

        # Use the table as the parent and its font for all new widgets
        table_as_parent = self.ranges_table
        table_font = table_as_parent.font()

        remove_btn = QPushButton("✖", table_as_parent)
        remove_btn.setFont(table_font)
        #remove_btn.setFixedSize(22, 20)
        remove_btn.clicked.connect(self.remove_range_row)

        if is_background:
            default_name = self._get_next_background_name()
        else:
            default_name = self._get_next_range_name()

        name_edit = SelectAllLineEdit(default_name, table_as_parent)
        name_edit.setFont(table_font)
        name_edit.textChanged.connect(self.update_background_options)

        start_spin = QDoubleSpinBox(table_as_parent)
        start_spin.setFont(table_font)
        start_spin.setRange(-1e6, 1e6)
        start_spin.setDecimals(2)
        start_spin.setValue(new_start_time)
        start_spin.valueChanged.connect(self.draw_range_indicators)
        start_spin.setFixedWidth(75)

        end_spin = QDoubleSpinBox(table_as_parent)
        end_spin.setFont(table_font)
        end_spin.setRange(-1e6, 1e6)
        end_spin.setDecimals(2)
        end_spin.setValue(new_end_time)
        end_spin.valueChanged.connect(self.draw_range_indicators)
        end_spin.setFixedWidth(75)

        analysis_widget = QWidget(table_as_parent)
        analysis_layout = QHBoxLayout(analysis_widget)
        analysis_layout.setContentsMargins(0, 0, 0, 0)

        analysis_combo = NoScrollComboBox(table_as_parent)
        analysis_combo.setFont(table_font)
        analysis_combo.addItems(["Average", "Peak"])

        peak_combo = NoScrollComboBox(table_as_parent)
        peak_combo.setFont(table_font)
        peak_combo.addItems(["Max", "Min", "Absolute Max"])
        peak_combo.setVisible(False)

        analysis_combo.currentTextChanged.connect(lambda text: peak_combo.setVisible(text == "Peak"))
        analysis_layout.addWidget(analysis_combo)
        analysis_layout.addWidget(peak_combo)

        bg_checkbox = QCheckBox(table_as_parent)
        bg_checkbox.setFont(table_font)
        bg_checkbox.stateChanged.connect(self.update_background_options)
        if is_background:
            bg_checkbox.setChecked(True)

        paired_combo = NoScrollComboBox(table_as_parent)
        paired_combo.setFont(table_font)
        paired_combo.addItem("None")

        self.ranges_table.setCellWidget(row, 0, remove_btn)
        self.ranges_table.setCellWidget(row, 1, name_edit)
        self.ranges_table.setCellWidget(row, 2, start_spin)
        self.ranges_table.setCellWidget(row, 3, end_spin)
        self.ranges_table.setCellWidget(row, 4, analysis_widget)
        self.ranges_table.setCellWidget(row, 5, self._center_widget(bg_checkbox))
        self.ranges_table.setCellWidget(row, 6, paired_combo)

        self.update_background_options()
        self.draw_range_indicators()

    def remove_range_row(self):
        """Removes the row corresponding to the clicked '✖' button."""
        sender = self.sender()
        if not sender:
            return
        for row in range(self.ranges_table.rowCount()):
            if self.ranges_table.cellWidget(row, 0) == sender:
                self.ranges_table.removeRow(row)
                self.update_background_options()
                self.draw_range_indicators()
                break

    def insert_mu_char(self):
        """Inserts 'μ' and restores focus without selecting all text."""
        editor = self.last_focused_editor
        if editor:
            editor.insert("μ")
            if isinstance(editor, SelectAllLineEdit):
                editor.setFocusAndDoNotSelect()
            else:
                editor.setFocus()

    def on_range_value_changed(self):
        """Called when range values are edited in the table."""
        self.draw_range_indicators()

    def draw_range_indicators(self):
        """Draws shaded regions and boundary lines for all ranges."""
        for line in self.range_lines:
            try:
                line.remove()
            except:
                pass
        self.range_lines.clear()

        for patch in self.range_patches:
            try:
                patch.remove()
            except:
                pass
        self.range_patches.clear()

        self.line_to_table_row_map.clear()
        if not self.ax or self.data_df is None:
            return

        colors = ANALYSIS_CONSTANTS['range_colors']

        for row in range(self.ranges_table.rowCount()):
            try:
                start_widget = self.ranges_table.cellWidget(row, 2)
                end_widget = self.ranges_table.cellWidget(row, 3)
                bg_widget = self.ranges_table.cellWidget(row, 5)

                if not start_widget or not end_widget:
                    continue

                start_val, end_val = start_widget.value(), end_widget.value()
                is_background = bg_widget.findChild(QCheckBox).isChecked()
                color_set = colors['background'] if is_background else colors['analysis']

                patch = self.ax.add_patch(mpatches.Rectangle(
                    (start_val, self.ax.get_ylim()[0]),
                    end_val - start_val,
                    self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                    facecolor=color_set['fill'], edgecolor='none', zorder=1
                ))
                self.range_patches.append(patch)

                start_line = self.ax.axvline(start_val, color=color_set['line'], ls='--', lw=1.5, picker=5, alpha=0.7)
                end_line = self.ax.axvline(end_val, color=color_set['line'], ls='--', lw=1.5, picker=5, alpha=0.7)

                self.range_lines.extend([start_line, end_line])
                self.line_to_table_row_map[start_line] = (row, 2)
                self.line_to_table_row_map[end_line] = (row, 3)

            except Exception as e:
                print(f"Error drawing range {row}: {e}")

        self.canvas.draw_idle()

    def on_click(self, event):
        """Handle mouse click events on the plot."""
        if event.inaxes != self.ax or not self.range_lines or event.xdata is None:
            return

        line_distances = [(line, abs(event.xdata - line.get_xdata()[0])) for line in self.range_lines]
        closest_line, min_dist = min(line_distances, key=lambda item: item[1])

        x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        if x_range > 0 and min_dist < x_range * 0.02:
            self.dragging_line = closest_line

    def on_drag(self, event):
        """Handle mouse drag events on the plot."""
        if self.dragging_line and event.xdata is not None:
            self.dragging_line.set_xdata([event.xdata, event.xdata])
            row, col = self.line_to_table_row_map.get(self.dragging_line, (None, None))
            if row is not None:
                spinbox = self.ranges_table.cellWidget(row, col)
                if spinbox:
                    spinbox.blockSignals(True)
                    spinbox.setValue(event.xdata)
                    spinbox.blockSignals(False)
            self.canvas.draw_idle()

    def on_release(self, event):
        """Handle mouse release events."""
        if self.dragging_line:
            self.dragging_line = None
            self.draw_range_indicators()

    def on_scroll_zoom(self, event):
        """Handle scroll events for zooming."""
        if event.inaxes != self.ax:
            return

        base_scale = ANALYSIS_CONSTANTS['zoom_scale_factor']
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        if event.button == 'up':
            # Zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # Zoom out
            scale_factor = base_scale
        else:
            # Unrecognized scroll event
            return

        # Get mouse coordinates
        xdata = event.xdata
        ydata = event.ydata

        # Calculate new axis limits
        new_xlim = [
            (x - xdata) * scale_factor + xdata for x in cur_xlim
        ]
        new_ylim = [
            (y - ydata) * scale_factor + ydata for y in cur_ylim
        ]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def on_pan_press(self, event):
        """Handle middle-mouse button press to start panning."""
        # We only pan with the middle mouse button
        if event.inaxes != self.ax or event.button != 2:
            return

        self.pan_active = True
        self.pan_start_pos = (event.xdata, event.ydata)
        self.pan_start_lim = (self.ax.get_xlim(), self.ax.get_ylim())
        self.canvas.setCursor(ANALYSIS_CONSTANTS['pan_cursor'])

    def on_pan_motion(self, event):
        """Handle mouse motion for panning."""
        if not self.pan_active or event.inaxes != self.ax:
            return

        # Calculate the change from the starting position
        dx = event.xdata - self.pan_start_pos[0]
        dy = event.ydata - self.pan_start_pos[1]

        # Get the original limits
        start_xlim, start_ylim = self.pan_start_lim

        # Apply the change to the original limits
        new_xlim = (start_xlim[0] - dx, start_xlim[1] - dx)
        new_ylim = (start_ylim[0] - dy, start_ylim[1] - dy)

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def on_pan_release(self, event):
        """Handle middle-mouse button release to stop panning."""
        if event.button == 2:
            self.pan_active = False
            self.pan_start_pos = None
            self.pan_start_lim = None
            self.canvas.setCursor(Qt.ArrowCursor)

    def update_background_options(self):
        """Updates the UI based on background range selections."""
        background_names = ["None"]
        for row in range(self.ranges_table.rowCount()):
            bg_widget = self.ranges_table.cellWidget(row, 5)
            name_widget = self.ranges_table.cellWidget(row, 1)
            analysis_widget = self.ranges_table.cellWidget(row, 4)

            if bg_widget and name_widget:
                is_checked = bg_widget.findChild(QCheckBox).isChecked()
                self._style_row(row, is_checked)
                combo = analysis_widget.findChild(NoScrollComboBox)
                if combo:
                    combo.setEnabled(not is_checked)
                if is_checked:
                    background_names.append(name_widget.text())
                    if combo:
                        combo.setCurrentText("Average")

        for row in range(self.ranges_table.rowCount()):
            paired_combo = self.ranges_table.cellWidget(row, 6)
            if paired_combo:
                current = paired_combo.currentText()
                paired_combo.clear()
                paired_combo.addItems(background_names)
                if current in background_names:
                    paired_combo.setCurrentText(current)

    def run_analysis(self):
        """Performs analysis for each loaded data column."""
        if self.data_df is None:
            QMessageBox.warning(self, "No File", "Please load a CSV file before running analysis.")
            return

        if self.ranges_table.rowCount() == 0:
            QMessageBox.warning(self, "No Ranges", "Please define at least one analysis range.")
            return

        ranges = []
        for row in range(self.ranges_table.rowCount()):
            try:
                analysis_widget = self.ranges_table.cellWidget(row, 4)
                combos = analysis_widget.findChildren(NoScrollComboBox)
                ranges.append({
                    'name': self.ranges_table.cellWidget(row, 1).text(),
                    'start': self.ranges_table.cellWidget(row, 2).value(),
                    'end': self.ranges_table.cellWidget(row, 3).value(),
                    'type': combos[0].currentText(),
                    'peak_type': combos[1].currentText() if len(combos) > 1 and combos[1].isVisible() else None,
                    'is_bg': self.ranges_table.cellWidget(row, 5).findChild(QCheckBox).isChecked(),
                    'paired_bg': self.ranges_table.cellWidget(row, 6).currentText()
                })
            except Exception as e:
                QMessageBox.critical(self, "Range Error", f"Error reading range at row {row + 1}: {e}")
                return

        bg_ranges = [r for r in ranges if r['is_bg']]
        non_bg_ranges = [r for r in ranges if not r['is_bg']]
        auto_paired = False
        if len(bg_ranges) == 1 and all(r['paired_bg'] == 'None' for r in non_bg_ranges):
            single_bg_name = bg_ranges[0]['name']
            for r in non_bg_ranges:
                r['paired_bg'] = single_bg_name
            auto_paired = True
            self.status_label.setText(f"Auto-paired all ranges to '{single_bg_name}' background")

        self.results_dfs.clear()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            df = self.data_df
            time_col = df.columns[0]

            for data_col_name in self.data_columns:
                all_results_for_trace = []
                bg_values = {}
                for r in bg_ranges:
                    mask = (df[time_col] >= r['start']) & (df[time_col] <= r['end'])
                    bg_values[r['name']] = df.loc[mask, data_col_name].mean() if mask.any() else 0.0

                for r in non_bg_ranges:
                    mask = (df[time_col] >= r['start']) & (df[time_col] <= r['end'])
                    subset = df.loc[mask, data_col_name]

                    if subset.empty:
                        raw_value = np.nan
                    elif r['type'] == 'Average':
                        raw_value = subset.mean()
                    else:  # Peak
                        if r['peak_type'] == 'Max':
                            raw_value = subset.max()
                        elif r['peak_type'] == 'Min':
                            raw_value = subset.min()
                        else:
                            raw_value = subset.loc[subset.abs().idxmax()]

                    bg_value = bg_values.get(r['paired_bg'], 0.0)

                    all_results_for_trace.append({
                        'File': self.filename, 'Data Trace': data_col_name, 'Range': r['name'],
                        'Raw Value': raw_value, 'Background': bg_value,
                        'Corrected Value': raw_value - bg_value
                    })

                if all_results_for_trace:
                    self.results_dfs[data_col_name] = pd.DataFrame(all_results_for_trace)

        finally:
            QApplication.restoreOverrideCursor()

        if self.results_dfs:
            self.display_results()
            self.export_btn.setEnabled(True)
            pairing_msg = "\n(Auto-paired to single background)" if auto_paired else ""
        else:
            QMessageBox.warning(self, "No Results", "No results were generated.")
            self.export_btn.setEnabled(False)

    def display_results(self):
        """Displays the analysis results in the results table for all traces."""
        self.results_table.setRowCount(0)
        if not self.results_dfs:
            return

        for trace_name, df in self.results_dfs.items():
            for idx, row_data in df.iterrows():
                row_pos = self.results_table.rowCount()
                self.results_table.insertRow(row_pos)

                for col_idx, col_name in enumerate(['File', 'Data Trace', 'Range', 'Raw Value', 'Background', 'Corrected Value']):
                    value = row_data[col_name]
                    text = f"{value:.4f}" if isinstance(value, float) and not np.isnan(value) else "N/A" if pd.isna(value) else str(value)
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if col_name == 'Corrected Value' and isinstance(value, float) and not np.isnan(value):
                        item.setBackground(QColor(220, 255, 220) if value >= 0 else QColor(255, 220, 220))
                    self.results_table.setItem(row_pos, col_idx, item)

    def export_results(self):
        """
        Exports results to CSVs in the source file's directory,
        handling filename conflicts automatically or via user prompt.
        """
        if not self.results_dfs or not self.filepath:
            QMessageBox.warning(self, "No Data to Export",
                                "Please load a file and run analysis before exporting.")
            return

        directory = os.path.dirname(self.filepath)
        base_filename = os.path.splitext(self.filename)[0]

        exported_files = []
        try:
            for trace_name, df in self.results_dfs.items():
                # Filename sanitization
                safe_trace_name = sanitize_filename(trace_name)

                output_filename = f"{base_filename}_{safe_trace_name}.csv"
                output_path = os.path.join(directory, output_filename)

                # Conflict resolution
                if os.path.exists(output_path):
                    msg_box = QMessageBox(self)
                    msg_box.setIcon(QMessageBox.Question)
                    msg_box.setWindowTitle("File Exists")
                    msg_box.setText(f"The file '{output_filename}' already exists.")
                    msg_box.setInformativeText("What would you like to do?")

                    overwrite_btn = msg_box.addButton("Overwrite", QMessageBox.AcceptRole)
                    rename_btn = msg_box.addButton("Save with New Name", QMessageBox.ActionRole)
                    cancel_btn = msg_box.addButton("Cancel Export", QMessageBox.RejectRole)

                    msg_box.setDefaultButton(rename_btn)
                    msg_box.exec_()

                    clicked_button = msg_box.clickedButton()

                    if clicked_button == overwrite_btn:
                        pass  # Use existing output_path
                    elif clicked_button == rename_btn:
                        output_path = get_next_available_filename(output_path)
                        output_filename = os.path.basename(output_path)
                    else:  # Cancel
                        self.status_label.setText("Export cancelled by user.")
                        QMessageBox.information(self, "Export Cancelled", "The export operation was cancelled.")
                        return

                # Data pivot and save
                export_data = {row['Range']: row['Corrected Value'] for _, row in df.iterrows()}
                export_df = pd.DataFrame([export_data])
                export_df.insert(0, '', '')

                export_df.to_csv(output_path, index=False, float_format='%.4f', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
                exported_files.append(output_filename)

            if exported_files:
                QMessageBox.information(self, "Export Successful",
                                        f"{len(exported_files)} file(s) saved to:\n{directory}\n\n"
                                        f"Files:\n- " + "\n- ".join(exported_files))
                self.status_label.setText(f"Results exported to {os.path.basename(directory)}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An unexpected error occurred during export: {str(e)}")

    def _center_widget(self, widget):
        """Helper to center a widget in a table cell."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return container

    def _style_row(self, row, is_background):
        """Styles a table row to indicate if it's a background range."""
        bg_color = QColor("#E3F2FD") if is_background else QColor(Qt.white)
        for col in range(self.ranges_table.columnCount()):
            widget = self.ranges_table.cellWidget(row, col)
            if widget:
                widget.setAutoFillBackground(True)
                palette = widget.palette()
                palette.setColor(widget.backgroundRole(), bg_color)
                widget.setPalette(palette)