import os
import re
import csv
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
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
from data_analysis_gui.widgets import NoScrollComboBox, SelectAllLineEdit, SelectAllSpinBox
from data_analysis_gui.config import ANALYSIS_CONSTANTS, TABLE_HEADERS
from data_analysis_gui.utils import get_next_available_filename, sanitize_filename


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class AnalysisRange:
    """Data model for an analysis range."""
    name: str
    start: float
    end: float
    analysis_type: str
    peak_type: Optional[str]
    is_background: bool
    paired_background: str
    row_index: int

@dataclass
class AnalysisResult:
    """Data model for analysis results."""
    file: str
    data_trace: str
    range_name: str
    raw_value: float
    background: float
    corrected_value: float


# ============================================================================
# HELPER CLASSES
# ============================================================================

class DataManager:
    """Manages data loading and processing."""
    
    def __init__(self):
        self.data_df = None
        self.filepath = None
        self.filename = None
        self.data_columns = []
        self.results_dfs = {}
    
    def load_csv(self, filepath: str) -> Tuple[bool, str]:
        """Load and validate CSV file."""
        try:
            df = pd.read_csv(filepath)
            
            if df.shape[1] < 2:
                return False, "CSV must have at least 2 columns (time and data)"
            
            self.data_df = df
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            self.data_columns = df.columns[1:].tolist()
            
            return True, f"Loaded {len(df)} points, {len(self.data_columns)} traces"
            
        except Exception as e:
            return False, str(e)
    
    def get_time_column(self) -> str:
        """Get the name of the time column (first column)."""
        return self.data_df.columns[0] if self.data_df is not None else None
    
    def get_filtered_data(self, start_time: float, end_time: float) -> pd.DataFrame:
        """Get data filtered by time range."""
        if self.data_df is None:
            return pd.DataFrame()
        
        time_col = self.get_time_column()
        mask = (self.data_df[time_col] >= start_time) & (self.data_df[time_col] <= end_time)
        return self.data_df.loc[mask].copy()
    
    def clear(self):
        """Clear all loaded data."""
        self.data_df = None
        self.filepath = None
        self.filename = None
        self.data_columns = []
        self.results_dfs = {}


class RangeManager:
    """Manages analysis ranges and their operations."""
    
    def __init__(self, table_widget: QTableWidget):
        self.table = table_widget
        self.ranges = []
    
    def get_all_ranges(self) -> List[AnalysisRange]:
        """Extract all ranges from the table."""
        ranges = []
        for row in range(self.table.rowCount()):
            try:
                analysis_widget = self.table.cellWidget(row, 4)
                combos = analysis_widget.findChildren(NoScrollComboBox)
                
                range_obj = AnalysisRange(
                    name=self.table.cellWidget(row, 1).text(),
                    start=self.table.cellWidget(row, 2).value(),
                    end=self.table.cellWidget(row, 3).value(),
                    analysis_type=combos[0].currentText(),
                    peak_type=combos[1].currentText() if len(combos) > 1 and combos[1].isVisible() else None,
                    is_background=self.table.cellWidget(row, 5).findChild(QCheckBox).isChecked(),
                    paired_background=self.table.cellWidget(row, 6).currentText(),
                    row_index=row
                )
                ranges.append(range_obj)
            except Exception as e:
                print(f"Error reading range at row {row}: {e}")
        
        return ranges
    
    def get_background_ranges(self) -> List[AnalysisRange]:
        """Get only background ranges."""
        return [r for r in self.get_all_ranges() if r.is_background]
    
    def get_analysis_ranges(self) -> List[AnalysisRange]:
        """Get only non-background ranges."""
        return [r for r in self.get_all_ranges() if not r.is_background]
    
    def get_next_range_name(self, is_background: bool = False) -> str:
        """Generate the next available range name."""
        existing_names = set()
        for row in range(self.table.rowCount()):
            name_widget = self.table.cellWidget(row, 1)
            if name_widget:
                existing_names.add(name_widget.text())
        
        if is_background:
            if "Background" not in existing_names:
                return "Background"
            i = 2
            while f"Background_{i}" in existing_names:
                i += 1
            return f"Background_{i}"
        else:
            i = 1
            while f"Range {i}" in existing_names:
                i += 1
            return f"Range {i}"
    
    def calculate_new_range_times(self) -> Tuple[float, float]:
        """Calculate start and end times for a new range."""
        all_end_times = [0.0]
        for row in range(self.table.rowCount()):
            end_spin = self.table.cellWidget(row, 3)
            if end_spin:
                all_end_times.append(end_spin.value())
        
        latest_time = max(all_end_times)
        new_start = latest_time + 5.0 if self.table.rowCount() > 0 else 0.0
        new_end = new_start + 5.0
        
        return new_start, new_end


class PlotManager:
    """Manages plot operations and visualization."""
    
    def __init__(self, figure: Figure, canvas: FigureCanvas, ax):
        self.figure = figure
        self.canvas = canvas
        self.ax = ax
        self.range_lines = []
        self.range_patches = []
        self.line_to_table_row_map = {}
        
    def clear_plot(self):
        """Clear the plot."""
        self.ax.clear()
        self.ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        self.canvas.draw()
    
    def plot_data(self, data_df: pd.DataFrame, data_columns: List[str], filename: str):
        """Plot the data."""
        self.ax.clear()
        
        if data_df is None or data_df.empty:
            self.ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            self.canvas.draw()
            return
        
        time_col = data_df.columns[0]
        color_cycle = plt.get_cmap('viridis')(np.linspace(0, 1, len(data_columns)))
        
        for i, data_col in enumerate(data_columns):
            self.ax.plot(data_df[time_col], data_df[data_col], 
                        lw=0.8, alpha=0.9, label=data_col, color=color_cycle[i])
        
        self.ax.set_xlabel(f"{time_col}")
        self.ax.set_ylabel(" and ".join(data_columns))
        self.ax.set_title(f"Data: {filename}", fontsize=10)
        self.ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        if len(data_columns) > 1:
            self.ax.legend()
        
        self.canvas.draw()
    
    def draw_range_indicators(self, ranges: List[AnalysisRange]):
        """Draw shaded regions and boundary lines for ranges."""
        # Clear existing indicators
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
        
        if not self.ax:
            return
        
        colors = ANALYSIS_CONSTANTS['range_colors']
        
        for range_obj in ranges:
            color_set = colors['background'] if range_obj.is_background else colors['analysis']
            
            # Add shaded region
            patch = self.ax.add_patch(mpatches.Rectangle(
                (range_obj.start, self.ax.get_ylim()[0]),
                range_obj.end - range_obj.start,
                self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                facecolor=color_set['fill'], edgecolor='none', zorder=1
            ))
            self.range_patches.append(patch)
            
            # Add boundary lines
            start_line = self.ax.axvline(range_obj.start, color=color_set['line'], 
                                        ls='--', lw=1.5, picker=5, alpha=0.7)
            end_line = self.ax.axvline(range_obj.end, color=color_set['line'], 
                                      ls='--', lw=1.5, picker=5, alpha=0.7)
            
            self.range_lines.extend([start_line, end_line])
            self.line_to_table_row_map[start_line] = (range_obj.row_index, 2)
            self.line_to_table_row_map[end_line] = (range_obj.row_index, 3)
        
        self.canvas.draw_idle()


class AnalysisProcessor:
    """Handles data analysis calculations."""
    
    @staticmethod
    def calculate_value(data: pd.Series, analysis_type: str, peak_type: Optional[str]) -> float:
        """Calculate a value from data based on analysis type."""
        if data.empty:
            return np.nan
        
        if analysis_type == 'Average':
            return data.mean()
        elif analysis_type == 'Peak':
            if peak_type == 'Max':
                return data.max()
            elif peak_type == 'Min':
                return data.min()
            else:  # Absolute Max
                return data.loc[data.abs().idxmax()]
        
        return np.nan
    
    @staticmethod
    def process_ranges(data_manager: DataManager, ranges: List[AnalysisRange]) -> Dict[str, pd.DataFrame]:
        """Process all ranges and return results."""
        results_dfs = {}
        
        if data_manager.data_df is None:
            return results_dfs
        
        time_col = data_manager.get_time_column()
        bg_ranges = [r for r in ranges if r.is_background]
        analysis_ranges = [r for r in ranges if not r.is_background]
        
        # Auto-pair if single background
        if len(bg_ranges) == 1 and all(r.paired_background == 'None' for r in analysis_ranges):
            single_bg_name = bg_ranges[0].name
            for r in analysis_ranges:
                r.paired_background = single_bg_name
        
        for data_col_name in data_manager.data_columns:
            all_results = []
            
            # Calculate background values
            bg_values = {}
            for bg_range in bg_ranges:
                mask = (data_manager.data_df[time_col] >= bg_range.start) & \
                       (data_manager.data_df[time_col] <= bg_range.end)
                subset = data_manager.data_df.loc[mask, data_col_name]
                bg_values[bg_range.name] = subset.mean() if not subset.empty else 0.0
            
            # Process analysis ranges
            for range_obj in analysis_ranges:
                mask = (data_manager.data_df[time_col] >= range_obj.start) & \
                       (data_manager.data_df[time_col] <= range_obj.end)
                subset = data_manager.data_df.loc[mask, data_col_name]
                
                raw_value = AnalysisProcessor.calculate_value(
                    subset, range_obj.analysis_type, range_obj.peak_type
                )
                
                bg_value = bg_values.get(range_obj.paired_background, 0.0)
                
                result = AnalysisResult(
                    file=data_manager.filename,
                    data_trace=data_col_name,
                    range_name=range_obj.name,
                    raw_value=raw_value,
                    background=bg_value,
                    corrected_value=raw_value - bg_value
                )
                
                all_results.append(result.__dict__)
            
            if all_results:
                results_dfs[data_col_name] = pd.DataFrame(all_results)
        
        return results_dfs


# ============================================================================
# UI BUILDERS
# ============================================================================

class UIBuilder:
    """Helper class for building UI components."""
    
    @staticmethod
    def create_file_group(load_callback) -> QGroupBox:
        """Create the file loading UI group."""
        group = QGroupBox("File")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(5, 5, 5, 5)
        
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("📁 Load CSV")
        load_btn.clicked.connect(load_callback)
        btn_layout.addWidget(load_btn)
        
        file_path_display = QLineEdit("No file loaded")
        file_path_display.setReadOnly(True)
        file_path_display.setStyleSheet("QLineEdit { color: #666; }")
        btn_layout.addWidget(file_path_display)
        
        layout.addLayout(btn_layout)
        
        # Store reference for later access
        group.file_path_display = file_path_display
        
        return group
    
    @staticmethod
    def create_ranges_table() -> QTableWidget:
        """Create and configure the ranges table."""
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(TABLE_HEADERS['ranges'])
        table.setMaximumHeight(250)
        table.setMinimumWidth(430)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        
        table.setColumnWidth(2, 75)
        table.setColumnWidth(3, 75)
        table.setColumnWidth(5, 35)
        
        return table
    
    @staticmethod
    def create_results_table() -> QTableWidget:
        """Create and configure the results table."""
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(TABLE_HEADERS['results'])
        table.setMaximumHeight(250)
        
        header = table.horizontalHeader()
        for i in range(6):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        
        return table
    
    @staticmethod
    def create_preview_table() -> QTableWidget:
        """Create and configure the data preview table."""
        table = QTableWidget()
        table.setMinimumHeight(300)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        return table
    
    @staticmethod
    def center_widget(widget: QWidget) -> QWidget:
        """Helper to center a widget in a container."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return container


# ============================================================================
# MAIN DIALOG
# ============================================================================

class ConcentrationResponseDialog(QDialog):
    """
    Enhanced dialog for analyzing patch-clamp time-series data.
    Refactored for better modularity and maintainability.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize managers
        self.data_manager = DataManager()
        self.analysis_processor = AnalysisProcessor()
        
        # UI state
        self.selected_range_row = None
        self.last_focused_editor = None
        
        # Interaction state
        self.dragging_line = None
        self.pan_active = False
        self.pan_start_pos = None
        self.pan_start_lim = None
        
        # Setup UI
        self._setup_window()
        self._setup_ui()
        self._setup_plot()
        
        # Initialize managers after UI is created
        self.range_manager = RangeManager(self.ranges_table)
        self.plot_manager = PlotManager(self.figure, self.canvas, self.ax)
        
        # Install event filter
        QApplication.instance().installEventFilter(self)
        
        # Add initial range
        self.add_range_row()
    
    # ========================================================================
    # INITIALIZATION METHODS
    # ========================================================================
    
    def _setup_window(self):
        """Configure window properties."""
        self.setWindowTitle("Patch-Clamp Concentration-Response Analysis")
        self.setGeometry(25, 50, 1850, 950)
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Status bar
        self.status_label = QLabel("Load a CSV file to begin")
        self.status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; font-size: 9pt; }")
        self.status_label.setMaximumHeight(20)
        main_layout.addWidget(self.status_label)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([700, 1150])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # File group
        self.file_group = UIBuilder.create_file_group(self.load_file)
        self.file_path_display = self.file_group.file_path_display
        layout.addWidget(self.file_group)
        
        # Ranges group
        layout.addWidget(self._create_ranges_group())
        
        # Results group
        layout.addWidget(self._create_results_group())
        
        layout.addStretch()
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with plot."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self._create_plot_group())
        return panel
    
    def _create_ranges_group(self) -> QGroupBox:
        """Create the ranges configuration group."""
        group = QGroupBox("Analysis Ranges (drag boundaries in plot)")
        layout = QVBoxLayout(group)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Ranges table
        self.ranges_table = UIBuilder.create_ranges_table()
        self.ranges_table.itemChanged.connect(self.on_range_value_changed)
        layout.addWidget(self.ranges_table)
        
        # Control buttons
        bottom_layout = QHBoxLayout()
        
        add_range_btn = QPushButton("➕ Add Range")
        add_range_btn.clicked.connect(lambda: self.add_range_row(is_background=False))
        
        add_bg_range_btn = QPushButton("➕ Add Background")
        add_bg_range_btn.clicked.connect(lambda: self.add_range_row(is_background=True))
        
        self.add_paired_bg_btn = QPushButton("➕ Add Paired Background")
        self.add_paired_bg_btn.clicked.connect(self.add_paired_background)
        self.add_paired_bg_btn.setToolTip("Add a background range paired to the currently selected analysis range")
        
        mu_button = QPushButton("Insert μ")
        mu_button.clicked.connect(self.insert_mu_char)
        
        bottom_layout.addWidget(add_range_btn)
        bottom_layout.addWidget(add_bg_range_btn)
        bottom_layout.addWidget(self.add_paired_bg_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(mu_button)
        
        layout.addLayout(bottom_layout)
        
        return group
    
    def _create_results_group(self) -> QGroupBox:
        """Create the results display group."""
        group = QGroupBox("Results")
        layout = QVBoxLayout(group)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.run_analysis_btn = QPushButton("▶ Run Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        btn_layout.addWidget(self.run_analysis_btn)
        
        self.export_btn = QPushButton("💾 Export CSV(s)")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        btn_layout.addWidget(self.export_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Results table
        self.results_table = UIBuilder.create_results_table()
        layout.addWidget(self.results_table)
        
        # Data preview section
        preview_label = QLabel("Data Preview (Selected Range):")
        preview_label.setStyleSheet("QLabel { font-weight: bold; margin-top: 5px; }")
        layout.addWidget(preview_label)
        
        self.preview_table = UIBuilder.create_preview_table()
        layout.addWidget(self.preview_table)
        
        return group
    
    def _create_plot_group(self) -> QGroupBox:
        """Create the plot visualization group."""
        group = QGroupBox("Data Visualization")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Setup figure and canvas
        self.figure = Figure(figsize=(14, 9), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Navigation toolbar
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        
        return group
    
    def _setup_plot(self):
        """Setup plot properties and event handlers."""
        self.ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        self.ax.set_xlabel("Time (s)", fontsize=10)
        self.ax.set_ylabel("Current (pA)", fontsize=10)
        
        # Connect event handlers
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect('scroll_event', self.on_scroll_zoom)
        self.canvas.mpl_connect('button_press_event', self.on_pan_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_pan_motion)
        self.canvas.mpl_connect('button_release_event', self.on_pan_release)
    
    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================
    
    def eventFilter(self, obj, event):
        """Event filter to capture focus events."""
        if event.type() == QEvent.FocusIn:
            if isinstance(obj, QLineEdit):
                self.last_focused_editor = obj
            
            if isinstance(obj, (QDoubleSpinBox, QLineEdit)):
                row_index = obj.property("row_index")
                if row_index is not None:
                    self.on_range_selected(row_index)
        
        return super().eventFilter(obj, event)
    
    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================
    
    def load_file(self):
        """Open dialog to select and load a CSV file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select a Patch-Clamp CSV File", "", "CSV files (*.csv)"
        )
        
        if filepath:
            success, message = self.data_manager.load_csv(filepath)
            
            if success:
                self.file_path_display.setText(self.data_manager.filename)
                self.status_label.setText(message)
                self.process_and_plot_file()
            else:
                QMessageBox.warning(self, "Load Error", f"Could not load file: {message}")
                self.status_label.setText("Error loading file")
    
    def process_and_plot_file(self):
        """Process and plot the loaded file."""
        if self.data_manager.data_df is None:
            self.plot_manager.clear_plot()
            return
        
        # Plot data
        self.plot_manager.plot_data(
            self.data_manager.data_df,
            self.data_manager.data_columns,
            self.data_manager.filename
        )
        
        # Update range indicators
        self.update_plot_ranges()
        
        # Auto-select first range for preview
        if self.ranges_table.rowCount() > 0:
            if self.selected_range_row is not None and self.selected_range_row < self.ranges_table.rowCount():
                self.on_range_selected(self.selected_range_row)
            else:
                self.on_range_selected(0)
        else:
            self.clear_preview()
    
    # ========================================================================
    # RANGE MANAGEMENT
    # ========================================================================
    
    def add_range_row(self, is_background=False):
        """Add a new range row to the table."""
        # Calculate timing
        new_start, new_end = self.range_manager.calculate_new_range_times()
        
        # Get row index and insert row
        row = self.ranges_table.rowCount()
        self.ranges_table.insertRow(row)
        self.ranges_table.setRowHeight(row, 24)
        
        # Create widgets
        table_parent = self.ranges_table
        table_font = table_parent.font()
        
        # Remove button
        remove_btn = QPushButton("✖", table_parent)
        remove_btn.setFont(table_font)
        remove_btn.clicked.connect(self.remove_range_row)
        
        # Name edit
        default_name = self.range_manager.get_next_range_name(is_background)
        name_edit = SelectAllLineEdit(default_name, table_parent)
        name_edit.setFont(table_font)
        name_edit.textChanged.connect(self.update_background_options)
        name_edit.setProperty("row_index", row)
        name_edit.installEventFilter(self)
        
        # Start/End spinboxes
        start_spin = self._create_time_spinbox(table_parent, table_font, new_start, row)
        end_spin = self._create_time_spinbox(table_parent, table_font, new_end, row)
        
        # Analysis type widget
        analysis_widget = self._create_analysis_widget(table_parent, table_font)
        
        # Background checkbox
        bg_checkbox = QCheckBox(table_parent)
        bg_checkbox.setFont(table_font)
        bg_checkbox.stateChanged.connect(self.update_background_options)
        if is_background:
            bg_checkbox.setChecked(True)
        
        # Paired background combo
        paired_combo = NoScrollComboBox(table_parent)
        paired_combo.setFont(table_font)
        paired_combo.addItem("None")
        
        # Add widgets to table
        self.ranges_table.setCellWidget(row, 0, remove_btn)
        self.ranges_table.setCellWidget(row, 1, name_edit)
        self.ranges_table.setCellWidget(row, 2, start_spin)
        self.ranges_table.setCellWidget(row, 3, end_spin)
        self.ranges_table.setCellWidget(row, 4, analysis_widget)
        self.ranges_table.setCellWidget(row, 5, UIBuilder.center_widget(bg_checkbox))
        self.ranges_table.setCellWidget(row, 6, paired_combo)
        
        # Configure for background ranges
        if is_background:
            paired_combo.setEnabled(False)
            paired_combo.setCurrentText("None")
            paired_combo.setStyleSheet("QComboBox { color: #999; background-color: #f0f0f0; }")
        
        # Update UI
        self.update_background_options()
        self.update_plot_ranges()
        self.on_range_selected(row)
    
    def _create_time_spinbox(self, parent, font, value, row):
        """Create a configured time spinbox."""
        spin = SelectAllSpinBox(parent)
        spin.setFont(font)
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(2)
        spin.setValue(value)
        spin.setFixedWidth(75)
        spin.setProperty("row_index", row)
        spin.installEventFilter(self)
        spin.valueChanged.connect(self.update_plot_ranges)
        spin.valueChanged.connect(lambda _, r=row: self.update_preview_if_selected(r))
        return spin
    
    def _create_analysis_widget(self, parent, font):
        """Create the analysis type selection widget."""
        widget = QWidget(parent)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        analysis_combo = NoScrollComboBox(parent)
        analysis_combo.setFont(font)
        analysis_combo.addItems(["Average", "Peak"])
        
        peak_combo = NoScrollComboBox(parent)
        peak_combo.setFont(font)
        peak_combo.addItems(["Max", "Min", "Absolute Max"])
        peak_combo.setVisible(False)
        
        analysis_combo.currentTextChanged.connect(lambda text: peak_combo.setVisible(text == "Peak"))
        
        layout.addWidget(analysis_combo)
        layout.addWidget(peak_combo)
        
        return widget
    
    def remove_range_row(self):
        """Remove the range row for the clicked button."""
        sender = self.sender()
        if not sender:
            return
        
        for row in range(self.ranges_table.rowCount()):
            if self.ranges_table.cellWidget(row, 0) == sender:
                # Update selection tracking
                if self.selected_range_row == row:
                    self.selected_range_row = None
                    self.update_data_preview()
                elif self.selected_range_row is not None and self.selected_range_row > row:
                    self.selected_range_row -= 1
                
                # Remove row and update
                self.ranges_table.removeRow(row)
                self.update_background_options()
                self.update_plot_ranges()
                break
    
    def add_paired_background(self):
        """Add a background range paired to the selected analysis range."""
        # Find target analysis range
        target_row = None
        
        if self.selected_range_row is not None:
            bg_widget = self.ranges_table.cellWidget(self.selected_range_row, 5)
            if bg_widget:
                is_bg = bg_widget.findChild(QCheckBox).isChecked()
                if not is_bg:
                    target_row = self.selected_range_row
        
        if target_row is None:
            for row in range(self.ranges_table.rowCount()):
                bg_widget = self.ranges_table.cellWidget(row, 5)
                if bg_widget:
                    is_bg = bg_widget.findChild(QCheckBox).isChecked()
                    if not is_bg:
                        target_row = row
                        break
        
        if target_row is None:
            QMessageBox.information(
                self, 
                "No Analysis Range", 
                "Please create or select an analysis range first before adding a paired background."
            )
            return
        
        # Get target range name
        name_widget = self.ranges_table.cellWidget(target_row, 1)
        target_range_name = name_widget.text() if name_widget else f"Range {target_row + 1}"
        
        # Generate unique background name
        existing_names = set()
        for row in range(self.ranges_table.rowCount()):
            name_widget = self.ranges_table.cellWidget(row, 1)
            if name_widget:
                existing_names.add(name_widget.text())
        
        base_bg_name = f"BG_{target_range_name}"
        bg_name = base_bg_name
        counter = 2
        while bg_name in existing_names:
            bg_name = f"{base_bg_name}_{counter}"
            counter += 1
        
        # Add background range
        bg_row = self.ranges_table.rowCount()
        self.add_range_row(is_background=True)
        
        # Set custom name
        bg_name_widget = self.ranges_table.cellWidget(bg_row, 1)
        if bg_name_widget:
            bg_name_widget.setText(bg_name)
        
        # Set pairing
        paired_combo = self.ranges_table.cellWidget(target_row, 6)
        if paired_combo:
            self.update_background_options()
            paired_combo.setCurrentText(bg_name)
        
        self.status_label.setText(f"Added '{bg_name}' paired to '{target_range_name}'")
        self.on_range_selected(bg_row)
    
    def update_background_options(self):
        """Update background-related UI elements."""
        background_names = ["None"]
        
        # Collect background names and update row styling
        for row in range(self.ranges_table.rowCount()):
            bg_widget = self.ranges_table.cellWidget(row, 5)
            name_widget = self.ranges_table.cellWidget(row, 1)
            analysis_widget = self.ranges_table.cellWidget(row, 4)
            paired_combo = self.ranges_table.cellWidget(row, 6)
            
            if bg_widget and name_widget:
                is_checked = bg_widget.findChild(QCheckBox).isChecked()
                self._style_row(row, is_checked)
                
                combo = analysis_widget.findChild(NoScrollComboBox)
                if combo:
                    combo.setEnabled(not is_checked)
                
                if paired_combo:
                    if is_checked:
                        paired_combo.setEnabled(False)
                        paired_combo.setCurrentText("None")
                        paired_combo.setStyleSheet("QComboBox { color: #999; background-color: #f0f0f0; }")
                    else:
                        paired_combo.setEnabled(True)
                        paired_combo.setStyleSheet("")
                
                if is_checked:
                    background_names.append(name_widget.text())
                    if combo:
                        combo.setCurrentText("Average")
        
        # Update paired background dropdowns
        for row in range(self.ranges_table.rowCount()):
            paired_combo = self.ranges_table.cellWidget(row, 6)
            bg_widget = self.ranges_table.cellWidget(row, 5)
            
            if paired_combo and bg_widget:
                is_bg = bg_widget.findChild(QCheckBox).isChecked()
                if not is_bg:
                    current = paired_combo.currentText()
                    paired_combo.clear()
                    paired_combo.addItems(background_names)
                    if current in background_names:
                        paired_combo.setCurrentText(current)
        
        # Update paired background button state
        self._update_paired_bg_button_state()
    
    def _update_paired_bg_button_state(self):
        """Update the state of the paired background button."""
        if not hasattr(self, 'add_paired_bg_btn'):
            return
        
        if self.selected_range_row is not None and self.selected_range_row < self.ranges_table.rowCount():
            bg_widget = self.ranges_table.cellWidget(self.selected_range_row, 5)
            if bg_widget:
                is_background = bg_widget.findChild(QCheckBox).isChecked()
                if is_background:
                    self.add_paired_bg_btn.setEnabled(False)
                    self.add_paired_bg_btn.setToolTip("Cannot pair a background to another background")
                    self.add_paired_bg_btn.setStyleSheet("QPushButton { color: #999; }")
                else:
                    self.add_paired_bg_btn.setEnabled(True)
                    self.add_paired_bg_btn.setToolTip("Add a background range paired to the currently selected analysis range")
                    self.add_paired_bg_btn.setStyleSheet("")
        else:
            self.add_paired_bg_btn.setEnabled(False)
            self.add_paired_bg_btn.setToolTip("Select an analysis range first")
            self.add_paired_bg_btn.setStyleSheet("QPushButton { color: #999; }")
    
    def _style_row(self, row, is_background):
        """Apply styling to indicate background ranges."""
        bg_color = QColor("#E3F2FD") if is_background else QColor(Qt.white)
        for col in range(self.ranges_table.columnCount()):
            widget = self.ranges_table.cellWidget(row, col)
            if widget:
                widget.setAutoFillBackground(True)
                palette = widget.palette()
                palette.setColor(widget.backgroundRole(), bg_color)
                widget.setPalette(palette)
    
    # ========================================================================
    # PLOT INTERACTION
    # ========================================================================
    
    def update_plot_ranges(self):
        """Update range indicators on the plot."""
        ranges = self.range_manager.get_all_ranges()
        self.plot_manager.draw_range_indicators(ranges)
    
    def on_click(self, event):
        """Handle mouse click on plot."""
        if event.inaxes != self.ax or not self.plot_manager.range_lines or event.xdata is None:
            return
        
        # Find closest line
        line_distances = [(line, abs(event.xdata - line.get_xdata()[0])) 
                         for line in self.plot_manager.range_lines]
        closest_line, min_dist = min(line_distances, key=lambda item: item[1])
        
        # Check if close enough to drag
        x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        if x_range > 0 and min_dist < x_range * 0.02:
            self.dragging_line = closest_line
            
            # Auto-select the range being dragged
            row, col = self.plot_manager.line_to_table_row_map.get(closest_line, (None, None))
            if row is not None:
                self.on_range_selected(row)
    
    def on_drag(self, event):
        """Handle mouse drag on plot."""
        if self.dragging_line and event.xdata is not None:
            self.dragging_line.set_xdata([event.xdata, event.xdata])
            row, col = self.plot_manager.line_to_table_row_map.get(self.dragging_line, (None, None))
            if row is not None:
                spinbox = self.ranges_table.cellWidget(row, col)
                if spinbox:
                    spinbox.blockSignals(True)
                    spinbox.setValue(event.xdata)
                    spinbox.blockSignals(False)
                
                if self.selected_range_row == row:
                    self.update_data_preview()
            
            self.canvas.draw_idle()
    
    def on_release(self, event):
        """Handle mouse release on plot."""
        if self.dragging_line:
            row, col = self.plot_manager.line_to_table_row_map.get(self.dragging_line, (None, None))
            self.dragging_line = None
            self.update_plot_ranges()
            
            if row is not None and self.selected_range_row == row:
                self.update_data_preview()
    
    def on_scroll_zoom(self, event):
        """Handle scroll wheel zoom."""
        if event.inaxes != self.ax:
            return
        
        base_scale = ANALYSIS_CONSTANTS['zoom_scale_factor']
        
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return
        
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        new_xlim = [(x - event.xdata) * scale_factor + event.xdata for x in cur_xlim]
        new_ylim = [(y - event.ydata) * scale_factor + event.ydata for y in cur_ylim]
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()
    
    def on_pan_press(self, event):
        """Handle pan start (middle mouse button)."""
        if event.inaxes != self.ax or event.button != 2:
            return
        
        self.pan_active = True
        self.pan_start_pos = (event.xdata, event.ydata)
        self.pan_start_lim = (self.ax.get_xlim(), self.ax.get_ylim())
        self.canvas.setCursor(ANALYSIS_CONSTANTS['pan_cursor'])
    
    def on_pan_motion(self, event):
        """Handle pan motion."""
        if not self.pan_active or event.inaxes != self.ax:
            return
        
        dx = event.xdata - self.pan_start_pos[0]
        dy = event.ydata - self.pan_start_pos[1]
        
        start_xlim, start_ylim = self.pan_start_lim
        
        new_xlim = (start_xlim[0] - dx, start_xlim[1] - dx)
        new_ylim = (start_ylim[0] - dy, start_ylim[1] - dy)
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()
    
    def on_pan_release(self, event):
        """Handle pan end."""
        if event.button == 2:
            self.pan_active = False
            self.pan_start_pos = None
            self.pan_start_lim = None
            self.canvas.setCursor(Qt.ArrowCursor)
    
    # ========================================================================
    # DATA PREVIEW
    # ========================================================================
    
    def on_range_selected(self, row):
        """Handle range selection for preview."""
        self.selected_range_row = row
        self.update_data_preview()
        self._update_paired_bg_button_state()
    
    def on_range_value_changed(self):
        """Handle range value changes."""
        self.update_plot_ranges()
    
    def update_preview_if_selected(self, row):
        """Update preview only if the given row is selected."""
        if self.selected_range_row == row:
            self.update_data_preview()
    
    def update_data_preview(self):
        """Update the data preview table."""
        if self.data_manager.data_df is None or self.selected_range_row is None:
            self.clear_preview()
            return
        
        if self.selected_range_row >= self.ranges_table.rowCount():
            self.selected_range_row = None
            self.clear_preview()
            return
        
        try:
            # Get range boundaries
            start_widget = self.ranges_table.cellWidget(self.selected_range_row, 2)
            end_widget = self.ranges_table.cellWidget(self.selected_range_row, 3)
            
            if not start_widget or not end_widget:
                self.clear_preview()
                return
            
            # Get filtered data
            filtered_data = self.data_manager.get_filtered_data(
                start_widget.value(), 
                end_widget.value()
            )
            
            if filtered_data.empty:
                self.show_empty_preview()
                return
            
            # Populate preview table
            self.populate_preview_table(filtered_data)
            
            # Update status
            range_name = self.ranges_table.cellWidget(self.selected_range_row, 1).text()
            self.status_label.setText(f"Previewing: {range_name} ({len(filtered_data)} points)")
            
        except Exception as e:
            print(f"Error updating data preview: {e}")
            self.clear_preview()
    
    def clear_preview(self):
        """Clear the preview table."""
        self.preview_table.clear()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
    
    def show_empty_preview(self):
        """Show empty preview message."""
        self.preview_table.setColumnCount(1)
        self.preview_table.setRowCount(1)
        item = QTableWidgetItem("No data in selected range")
        item.setTextAlignment(Qt.AlignCenter)
        self.preview_table.setItem(0, 0, item)
    
    def populate_preview_table(self, data):
        """Populate the preview table with data."""
        time_col = self.data_manager.get_time_column()
        columns_to_show = [time_col] + self.data_manager.data_columns
        
        self.preview_table.setColumnCount(len(columns_to_show))
        self.preview_table.setHorizontalHeaderLabels(columns_to_show)
        self.preview_table.setRowCount(len(data))
        
        for row_idx, (data_idx, row_data) in enumerate(data.iterrows()):
            for col_idx, col_name in enumerate(columns_to_show):
                value = row_data[col_name]
                if isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                self.preview_table.setItem(row_idx, col_idx, item)
        
        self.preview_table.resizeColumnsToContents()
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    
    def run_analysis(self):
        """Run the analysis on defined ranges."""
        if self.data_manager.data_df is None:
            QMessageBox.warning(self, "No File", "Please load a CSV file before running analysis.")
            return
        
        if self.ranges_table.rowCount() == 0:
            QMessageBox.warning(self, "No Ranges", "Please define at least one analysis range.")
            return
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Get all ranges
            ranges = self.range_manager.get_all_ranges()
            
            # Process ranges
            self.data_manager.results_dfs = self.analysis_processor.process_ranges(
                self.data_manager, 
                ranges
            )
            
            # Check for auto-pairing
            bg_ranges = [r for r in ranges if r.is_background]
            non_bg_ranges = [r for r in ranges if not r.is_background]
            auto_paired = (len(bg_ranges) == 1 and 
                          all(r.paired_background == 'None' for r in non_bg_ranges))
            
            if auto_paired:
                self.status_label.setText(f"Auto-paired all ranges to '{bg_ranges[0].name}' background")
            
        finally:
            QApplication.restoreOverrideCursor()
        
        if self.data_manager.results_dfs:
            self.display_results()
            self.export_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "No Results", "No results were generated.")
            self.export_btn.setEnabled(False)
    
    def display_results(self):
        """Display analysis results in the results table."""
        self.results_table.setRowCount(0)
        if not self.data_manager.results_dfs:
            return
        
        for trace_name, df in self.data_manager.results_dfs.items():
            for idx, row_data in df.iterrows():
                row_pos = self.results_table.rowCount()
                self.results_table.insertRow(row_pos)
                
                columns = ['file', 'data_trace', 'range_name', 'raw_value', 'background', 'corrected_value']
                for col_idx, col_name in enumerate(columns):
                    value = row_data[col_name]
                    
                    if isinstance(value, float) and not np.isnan(value):
                        text = f"{value:.4f}"
                    elif pd.isna(value):
                        text = "N/A"
                    else:
                        text = str(value)
                    
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    
                    # Color code corrected values
                    if col_name == 'corrected_value' and isinstance(value, float) and not np.isnan(value):
                        color = QColor(220, 255, 220) if value >= 0 else QColor(255, 220, 220)
                        item.setBackground(color)
                    
                    self.results_table.setItem(row_pos, col_idx, item)
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    def export_results(self):
        """Export results to CSV files."""
        if not self.data_manager.results_dfs or not self.data_manager.filepath:
            QMessageBox.warning(self, "No Data to Export",
                              "Please load a file and run analysis before exporting.")
            return
        
        directory = os.path.dirname(self.data_manager.filepath)
        base_filename = os.path.splitext(self.data_manager.filename)[0]
        
        exported_files = []
        try:
            for trace_name, df in self.data_manager.results_dfs.items():
                # Prepare filename
                safe_trace_name = sanitize_filename(trace_name)
                output_filename = f"{base_filename}_{safe_trace_name}.csv"
                output_path = os.path.join(directory, output_filename)
                
                # Handle file conflicts
                output_path = self._handle_file_conflict(output_path, output_filename)
                if output_path is None:
                    return  # User cancelled
                
                # Export data
                export_data = {row['range_name']: row['corrected_value'] 
                             for _, row in df.iterrows()}
                export_df = pd.DataFrame([export_data])
                export_df.insert(0, '', '')
                
                export_df.to_csv(output_path, index=False, float_format='%.4f', 
                               encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
                exported_files.append(os.path.basename(output_path))
            
            # Show success message
            if exported_files:
                QMessageBox.information(
                    self, "Export Successful",
                    f"{len(exported_files)} file(s) saved to:\n{directory}\n\n"
                    f"Files:\n- " + "\n- ".join(exported_files)
                )
                self.status_label.setText(f"Results exported to {os.path.basename(directory)}")
        
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"An unexpected error occurred during export: {str(e)}")
    
    def _handle_file_conflict(self, output_path, output_filename):
        """Handle file naming conflicts during export."""
        if not os.path.exists(output_path):
            return output_path
        
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
            return output_path
        elif clicked_button == rename_btn:
            return get_next_available_filename(output_path)
        else:
            self.status_label.setText("Export cancelled by user.")
            QMessageBox.information(self, "Export Cancelled", 
                                   "The export operation was cancelled.")
            return None
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def insert_mu_char(self):
        """Insert μ character at last focused position."""
        editor = self.last_focused_editor
        if editor:
            editor.insert("μ")
            if isinstance(editor, SelectAllLineEdit):
                editor.setFocusAndDoNotSelect()
            else:
                editor.setFocus()