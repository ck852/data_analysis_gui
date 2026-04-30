"""
PatchBatch Electrophysiology Data Analysis Tool - Kinetics Dialog

Non-modal dialog displaying mono- and bi-exponential fit results for the
currently displayed sweep over Range 1. Text-only presentation; no plotting.

Consumes a KineticsResult from services/kinetics_service.py and renders
per-fit parameter tables, goodness-of-fit statistics, and AIC/BIC comparison.

Includes an "Auto-detect fit region within Range 1" checkbox. When checked,
reanalysis narrows the fit window to a peak/trough-anchored sub-region inside
Range 1 (see services/kinetics_service.py for details). When unchecked, the
full Range 1 is used. The checkbox state is passed to the reanalyze callback.

Supports a Reanalyze button: when provided with a reanalyze_callback, the
dialog can refresh its contents in-place using the latest state of MainWindow
(current sweep, Range 1 bounds, file, units) and the current auto-detect
checkbox state, without being closed and reopened.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QWidget, QApplication,
    QCheckBox,
)
from PySide6.QtCore import Qt

from data_analysis_gui.config.themes import (
    apply_modern_theme, style_label, style_group_box, style_button,
    apply_compact_layout, style_checkbox,
)
from data_analysis_gui.config.logging import get_logger
from data_analysis_gui.services.kinetics_service import (
    KineticsResult, SingleFitResult,
)

logger = get_logger(__name__)


# Type alias: callback takes the auto_detect_region flag and returns
# (result, file_path, sweep_index, current_units) or None on precondition failure.
ReanalyzeCallback = Callable[
    [bool],
    Optional[Tuple[KineticsResult, Optional[str], str, str]]
]


# Display labels for parameter keys
_PARAM_DISPLAY = {
    "A": "A",
    "tau": "\u03C4",           # tau
    "tau1": "\u03C4\u2081",    # tau subscript 1
    "tau2": "\u03C4\u2082",    # tau subscript 2
    "A1": "A\u2081",
    "A2": "A\u2082",
    "C": "C",
}


def _param_units(key: str, current_units: str) -> str:
    if key.startswith("tau"):
        return "ms"
    if key.startswith("A") or key == "C":
        return current_units
    return ""


class KineticsDialog(QDialog):
    """Non-modal results window for mono/bi-exponential kinetics fits."""

    def __init__(
        self,
        result: KineticsResult,
        file_path: Optional[str],
        sweep_index: str,
        current_units: str = "pA",
        reanalyze_callback: Optional[ReanalyzeCallback] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.result = result
        self.file_path = file_path
        self.sweep_index = sweep_index
        self.current_units = current_units
        self._reanalyze_callback = reanalyze_callback

        # The auto-detect checkbox is created in _build_button_bar(). Its state
        # is the source of truth for what reanalysis should do.
        self._auto_detect_cb: Optional[QCheckBox] = None

        self.setWindowTitle("Kinetics Analysis")
        self.setWindowModality(Qt.WindowModality.NonModal)

        apply_modern_theme(self)
        self._build_ui()
        self._size_to_screen()

    def _size_to_screen(self):
        """Pick a reasonable default size, capped to available screen area."""
        app = QApplication.instance()
        screen = app.primaryScreen() if app else None

        default_w, default_h = 820, 480

        if screen is not None:
            avail = screen.availableGeometry()
            max_w = int(avail.width() * 0.9)
            max_h = int(avail.height() * 0.85)
            w = min(default_w, max_w)
            h = min(default_h, max_h)
        else:
            w, h = default_w, default_h

        self.setMinimumSize(560, 340)
        self.resize(w, h)

    def _build_ui(self):
        """Build the dialog shell. Content is populated separately so it can be refreshed."""
        outer = QVBoxLayout(self)
        apply_compact_layout(self, spacing=6, margin=8)

        # Container for variable content (header + fit panels + comparison)
        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)
        outer.addWidget(self._content_widget)

        self._populate_content()

        # Persistent button bar (includes auto-detect checkbox)
        outer.addWidget(self._build_button_bar())

    def _populate_content(self):
        """Fill the content container based on the current self.result."""
        self._content_layout.addWidget(self._build_header())

        if not self.result.success:
            err_label = QLabel(
                f"Kinetics analysis failed:\n\n{self.result.error_message}"
            )
            err_label.setWordWrap(True)
            style_label(err_label, "normal")
            self._content_layout.addWidget(err_label)
            return

        # Two fit panels side-by-side, wrapped in a container so clearing is clean
        fits_container = QWidget()
        fits_row = QHBoxLayout(fits_container)
        fits_row.setContentsMargins(0, 0, 0, 0)
        fits_row.setSpacing(8)
        fits_row.addWidget(self._build_fit_panel(self.result.mono))
        fits_row.addWidget(self._build_fit_panel(self.result.biexp))
        self._content_layout.addWidget(fits_container)

        self._content_layout.addWidget(self._build_comparison_panel())

    def _clear_content(self):
        """Remove all widgets from the content container."""
        while self._content_layout.count() > 0:
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def update_result(
        self,
        result: KineticsResult,
        file_path: Optional[str],
        sweep_index: str,
        current_units: str,
    ):
        """Replace the dialog's contents with a fresh KineticsResult."""
        self.result = result
        self.file_path = file_path
        self.sweep_index = sweep_index
        self.current_units = current_units

        self._clear_content()
        self._populate_content()

    def _build_header(self) -> QWidget:
        """Compact header showing file/sweep and the fit region actually used."""
        header = QWidget()
        vbox = QVBoxLayout(header)
        vbox.setContentsMargins(4, 2, 4, 2)
        vbox.setSpacing(2)

        file_name = Path(self.file_path).name if self.file_path else "Unknown"

        line1 = QLabel(
            f"<b>File:</b> {file_name}   "
            f"<b>Sweep:</b> {self.sweep_index}"
        )
        line1.setTextFormat(Qt.TextFormat.RichText)
        style_label(line1, "normal")
        vbox.addWidget(line1)

        # Fit region line: show requested Range 1 and, if different, the
        # actual sub-region used after auto-detection.
        fit_region_text = self._format_fit_region_text()
        line2 = QLabel(fit_region_text)
        line2.setTextFormat(Qt.TextFormat.RichText)
        style_label(line2, "muted")
        vbox.addWidget(line2)

        return header

    def _format_fit_region_text(self) -> str:
        """Build the rich-text summary of the fit region."""
        r = self.result
        n = len(r.time_fit_ms)

        if r.auto_detected:
            # Show the user range and the auto-detected sub-region
            return (
                f"<b>Range 1:</b> {r.range_start_ms:.2f} \u2013 "
                f"{r.range_end_ms:.2f} ms   "
                f"<b>Fit region (auto-detected):</b> "
                f"{r.fit_region_start_ms:.2f} \u2013 "
                f"{r.fit_region_end_ms:.2f} ms   "
                f"<b>Direction:</b> {r.direction}   "
                f"<b>N:</b> {n}"
            )
        else:
            return (
                f"<b>Fit region:</b> {r.fit_region_start_ms:.2f} \u2013 "
                f"{r.fit_region_end_ms:.2f} ms "
                f"(t=0 at {r.fit_region_start_ms:.2f} ms)   "
                f"<b>Direction:</b> {r.direction}   "
                f"<b>N:</b> {n}"
            )

    def _build_fit_panel(self, fit: SingleFitResult) -> QWidget:
        """Build a group box for a single fit (mono or bi-exp)."""
        group = QGroupBox(fit.model_name)
        style_group_box(group)
        vbox = QVBoxLayout(group)
        apply_compact_layout(group, spacing=4, margin=6)

        if not fit.success:
            err = QLabel(f"Fit failed: {fit.error_message}")
            err.setWordWrap(True)
            style_label(err, "normal")
            vbox.addWidget(err)
            return group

        # Parameter table
        param_table = QTableWidget()
        param_table.setColumnCount(4)
        param_table.setHorizontalHeaderLabels(
            ["Param", "Value", "Std. Err.", "Units"]
        )
        param_table.setRowCount(len(fit.params))
        param_table.verticalHeader().setVisible(False)
        param_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        param_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        param_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        param_table.setAlternatingRowColors(True)

        header = param_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        for row, (key, val) in enumerate(fit.params.items()):
            display_name = _PARAM_DISPLAY.get(key, key)
            stderr = fit.param_stderr.get(key, float("nan"))
            units = _param_units(key, self.current_units)

            name_item = QTableWidgetItem(display_name)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            param_table.setItem(row, 0, name_item)

            val_item = QTableWidgetItem(self._fmt_value(val))
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            param_table.setItem(row, 1, val_item)

            err_item = QTableWidgetItem(self._fmt_stderr(stderr))
            err_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            param_table.setItem(row, 2, err_item)

            unit_item = QTableWidgetItem(units)
            unit_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            param_table.setItem(row, 3, unit_item)

        self._shrink_table_rows(param_table)
        param_table.setFixedHeight(self._exact_table_height(param_table))
        vbox.addWidget(param_table)

        stats_grid = self._build_stats_grid(fit)
        vbox.addWidget(stats_grid)

        return group

    def _build_stats_grid(self, fit: SingleFitResult) -> QWidget:
        """Compact stats display using a 2-column table (label, value pairs)."""
        stats = [
            ("R\u00B2", self._fmt_r2(fit.r_squared)),
            ("Adj. R\u00B2", self._fmt_r2(fit.adjusted_r_squared)),
            ("RMSE", self._fmt_value(fit.rmse)),
            ("SS_res", self._fmt_value(fit.ss_res)),
            ("AIC", self._fmt_value(fit.aic)),
            ("BIC", self._fmt_value(fit.bic)),
        ]

        table = QTableWidget()
        table.setColumnCount(2)
        table.setRowCount(len(stats))
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)

        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )

        for row, (label, value) in enumerate(stats):
            label_item = QTableWidgetItem(label)
            label_item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(row, 0, label_item)

            value_item = QTableWidgetItem(value)
            value_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(row, 1, value_item)

        self._shrink_table_rows(table)
        table.setFixedHeight(self._exact_table_height(table, has_header=False))
        return table

    def _build_comparison_panel(self) -> QWidget:
        """AIC/BIC delta comparison, single-line where possible."""
        group = QGroupBox("Model Comparison (Bi-exp vs. Mono-exp)")
        style_group_box(group)
        vbox = QVBoxLayout(group)
        apply_compact_layout(group, spacing=2, margin=6)

        if self.result.delta_aic is None or self.result.delta_bic is None:
            note = QLabel(
                "Comparison unavailable: one or both fits did not converge."
            )
            note.setWordWrap(True)
            style_label(note, "muted")
            vbox.addWidget(note)
            return group

        d_aic = self.result.delta_aic
        d_bic = self.result.delta_bic

        metrics = QLabel(
            f"<b>\u0394AIC:</b> {d_aic:+.2f} "
            f"({'favors bi-exp' if d_aic < 0 else 'favors mono-exp'})   "
            f"\u2003"
            f"<b>\u0394BIC:</b> {d_bic:+.2f} "
            f"({'favors bi-exp' if d_bic < 0 else 'favors mono-exp'})"
        )
        metrics.setTextFormat(Qt.TextFormat.RichText)
        style_label(metrics, "normal")

        note = QLabel(
            "Negative \u0394 favors bi-exponential. "
            "Rule of thumb: |\u0394| < 2 negligible; 2\u201310 moderate; "
            "> 10 strong evidence."
        )
        note.setWordWrap(True)
        style_label(note, "muted")

        vbox.addWidget(metrics)
        vbox.addWidget(note)
        return group

    def _build_button_bar(self) -> QWidget:
        """Button bar with auto-detect checkbox, Reanalyze (if callback provided), and Close."""
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 4, 0, 0)
        h.setSpacing(8)

        # Auto-detect checkbox, left-aligned. Unchecked on first open per spec.
        self._auto_detect_cb = QCheckBox("Auto-detect fit region within Range 1")
        self._auto_detect_cb.setChecked(False)
        self._auto_detect_cb.setToolTip(
            "When checked, reanalysis narrows the fit window to a sub-region "
            "anchored at the peak (decaying trace) or trough (rising trace) "
            "inside Range 1. Click Reanalyze to apply."
        )
        style_checkbox(self._auto_detect_cb)
        h.addWidget(self._auto_detect_cb)

        h.addStretch()

        if self._reanalyze_callback is not None:
            reanalyze_btn = QPushButton("Reanalyze")
            style_button(reanalyze_btn, "primary")
            reanalyze_btn.setToolTip(
                "Re-run the fit using the current Range 1 bounds, displayed "
                "sweep, and the auto-detect setting above."
            )
            reanalyze_btn.clicked.connect(self._on_reanalyze_clicked)
            h.addWidget(reanalyze_btn)

        close_btn = QPushButton("Close")
        style_button(close_btn, "secondary")
        close_btn.clicked.connect(self.close)
        h.addWidget(close_btn)

        return container

    def _on_reanalyze_clicked(self):
        """Invoke the reanalyze callback and refresh dialog contents with new results."""
        if self._reanalyze_callback is None:
            return

        auto_detect = (
            self._auto_detect_cb.isChecked()
            if self._auto_detect_cb is not None else False
        )

        try:
            new_data = self._reanalyze_callback(auto_detect)
        except Exception as e:
            logger.error(f"Reanalyze callback raised: {e}", exc_info=True)
            return

        # Callback returns None when preconditions fail (it handles user warnings itself)
        if new_data is None:
            return

        result, file_path, sweep_index, current_units = new_data
        self.update_result(result, file_path, sweep_index, current_units)
        logger.info(
            f"Kinetics dialog refreshed (sweep {sweep_index}, "
            f"auto_detect={auto_detect})"
        )

    # --- Table sizing helpers ---

    @staticmethod
    def _shrink_table_rows(table: QTableWidget, row_height: int = 22) -> None:
        """Set a compact row height on all rows."""
        for r in range(table.rowCount()):
            table.setRowHeight(r, row_height)

    @staticmethod
    def _exact_table_height(table: QTableWidget, has_header: bool = True) -> int:
        """Compute the height needed to show all rows without a scrollbar."""
        total = 2  # frame fudge
        if has_header:
            total += table.horizontalHeader().height()
        for r in range(table.rowCount()):
            total += table.rowHeight(r)
        return total

    # --- Number formatting helpers ---

    @staticmethod
    def _fmt_value(v: float) -> str:
        """Format a floating-point value with sensible precision."""
        if v is None:
            return "\u2014"
        try:
            if not (v == v):  # NaN
                return "\u2014"
        except TypeError:
            return "\u2014"
        abs_v = abs(v)
        if abs_v == 0:
            return "0"
        if abs_v < 1e-3 or abs_v >= 1e5:
            return f"{v:.4e}"
        return f"{v:.4g}"

    @staticmethod
    def _fmt_stderr(v: float) -> str:
        if v is None:
            return "\u2014"
        try:
            if not (v == v):
                return "\u2014"
        except TypeError:
            return "\u2014"
        return KineticsDialog._fmt_value(v)

    @staticmethod
    def _fmt_r2(v: float) -> str:
        if v is None:
            return "\u2014"
        try:
            if not (v == v):
                return "\u2014"
        except TypeError:
            return "\u2014"
        return f"{v:.5f}"