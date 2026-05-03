"""
PatchBatch Electrophysiology Data Analysis Tool - Kinetics Batch Export Dialog

Batch-export-focused dialog: runs mono- and bi-exponential kinetics fits on
every sweep in the currently loaded file using Range 1, then writes selected
per-sweep parameters to a CSV file or copies them as TSV to the clipboard.
Output format mirrors PatchBatch's standard analysis exports: column 1 is the
sweep timestamp in seconds (from dataset metadata), followed by one column
per selected measure.

Available measures:
    - tau_mono           Mono-exponential time constant (ms)
    - tau1_biexp_fast    Bi-exponential FAST time constant (ms)
    - tau2_biexp_slow    Bi-exponential SLOW time constant (ms)
    - fast_fraction      |A_fast| / (|A_fast| + |A_slow|) * 100  (%)
    - slow_fraction      |A_slow| / (|A_fast| + |A_slow|) * 100  (%)

Note on bi-exponential ordering: scipy curve_fit does NOT enforce tau1 < tau2,
so the "fast" component is defined empirically per-fit as whichever component
has the smaller tau. tau1/A1 and tau2/A2 from the raw fit result are sorted by
tau magnitude before the export columns are filled, so tau1 in the output is
ALWAYS the fast component by definition (and tau2 is the slow). Amplitudes
are sorted in lockstep so the fast/slow fractions correspond correctly.

Failed fits produce empty cells for that sweep's affected columns.

Performance: per-sweep KineticsResults are cached after the first batch run.
Subsequent Export/Copy actions reuse the cache as long as the auto-detect
setting is unchanged; toggling auto-detect invalidates it. This makes a
second action (e.g. Export then Copy) essentially instant after the initial
1-2 minute fit on large files.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from pathlib import Path
import csv
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton,
    QCheckBox, QApplication, QMessageBox, QWidget, QFileDialog, QProgressBar,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor

from data_analysis_gui.config.themes import (
    apply_modern_theme, style_label, style_group_box, style_button,
    apply_compact_layout, style_checkbox,
)
from data_analysis_gui.config.logging import get_logger
from data_analysis_gui.core.data_extractor import DataExtractor
from data_analysis_gui.services.kinetics_service import (
    fit_kinetics, KineticsResult,
)

logger = get_logger(__name__)


# (internal_key, checkbox_label, csv_header). Order is preserved in the UI
# and the output CSV.
_MEASURES: List[Tuple[str, str, str]] = [
    ("tau_mono",      "Mono-exp \u03C4",                "tau_mono (ms)"),
    ("tau1_biexp",    "Bi-exp \u03C4\u2081 (fast)",     "tau1_biexp_fast (ms)"),
    ("tau2_biexp",    "Bi-exp \u03C4\u2082 (slow)",     "tau2_biexp_slow (ms)"),
    ("fast_fraction", "Fast component fraction (%)",    "fast_fraction (%)"),
    ("slow_fraction", "Slow component fraction (%)",    "slow_fraction (%)"),
]


class KineticsBatchExportDialog(QDialog):
    """
    Pick which kinetics measures to export, optionally enable peak/trough
    auto-detection, then either write a CSV with one row per sweep or copy
    the same data to the clipboard as TSV.
    """

    def __init__(
        self,
        dataset,
        file_path: Optional[str],
        range_start_ms: float,
        range_end_ms: float,
        current_units: str = "pA",
        file_dialog_service=None,
        parent=None,
    ):
        super().__init__(parent)

        self.dataset = dataset
        self.file_path = file_path
        self.range_start_ms = range_start_ms
        self.range_end_ms = range_end_ms
        self.current_units = current_units
        self._file_dialog_service = file_dialog_service

        self._checkboxes: Dict[str, QCheckBox] = {}
        self._auto_detect_cb: Optional[QCheckBox] = None
        self._export_btn: Optional[QPushButton] = None
        self._copy_btn: Optional[QPushButton] = None
        self._progress_widget: Optional[QWidget] = None
        self._progress_bar: Optional[QProgressBar] = None
        self._progress_label: Optional[QLabel] = None

        # Cache of per-sweep fit results, keyed by the auto_detect flag used
        # when they were computed. None = no cache yet.
        self._results_cache: Optional[Dict[str, Optional[KineticsResult]]] = None
        self._cache_auto_detect: Optional[bool] = None

        self.setWindowTitle("Kinetics Batch Export")
        self.setWindowModality(Qt.WindowModality.NonModal)

        apply_modern_theme(self)
        self._build_ui()
        self.setMinimumWidth(440)

    # --- UI construction ---

    def _build_ui(self):
        outer = QVBoxLayout(self)
        apply_compact_layout(self, spacing=6, margin=8)

        outer.addWidget(self._build_header())
        outer.addWidget(self._build_measures_group())
        outer.addWidget(self._build_options_group())
        outer.addStretch()
        outer.addWidget(self._build_progress_widget())
        outer.addWidget(self._build_button_bar())

    def _build_header(self) -> QWidget:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 2, 4, 2)
        v.setSpacing(2)

        file_name = Path(self.file_path).name if self.file_path else "Unknown"
        n_sweeps = self.dataset.sweep_count() if self.dataset else 0

        l1 = QLabel(f"<b>File:</b> {file_name}   <b>Sweeps:</b> {n_sweeps}")
        l1.setTextFormat(Qt.TextFormat.RichText)
        style_label(l1, "normal")
        v.addWidget(l1)

        l2 = QLabel(
            f"<b>Range 1:</b> {self.range_start_ms:.2f} \u2013 "
            f"{self.range_end_ms:.2f} ms"
        )
        l2.setTextFormat(Qt.TextFormat.RichText)
        style_label(l2, "muted")
        v.addWidget(l2)

        return container

    def _build_measures_group(self) -> QGroupBox:
        group = QGroupBox("Measures to export")
        style_group_box(group)
        v = QVBoxLayout(group)
        apply_compact_layout(group, spacing=4, margin=8)

        for key, label_text, _csv_header in _MEASURES:
            cb = QCheckBox(label_text)
            cb.setChecked(True)  # default: all selected
            style_checkbox(cb)
            v.addWidget(cb)
            self._checkboxes[key] = cb

        return group

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Fit options")
        style_group_box(group)
        v = QVBoxLayout(group)
        apply_compact_layout(group, spacing=4, margin=8)

        self._auto_detect_cb = QCheckBox(
            "Auto-detect fit region within Range 1"
        )
        self._auto_detect_cb.setChecked(False)
        self._auto_detect_cb.setToolTip(
            "When checked, each sweep's fit window is narrowed to a sub-region "
            "anchored at the peak (decaying) or trough (rising) inside Range 1. "
            "Detected independently for every sweep."
        )
        style_checkbox(self._auto_detect_cb)
        # Toggling auto-detect invalidates the cached fit results.
        self._auto_detect_cb.stateChanged.connect(self._invalidate_cache)
        v.addWidget(self._auto_detect_cb)

        return group

    def _build_progress_widget(self) -> QWidget:
        """Status label + progress bar, hidden until a batch fit starts."""
        self._progress_widget = QWidget()
        v = QVBoxLayout(self._progress_widget)
        v.setContentsMargins(4, 4, 4, 0)
        v.setSpacing(2)

        self._progress_label = QLabel("")
        style_label(self._progress_label, "muted")
        v.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        v.addWidget(self._progress_bar)

        self._progress_widget.setVisible(False)
        return self._progress_widget

    def _build_button_bar(self) -> QWidget:
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 4, 0, 0)
        h.setSpacing(8)
        h.addStretch()

        self._copy_btn = QPushButton("Copy Data")
        style_button(self._copy_btn, "secondary")
        self._copy_btn.setToolTip(
            "Copy the same data as tab-separated text to the clipboard "
            "(paste-ready for Excel, Prism, etc.)."
        )
        self._copy_btn.clicked.connect(self._on_copy)
        h.addWidget(self._copy_btn)

        self._export_btn = QPushButton("Export CSV")
        style_button(self._export_btn, "primary")
        self._export_btn.clicked.connect(self._on_export)
        h.addWidget(self._export_btn)

        close_btn = QPushButton("Close")
        style_button(close_btn, "secondary")
        close_btn.clicked.connect(self.close)
        h.addWidget(close_btn)

        return container

    # --- Cache management ---

    def _invalidate_cache(self):
        """Drop cached fit results. Called when auto-detect setting changes."""
        if self._results_cache is not None:
            logger.debug("Kinetics batch cache invalidated (auto-detect changed)")
        self._results_cache = None
        self._cache_auto_detect = None

    def _ensure_results(
        self, auto_detect: bool
    ) -> Optional[Dict[str, Optional[KineticsResult]]]:
        """
        Return cached results if valid for this auto_detect setting; otherwise
        run the batch fit (with progress UI) and cache before returning.
        Returns None on unexpected failure.
        """
        if (self._results_cache is not None
                and self._cache_auto_detect == auto_detect):
            logger.debug("Kinetics batch cache hit")
            return self._results_cache

        results = self._compute_fit_results(auto_detect)
        if results is None:
            return None

        self._results_cache = results
        self._cache_auto_detect = auto_detect
        return results

    # --- Batch fit (with progress UI) ---

    def _compute_fit_results(
        self, auto_detect: bool
    ) -> Optional[Dict[str, Optional[KineticsResult]]]:
        """
        Run fit_kinetics on every sweep with progress display.

        Returns dict {sweep_name: KineticsResult or None}, in original sweep
        order. None means an exception was raised during extract/fit for that
        sweep -- distinct from a successful KineticsResult whose individual
        mono/biexp fits may still have failed internally.
        """
        sweep_names = sorted(
            self.dataset.sweeps(),
            key=lambda x: int(x) if x.isdigit() else 0
        )
        total = len(sweep_names)
        if total == 0:
            QMessageBox.warning(
                self, "No Sweeps",
                "The loaded dataset contains no sweeps."
            )
            return None

        extractor = DataExtractor()
        results: Dict[str, Optional[KineticsResult]] = {}

        self._begin_busy_state(total)
        try:
            for i, sweep in enumerate(sweep_names):
                self._update_progress(i, total, sweep)

                try:
                    sweep_data = extractor.extract_sweep_data(self.dataset, sweep)
                    result = fit_kinetics(
                        sweep_data["time_ms"],
                        sweep_data["current"],
                        self.range_start_ms,
                        self.range_end_ms,
                        auto_detect_region=auto_detect,
                    )
                except Exception as e:
                    logger.warning(f"Kinetics fit raised for sweep {sweep}: {e}")
                    result = None

                results[sweep] = result
                QApplication.processEvents()  # keep UI responsive

            self._update_progress(total, total, None)
        finally:
            self._end_busy_state()

        return results

    def _begin_busy_state(self, total: int):
        """Show progress UI and disable interactive controls during fit."""
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(0)
        self._progress_label.setText(f"Fitting sweep 1 of {total}...")
        self._progress_widget.setVisible(True)

        self._copy_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._auto_detect_cb.setEnabled(False)

        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        QApplication.processEvents()

    def _end_busy_state(self):
        """Hide progress UI and re-enable interactive controls."""
        QApplication.restoreOverrideCursor()

        self._copy_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._auto_detect_cb.setEnabled(True)

        self._progress_widget.setVisible(False)

    def _update_progress(self, completed: int, total: int, current_sweep):
        """Update the progress bar and status label."""
        self._progress_bar.setValue(completed)
        if completed >= total:
            self._progress_label.setText(f"Done -- {total} sweeps fit.")
        else:
            sweep_str = f" (sweep {current_sweep})" if current_sweep else ""
            self._progress_label.setText(
                f"Fitting sweep {completed + 1} of {total}{sweep_str}..."
            )

    # --- Action handlers ---

    def _on_export(self):
        """Save results to a CSV file."""
        selected = self._get_selected_measures()
        if not selected:
            QMessageBox.warning(
                self, "No Measures Selected",
                "Select at least one measure to export."
            )
            return

        save_path = self._prompt_save_path()
        if not save_path:
            return

        auto_detect = self._auto_detect_cb.isChecked()
        results = self._ensure_results(auto_detect)
        if results is None:
            return

        sweep_times = self.dataset.metadata.get("sweep_times", {})
        rows, empty_rows = self._build_rows(results, selected, sweep_times)

        try:
            self._write_csv(save_path, rows, selected)
        except Exception as e:
            logger.error(f"Kinetics CSV write failed: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Export Failed",
                f"Failed to write CSV:\n{str(e)}"
            )
            return

        msg = f"Exported {len(rows)} sweeps to {Path(save_path).name}."
        if empty_rows:
            msg += (
                f"\n\n{empty_rows} sweep(s) had no usable fit results "
                "(empty cells in the CSV)."
            )
        QMessageBox.information(self, "Export Complete", msg)
        logger.info(
            f"Kinetics batch export: wrote {len(rows)} rows "
            f"({empty_rows} fully empty), file={save_path}"
        )

    def _on_copy(self):
        """Copy results as TSV (tab-separated) to the clipboard."""
        selected = self._get_selected_measures()
        if not selected:
            QMessageBox.warning(
                self, "No Measures Selected",
                "Select at least one measure to copy."
            )
            return

        auto_detect = self._auto_detect_cb.isChecked()
        results = self._ensure_results(auto_detect)
        if results is None:
            return

        sweep_times = self.dataset.metadata.get("sweep_times", {})
        rows, empty_rows = self._build_rows(results, selected, sweep_times)

        text = self._build_tsv(rows, selected)
        QApplication.clipboard().setText(text)

        msg = f"Copied {len(rows)} sweeps to clipboard."
        if empty_rows:
            msg += f"\n\n{empty_rows} sweep(s) had no usable fit results (empty cells)."
        QMessageBox.information(self, "Copied", msg)
        logger.info(
            f"Kinetics batch copy: copied {len(rows)} rows "
            f"({empty_rows} fully empty)"
        )

    # --- Helpers ---

    def _get_selected_measures(self) -> List[str]:
        return [k for k, _, _ in _MEASURES if self._checkboxes[k].isChecked()]

    def _prompt_save_path(self) -> Optional[str]:
        """Get save path via FileDialogService if available; otherwise QFileDialog."""
        suggested = (
            Path(self.file_path).stem + "_kinetics.csv"
            if self.file_path else "kinetics.csv"
        )
        fallback = (
            str(Path(self.file_path).parent) if self.file_path else None
        )

        if self._file_dialog_service is not None:
            return self._file_dialog_service.get_export_path(
                parent=self,
                suggested_name=suggested,
                default_directory=fallback,
                file_types="CSV files (*.csv);;All files (*.*)",
                dialog_type="export",
            )

        # Fallback when no service is provided
        initial = str(Path(fallback) / suggested) if fallback else suggested
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Kinetics CSV", initial,
            "CSV files (*.csv);;All files (*.*)"
        )
        return path or None

    def _build_rows(
        self,
        results: Dict[str, Optional[KineticsResult]],
        selected: List[str],
        sweep_times: dict,
    ) -> Tuple[List[dict], int]:
        """
        Convert per-sweep KineticsResults into row dicts ready for output.

        Returns (rows, empty_rows_count). empty_rows_count is the number of
        sweeps for which every selected measure ended up None.
        """
        sweep_names = sorted(
            results.keys(),
            key=lambda x: int(x) if x.isdigit() else 0
        )

        rows: List[dict] = []
        empty_rows = 0
        for sweep in sweep_names:
            result = results[sweep]
            row = {"_time_s": float(sweep_times.get(sweep, 0.0))}
            all_empty = True
            for key in selected:
                v = self._extract_measure(result, key)
                row[key] = v
                if v is not None:
                    all_empty = False
            if all_empty:
                empty_rows += 1
            rows.append(row)

        return rows, empty_rows

    @staticmethod
    def _extract_measure(
        result: Optional[KineticsResult], key: str
    ) -> Optional[float]:
        """
        Pull a single export value from a KineticsResult.

        Convention: tau1 in the export = FAST component (smaller tau),
        tau2 = SLOW component. Amplitudes are sorted in lockstep so the
        fast_fraction / slow_fraction values correspond to the correct
        component regardless of the raw scipy parameter order.
        """
        if result is None or not result.success:
            return None

        if key == "tau_mono":
            if result.mono and result.mono.success:
                return result.mono.params.get("tau")
            return None

        # All remaining keys depend on the bi-exponential fit
        if not result.biexp or not result.biexp.success:
            return None
        p = result.biexp.params
        tau1, tau2 = p["tau1"], p["tau2"]
        A1, A2 = p["A1"], p["A2"]

        # Define fast = component with smaller tau (scipy curve_fit does not
        # enforce ordering on tau1/tau2). Sort A and tau together.
        if tau1 <= tau2:
            tau_fast, tau_slow = tau1, tau2
            A_fast, A_slow = A1, A2
        else:
            tau_fast, tau_slow = tau2, tau1
            A_fast, A_slow = A2, A1

        if key == "tau1_biexp":
            return tau_fast
        if key == "tau2_biexp":
            return tau_slow

        denom = abs(A_fast) + abs(A_slow)
        # Guard against zero or NaN denominator
        if denom == 0 or denom != denom:
            return None
        if key == "fast_fraction":
            return abs(A_fast) / denom * 100.0
        if key == "slow_fraction":
            return abs(A_slow) / denom * 100.0

        return None

    @staticmethod
    def _build_table(rows: List[dict], selected: List[str]) -> List[List[str]]:
        """Build a 2D list of strings (header row first, then data rows)."""
        csv_headers_by_key = {key: csv_h for key, _, csv_h in _MEASURES}
        headers = ["Time (s)"] + [csv_headers_by_key[k] for k in selected]

        fmt = KineticsBatchExportDialog._fmt
        table: List[List[str]] = [headers]
        for row in rows:
            cells = [fmt(row["_time_s"])]
            for key in selected:
                cells.append(fmt(row.get(key)))
            table.append(cells)
        return table

    @staticmethod
    def _write_csv(save_path: str, rows: List[dict], selected: List[str]):
        """Write the per-sweep rows to a CSV at save_path."""
        table = KineticsBatchExportDialog._build_table(rows, selected)
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(table)

    @staticmethod
    def _build_tsv(rows: List[dict], selected: List[str]) -> str:
        """Build a tab-separated string for clipboard paste."""
        table = KineticsBatchExportDialog._build_table(rows, selected)
        return "\n".join("\t".join(r) for r in table)

    @staticmethod
    def _fmt(v) -> str:
        """Format a value for output. None / NaN become empty strings."""
        if v is None:
            return ""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return ""
        if f != f:  # NaN
            return ""
        return f"{f:.6f}"