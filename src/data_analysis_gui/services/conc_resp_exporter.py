"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Export functionality for concentration-response analysis results.

Handles CSV export with filename sanitization, conflict resolution,
and user-friendly dialogs. Complements ConcentrationResponseService
by managing the file I/O layer.
"""

import os
import re
import csv
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from PySide6.QtWidgets import QWidget, QMessageBox

from data_analysis_gui.services.conc_resp_service import ConcentrationResponseService
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ConcentrationResponseExporter:
    """
    Stateless exporter for concentration-response analysis results.
    
    Handles all file export operations including filename sanitization,
    conflict resolution, and CSV writing. UI-aware for dialog interactions
    but contains no dialog-specific state.
    """
    
    @staticmethod
    def export_results(
        results_dfs: Dict[str, pd.DataFrame],
        source_filepath: str,
        output_directory: str,
        parent_widget: Optional[QWidget] = None
    ) -> tuple[bool, str]:
        """
        Export analysis results to CSV files.
        
        Creates one CSV file per data trace, with sanitized filenames based
        on the source file. Handles file conflicts with user dialog.
        """
        if not results_dfs:
            return False, "No results to export"
        
        # Get base filename from source (for naming only)
        filename = os.path.basename(source_filepath)
        base_filename = os.path.splitext(filename)[0]
        
        exported_files = []
        
        try:
            for trace_name, df in results_dfs.items():
                # Sanitize trace name for filename
                safe_trace_name = ConcentrationResponseExporter.sanitize_trace_name(trace_name)
                
                output_filename = f"{base_filename}_{safe_trace_name}.csv"
                output_path = os.path.join(output_directory, output_filename)
                
                # Handle file conflicts
                resolved_path = ConcentrationResponseExporter._resolve_filename_conflict(
                    output_path, parent_widget
                )
                
                if resolved_path is None:
                    # User cancelled export
                    return False, "Export cancelled by user"
                
                output_filename = os.path.basename(resolved_path)
                
                # Pivot data for export
                export_df = ConcentrationResponseService.pivot_for_export(df)
                
                # Save to CSV
                export_df.to_csv(
                    resolved_path,
                    index=False,
                    float_format='%.4f',
                    encoding='utf-8',
                    quoting=csv.QUOTE_NONNUMERIC
                )
                
                exported_files.append(output_filename)
                logger.info(f"Exported: {output_filename}")
            
            # Show success message
            if exported_files:
                QMessageBox.information(
                    parent_widget,
                    "Export Successful",
                    f"{len(exported_files)} file(s) saved to:\n{output_directory}\n\n"
                    f"Files:\n- " + "\n- ".join(exported_files)
                )
                
                return True, f"Exported {len(exported_files)} file(s) to {Path(output_directory).name}"
        
        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            QMessageBox.critical(
                parent_widget,
                "Export Error",
                f"An unexpected error occurred during export:\n\n{str(e)}"
            )
            return False, f"Export failed: {str(e)}"
    
    @staticmethod
    def sanitize_trace_name(trace_name: str) -> str:
        """
        Sanitize trace name for use in filename.
        
        Removes or transforms parentheses content, replaces non-word characters,
        and cleans up spacing. Preserves voltage indicators like +/-.
        Also handles Greek characters like µ.
        
        Args:
            trace_name: Raw trace name from CSV header
            
        Returns:
            Sanitized filename-safe string
        
        Examples:
            >>> ConcentrationResponseExporter.sanitize_trace_name("Current (pA)")
            'Current'
            >>> ConcentrationResponseExporter.sanitize_trace_name("Voltage (+80 mV)")
            'Voltage_+80_mV'
            >>> ConcentrationResponseExporter.sanitize_trace_name("I-Ca (L-type)")
            'I-Ca'
            >>> ConcentrationResponseExporter.sanitize_trace_name("+100 mV")
            '+100_mV'
        """
        # Helper function for parentheses content
        def replacer(match):
            content = match.group(1)
            # Keep voltage indicators like +80 or -60
            if '+' in content or '-' in content:
                return '_' + content
            # Remove unit labels like (pA), (mV)
            return ''
        
        # Remove or transform parentheses content
        name_after_parens = re.sub(r'\s*\((.*?)\)', replacer, trace_name).strip()
        
        # Replace Greek µ with 'u' for filename safety
        name_after_parens = name_after_parens.replace('µ', 'u')
        
        # Replace non-word characters (except + and -)
        safe_trace_name = re.sub(r'[^\w+-]', '_', name_after_parens)
        
        # Remove double underscores
        safe_trace_name = safe_trace_name.replace('__', '_')
        
        # Remove leading/trailing underscores
        safe_trace_name = safe_trace_name.strip('_')
        
        return safe_trace_name
    
    @staticmethod
    def _resolve_filename_conflict(
        filepath: str,
        parent_widget: Optional[QWidget] = None
    ) -> Optional[str]:
        """
        Resolve filename conflict with user-friendly 3-button dialog.
        
        If file exists, presents options to:
        - Overwrite existing file
        - Save with new name (auto-incremented)
        - Cancel export
        
        Args:
            filepath: Proposed file path
            parent_widget: Parent widget for dialog (optional)
            
        Returns:
            Resolved file path, or None if user cancelled
        
        Example:
            >>> path = exporter._resolve_filename_conflict(
            ...     "/data/experiment_Current.csv",
            ...     parent=dialog
            ... )
            >>> # If file exists, shows dialog
            >>> # Returns "/data/experiment_Current_1.csv" if user chose rename
        """
        if not os.path.exists(filepath):
            return filepath
        
        filename = os.path.basename(filepath)
        
        # Show conflict dialog
        msg_box = QMessageBox(parent_widget)
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setWindowTitle("File Exists")
        msg_box.setText(f"The file '{filename}' already exists.")
        msg_box.setInformativeText("What would you like to do?")
        
        overwrite_btn = msg_box.addButton("Overwrite", QMessageBox.ButtonRole.AcceptRole)
        rename_btn = msg_box.addButton("Save with New Name", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = msg_box.addButton("Cancel Export", QMessageBox.ButtonRole.RejectRole)
        
        msg_box.setDefaultButton(rename_btn)
        msg_box.exec()
        
        clicked_button = msg_box.clickedButton()
        
        if clicked_button == overwrite_btn:
            # User chose to overwrite
            logger.info(f"Overwriting existing file: {filename}")
            return filepath
        
        elif clicked_button == rename_btn:
            # Find next available filename
            new_path = ConcentrationResponseExporter._get_next_available_filename(filepath)
            logger.info(f"Renamed to avoid conflict: {os.path.basename(new_path)}")
            return new_path
        
        else:  # Cancel
            logger.info("Export cancelled by user at file conflict dialog")
            return None
    
    @staticmethod
    def _get_next_available_filename(filepath: str) -> str:
        """
        Find next available filename by appending _1, _2, etc.
        
        Args:
            filepath: Initial file path
            
        Returns:
            Available file path that doesn't exist
        
        Example:
            >>> path = exporter._get_next_available_filename("/data/test.csv")
            >>> # If test.csv exists, returns "/data/test_1.csv"
            >>> # If test_1.csv exists, returns "/data/test_2.csv"
            >>> # etc.
        """
        if not os.path.exists(filepath):
            return filepath
        
        base, ext = os.path.splitext(filepath)
        i = 1
        
        while True:
            new_path = f"{base}_{i}{ext}"
            if not os.path.exists(new_path):
                return new_path
            i += 1