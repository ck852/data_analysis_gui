# src/data_analysis_gui/services/export_service.py
"""
Centralized export service for handling all export operations across dialogs.
This service provides a unified interface for exporting plots and data,
eliminating code duplication across dialog classes.
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget

# Import the core exporter module
from data_analysis_gui.core import exporter

@dataclass
class ExportResult:
    """Result of an export operation"""
    success: bool
    file_path: Optional[str] = None
    error_message: Optional[str] = None

class ExportService:
    """Centralized service for handling all export operations"""

    @staticmethod
    def export_plot_image(
        figure,
        parent: QWidget,
        default_path: str = "",
        title: str = "Export Plot",
        dpi: int = 300
    ) -> ExportResult:
        """
        Export a matplotlib figure as an image file.
        
        Args:
            figure: Matplotlib figure to export
            parent: Parent widget for dialogs
            default_path: Default file path/name
            title: Title for the file dialog
            dpi: Resolution for the exported image
            
        Returns:
            ExportResult indicating success/failure
        """
        file_path, _ = QFileDialog.getSaveFileName(
            parent, title, default_path,
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )

        if not file_path:
            return ExportResult(success=False)

        try:
            figure.savefig(file_path, dpi=dpi, bbox_inches='tight')
            QMessageBox.information(
                parent, "Export Successful",
                f"Plot saved to {file_path}"
            )
            return ExportResult(success=True, file_path=file_path)
        except Exception as e:
            error_msg = f"Failed to save plot: {str(e)}"
            QMessageBox.critical(parent, "Export Failed", error_msg)
            return ExportResult(success=False, error_message=error_msg)

    @staticmethod
    def export_data_to_csv(
        data: np.ndarray,
        headers: List[str],
        parent: QWidget,
        default_path: str = "",
        title: str = "Export Data",
        format_spec: str = '%.6f'
    ) -> ExportResult:
        """
        Export data to a CSV file using the core exporter.
        
        Args:
            data: NumPy array of data to export
            headers: List of column headers
            parent: Parent widget for dialogs
            default_path: Default file path/name
            title: Title for the file dialog
            format_spec: Format specification for numeric data
            
        Returns:
            ExportResult indicating success/failure
        """
        file_path, _ = QFileDialog.getSaveFileName(
            parent, title, default_path, "CSV files (*.csv)"
        )

        if not file_path:
            return ExportResult(success=False)

        try:
            table_data = {
                'headers': headers,
                'data': data,
                'format_spec': format_spec
            }
            
            # Use the core exporter
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            destination_folder = os.path.dirname(file_path)
            
            # The core exporter handles filename collisions, but we pass the exact path
            # For simplicity here, we'll call the internal write function directly.
            # A better approach would be to adapt write_single_table if needed.
            exporter._write_csv(file_path, table_data)
            
            QMessageBox.information(
                parent, "Export Successful",
                f"Data successfully saved to {file_path}"
            )
            return ExportResult(success=True, file_path=file_path)
        except Exception as e:
            error_msg = f"An error occurred while saving the file:\n{e}"
            QMessageBox.critical(parent, "Export Error", error_msg)
            return ExportResult(success=False, error_message=error_msg)

    @staticmethod
    def export_dict_to_csv(
        data_dict: Dict[str, Any],
        parent: QWidget,
        default_path: str = "",
        title: str = "Export Data"
    ) -> ExportResult:
        """
        Export a dictionary with consistent structure to CSV.
        Expects dict with 'headers' and 'data' keys.
        
        Args:
            data_dict: Dictionary containing 'headers' and 'data'
            parent: Parent widget for dialogs
            default_path: Default file path/name
            title: Title for the file dialog
            
        Returns:
            ExportResult indicating success/failure
        """
        if 'headers' not in data_dict or 'data' not in data_dict:
            error_msg = "Data dictionary must contain 'headers' and 'data' keys"
            QMessageBox.critical(parent, "Export Error", error_msg)
            return ExportResult(success=False, error_message=error_msg)
        
        return ExportService.export_data_to_csv(
            data=data_dict['data'],
            headers=data_dict['headers'],
            parent=parent,
            default_path=default_path,
            title=title,
            format_spec=data_dict.get('format_spec', '%.6f')
        )

    @staticmethod
    def select_export_folder(
        parent: QWidget,
        title: str = "Select Output Folder",
        default_folder: str = ""
    ) -> Optional[str]:
        """
        Open a folder selection dialog.
        
        Args:
            parent: Parent widget for dialog
            title: Title for the folder dialog
            default_folder: Default folder path
            
        Returns:
            Selected folder path or None if cancelled
        """
        folder = QFileDialog.getExistingDirectory(
            parent, title, default_folder
        )
        return folder if folder else None

    @staticmethod
    def ensure_folder_exists(folder_path: str) -> Tuple[bool, Optional[str]]:
        """
        Ensure a folder exists, creating it if necessary.
        
        Args:
            folder_path: Path to the folder
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            os.makedirs(folder_path, exist_ok=True)
            return True, None
        except OSError as e:
            return False, f"Could not create folder '{folder_path}': {e}"

    @staticmethod
    def export_multiple_files(
        files_data: List[Dict[str, Any]],
        output_folder: str,
        parent: QWidget,
        file_prefix: str = "",
        show_summary: bool = True
    ) -> List[ExportResult]:
        """
        Export multiple files to a folder using the core exporter.
        
        Args:
            files_data: List of dicts, each containing 'filename', 'data', 'headers'
            output_folder: Folder to save files to
            parent: Parent widget for dialogs
            file_prefix: Optional prefix for all filenames
            show_summary: Whether to show a summary message
            
        Returns:
            List of ExportResult objects
        """
        results = []

        # Ensure folder exists
        success, error = ExportService.ensure_folder_exists(output_folder)
        if not success:
            QMessageBox.critical(parent, "Export Error", error)
            return [ExportResult(success=False, error_message=error)]

        # Export each file using the core exporter
        for file_info in files_data:
            filename = file_info.get('filename', 'data.csv')
            if file_prefix:
                filename = f"{file_prefix}_{filename}"

            table_data = {
                'headers': file_info.get('headers', []),
                'data': file_info.get('data', np.array([[]])),
                'format_spec': file_info.get('format_spec', '%.6f')
            }

            outcome = exporter.write_single_table(
                table=table_data,
                base_name=os.path.splitext(filename)[0],
                destination_folder=output_folder
            )

            results.append(ExportResult(
                success=outcome.success,
                file_path=outcome.path,
                error_message=outcome.error_message
            ))

        # Show summary if requested
        if show_summary and results:
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            message = f"Exported {successful} files successfully."
            if failed > 0:
                message += f"\n{failed} files failed to export."
                
            QMessageBox.information(parent, "Export Complete", message)
        
        return results

    @staticmethod
    def get_suggested_filename(
        base_name: str,
        suffix: str = "",
        extension: str = "csv",
        destination_folder: str = ""
    ) -> str:
        """
        Generate a suggested filename with optional suffix and folder.
        
        Args:
            base_name: Base name for the file
            suffix: Optional suffix to add
            extension: File extension (without dot)
            destination_folder: Optional folder path
            
        Returns:
            Complete suggested file path
        """
        if suffix:
            filename = f"{base_name}_{suffix}.{extension}"
        else:
            filename = f"{base_name}.{extension}"
        
        if destination_folder:
            return os.path.join(destination_folder, filename)
        return filename