"""
PatchBatch Electrophysiology Data Analysis Tool

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Service for multi-file concentration-response dataset building.

Manages accumulation of analysis results across multiple files with locked
concentrations. Creates separate datasets for each data trace (by column position).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class DatasetBuilderService:
    """
    Stateful service for building multi-file concentration-response datasets.
    
    Maintains state across multiple file analyses:
    - Locks concentration names after first analysis
    - Accumulates data by trace (column position)
    - Validates subsequent files match locked concentrations
    - Exports formatted datasets
    """
    
    def __init__(self):
        """Initialize an empty dataset builder session."""
        self.reset_session()
    
    def reset_session(self):
        """Reset to a new dataset building session."""
        # Concentration locking
        self.locked_range_names: List[str] = []
        self.is_locked: bool = False
        
        # Data accumulation - separate datasets per trace
        # Structure: {trace_name: {concentration_value: [file1, file2, ...]}}
        self.datasets: Dict[str, Dict[str, List[float]]] = {}
        
        # Track file contributions
        self.file_names: List[str] = []
        
        # Track trace names from first file
        self.trace_names: List[str] = []
        
        logger.info("Started new dataset building session")
    
    def get_file_count(self) -> int:
        """Get the number of files added to the dataset."""
        return len(self.file_names)
    
    def is_first_file(self) -> bool:
        """Check if this is the first file (concentrations not locked yet)."""
        return not self.is_locked
    
    def lock_concentrations(self, concentrations: List[float], trace_names: List[str]):
        """
        Lock concentration values from the first file analysis.
        
        Args:
            concentrations: List of concentration values in µM (e.g., [10.0, 100.0, 1000.0])
            trace_names: List of data trace names (e.g., ["Current (pA) (+80mV)"])
        """
        if self.is_locked:
            logger.warning("Attempted to lock concentrations when already locked")
            return
        
        self.locked_range_names = [str(c) for c in concentrations]
        self.trace_names = trace_names.copy()
        self.is_locked = True
        
        # Initialize datasets for each trace
        for trace_name in trace_names:
            self.datasets[trace_name] = {
                str(conc): []
                for conc in concentrations
            }
        
        logger.info(
            f"Locked {len(concentrations)} concentrations across {len(trace_names)} trace(s)"
        )
    
    def validate_ranges(self, concentrations: List[float]) -> Tuple[bool, str]:
        """
        Validate that concentration values match locked concentrations.
        
        Args:
            concentrations: List of concentration values from current file
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if not self.is_locked:
            return True, ""
        
        # Convert to strings for comparison
        conc_strs = [str(c) for c in concentrations]
        
        if len(conc_strs) != len(self.locked_range_names):
            return False, (
                f"Range count mismatch: expected {len(self.locked_range_names)} ranges, "
                f"got {len(conc_strs)}"
            )
        
        # Check exact concentration matching
        if set(conc_strs) != set(self.locked_range_names):
            missing = set(self.locked_range_names) - set(conc_strs)
            extra = set(conc_strs) - set(self.locked_range_names)
            
            msg = "Concentrations do not match locked values:\n"
            if missing:
                msg += f"  Missing: {', '.join(missing)} µM\n"
            if extra:
                msg += f"  Extra: {', '.join(extra)} µM"
            
            return False, msg
        
        return True, ""
    
    def check_duplicate_filename(self, filename: str) -> bool:
        """
        Check if filename already exists in dataset.
        
        Args:
            filename: Name of file to check
            
        Returns:
            True if filename is a duplicate
        """
        return filename in self.file_names
    
    def add_file_results(
        self,
        filename: str,
        results_dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[bool, str]:
        """
        Add analysis results from a file to the dataset.
        
        Args:
            filename: Name of the source file
            results_dfs: Dictionary mapping trace names to result DataFrames
                        (from ConcentrationResponseService.run_analysis)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Track filename
            self.file_names.append(filename)
            
            # Process each trace's results
            for trace_idx, (trace_name, results_df) in enumerate(results_dfs.items()):
                # Determine which dataset this belongs to (by column position)
                if trace_idx < len(self.trace_names):
                    dataset_key = self.trace_names[trace_idx]
                else:
                    # Shouldn't happen if validation passed, but handle gracefully
                    logger.warning(f"Unexpected trace index {trace_idx}")
                    continue
                
                # Add data to appropriate dataset
                for _, row in results_df.iterrows():
                    concentration = row['Concentration (µM)']
                    corrected_value = row['Corrected Value']
                    
                    # Convert concentration to string for dict key
                    conc_key = str(concentration)
                    
                    if conc_key in self.datasets[dataset_key]:
                        self.datasets[dataset_key][conc_key].append(corrected_value)
                    else:
                        logger.warning(
                            f"Unexpected concentration '{conc_key}' not in locked ranges"
                        )
            
            logger.info(f"Added results from {filename} to dataset ({self.get_file_count()} files total)")
            return True, f"Added {filename} to dataset"
        
        except Exception as e:
            logger.error(f"Error adding file results: {e}", exc_info=True)
            return False, f"Failed to add results: {str(e)}"
    
    def get_dataset_preview(self, trace_name: str, max_files: int = 5) -> pd.DataFrame:
        """
        Get a preview of the accumulated dataset for a specific trace.
        
        Args:
            trace_name: Name of trace to preview
            max_files: Maximum number of file columns to show
            
        Returns:
            DataFrame for display (Concentration column + file columns)
        """
        if trace_name not in self.datasets:
            return pd.DataFrame()
        
        dataset = self.datasets[trace_name]
        
        # Sort concentrations numerically
        try:
            sorted_concentrations = sorted(dataset.keys(), key=float)
        except ValueError:
            sorted_concentrations = sorted(dataset.keys())
        
        # Build preview DataFrame
        preview_data = {
            'Concentration (uM)': [float(c) for c in sorted_concentrations]
        }
        
        # Add file columns (limited by max_files)
        for file_idx, filename in enumerate(self.file_names[:max_files]):
            preview_data[f'File {file_idx + 1}'] = [
                dataset[conc][file_idx] if file_idx < len(dataset[conc]) else np.nan
                for conc in sorted_concentrations
            ]
        
        if len(self.file_names) > max_files:
            preview_data['...'] = ['...' for _ in dataset.keys()]
        
        return pd.DataFrame(preview_data)
    
    def export_dataset(
        self, 
        base_output_path: str, 
        base_filename: str = "dataset"
    ) -> Tuple[bool, List[str]]:
        """
        Export all accumulated datasets to CSV files.
        
        Creates one CSV per trace, with filename suffix based on trace name.
        Format: "Concentration (uM)" column + one column per file (no headers).
        
        Args:
            base_output_path: Output directory path (e.g., "/path/to/output_dir")
            base_filename: Base filename without extension (e.g., "dataset")
            
        Returns:
            Tuple of (success: bool, list of exported file paths)
        """
        if not self.datasets:
            return False, []
        
        from pathlib import Path
        
        output_dir = Path(base_output_path)
        
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        try:
            for trace_idx, (trace_name, dataset) in enumerate(self.datasets.items()):
                # Generate filename suffix based on trace
                trace_suffix = f"_trace{trace_idx + 1}"
                output_path = output_dir / f"{base_filename}{trace_suffix}.csv"
                
                # Build export DataFrame
                export_df = self._build_export_dataframe(dataset)
                
                # Write CSV
                export_df.to_csv(
                    output_path,
                    index=False,
                    header=True,  # Only first column has header
                    float_format='%.4f',
                    encoding='utf-8'
                )
                
                exported_files.append(str(output_path))
                logger.info(f"Exported dataset to {output_path}")
            
            return True, exported_files
        
        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            return False, []
    
    def _build_export_dataframe(self, dataset: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Build export DataFrame with proper formatting.
        
        Format:
        - First column: "Concentration (uM)" with numeric values
        - Remaining columns: No headers, just data values
        
        Args:
            dataset: Dictionary mapping concentration strings to lists of results
            
        Returns:
            DataFrame ready for CSV export
        """
        # Sort concentrations numerically
        try:
            sorted_concentrations = sorted(dataset.keys(), key=float)
        except ValueError:
            # Fallback to string sorting if conversion fails
            sorted_concentrations = sorted(dataset.keys())
        
        # Build data dictionary
        data = {
            'Concentration (uM)': [float(c) for c in sorted_concentrations]
        }
        
        # Add file columns with empty string headers
        num_files = len(self.file_names)
        for file_idx in range(num_files):
            # Create unique empty header using spaces
            if file_idx == 0:
                col_name = ''
            else:
                col_name = ' ' * file_idx
            
            data[col_name] = [
                dataset[conc][file_idx] if file_idx < len(dataset[conc]) else np.nan
                for conc in sorted_concentrations
            ]
        
        return pd.DataFrame(data)