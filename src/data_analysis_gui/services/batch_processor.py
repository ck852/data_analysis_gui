"""
Simplified batch processing with direct method calls.

This module provides straightforward batch analysis functionality without
complex dependency injection patterns.

Author: Data Analysis GUI Contributors
License: MIT
"""

import time
import re
from pathlib import Path
from typing import List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.models import (
    FileAnalysisResult, BatchAnalysisResult, BatchExportResult
)
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.config.logging import get_logger

# Direct imports of managers
from data_analysis_gui.services.data_manager import DataManager
from data_analysis_gui.services.analysis_manager import AnalysisManager

logger = get_logger(__name__)


class BatchProcessor:
    """
    Processes multiple files with the same analysis parameters.
    
    Simple, direct implementation without complex dependency patterns.
    Scientists can easily understand and modify the batch processing logic.
    """
    
    def __init__(self, channel_definitions: ChannelDefinitions):
        """
        Initialize with channel definitions.
        
        Args:
            channel_definitions: Channel configuration
        """
        self.channel_definitions = channel_definitions
        self.data_manager = DataManager()  # Direct instantiation
        
        # Progress callbacks (optional)
        self.on_progress: Optional[Callable] = None
        self.on_file_complete: Optional[Callable] = None
        
        logger.info("BatchProcessor initialized")
    
    def process_files(self,
                     file_paths: List[str],
                     params: AnalysisParameters,
                     parallel: bool = False,
                     max_workers: int = 4) -> BatchAnalysisResult:
        """
        Process multiple files with the same parameters.
        
        Args:
            file_paths: List of files to process
            params: Analysis parameters
            parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
            
        Returns:
            BatchAnalysisResult with all results
        """
        if not file_paths:
            raise ValueError("No files provided")
        
        logger.info(f"Processing {len(file_paths)} files")
        start_time = time.time()
        
        successful_results = []
        failed_results = []
        
        if parallel and len(file_paths) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_single_file, path, params): path
                    for path in file_paths
                }
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    path = futures[future]
                    
                    if self.on_progress:
                        self.on_progress(completed, len(file_paths), Path(path).name)
                    
                    try:
                        result = future.result()
                        if result.success:
                            successful_results.append(result)
                        else:
                            failed_results.append(result)
                        
                        if self.on_file_complete:
                            self.on_file_complete(result)
                            
                    except Exception as e:
                        logger.error(f"Failed to process {path}: {e}")
                        failed_results.append(
                            FileAnalysisResult(
                                file_path=path,
                                base_name=Path(path).stem,
                                success=False,
                                error_message=str(e)
                            )
                        )
        else:
            # Sequential processing
            for i, path in enumerate(file_paths):
                if self.on_progress:
                    self.on_progress(i + 1, len(file_paths), Path(path).name)
                
                result = self._process_single_file(path, params)
                
                if result.success:
                    successful_results.append(result)
                else:
                    failed_results.append(result)
                
                if self.on_file_complete:
                    self.on_file_complete(result)
        
        end_time = time.time()
        
        logger.info(
            f"Batch complete: {len(successful_results)} succeeded, "
            f"{len(failed_results)} failed in {end_time - start_time:.2f}s"
        )
        
        return BatchAnalysisResult(
            successful_results=successful_results,
            failed_results=failed_results,
            parameters=params,
            start_time=start_time,
            end_time=end_time
        )
    
    def _process_single_file(self,
                           file_path: str,
                           params: AnalysisParameters) -> FileAnalysisResult:
        """
        Process a single file.
        
        Args:
            file_path: Path to file
            params: Analysis parameters
            
        Returns:
            FileAnalysisResult with status
        """
        base_name = self._clean_filename(file_path)
        start_time = time.time()
        
        try:
            # Load dataset
            dataset = self.data_manager.load_dataset(file_path, self.channel_definitions)
            
            # Create analysis manager for this file
            analysis_manager = AnalysisManager(self.channel_definitions)
            
            # Perform analysis
            analysis_result = analysis_manager.analyze(dataset, params)
            
            # Get export table
            export_table = analysis_manager.get_export_table(dataset, params)
            
            processing_time = time.time() - start_time
            
            return FileAnalysisResult(
                file_path=file_path,
                base_name=base_name,
                success=True,
                x_data=analysis_result.x_data,
                y_data=analysis_result.y_data,
                x_data2=analysis_result.x_data2 if params.use_dual_range else None,
                y_data2=analysis_result.y_data2 if params.use_dual_range else None,
                export_table=export_table,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process {base_name}: {e}")
            return FileAnalysisResult(
                file_path=file_path,
                base_name=base_name,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def export_results(self,
                      batch_result: BatchAnalysisResult,
                      output_dir: str) -> BatchExportResult:
        """
        Export all successful results to CSV files.
        
        Args:
            batch_result: Batch analysis results
            output_dir: Output directory
            
        Returns:
            BatchExportResult with export status
        """
        export_results = []
        total_records = 0
        
        for file_result in batch_result.successful_results:
            if file_result.export_table:
                output_path = Path(output_dir) / f"{file_result.base_name}.csv"
                
                # Export using DataManager
                export_result = self.data_manager.export_to_csv(
                    file_result.export_table,
                    str(output_path)
                )
                
                export_results.append(export_result)
                if export_result.success:
                    total_records += export_result.records_exported
        
        logger.info(f"Exported {len(export_results)} files, {total_records} total records")
        
        return BatchExportResult(
            export_results=export_results,
            output_directory=output_dir,
            total_records=total_records
        )
    
    @staticmethod
    def _clean_filename(file_path: str) -> str:
        """
        Clean filename for display/export.
        
        Args:
            file_path: Full file path
            
        Returns:
            Cleaned filename without extension and brackets
        """
        stem = Path(file_path).stem
        # Remove bracketed content
        cleaned = re.sub(r"\[.*?\]", "", stem).strip()
        return cleaned