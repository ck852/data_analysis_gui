# services/batch_service.py

from typing import List, Callable, Optional
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_analysis_gui.core.params import AnalysisParameters
from data_analysis_gui.core.models import (
    FileAnalysisResult, BatchAnalysisResult, BatchExportResult
)
from data_analysis_gui.services.dataset_service import DatasetService
from data_analysis_gui.services.analysis_service import AnalysisService
from data_analysis_gui.services.export_service import ExportService
from data_analysis_gui.core.exceptions import ValidationError
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class BatchService:
    """
    Service for batch processing multiple files with the same parameters.
    
    This service orchestrates the analysis of multiple files using the
    existing single-file infrastructure, maintaining clean architecture
    and fail-closed principles.
    """
    
    def __init__(self,
                 dataset_service: DatasetService,
                 analysis_service: AnalysisService,
                 export_service: ExportService,
                 channel_definitions):
        """Initialize with injected dependencies."""
        self.dataset_service = dataset_service
        self.analysis_service = analysis_service
        self.export_service = export_service
        self.channel_definitions = channel_definitions
        
        # Progress callback (set by GUI)
        self.on_progress: Optional[Callable[[int, int, str], None]] = None
        self.on_file_complete: Optional[Callable[[FileAnalysisResult], None]] = None
    
    def analyze_files(self,
                      file_paths: List[str],
                      params: AnalysisParameters,
                      parallel: bool = False,
                      max_workers: int = 4) -> BatchAnalysisResult:
        """
        Analyze multiple files with the same parameters.
        
        Args:
            file_paths: List of file paths to analyze
            params: Analysis parameters to use for all files
            parallel: Whether to process files in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            BatchAnalysisResult with all results
        """
        if not file_paths:
            raise ValidationError("No files provided for batch analysis")
        
        logger.info(f"Starting batch analysis of {len(file_paths)} files")
        start_time = time.time()
        
        successful_results = []
        failed_results = []
        
        if parallel and len(file_paths) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._analyze_single_file, path, params): path
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
                        logger.error(f"Unexpected error processing {path}: {e}")
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
                
                result = self._analyze_single_file(path, params)
                
                if result.success:
                    successful_results.append(result)
                else:
                    failed_results.append(result)
                
                if self.on_file_complete:
                    self.on_file_complete(result)
        
        end_time = time.time()
        
        logger.info(
            f"Batch analysis complete: {len(successful_results)} succeeded, "
            f"{len(failed_results)} failed in {end_time - start_time:.2f}s"
        )
        
        return BatchAnalysisResult(
            successful_results=successful_results,
            failed_results=failed_results,
            parameters=params,
            start_time=start_time,
            end_time=end_time
        )
    
    def _analyze_single_file(self,
                            file_path: str,
                            params: AnalysisParameters) -> FileAnalysisResult:
        """Analyze a single file and return result."""
        base_name = Path(file_path).stem
        start_time = time.time()
        
        try:
            # Load dataset
            dataset = self.dataset_service.load_dataset(
                file_path, self.channel_definitions
            )
            
            # Perform analysis
            analysis_result = self.analysis_service.perform_analysis(
                dataset, params
            )
            
            # Get export table
            export_table = self.analysis_service.get_export_table(
                dataset, params
            )
            
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
            logger.error(f"Failed to analyze {base_name}: {e}")
            return FileAnalysisResult(
                file_path=file_path,
                base_name=base_name,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def export_batch_results(self,
                            batch_result: BatchAnalysisResult,
                            output_directory: str) -> BatchExportResult:
        """Export all successful results to CSV files."""
        export_results = []
        total_records = 0
        
        for file_result in batch_result.successful_results:
            if file_result.export_table:
                output_path = Path(output_directory) / f"{file_result.base_name}.csv"
                export_result = self.export_service.export_analysis_data(
                    file_result.export_table,
                    str(output_path)
                )
                export_results.append(export_result)
                if export_result.success:
                    total_records += export_result.records_exported
        
        return BatchExportResult(
            export_results=export_results,
            output_directory=output_directory,
            total_records=total_records
        )