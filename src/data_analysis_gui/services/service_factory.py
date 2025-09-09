"""
Service factory for dependency injection and wiring.

This module provides factory functions to create properly configured services
with all their dependencies injected, maintaining clean architecture while
integrating with the existing codebase.

Phase 5 Refactor: Created to wire together the refactored services.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Optional

from data_analysis_gui.services.dataset_service import DatasetService
from data_analysis_gui.services.export_service import ExportService
from data_analysis_gui.infrastructure.file_io import (
    FileDatasetLoader, CsvFileWriter, FileSystemOperations, PathUtilities
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class ServiceFactory:
    """
    Factory for creating configured service instances.
    
    This factory creates services with proper dependency injection,
    allowing for easy testing and configuration changes.
    """
    
    @staticmethod
    def create_dataset_service() -> DatasetService:
        """
        Create a DatasetService with default infrastructure implementations.
        
        Returns:
            Configured DatasetService instance
        """
        logger.debug("Creating DatasetService with default dependencies")
        
        # Create infrastructure implementations
        dataset_loader = FileDatasetLoader()
        file_system = FileSystemOperations()
        
        # Create and return service
        return DatasetService(
            dataset_loader=dataset_loader,
            file_system=file_system
        )
    
    @staticmethod
    def create_export_service() -> ExportService:
        """
        Create an ExportService with default infrastructure implementations.
        
        Returns:
            Configured ExportService instance
        """
        logger.debug("Creating ExportService with default dependencies")
        
        # Create infrastructure implementations
        file_writer = CsvFileWriter()
        file_system = FileSystemOperations()
        path_utilities = PathUtilities(file_system)
        
        # Create and return service
        return ExportService(
            file_writer=file_writer,
            file_system=file_system,
            path_utilities=path_utilities
        )
    
    @staticmethod
    def create_services():
        """
        Create all services with proper wiring.
        
        Returns:
            Tuple of (DatasetService, ExportService)
        """
        return (
            ServiceFactory.create_dataset_service(),
            ServiceFactory.create_export_service()
        )
    
    @staticmethod
    def create_batch_service(dataset_service, analysis_service, 
                            export_service, channel_definitions):
        """Create a BatchService with dependencies."""
        from data_analysis_gui.services.batch_service import BatchService
        return BatchService(
            dataset_service,
            analysis_service,
            export_service,
            channel_definitions
        )


# Compatibility layer for existing code
def create_export_service_for_analysis() -> ExportService:
    """
    Create an ExportService for use with AnalysisService.
    
    This function provides compatibility with the existing architecture
    where AnalysisService expects an ExportService instance.
    
    Returns:
        Configured ExportService instance
    """
    return ServiceFactory.create_export_service()


def create_loader_service() -> DatasetService:
    """
    Create a DatasetService (replacement for LoaderService).
    
    This provides a migration path from the old LoaderService
    to the new DatasetService with proper separation of concerns.
    
    Returns:
        Configured DatasetService instance
    """
    return ServiceFactory.create_dataset_service()