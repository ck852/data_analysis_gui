"""
Business layer service for dataset operations.

This service handles the business logic for loading and validating datasets,
delegating infrastructure concerns to injected dependencies.

Phase 5 Refactor: Created to properly separate business logic from infrastructure.

Author: Data Analysis GUI Contributors
License: MIT
"""

from typing import Optional

from data_analysis_gui.core.dataset import ElectrophysiologyDataset
from data_analysis_gui.core.channel_definitions import ChannelDefinitions
from data_analysis_gui.core.interfaces import IDatasetLoader, IFileSystem
from data_analysis_gui.core.exceptions import (
    ValidationError, FileError, DataError, 
    validate_not_none, validate_file_exists
)
from data_analysis_gui.config.logging import get_logger, log_performance

logger = get_logger(__name__)


class DatasetService:
    """
    Business service for dataset operations.
    
    This service contains only business logic, delegating all I/O operations
    to injected dependencies. This follows the Dependency Inversion Principle
    and makes the service easily testable.
    """
    
    def __init__(self, 
                 dataset_loader: IDatasetLoader,
                 file_system: IFileSystem):
        """
        Initialize with injected dependencies.
        
        Args:
            dataset_loader: Implementation for loading datasets
            file_system: Implementation for file system operations
        """
        validate_not_none(dataset_loader, "dataset_loader")
        validate_not_none(file_system, "file_system")
        
        self.dataset_loader = dataset_loader
        self.file_system = file_system
        
        logger.info("DatasetService initialized")
    
    def load_dataset(self, 
                    filepath: str,
                    channel_config: Optional[ChannelDefinitions] = None) -> ElectrophysiologyDataset:
        """
        Load and validate a dataset from a file.
        
        This method contains the business logic for dataset loading:
        1. Validate the file path
        2. Check file accessibility
        3. Load the dataset
        4. Validate the loaded data
        
        Args:
            filepath: Path to the data file
            channel_config: Optional channel configuration
            
        Returns:
            Validated dataset ready for analysis
            
        Raises:
            ValidationError: If inputs are invalid
            FileError: If file is not accessible
            DataError: If dataset is invalid or empty
        """
        # Business rule: filepath must be provided
        validate_not_none(filepath, "filepath")
        if not filepath.strip():
            raise ValidationError("Filepath cannot be empty")
        
        # Business rule: file must exist and be readable
        if not self.file_system.exists(filepath):
            raise FileError(
                f"File not found: {filepath}",
                details={'filepath': filepath}
            )
        
        if not self.file_system.is_readable(filepath):
            raise FileError(
                f"File is not readable: {filepath}",
                details={'filepath': filepath, 'permission': 'read'}
            )
        
        # Get file info for logging and validation
        file_info = self.file_system.get_info(filepath)
        file_name = file_info['name']
        file_size = file_info['size']
        
        # Business rule: file must not be empty
        if file_size == 0:
            raise DataError(
                f"File is empty: {file_name}",
                details={'filepath': filepath, 'size': 0}
            )
        
        logger.info(f"Loading dataset from {file_name} (size: {file_size:,} bytes)")
        
        # Delegate actual loading to infrastructure
        with log_performance(logger, f"load dataset from {file_name}"):
            dataset = self.dataset_loader.load(filepath, channel_config)
        
        # Business validation of loaded dataset
        self._validate_dataset(dataset, file_name)
        
        # Log success metrics
        sweep_count = dataset.sweep_count() if hasattr(dataset, 'sweep_count') else 'unknown'
        channel_count = dataset.channel_count() if hasattr(dataset, 'channel_count') else 'unknown'
        
        logger.info(
            f"Successfully loaded {file_name}: "
            f"{sweep_count} sweeps, {channel_count} channels"
        )
        
        return dataset
    
    def _validate_dataset(self, dataset: ElectrophysiologyDataset, file_name: str) -> None:
        """
        Apply business rules to validate a loaded dataset.
        
        Args:
            dataset: Dataset to validate
            file_name: Name of source file for error messages
            
        Raises:
            DataError: If dataset violates business rules
        """
        # Business rule: dataset must not be None
        if dataset is None:
            raise DataError(
                f"Failed to load dataset from {file_name}",
                details={'file_name': file_name}
            )
        
        # Business rule: dataset must contain data
        if dataset.is_empty():
            sweep_count = dataset.sweep_count() if hasattr(dataset, 'sweep_count') else 0
            raise DataError(
                f"Dataset is empty - no valid sweeps found in {file_name}",
                details={
                    'file_name': file_name,
                    'sweep_count': sweep_count
                }
            )
        
        # Business rule: dataset must have at least one channel
        if hasattr(dataset, 'channel_count'):
            channel_count = dataset.channel_count()
            if channel_count < 1:
                raise DataError(
                    f"Dataset has no channels in {file_name}",
                    details={
                        'file_name': file_name,
                        'channel_count': channel_count
                    }
                )
        
        logger.debug(f"Dataset validation passed for {file_name}")