"""
Dataset identifier with flexible, layered fingerprinting.

This module provides content-based identification for datasets using a layered
approach that gracefully handles varying levels of information availability.
Replaces the buggy memory-address-based caching with robust content hashing.

Phase 5 Refactor: Created to fix cache key generation bugs and provide
deterministic, content-based dataset identification.

Author: Data Analysis GUI Contributors
License: MIT
"""

import hashlib
import os
from dataclasses import dataclass
from enum import Flag, auto
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class FingerprintLevel(Flag):
    """
    Flags for fingerprint detail levels.
    Can be combined with | operator for multiple levels.
    
    Example:
        level = FingerprintLevel.FAST | FingerprintLevel.HEADER
    """
    NONE = 0
    FAST = auto()           # File metadata: path, mtime, size
    HEADER = auto()         # Dataset structure: sweeps, channels, samples
    CONTENT_PARTIAL = auto() # Sampled content hash
    CONTENT_FULL = auto()   # Complete content hash (expensive)
    
    # Convenience combinations
    DEFAULT = FAST | HEADER
    ROBUST = FAST | HEADER | CONTENT_PARTIAL
    PARANOID = FAST | HEADER | CONTENT_PARTIAL | CONTENT_FULL


@dataclass
class DatasetFingerprint:
    """
    Structured fingerprint containing all available dataset information.
    Fields are Optional to handle missing information gracefully.
    """
    # FAST level - file metadata
    file_path: Optional[str] = None
    file_mtime: Optional[float] = None
    file_size: Optional[int] = None
    
    # HEADER level - dataset structure
    sweep_count: Optional[int] = None
    channel_count: Optional[int] = None
    samples_per_sweep: Optional[int] = None
    sampling_rate_hz: Optional[float] = None
    max_time_ms: Optional[float] = None
    
    # CONTENT level - data hashes
    content_partial_hash: Optional[str] = None
    content_full_hash: Optional[str] = None
    
    # Metadata
    fingerprint_level: Optional[FingerprintLevel] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def to_deterministic_string(self) -> str:
        """
        Convert to deterministic string for hashing.
        Sorts keys to ensure same fingerprint produces same string.
        """
        data = self.to_dict()
        # Sort keys for deterministic ordering
        sorted_items = sorted(data.items())
        # Format as key=value pairs
        parts = [f"{k}={v}" for k, v in sorted_items]
        return "|".join(parts)


class DatasetIdentifier:
    """
    Generates content-based identifiers for electrophysiology datasets.
    
    Uses a flexible, layered approach that works with whatever information
    is available from the dataset. Higher levels provide more confidence
    but require more computation.
    
    Example:
        >>> identifier = DatasetIdentifier()
        >>> 
        >>> # Fast identification (file metadata only)
        >>> fast_id = identifier.get_identifier(dataset, FingerprintLevel.FAST)
        >>> 
        >>> # Robust identification (includes content sampling)
        >>> robust_id = identifier.get_identifier(dataset, FingerprintLevel.ROBUST)
        >>> 
        >>> # Get structured fingerprint for debugging
        >>> fingerprint = identifier.get_fingerprint(dataset, FingerprintLevel.DEFAULT)
    """
    
    def __init__(self, partial_sample_size: int = 10240, sample_points: int = 5):
        """
        Initialize the identifier.
        
        Args:
            partial_sample_size: Bytes to sample for CONTENT_PARTIAL (default 10KB)
            sample_points: Number of positions to sample from (default 5)
        """
        self.partial_sample_size = partial_sample_size
        self.sample_points = sample_points
        logger.debug(f"DatasetIdentifier initialized with sample_size={partial_sample_size}")
    
    def get_identifier(
        self,
        dataset: Any,
        level: FingerprintLevel = FingerprintLevel.DEFAULT
    ) -> str:
        """
        Generate a unique identifier for the dataset.
        
        Args:
            dataset: ElectrophysiologyDataset instance
            level: Fingerprint detail level(s) to include
            
        Returns:
            Hexadecimal hash string identifier
        """
        fingerprint = self.get_fingerprint(dataset, level)
        identifier_string = fingerprint.to_deterministic_string()
        return self.create_hash(identifier_string)
    
    def get_fingerprint(
        self,
        dataset: Any,
        level: FingerprintLevel = FingerprintLevel.DEFAULT
    ) -> DatasetFingerprint:
        """
        Extract a structured fingerprint from the dataset.
        
        Args:
            dataset: ElectrophysiologyDataset instance
            level: Fingerprint detail level(s) to include
            
        Returns:
            DatasetFingerprint with available information
        """
        fingerprint = DatasetFingerprint(fingerprint_level=level)
        
        # Extract each level if requested
        if level & FingerprintLevel.FAST:
            self._add_fast_info(dataset, fingerprint)
        
        if level & FingerprintLevel.HEADER:
            self._add_header_info(dataset, fingerprint)
        
        if level & FingerprintLevel.CONTENT_PARTIAL:
            self._add_partial_content(dataset, fingerprint)
        
        if level & FingerprintLevel.CONTENT_FULL:
            self._add_full_content(dataset, fingerprint)
        
        logger.debug(f"Generated fingerprint with {len(fingerprint.to_dict())} fields")
        return fingerprint
    
    def create_hash(self, content: str) -> str:
        """
        Create a consistent hash from string content.
        
        Args:
            content: String to hash
            
        Returns:
            MD5 hexadecimal hash string
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    # =========================================================================
    # Private extraction methods - each handles missing info gracefully
    # =========================================================================
    
    def _add_fast_info(self, dataset: Any, fingerprint: DatasetFingerprint) -> None:
        """Extract file-level metadata if available."""
        # Try to get file path from dataset
        file_path = None
        if hasattr(dataset, 'source_file'):
            file_path = dataset.source_file
        elif hasattr(dataset, 'filepath'):
            file_path = dataset.filepath
        elif hasattr(dataset, 'filename'):
            file_path = dataset.filename
        
        if file_path and os.path.exists(file_path):
            fingerprint.file_path = str(Path(file_path).resolve())
            
            try:
                stat = os.stat(file_path)
                fingerprint.file_mtime = stat.st_mtime
                fingerprint.file_size = stat.st_size
            except OSError as e:
                logger.debug(f"Could not stat file {file_path}: {e}")
    
    def _add_header_info(self, dataset: Any, fingerprint: DatasetFingerprint) -> None:
        """Extract dataset structure information."""
        # Sweep count - most datasets have this
        if hasattr(dataset, 'sweep_count'):
            try:
                fingerprint.sweep_count = dataset.sweep_count()
            except:
                pass
        elif hasattr(dataset, 'n_sweeps'):
            fingerprint.sweep_count = dataset.n_sweeps
        
        # Channel count
        if hasattr(dataset, 'channel_count'):
            try:
                fingerprint.channel_count = dataset.channel_count()
            except:
                pass
        elif hasattr(dataset, 'n_channels'):
            fingerprint.channel_count = dataset.n_channels
        
        # Samples per sweep (if uniform)
        if hasattr(dataset, 'samples_per_sweep'):
            try:
                fingerprint.samples_per_sweep = dataset.samples_per_sweep()
            except:
                # Try to get from first sweep
                self._try_get_samples_from_sweep(dataset, fingerprint)
        else:
            self._try_get_samples_from_sweep(dataset, fingerprint)
        
        # Sampling rate (optional enhancement)
        if hasattr(dataset, 'sampling_rate'):
            try:
                fingerprint.sampling_rate_hz = dataset.sampling_rate
            except:
                pass
        elif hasattr(dataset, 'sample_rate'):
            fingerprint.sampling_rate_hz = dataset.sample_rate
        
        # Max time (optional enhancement)
        if hasattr(dataset, 'get_max_sweep_time'):
            try:
                fingerprint.max_time_ms = dataset.get_max_sweep_time()
            except:
                pass
    
    def _try_get_samples_from_sweep(self, dataset: Any, fingerprint: DatasetFingerprint) -> None:
        """Try to get sample count from first sweep."""
        try:
            if hasattr(dataset, 'sweeps') and hasattr(dataset, 'get_channel_vector'):
                sweeps = dataset.sweeps()
                if sweeps:
                    time_data, _ = dataset.get_channel_vector(sweeps[0], 0)
                    if time_data is not None:
                        fingerprint.samples_per_sweep = len(time_data)
        except:
            pass  # Gracefully handle any errors
    
    def _add_partial_content(self, dataset: Any, fingerprint: DatasetFingerprint) -> None:
        """Add partial content hash by sampling strategic positions."""
        try:
            samples = self._collect_content_samples(dataset)
            if samples:
                # Create hash from samples
                hasher = hashlib.md5()
                for sample in samples:
                    hasher.update(sample.tobytes())
                fingerprint.content_partial_hash = hasher.hexdigest()
        except Exception as e:
            logger.debug(f"Could not create partial content hash: {e}")
    
    def _add_full_content(self, dataset: Any, fingerprint: DatasetFingerprint) -> None:
        """Add complete content hash (expensive operation)."""
        logger.warning("Full content hashing requested - this may be slow for large datasets")
        
        try:
            hasher = hashlib.md5()
            sweep_count = 0
            
            # Hash all sweep data
            if hasattr(dataset, 'sweeps') and hasattr(dataset, 'get_sweep_data'):
                for sweep in dataset.sweeps():
                    try:
                        sweep_data = dataset.get_sweep_data(sweep)
                        if sweep_data is not None:
                            hasher.update(sweep_data.tobytes())
                            sweep_count += 1
                    except:
                        continue
            
            if sweep_count > 0:
                fingerprint.content_full_hash = hasher.hexdigest()
                logger.debug(f"Created full content hash from {sweep_count} sweeps")
        except Exception as e:
            logger.debug(f"Could not create full content hash: {e}")
    
    def _collect_content_samples(self, dataset: Any) -> List[np.ndarray]:
        """
        Collect strategic samples from dataset content.
        
        Samples from:
        - First sweep
        - Last sweep
        - Middle sweep(s)
        - Random positions if many sweeps
        
        Returns:
            List of numpy arrays containing sampled data
        """
        samples = []
        
        try:
            if not hasattr(dataset, 'sweeps'):
                return samples
            
            sweeps = dataset.sweeps()
            if not sweeps:
                return samples
            
            # Determine which sweeps to sample
            sweep_indices = self._get_sample_indices(len(sweeps))
            
            for idx in sweep_indices:
                if idx < len(sweeps):
                    sweep_id = sweeps[idx]
                    sample = self._sample_sweep_data(dataset, sweep_id)
                    if sample is not None:
                        samples.append(sample)
        
        except Exception as e:
            logger.debug(f"Error collecting content samples: {e}")
        
        return samples
    
    def _get_sample_indices(self, total_count: int) -> List[int]:
        """
        Get indices of items to sample from a collection.
        
        Strategy:
        - Always include first and last
        - Include middle
        - Include evenly spaced points for larger collections
        
        Args:
            total_count: Total number of items
            
        Returns:
            List of indices to sample
        """
        if total_count <= 0:
            return []
        
        if total_count <= self.sample_points:
            # Sample everything if small enough
            return list(range(total_count))
        
        # Always include first and last
        indices = [0, total_count - 1]
        
        # Add evenly spaced middle points
        remaining_points = self.sample_points - 2
        if remaining_points > 0:
            step = (total_count - 1) / (remaining_points + 1)
            for i in range(1, remaining_points + 1):
                idx = int(i * step)
                if idx not in indices:
                    indices.append(idx)
        
        return sorted(indices)
    
    def _sample_sweep_data(self, dataset: Any, sweep_id: str) -> Optional[np.ndarray]:
        """
        Extract a sample of data from a single sweep.
        
        Args:
            dataset: Dataset to sample from
            sweep_id: Identifier of sweep to sample
            
        Returns:
            Numpy array with sampled data, or None if extraction fails
        """
        try:
            # Try different methods to get sweep data
            if hasattr(dataset, 'get_sweep_data'):
                data = dataset.get_sweep_data(sweep_id)
            elif hasattr(dataset, 'get_channel_vector'):
                # Get first channel as sample
                _, data = dataset.get_channel_vector(sweep_id, 0)
            else:
                return None
            
            if data is None or len(data) == 0:
                return None
            
            # Sample a portion of the data
            if len(data) <= self.partial_sample_size:
                return data
            
            # Take samples from beginning, middle, and end
            indices = self._get_sample_indices(len(data))
            sampled = data[indices]
            
            return sampled
            
        except Exception as e:
            logger.debug(f"Could not sample sweep {sweep_id}: {e}")
            return None
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DatasetIdentifier(partial_sample_size={self.partial_sample_size}, "
            f"sample_points={self.sample_points})"
        )